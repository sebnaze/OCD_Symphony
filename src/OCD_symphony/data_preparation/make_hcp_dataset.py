# Make HCP dataset from pre-saved patients' tensor files
# Optimized version: parallel loading, one-time preprocessing, full RAM/VRAM caching.
#
# Key changes vs. original:
#   1. Parallel file loading  (ThreadPoolExecutor – I/O bound)
#   2. All data loaded & preprocessed ONCE at construction time
#   3. __getitem__ is O(1) – just an index into pre-allocated tensors
#   4. Stats computed with vectorised torch ops (no Dask DataFrames)
#   5. Optional: tensors pinned in GPU VRAM (device='cuda')
#   6. DataLoader num_workers auto-set to 0 for GPU tensors
#   7. Subject-level splitting: K-fold CV and single train/val/test split
#      guarantee that every sample from a given subject lands in exactly
#      one partition (no leakage between train and val/test).

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

import OCD_symphony
from OCD_symphony.utils import proj_dir


# ---------------------------------------------------------------------------
# File-level loader (runs in threads – keep it side-effect-free)
# ---------------------------------------------------------------------------

def _load_raw_file(fname: str) -> Optional[Dict]:
    """
    Load one .pt file from disk and return minimally reshaped tensors + metadata.
    No normalisation or ROI reindexing yet – those happen once on the full batch.
    """
    try:
        tnsr = torch.load(fname, map_location="cpu", weights_only=True)
        subj = os.path.basename(fname)[:-3]

        fmri = tnsr["fmri_data"]  # e.g. (sessions, runs, 450, 200)
        fc   = tnsr["fc_data"]

        # Flatten leading dims → (n_samples, n_rois, T_or_rois)
        n = fmri.shape[0] * fmri.shape[1]
        fmri = fmri.reshape(n, fmri.shape[-2], fmri.shape[-1])  # (n, 450, 200)
        fc   = fc.reshape(  n, fc.shape[-2],   fc.shape[-1])    # (n, 450, 450)

        age, gender = load_hcp_age_gender([subj])
        return dict(
            fmri=fmri,
            fc=fc,
            subj=subj,
            n=n,
            age=age[0]    if age    else "unknown",
            gender=gender[0] if gender else "unknown",
        )
    except Exception as exc:
        print(f"[WARN] Skipping {fname}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FMRIDataset(Dataset):
    """
    Fully RAM- (or VRAM-) cached map dataset.

    All data is loaded and preprocessed once at construction time.
    __getitem__ is O(1) – just an index into pre-allocated contiguous tensors.

    Returns: (fmri_data, fc_data, G_tensor, label)
    """

    def __init__(
        self,
        fmri_data:  torch.Tensor,  # (N, 450, 200)  – already normalised
        fc_data:    torch.Tensor,  # (N, 450, 450)
        G_tensor:   torch.Tensor,  # (N, 450, 10)
        ages:       np.ndarray,
        genders:    np.ndarray,
        subjs:      np.ndarray,
        label:      str,
    ) -> None:
        self.fmri_data = fmri_data
        self.fc_data   = fc_data
        self.G_tensor  = G_tensor
        self.subjs     = subjs
        self.label     = label

        self.label_encoder = LabelEncoder()
        if label == "age":
            self.encoded_labels = torch.from_numpy(
                self.label_encoder.fit_transform(ages)
            ).long()
        elif label == "gender":
            self.encoded_labels = torch.from_numpy(
                self.label_encoder.fit_transform(genders)
            ).long()
        else:
            raise ValueError(f"Unknown label: {label!r}. Expected 'age' or 'gender'.")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.fmri_data.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.fmri_data[idx],
            self.fc_data[idx],
            self.G_tensor[idx],
            self.encoded_labels[idx],
        )


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def make_hcp_dataset_from_pt(
    path:        str,
    label:       str,
    include:     str = "",
    n_subjs:     Optional[int] = None,
    start_idx:   int = 0,
    num_io_workers: int = 16,     # threads for parallel disk reads
    device:      str = "cpu",     # "cuda" to keep tensors in VRAM
    approx_quantile_samples: Optional[int] = 200_000,  # None = exact (slower)
) -> Tuple[FMRIDataset, np.ndarray]:
    """
    Build a fully cached FMRIDataset.

    Pipeline
    --------
    1. Discover .pt files
    2. Load all files in parallel (ThreadPoolExecutor)
    3. Concatenate into contiguous tensors
    4. ROI reindexing (once, on the whole dataset)
    5. Per-ROI robust statistics  (median / IQR / mean / std)
    6. Compute W, B and apply normalisation in-place
    7. Optionally move everything to GPU

    Parameters
    ----------
    approx_quantile_samples : int or None
        If an int, draw this many time-point samples per ROI before computing
        quantiles (much faster, negligibly different in practice for >1M pts).
        Pass None for the exact calculation.
    """

    # ── 1. File discovery ────────────────────────────────────────────────────
    fnames = sorted([
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".pt") and include in f
    ])
    if n_subjs is not None:
        fnames = fnames[start_idx : start_idx + n_subjs]
    print(f"Found {len(fnames)} files to load.")

    # ── 2. Parallel file load ─────────────────────────────────────────────────
    records: List[Dict] = []
    with ThreadPoolExecutor(max_workers=num_io_workers) as ex:
        futures = {ex.submit(_load_raw_file, f): f for f in fnames}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Loading files"):
            rec = fut.result()
            if rec is not None:
                records.append(rec)

    if not records:
        raise RuntimeError("No files loaded successfully.")

    # Keep deterministic order (as_completed is non-deterministic)
    records.sort(key=lambda r: r["subj"])

    n_unique_subjs  = len(records)
    n_total_samples = sum(r["n"] for r in records)
    print(
        f"Loaded {n_unique_subjs} subjects | "
        f"{n_total_samples} total samples | "
        f"avg {n_total_samples / n_unique_subjs:.1f} samples/subject"
    )

    # ── 3. Concatenate ────────────────────────────────────────────────────────
    print("Concatenating tensors …")
    all_fmri    = torch.cat([r["fmri"] for r in records], dim=0).float()  # (N, 450, 200)
    all_fc      = torch.cat([r["fc"]   for r in records], dim=0).float()  # (N, 450, 450)
    all_subjs   = np.concatenate([[r["subj"]]   * r["n"] for r in records])
    all_ages    = np.concatenate([[r["age"]]    * r["n"] for r in records])
    all_genders = np.concatenate([[r["gender"]] * r["n"] for r in records])
    del records  # free per-file dicts

    N = all_fmri.shape[0]

    # ── 4. ROI reindexing (once) ──────────────────────────────────────────────
    # Order: sub-ctx_R, sub-ctx_L, ctx_L, ctx_R
    roi_inds = torch.from_numpy(
        np.concatenate([np.arange(425, 450), np.arange(200, 225),
                        np.arange(0,   200), np.arange(225, 425)])
    )
    all_fmri = all_fmri[:, roi_inds, :]                          # (N, 450, 200)
    all_fc   = all_fc[:, roi_inds, :][:, :, roi_inds]            # (N, 450, 450)

    # ── 5. Robust statistics (post-reindex, single pass) ─────────────────────
    # Reshape to (450, N*200) so stats are over all time-points across all samples.
    print("Computing robust statistics …")
    fmri_flat = all_fmri.permute(1, 0, 2).reshape(450, -1)      # (450, N*200)

    if approx_quantile_samples is not None and fmri_flat.shape[1] > approx_quantile_samples:
        # Fast path: random sub-sample columns per ROI (results are very close to exact)
        col_idx = torch.randperm(fmri_flat.shape[1])[:approx_quantile_samples]
        q = torch.quantile(
            fmri_flat[:, col_idx],
            torch.tensor([0.25, 0.5, 0.75]),
            dim=1,
        )
    else:
        q = torch.quantile(fmri_flat, torch.tensor([0.25, 0.5, 0.75]), dim=1)

    median = q[1].numpy()                                        # (450,)
    iqr    = (q[2] - q[0]).numpy()                               # (450,)
    iqr    = np.where(iqr == 0, 1e-8, iqr)

    raw_mu  = fmri_flat.mean(dim=1).numpy()                      # (450,)
    raw_std = fmri_flat.std(dim=1).numpy()                       # (450,)
    del fmri_flat                                                 # free ~3 GB

    # ── 6. Compute W, B and normalise ─────────────────────────────────────────
    mu_robust    = (raw_mu - median) / iqr
    sigma_robust = raw_std / iqr
    sigma_robust = np.where(sigma_robust == 0, 1e-8, sigma_robust)

    denom = iqr * sigma_robust
    W = torch.from_numpy(1.0 / denom).float()                    # (450,)
    B = torch.from_numpy((-median / denom) - (mu_robust / sigma_robust)).float()

    print("Applying normalisation …")
    all_fmri = all_fmri * W.reshape(1, -1, 1) + B.reshape(1, -1, 1)

    # ── 7. G_tensor placeholder ───────────────────────────────────────────────
    all_G = torch.randn(N, 450, 10)

    # ── 8. Optional: move to GPU ──────────────────────────────────────────────
    if device != "cpu":
        print(f"Transferring tensors to {device} …")
        all_fmri = all_fmri.to(device, non_blocking=True)
        all_fc   = all_fc.to(device,   non_blocking=True)
        all_G    = all_G.to(device,    non_blocking=True)
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()

    ds = FMRIDataset(all_fmri, all_fc, all_G, all_ages, all_genders, all_subjs, label)
    return ds, all_subjs


# ---------------------------------------------------------------------------
# Internal helpers for subject-level splitting
# ---------------------------------------------------------------------------

def _subject_level_arrays(ds: FMRIDataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (unique_subjects, per_subject_label).

    All samples from one subject share the same label (age/gender is fixed per
    person), so we take the label of the first sample for each subject.
    """
    subjs  = ds.subjs
    labels = ds.encoded_labels.numpy()
    unique_subjs = np.unique(subjs)
    subj_labels  = np.array([labels[subjs == s][0] for s in unique_subjs])
    return unique_subjs, subj_labels


def _sample_indices_for_subjects(
    all_subjs: np.ndarray,
    selected:  np.ndarray,
) -> np.ndarray:
    """Return the sample-level integer indices that belong to `selected` subjects."""
    return np.where(np.isin(all_subjs, selected))[0]


def _assert_no_subject_leakage(*subject_arrays: np.ndarray) -> None:
    """Raise if any two arrays share a subject ID."""
    for i, a in enumerate(subject_arrays):
        for j, b in enumerate(subject_arrays):
            if i >= j:
                continue
            overlap = np.intersect1d(a, b)
            assert len(overlap) == 0, (
                f"Subject leakage detected between partitions {i} and {j}: "
                f"{overlap[:5]} …"
            )


def _is_gpu_dataset(ds: Dataset) -> bool:
    root = ds.dataset if isinstance(ds, Subset) else ds  # type: ignore[attr-defined]
    return isinstance(root, FMRIDataset) and root.fmri_data.is_cuda


def _print_split_table(
    ds:          FMRIDataset,
    split_names: List[str],
    subj_groups: List[np.ndarray],   # per-split unique subjects
    idx_groups:  List[np.ndarray],   # per-split sample indices
) -> None:
    class_names = list(ds.label_encoder.classes_)
    labels_all  = ds.encoded_labels.numpy()

    header_cls  = "  ".join(f"n_{c}" for c in class_names)
    print(f"\n{'Split':>6}  {'Subjects':>8}  {'Samples':>8}  {header_cls}")
    print("─" * (30 + 8 * len(class_names)))

    for name, subjs, idx in zip(split_names, subj_groups, idx_groups):
        lbl = labels_all[idx]
        cc  = [int(np.sum(lbl == c)) for c in range(len(class_names))]
        row_cls = "  ".join(f"{c:>{5+len(str(cls))}}" for cls, c in zip(class_names, cc))
        print(f"{name:>6}  {len(subjs):>8}  {len(idx):>8}  {row_cls}")
    print()


# ---------------------------------------------------------------------------
# Subject-level splitting API
# ---------------------------------------------------------------------------

def subject_kfold(
    ds:       FMRIDataset,
    n_splits: int = 5,
    seed:     int = 42,
) -> List[Tuple[Subset, Subset]]:
    """
    Stratified K-fold cross-validation with strict subject-level partitioning.

    Every sample from a given subject appears in **exactly one** val fold and
    never in the corresponding train fold. Class proportions are approximately
    preserved across folds via stratification.

    Strategy
    --------
    We run StratifiedGroupKFold at the *subject* level (one row per subject,
    group == subject). This is cleaner than running it at the sample level with
    groups=subjs because the fraction arithmetic is exact and the guarantee is
    trivially verifiable: if sklearn puts subject S in the val fold, every one
    of S's samples goes to val.

    Parameters
    ----------
    ds       : FMRIDataset (fully cached, with encoded_labels + subjs)
    n_splits : number of folds  (K)
    seed     : random seed passed to StratifiedGroupKFold

    Returns
    -------
    List of length n_splits, each element is (train_subset, val_subset).
    """
    unique_subjs, subj_labels = _subject_level_arrays(ds)
    n_subjs = len(unique_subjs)

    if n_splits > n_subjs:
        raise ValueError(
            f"n_splits={n_splits} > n_subjects={n_subjs}. Reduce n_splits."
        )

    # StratifiedGroupKFold on subjects: X=subjects (dummy), y=subject_label,
    # groups=subjects (each subject is its own group → no subject is split).
    sgkf   = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(sgkf.split(unique_subjs, subj_labels, groups=unique_subjs))

    # ── Print summary ─────────────────────────────────────────────────────────
    class_names = list(ds.label_encoder.classes_)
    col_w = max(len(c) for c in class_names) + 4
    header = (f"{'Fold':>4}  {'Train subjs':>11}  {'Val subjs':>9}  "
              + "  ".join(f"{'val_'+c:>{col_w}}" for c in class_names))
    print(f"\nStratified {n_splits}-fold CV  "
          f"({n_subjs} subjects, label='{ds.label}')")
    print(header)
    print("─" * len(header))

    folds: List[Tuple[Subset, Subset]] = []
    for fold_idx, (train_subj_idx, val_subj_idx) in enumerate(splits):
        train_subjs = unique_subjs[train_subj_idx]
        val_subjs   = unique_subjs[val_subj_idx]
        val_labels  = subj_labels[val_subj_idx]

        _assert_no_subject_leakage(train_subjs, val_subjs)

        train_sample_idx = _sample_indices_for_subjects(ds.subjs, train_subjs)
        val_sample_idx   = _sample_indices_for_subjects(ds.subjs, val_subjs)

        cc = [int(np.sum(val_labels == c)) for c in range(len(class_names))]
        row_cls = "  ".join(f"{c:>{col_w}}" for c in cc)
        print(f"{fold_idx+1:>4}  {len(train_subjs):>11}  "
              f"{len(val_subjs):>9}  {row_cls}")

        folds.append((Subset(ds, train_sample_idx), Subset(ds, val_sample_idx)))

    print()
    return folds


def subject_train_val_test_split(
    ds:        FMRIDataset,
    test_frac: float = 0.15,
    val_frac:  float = 0.15,
    seed:      int   = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Single deterministic train / val / test split at the subject level.

    Guarantees
    ----------
    * All samples from one subject land in exactly one partition.
    * No subject appears in both train and val/test (no leakage).
    * Class proportions are approximately preserved in every partition.

    Strategy
    --------
    Two sequential StratifiedGroupKFold passes, both operating at subject level:

      Pass 1 – carve out test subjects
        n_splits = round(1 / test_frac)   → one fold ≈ test_frac of subjects
        Take fold 0 as test; the rest become the trainval pool.

      Pass 2 – carve out val subjects from the trainval pool
        effective_val_frac = val_frac / (1 - test_frac)
        n_splits = round(1 / effective_val_frac)
        Take fold 0 as val; the rest become train.

    Parameters
    ----------
    test_frac : fraction of subjects reserved for test  (default 0.15)
    val_frac  : fraction of ALL subjects reserved for val (default 0.15)
                (internally rescaled to the trainval pool)
    seed      : random seed; val pass uses seed+1 for independence

    Returns
    -------
    (train_subset, val_subset, test_subset) – each a torch.utils.data.Subset
    """
    unique_subjs, subj_labels = _subject_level_arrays(ds)

    # ── Pass 1: carve out test subjects ──────────────────────────────────────
    n_test_splits = max(2, round(1.0 / test_frac))
    sgkf_test = StratifiedGroupKFold(
        n_splits=n_test_splits, shuffle=True, random_state=seed
    )
    trainval_subj_idx, test_subj_idx = next(
        sgkf_test.split(unique_subjs, subj_labels, groups=unique_subjs)
    )
    trainval_subjs  = unique_subjs[trainval_subj_idx]
    trainval_labels = subj_labels[trainval_subj_idx]
    test_subjs      = unique_subjs[test_subj_idx]

    # ── Pass 2: carve out val from trainval pool ──────────────────────────────
    effective_val_frac = val_frac / (1.0 - test_frac)      # re-scale to trainval pool
    n_val_splits = max(2, round(1.0 / effective_val_frac))
    sgkf_val = StratifiedGroupKFold(
        n_splits=n_val_splits, shuffle=True, random_state=seed + 1
    )
    train_subj_idx, val_subj_idx = next(
        sgkf_val.split(trainval_subjs, trainval_labels, groups=trainval_subjs)
    )
    train_subjs = trainval_subjs[train_subj_idx]
    val_subjs   = trainval_subjs[val_subj_idx]

    # ── Verify zero leakage (cheap, raises immediately if broken) ─────────────
    _assert_no_subject_leakage(train_subjs, val_subjs, test_subjs)

    # ── Map subject IDs → flat sample indices ─────────────────────────────────
    train_idx = _sample_indices_for_subjects(ds.subjs, train_subjs)
    val_idx   = _sample_indices_for_subjects(ds.subjs, val_subjs)
    test_idx  = _sample_indices_for_subjects(ds.subjs, test_subjs)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Subject-level train/val/test split  (label='{ds.label}')")
    _print_split_table(
        ds,
        split_names=["train", "val", "test"],
        subj_groups=[train_subjs, val_subjs, test_subjs],
        idx_groups =[train_idx,  val_idx,   test_idx],
    )

    return Subset(ds, train_idx), Subset(ds, val_idx), Subset(ds, test_idx)


# ---------------------------------------------------------------------------
# DataLoader factory (works with both raw FMRIDataset and Subset objects)
# ---------------------------------------------------------------------------

def _make_loader(
    ds:            Dataset,
    shuffle:       bool,
    batch_size:    int,
    num_workers:   int,
    pin_memory:    bool,
) -> DataLoader:
    if _is_gpu_dataset(ds):
        # CUDA tensors cannot be shared across worker processes
        num_workers = 0
        pin_memory  = False
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def make_dataloaders(
    train_ds:      Dataset,
    val_ds:        Dataset,
    test_ds:       Dataset,
    batch_size:    int  = 16,
    num_workers:   int  = 4,
    pin_memory:    bool = True,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Wrap three pre-split Dataset / Subset objects in DataLoaders.

    Typical usage with subject_train_val_test_split
    ------------------------------------------------
    >>> train, val, test = subject_train_val_test_split(ds)
    >>> train_dl, val_dl, test_dl = make_dataloaders(train, val, test)

    Typical usage with subject_kfold
    ---------------------------------
    >>> for train, val in subject_kfold(ds):
    ...     train_dl, val_dl, _ = make_dataloaders(train, val, val)

    If the underlying tensors are on GPU, num_workers is forced to 0 and
    pin_memory to False automatically.
    """
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    return (
        _make_loader(train_ds, shuffle=shuffle_train, **kw),
        _make_loader(val_ds,   shuffle=False,         **kw),
        _make_loader(test_ds,  shuffle=False,         **kw),
    )


def cv_dataloaders(
    ds:            FMRIDataset,
    n_splits:      int  = 5,
    seed:          int  = 42,
    batch_size:    int  = 16,
    num_workers:   int  = 4,
    pin_memory:    bool = True,
    shuffle_train: bool = True,
) -> Iterator[Tuple[int, DataLoader, DataLoader]]:
    """
    Generator that yields (fold_index, train_loader, val_loader) for each fold.

    Example
    -------
    >>> for fold, train_dl, val_dl in cv_dataloaders(ds, n_splits=5):
    ...     for batch in train_dl:
    ...         fmri, fc, G, label = batch
    ...         ...
    """
    folds = subject_kfold(ds, n_splits=n_splits, seed=seed)
    kw    = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    for fold_idx, (train_sub, val_sub) in enumerate(folds):
        yield (
            fold_idx,
            _make_loader(train_sub, shuffle=shuffle_train, **kw),
            _make_loader(val_sub,   shuffle=False,         **kw),
        )


# ---------------------------------------------------------------------------
# Demographics helper
# ---------------------------------------------------------------------------

def load_hcp_age_gender(subjs):
    """Return (ages, genders) lists aligned with `subjs`."""
    hcp_demographics = pd.read_csv(
        os.path.join(proj_dir, "data/HCP_demographic_data.csv")
    )
    hcp_gender, hcp_age = [], []
    for subj in subjs:
        row = hcp_demographics[hcp_demographics.Subject.astype(str) == subj]
        if len(row):
            hcp_gender.append(row.iloc[0]["Gender"])
            hcp_age.append(row.iloc[0]["Age"])
        else:
            print(f"HCP {subj} not found in demographic file.")
    return hcp_age, hcp_gender


# ---------------------------------------------------------------------------
# Backward-compatibility helpers
# ---------------------------------------------------------------------------

def subject_kfold_split(
    ds: Dataset,
    subjs: np.ndarray = None,
    labels: np.ndarray = None,
    n_folds: int = 5,
    fold_idx: int = 0,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Backward-compatibility function matching the legacy signature.
    Performs subject-level K-fold cross-validation and splits the training fold
    into train and val subsets without subject leakage.
    """
    if isinstance(ds, FMRIDataset):
        folds = subject_kfold(ds, n_splits=n_folds, seed=seed)
        trainset, testset = folds[fold_idx]

        unique_subjs, subj_labels = _subject_level_arrays(ds)
        train_subjs_set = set(ds.subjs[list(trainset.indices)])
        unique_train_subjs = np.array([s for s in unique_subjs if s in train_subjs_set])

        # Sub-split train into train & val
        n_inner = max(2, round(1.0 / val_frac))
        inner_sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + fold_idx)
        train_consensus = []
        for s in unique_train_subjs:
            mask = ds.subjs == s
            lbls = ds.encoded_labels.numpy()[mask]
            train_consensus.append(np.argmax(np.bincount(lbls)))
        train_consensus = np.array(train_consensus)

        inner_splits = list(inner_sgkf.split(unique_train_subjs, train_consensus, groups=unique_train_subjs))
        _, val_subj_idx = inner_splits[-1]
        train_subj_idx = np.setdiff1d(np.arange(len(unique_train_subjs)), val_subj_idx)

        val_subjs = unique_train_subjs[val_subj_idx]
        new_train_subjs = unique_train_subjs[train_subj_idx]

        val_sample_idx = _sample_indices_for_subjects(ds.subjs, val_subjs)
        train_sample_idx = _sample_indices_for_subjects(ds.subjs, new_train_subjs)

        return Subset(ds, train_sample_idx), Subset(ds, val_sample_idx), testset
    else:
        raise TypeError('Expected FMRIDataset for subject_kfold_split')


def make_lso_dataloaders(
    ds: Dataset,
    subjs: np.ndarray = None,
    labels: np.ndarray = None,
    n_folds: int = 5,
    fold_idx: int = 0,
    val_frac: float = 0.1,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Legacy convenience wrapper returning (train_loader, val_loader, test_loader)."""
    train_sub, val_sub, test_sub = subject_kfold_split(
        ds, subjs=subjs, labels=labels,
        n_folds=n_folds, fold_idx=fold_idx,
        val_frac=val_frac, seed=seed,
    )
    return make_dataloaders(
        train_sub, val_sub, test_sub,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle_train=shuffle_train,
    )


if __name__ == "__main__":
    dataset, subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, "data/hcp"),
        label="gender",
        n_subjs=50,
        device="cpu",   # swap to "cuda" to pin everything in VRAM
    )

    # ── Option A: single train / val / test split ─────────────────────────────
    train_sub, val_sub, test_sub = subject_train_val_test_split(
        dataset, test_frac=0.15, val_frac=0.15
    )
    train_dl, val_dl, test_dl = make_dataloaders(train_sub, val_sub, test_sub, batch_size=16)

    for batch in train_dl:
        fmri, fc, G, label = batch
        print(f"[single split] fmri: {fmri.shape}  label: {label[:4]}")
        break

    # ── Option B: K-fold cross-validation ─────────────────────────────────────
    for fold, train_dl, val_dl in cv_dataloaders(dataset, n_splits=5, batch_size=16):
        for batch in train_dl:
            fmri, fc, G, label = batch
            print(f"[fold {fold}] fmri: {fmri.shape}  label: {label[:4]}")
            break
