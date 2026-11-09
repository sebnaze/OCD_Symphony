"""Reconstruct BOLD time series from a Stage-3.1 (DAPT) checkpoint.

The reconstruction shown is the Perceiver output (the head whose loss ``l_pc``
is ``MSE(recon, fmri)`` in ``finetuning.dapt``). Masking is disabled, so the
model sees the full, unmasked BOLD.

Speed
-----
* Normalisation is a per-ROI affine map ``x -> W*x + B`` fitted on *all* subjects
  (see ``make_hcp_dataset_from_pt``). Re-fitting on a subset would change the model
  inputs, so W and B are fitted once and cached in ``results/``.
* Only the selected subjects' raw files are read.
* Inference runs on the GPU when available. Models are built per request and released
  (VRAM returned) before the request returns.
"""
import gc
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from OCD_symphony.utils import proj_dir
from OCD_symphony.data_preparation.make_hcp_dataset import (
    DATASET_DIRS,
    WINDOW_DEMEAN_DATASETS,
    FMRIDataset,
    _load_raw_file,
    make_hcp_dataset_from_pt,
    subject_train_val_test_split,
)
from OCD_symphony.data_validation.validate_preprocessing import compute_roi_correlation
from OCD_symphony.model.finetuning import build_pretrained_models

DATA_DIR = DATASET_DIRS["hcp_ya"]  # kept for backward compatibility
RESULTS_DIR = os.path.join(proj_dir, "results")

# Subject-level split label per dataset. It stratifies the held-out test subjects, so it is
# fixed here and does not depend on the target chosen for training.
RECON_SPLIT_LABEL = {"hcp_ya": "gender", "ocd": "condition"}

# ROI reindexing used by make_hcp_dataset (order: sub-ctx_R, sub-ctx_L, ctx_L, ctx_R)
_ROI_INDS = torch.from_numpy(np.concatenate([
    np.arange(425, 450), np.arange(200, 225), np.arange(0, 200), np.arange(225, 425),
]))


def _cache_key(n_subjs: Optional[int], max_abs: float, test_frac: float,
               val_frac: float, seed: int, dataset: str = "hcp_ya") -> str:
    """Stable key for cached files; any change of data/split args gives a new key.

    HCP-YA keeps the legacy key format so existing caches stay valid.
    """
    cfg = dict(n_subjs=n_subjs, max_abs=max_abs, test_frac=test_frac, val_frac=val_frac, seed=seed)
    if dataset != "hcp_ya":
        cfg["dataset"] = dataset
        # Bumped whenever the preprocessing changes, so stale statistics are never reused.
        cfg["preprocess"] = "window_demean_v1" if dataset in WINDOW_DEMEAN_DATASETS else "none"
    digest = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    prefix = "" if dataset == "hcp_ya" else f"{dataset}_"
    return f"{prefix}n{n_subjs}_seed{seed}_{digest}"


def fit_normalisation_stats(
    n_subjs: Optional[int] = 500,
    max_abs: float = 1e4,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
    cache: bool = True,
    dataset: str = "hcp_ya",
) -> Dict[str, np.ndarray]:
    """Return ``{"W", "B", "test_subjs"}`` for the reconstruction data configuration.

    ``dataset`` is 'hcp_ya' or 'ocd'. The OCD cohort always uses all subjects, so W,B are
    fitted on the whole cohort and the test subjects come from the OCD split.

    The first call loads the full dataset once (~5 min) to fit the global
    per-ROI normalisation and the held-out test subject list, then caches them in
    ``results/``. Later calls read the cache in milliseconds.
    """
    if dataset == "ocd":
        n_subjs = None
    key = _cache_key(n_subjs, max_abs, test_frac, val_frac, seed, dataset)
    path = os.path.join(RESULTS_DIR, f"dapt_recon_stats_{key}.npz")
    if cache and os.path.isfile(path):
        d = np.load(path, allow_pickle=False)
        return {"W": d["W"], "B": d["B"], "test_subjs": d["test_subjs"]}

    dataset_obj, _ = make_hcp_dataset_from_pt(
        path=DATASET_DIRS[dataset], label=RECON_SPLIT_LABEL[dataset], n_subjs=n_subjs,
        max_abs=max_abs, dataset=dataset,
    )
    _, _, testset = subject_train_val_test_split(
        dataset_obj, test_frac=test_frac, val_frac=val_frac, seed=seed,
    )
    test_subjs = sorted(set(np.asarray(dataset_obj.subjs)[np.asarray(testset.indices)].tolist()))
    stats = {
        "W": dataset_obj.norm_W.numpy(),
        "B": dataset_obj.norm_B.numpy(),
        "test_subjs": np.array(test_subjs),
    }
    if cache:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.savez(path, **stats)
    return stats


def select_test_subject_subset(test_subjs, frac: float = 0.10, seed: int = 0) -> List[str]:
    """Reproducible ``frac`` of the held-out test subjects (at least one).

    The subjects are taken from one fixed random permutation (``seed``), so a larger
    ``frac`` always contains the smaller ones: 10% ⊂ 20% ⊂ ... ⊂ 100%.
    """
    test_subjs = np.asarray(sorted(str(s) for s in test_subjs))
    order = np.random.default_rng(seed).permutation(len(test_subjs))
    k = min(len(test_subjs), max(1, int(round(frac * len(test_subjs)))))
    return sorted(str(s) for s in test_subjs[order[:k]])


_ENCODER_SECTIONS = ("spatial", "temporal", "conv")
_DAPT_SECTIONS = _ENCODER_SECTIONS + ("perceiver",)


class MissingEncoderError(ValueError):
    """The run cannot be reconstructed because its encoder weights are not available."""


def _find_run_config(ckpt_path: str) -> Optional[dict]:
    """Return the ``config`` of the results/runs JSON whose ``history.ckpt_path`` is ``ckpt_path``.

    Used for checkpoints saved before the run recorded its own DAPT source. Matches on
    basename so that a moved project directory still resolves.
    """
    runs_dir = os.path.join(RESULTS_DIR, "runs")
    if not os.path.isdir(runs_dir):
        return None
    target = os.path.basename(ckpt_path)
    for name in sorted(os.listdir(runs_dir)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(runs_dir, name)) as f:
                run = json.load(f)
        except (OSError, ValueError):
            continue
        saved = (run.get("history") or {}).get("ckpt_path")
        if saved and os.path.basename(saved) == target:
            meta = run.get("meta") or {}
            return meta.get("config") or run.get("config") or {}
    return None


def load_reconstruction_checkpoint(ckpt_path: str, device: torch.device):
    """Return ``(encoder_ckpt, description)`` for ``build_pretrained_models``.

    Accepted inputs:

    * **DAPT checkpoint** (Stage 3.1): contains ``spatial``, ``temporal``, ``conv``
      and ``perceiver`` sections. Used as-is.
    * **Classifier checkpoint with stored encoders** (Stage 3.2 with ``encoder_lr > 0``):
      contains the same DAPT-format sections next to the flat classifier keys.
    * **Classifier checkpoint without encoders** (Stage 3.2 with ``encoder_lr == 0``):
      the encoders are the frozen DAPT encoders. Their source is the ``dapt_ckpt`` of the
      run, read from the checkpoint or the run JSON. The Perceiver comes from this file.

    Raises ``ValueError`` with an explicit message for anything else.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"{ckpt_path}: expected a dict checkpoint, got {type(raw).__name__}.")

    # Case 1 and 2: DAPT-format sections present.
    if all(isinstance(raw.get(s), dict) for s in _DAPT_SECTIONS):
        kind = "DAPT checkpoint" if "encoder_lr" not in raw else "Stage 3.2 checkpoint (encoders stored)"
        return {s: raw[s] for s in _DAPT_SECTIONS}, f"{kind}: {os.path.basename(ckpt_path)}"

    # Case 3: flat classifier checkpoint.
    perceiver_prefix = "perceiver."
    perceiver_state = {k[len(perceiver_prefix):]: v for k, v in raw.items()
                       if isinstance(k, str) and k.startswith(perceiver_prefix)}
    if not perceiver_state:
        raise ValueError(
            f"{ckpt_path}: unrecognised checkpoint. Expected DAPT sections "
            f"{_DAPT_SECTIONS} or classifier 'perceiver.*' keys."
        )

    run_cfg = raw.get("config") if isinstance(raw.get("config"), dict) else None
    if run_cfg is None:
        run_cfg = _find_run_config(ckpt_path) or {}
    encoder_lr = float(run_cfg.get("encoder_lr", 0.0) or 0.0)
    if encoder_lr > 0.0:
        raise MissingEncoderError(
            f"{ckpt_path}: Stage 3.2 run used encoder_lr={encoder_lr:g} but its fine-tuned "
            "encoders were not saved. Re-run Stage 3.2 with the current code to reconstruct it."
        )

    dapt_path = raw.get("dapt_ckpt") or run_cfg.get("dapt_ckpt")
    if not dapt_path:
        raise MissingEncoderError(
            f"{ckpt_path}: no DAPT encoder source found (checked the checkpoint and its run JSON)."
        )
    if not os.path.isfile(dapt_path):
        raise MissingEncoderError(f"DAPT encoder checkpoint referenced by run not found: {dapt_path}")

    dapt = torch.load(dapt_path, map_location=device, weights_only=False)
    missing = [s for s in _DAPT_SECTIONS if not isinstance(dapt.get(s), dict)]
    if missing:
        raise ValueError(f"{dapt_path}: not a DAPT checkpoint, missing sections {missing}.")

    encoder_ckpt = {s: dapt[s] for s in _ENCODER_SECTIONS}
    encoder_ckpt["perceiver"] = perceiver_state  # Stage 3.2 Perceiver, not the DAPT one
    desc = (f"Stage 3.2 classifier {os.path.basename(ckpt_path)} "
            f"(encoders from DAPT {os.path.basename(dapt_path)})")
    return encoder_ckpt, desc


def get_dapt_models(ckpt_path: Optional[str], device: torch.device):
    """Build the four modules (eval mode).

    ``ckpt_path`` is a DAPT or Stage 3.2 checkpoint (see ``load_reconstruction_checkpoint``),
    or ``None`` for the baseline: the original HCP-AGING pretrained weights loaded by
    ``build_pretrained_models``.

    Not cached: the caller owns the returned models and must release them with
    ``release_gpu_memory()`` once inference is finished.
    """
    ckpt = None
    if ckpt_path is not None:
        ckpt, _ = load_reconstruction_checkpoint(ckpt_path, device)
    models = build_pretrained_models(
        args=None, device=device, mask_ratio=0.0, encoder_ckpt=ckpt,
    )
    for m in models:
        m.eval()
    return models



def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_subjects_normalised(subject_ids: List[str], W, B, max_abs: float = 1e4,
                             num_io_workers: int = 8, dataset: str = "hcp_ya") -> FMRIDataset:
    """Read only ``subject_ids`` from ``dataset``, reindex ROIs and apply the global (W, B).

    Same transform as ``make_hcp_dataset_from_pt`` steps 3-6, so inputs match training.
    """
    data_dir = DATASET_DIRS[dataset]
    fnames = [os.path.join(data_dir, f"{s}.pt") for s in subject_ids]
    with ThreadPoolExecutor(max_workers=num_io_workers) as ex:
        records = list(ex.map(lambda f: _load_raw_file(f, max_abs, dataset), fnames))
    records = sorted((r for r in records if not r.get("excluded")), key=lambda r: r["subj"])
    if not records:
        raise RuntimeError("None of the selected subjects passed validation.")

    fmri = torch.cat([r["fmri"] for r in records], dim=0).float()[:, _ROI_INDS, :]
    fc = torch.cat([r["fc"] for r in records], dim=0).float()
    fc = fc[:, _ROI_INDS, :][:, :, _ROI_INDS]

    fmri = fmri * torch.as_tensor(W).float().reshape(1, -1, 1) + torch.as_tensor(B).float().reshape(1, -1, 1)

    subjs = np.concatenate([[r["subj"]] * r["n"] for r in records])
    ages = np.concatenate([[r["age"]] * r["n"] for r in records])
    genders = np.concatenate([[r["gender"]] * r["n"] for r in records])
    conditions = np.concatenate([[r["condition"]] * r["n"] for r in records])
    G = torch.zeros(fmri.shape[0], 450, 10)  # unused by the learnable spatial PE
    label = RECON_SPLIT_LABEL[dataset] if dataset == "ocd" else "gender"
    return FMRIDataset(
        fmri, fc, G, ages, genders, subjs, label,
        conditions=conditions if dataset == "ocd" else None,
    )


def estimate_seconds_per_subject(device: torch.device) -> float:
    """Planning figure for the UI, measured on an NVIDIA L40S (72 subjects in 21.5 s on GPU,
    7 subjects in 31 s on CPU). Used only for the displayed time estimate."""
    return 0.3 if device.type == "cuda" else 4.5


def select_representative_subjects(scores: Dict[str, float], k: int = 6) -> List[str]:
    """Pick ``k`` subjects: best (highest score), worst (lowest), and the remaining
    ``k - 2`` closest to the median score (ties broken by subject id).

    Subjects with a non-finite score are ignored. If there are at most ``k`` valid
    subjects, all of them are returned (best first).
    """
    if k < 2:
        raise ValueError("k must be >= 2 (best and worst subject)")
    valid = sorted(((s, float(v)) for s, v in scores.items() if np.isfinite(v)), key=lambda x: x[0])
    if len(valid) <= k:
        return [s for s, _ in sorted(valid, key=lambda x: -x[1])]
    best = max(valid, key=lambda x: x[1])
    worst = min(valid, key=lambda x: x[1])
    median = float(np.median([v for _, v in valid]))
    rest = [x for x in valid if x[0] not in (best[0], worst[0])]
    rest.sort(key=lambda x: (abs(x[1] - median), x[0]))
    middle = [s for s, _ in rest[: k - 2]]
    return [best[0]] + middle + [worst[0]]


def release_gpu_memory() -> None:
    """Drop Python references and return cached CUDA blocks to the driver."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _forward_subject(models, sid: str, W, B, max_abs: float, batch_size: int, device,
                     dataset: str = "hcp_ya"):
    """Yield ``(orig, recon)`` numpy batches (n, R, T) for one subject, masking off."""
    spatial_model, temporal_model, conv_model, perceiver = models
    ds = load_subjects_normalised([sid], W, B, max_abs=max_abs, dataset=dataset)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    for fmri, fc, gm, _label in loader:
        fmri = fmri.to(device, non_blocking=True).float()
        fc = fc.to(device, non_blocking=True).float()
        gm = gm.to(device, non_blocking=True).float()
        _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
        _, _, _, _, time_embeds, _ = temporal_model(fmri)
        _, signal_context = conv_model(fmri, mask=None)
        recon, _, _ = perceiver(roi_embeds, time_embeds, signal_context, roi_pos)
        yield fmri.cpu().numpy(), recon.cpu().numpy()


def _add_r2_sums(sums, orig, recon):
    """Accumulate per-ROI running sums (count, sum y, sum y², SS_res) for a (n, R, T) batch."""
    o = np.asarray(orig, dtype=np.float64)
    r = np.asarray(recon, dtype=np.float64)
    part = np.stack([
        np.full(o.shape[1], o.shape[0] * o.shape[2], dtype=np.float64),
        o.sum(axis=(0, 2)),
        (o ** 2).sum(axis=(0, 2)),
        ((o - r) ** 2).sum(axis=(0, 2)),
    ])
    return part if sums is None else sums + part


def _r2_from_sums(sums) -> np.ndarray:
    """ROI-wise R² = 1 - SS_res / SS_tot, identical to ``compute_r2`` over all samples/time."""
    counts, sum_o, sum_o2, ss_res = sums
    ss_tot = sum_o2 - sum_o ** 2 / counts
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - ss_res / ss_tot


@torch.no_grad()
def reconstruct_subject_stats(
    ckpt_path: Optional[str],
    subject_ids: List[str],
    W,
    B,
    keep_k: int = 6,
    max_abs: float = 1e4,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    dataset: str = "hcp_ya",
) -> Dict[str, object]:
    """Reconstruct ``subject_ids`` and return aggregate statistics plus traces for a
    small set of representative subjects. The GPU models are released before returning.

    Pass 1 streams every subject once: per-sample ROI-wise Pearson r is kept (small),
    ROI-wise R² is accumulated from running sums (the full BOLD arrays are never held),
    and a per-subject score (mean r over ROIs and time) is recorded.
    Pass 2 re-runs inference for only the ``keep_k`` representative subjects to keep
    their full original/reconstructed traces (chosen after pass 1, so the median is known).

    Returns a dict with:
      r2 (R,), corr (N, R), subjs (N,) — over all reconstructed samples;
      subject_scores {subj: score}; representative (list, best first, worst last);
      traces {subj: {"orig": (n, R, T), "recon": (n, R, T)}}.
    """
    if ckpt_path is not None and not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"DAPT checkpoint not found: {ckpt_path}")
    device = device or default_device()
    models = get_dapt_models(ckpt_path, device)
    try:
        total = len(subject_ids)
        n_roi = None
        sums = None  # per-ROI running sums in float64
        corr_rows, subjs_rows = [], []
        subject_scores: Dict[str, float] = {}
        n_t = None
        n_done = 0

        # ── Pass 1: stream all subjects ─────────────────────────────────────
        for sid in subject_ids:
            sid_corr = []
            for o, r in _forward_subject(models, sid, W, B, max_abs, batch_size, device, dataset):
                n_t = o.shape[2]
                sums = _add_r2_sums(sums, o, r)
                sid_corr.append(compute_roi_correlation(o, r))
                subjs_rows.extend([sid] * o.shape[0])
            if sid_corr:
                corr_all = np.concatenate(sid_corr, axis=0)
                corr_rows.append(corr_all)
                finite = corr_all[np.isfinite(corr_all)]
                subject_scores[sid] = float(finite.mean()) if finite.size else float("nan")
            n_done += 1
            if progress is not None:
                progress(n_done, total + keep_k)

        if not corr_rows:
            raise RuntimeError("No subject could be reconstructed.")

        r2 = _r2_from_sums(sums)

        # ── Pass 2: traces for representative subjects only ─────────────────
        representative = select_representative_subjects(subject_scores, k=keep_k)
        traces: Dict[str, Dict[str, np.ndarray]] = {}
        for sid in representative:
            o_list, r_list = [], []
            for o, r in _forward_subject(models, sid, W, B, max_abs, batch_size, device, dataset):
                o_list.append(o)
                r_list.append(r)
            traces[sid] = {"orig": np.concatenate(o_list, axis=0),
                           "recon": np.concatenate(r_list, axis=0)}
            n_done += 1
            if progress is not None:
                progress(n_done, total + keep_k)

        return {
            "r2": r2,
            "corr": np.concatenate(corr_rows, axis=0),
            "subjs": np.asarray(subjs_rows),
            "subject_scores": subject_scores,
            "representative": representative,
            "traces": traces,
            "n_timepoints": n_t,
            "source": "dapt" if ckpt_path is not None else "baseline",
            "dataset": dataset,
        }
    finally:
        del models
        release_gpu_memory()
