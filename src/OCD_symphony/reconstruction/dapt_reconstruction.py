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
* Inference runs on the GPU when available, and the built models are cached per
  checkpoint/device so repeated requests do not rebuild them.
"""
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
    FMRIDataset,
    _load_raw_file,
    make_hcp_dataset_from_pt,
    subject_train_val_test_split,
)
from OCD_symphony.model.finetuning import build_pretrained_models

DATA_DIR = os.path.join(proj_dir, "data", "hcp")
RESULTS_DIR = os.path.join(proj_dir, "results")

# ROI reindexing used by make_hcp_dataset (order: sub-ctx_R, sub-ctx_L, ctx_L, ctx_R)
_ROI_INDS = torch.from_numpy(np.concatenate([
    np.arange(425, 450), np.arange(200, 225), np.arange(0, 200), np.arange(225, 425),
]))


def _cache_key(n_subjs: Optional[int], max_abs: float, test_frac: float,
               val_frac: float, seed: int) -> str:
    """Stable key for cached files; any change of data/split args gives a new key."""
    cfg = dict(n_subjs=n_subjs, max_abs=max_abs, test_frac=test_frac, val_frac=val_frac, seed=seed)
    digest = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]
    return f"n{n_subjs}_seed{seed}_{digest}"


def fit_normalisation_stats(
    n_subjs: Optional[int] = 500,
    max_abs: float = 1e4,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
    cache: bool = True,
) -> Dict[str, np.ndarray]:
    """Return ``{"W", "B", "test_subjs"}`` for the DAPT data configuration.

    The first call loads the full dataset once (~5 min) to fit the global
    per-ROI normalisation and the held-out test subject list, then caches them in
    ``results/``. Later calls read the cache in milliseconds.
    """
    key = _cache_key(n_subjs, max_abs, test_frac, val_frac, seed)
    path = os.path.join(RESULTS_DIR, f"dapt_recon_stats_{key}.npz")
    if cache and os.path.isfile(path):
        d = np.load(path, allow_pickle=False)
        return {"W": d["W"], "B": d["B"], "test_subjs": d["test_subjs"]}

    dataset, _ = make_hcp_dataset_from_pt(
        path=DATA_DIR, label="gender", n_subjs=n_subjs, max_abs=max_abs,
    )
    _, _, testset = subject_train_val_test_split(
        dataset, test_frac=test_frac, val_frac=val_frac, seed=seed,
    )
    test_subjs = sorted(set(np.asarray(dataset.subjs)[np.asarray(testset.indices)].tolist()))
    stats = {
        "W": dataset.norm_W.numpy(),
        "B": dataset.norm_B.numpy(),
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


_MODEL_CACHE: Dict[tuple, tuple] = {}


def get_dapt_models(ckpt_path: str, device: torch.device):
    """Build the four DAPT modules once per (checkpoint, device) and reuse them."""
    key = (os.path.abspath(ckpt_path), str(device))
    if key not in _MODEL_CACHE:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        models = build_pretrained_models(
            args=None, device=device, mask_ratio=0.0, encoder_ckpt=ckpt,
        )
        for m in models:
            m.eval()
        _MODEL_CACHE[key] = models
    return _MODEL_CACHE[key]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_subjects_normalised(subject_ids: List[str], W, B, max_abs: float = 1e4,
                             num_io_workers: int = 8) -> FMRIDataset:
    """Read only ``subject_ids``, reindex ROIs and apply the global (W, B).

    Same transform as ``make_hcp_dataset_from_pt`` steps 3-6, so inputs match training.
    """
    fnames = [os.path.join(DATA_DIR, f"{s}.pt") for s in subject_ids]
    with ThreadPoolExecutor(max_workers=num_io_workers) as ex:
        records = list(ex.map(lambda f: _load_raw_file(f, max_abs), fnames))
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
    G = torch.zeros(fmri.shape[0], 450, 10)  # unused by the learnable spatial PE
    return FMRIDataset(fmri, fc, G, ages, genders, subjs, "gender")


def estimate_seconds_per_subject(device: torch.device) -> float:
    """Planning figure for the UI, measured on an NVIDIA L40S (72 subjects in 21.5 s on GPU,
    7 subjects in 31 s on CPU). Used only for the displayed time estimate."""
    return 0.3 if device.type == "cuda" else 4.5


@torch.no_grad()
def reconstruct_subjects(
    ckpt_path: str,
    subject_ids: List[str],
    W,
    B,
    max_abs: float = 1e4,
    batch_size: int = 16,
    device: Optional[torch.device] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, np.ndarray]:
    """Reconstruct BOLD for ``subject_ids`` with the DAPT checkpoint (masking off).

    Subjects are read in chunks of one subject each; ``progress(done, total)`` is
    called after every subject so the UI can show a progress bar.
    Returns ``orig`` (N, R, T), ``recon`` (N, R, T) and ``subjs`` (N,).
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"DAPT checkpoint not found: {ckpt_path}")
    device = device or default_device()
    spatial_model, temporal_model, conv_model, perceiver = get_dapt_models(ckpt_path, device)

    origs, recons, subjs = [], [], []
    total = len(subject_ids)
    for done, sid in enumerate(subject_ids, start=1):
        ds = load_subjects_normalised([sid], W, B, max_abs=max_abs)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        for fmri, fc, gm, _label in loader:
            fmri = fmri.to(device, non_blocking=True).float()
            fc = fc.to(device, non_blocking=True).float()
            gm = gm.to(device, non_blocking=True).float()
            _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
            _, _, _, _, time_embeds, _ = temporal_model(fmri)
            _, signal_context = conv_model(fmri, mask=None)
            recon, _, _ = perceiver(roi_embeds, time_embeds, signal_context, roi_pos)
            origs.append(fmri.cpu().numpy())
            recons.append(recon.cpu().numpy())
        subjs.extend(ds.subjs.tolist())
        if progress is not None:
            progress(done, total)

    if not origs:
        raise RuntimeError("No subject could be reconstructed.")
    return {
        "orig": np.concatenate(origs, axis=0),
        "recon": np.concatenate(recons, axis=0),
        "subjs": np.asarray(subjs),
    }
