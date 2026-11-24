# Shared helpers for paper-mode torch file extraction (HCP-YA and OCD).
# Output schema matches the legacy .pt files consumed by make_hcp_dataset._load_raw_file:
#   fmri_data (1, n_windows, R, W), fc_data (1, n_windows, R, R), session (1, n_windows),
# plus n_flagged (1, n_windows) and window_start (1, n_windows) for QC.

import json
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import torch
from nilearn.input_data import NiftiLabelsMasker

from OCD_symphony.data_preparation.fmri_cleaning import (
    CleaningConfig, cleaning_metadata, process_runs,
)

METADATA_FILE = "metadata.json"
QC_DIR = "qc"


def parcellate(nifti_path: str, atlas_path: str, tr: float) -> np.ndarray:
    """Mean ROI time series of a 4D NIfTI, shape (R, T). No nilearn filtering or standardising."""
    masker = NiftiLabelsMasker(atlas_path, smoothing_fwhm=0, t_r=tr,
                               memory="nilearn_cache", memory_level=1, verbose=0)
    return masker.fit_transform(nifti_path).T


def save_subject(out_dir: str, subj: str, out: Dict) -> Optional[str]:
    """Write one subject's windows to ``<out_dir>/<subj>.pt`` (atomic). Returns the path,
    or None when the subject has no kept windows."""
    if out.get("excluded") or "fmri" not in out:
        return None
    pt = {
        "fmri_data": torch.tensor(out["fmri"], dtype=torch.float32)[None],
        "fc_data": torch.tensor(out["fc"], dtype=torch.float32)[None],
        "session": torch.tensor(out["session"], dtype=torch.int64)[None],
        "n_flagged": torch.tensor(out["n_flagged"], dtype=torch.int64)[None],
        "window_start": torch.tensor(out["window_start"], dtype=torch.int64)[None],
    }
    path = os.path.join(out_dir, subj + ".pt")
    tmp = path + ".tmp"
    torch.save(pt, tmp)
    os.replace(tmp, path)
    return path


def write_qc(out_dir: str, subj: str, out: Dict) -> None:
    """Per-subject QC sidecar: run statistics and exclusion status."""
    qc_dir = os.path.join(out_dir, QC_DIR)
    os.makedirs(qc_dir, exist_ok=True)
    rec = dict(subject=subj, excluded=bool(out.get("excluded")),
               n_windows=int(out.get("n_windows", 0)), runs=out.get("run_stats", []),
               reason=out.get("reason"))
    with open(os.path.join(qc_dir, subj + ".json"), "w") as f:
        json.dump(rec, f, indent=2)


def prepare_output_dir(out_dir: str, metadata: Dict, overwrite: bool) -> None:
    """Create the output folder and guard against mixing outputs of different settings.

    * empty or new folder: write ``metadata.json`` and continue;
    * folder with a ``metadata.json`` that differs from the current settings: refuse unless
      ``overwrite``;
    * folder with files from a different (or unknown) origin and no matching metadata: refuse
      unless ``overwrite``.
    Subject files that already exist are skipped by the caller unless ``overwrite``.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta_path = os.path.join(out_dir, METADATA_FILE)
    existing = [f for f in os.listdir(out_dir) if f.endswith(".pt")]
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            prev = json.load(f)
        if _settings_key(prev) != _settings_key(metadata) and not overwrite:
            raise RuntimeError(
                f"{out_dir} holds outputs with different settings "
                f"(see {METADATA_FILE}). Pass --overwrite or choose another --out_dir.")
    elif existing and not overwrite:
        raise RuntimeError(
            f"{out_dir} is not empty and has no {METADATA_FILE}; refusing to write. "
            "Pass --overwrite or choose another --out_dir.")
    if not os.path.exists(meta_path) or overwrite:
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)


def _settings_key(meta: Dict) -> str:
    keep = {k: meta.get(k) for k in ("preprocessing", "steps", "cleaning", "tr_seconds",
                                     "motion_parameters", "fd_radius_mm", "dataset")}
    return json.dumps(keep, sort_keys=True)


def dropped_subjects(out_dir: str) -> List[Dict]:
    """Collect exclusion records from the QC folder."""
    qc_dir = os.path.join(out_dir, QC_DIR)
    if not os.path.isdir(qc_dir):
        return []
    recs = []
    for f in sorted(os.listdir(qc_dir)):
        with open(os.path.join(qc_dir, f)) as fh:
            rec = json.load(fh)
        if rec.get("excluded"):
            recs.append(dict(subject=rec["subject"], n_windows=rec["n_windows"]))
    return recs


def update_metadata_dropped(out_dir: str) -> None:
    """Refresh the list of dropped subjects in ``metadata.json``."""
    meta_path = os.path.join(out_dir, METADATA_FILE)
    if not os.path.exists(meta_path):
        return
    with open(meta_path) as f:
        meta = json.load(f)
    meta["dropped_subjects"] = dropped_subjects(out_dir)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def extract_subject_paper(subj: str, runs: List[Dict], out_dir: str, tr: float,
                          cfg: CleaningConfig, overwrite: bool,
                          logger: logging.Logger) -> Optional[str]:
    """Clean and window the runs of one subject and write its .pt file and QC sidecar."""
    target = os.path.join(out_dir, subj + ".pt")
    if os.path.exists(target) and not overwrite:
        logger.info(f"{subj}: {target} exists, skipped (use --overwrite to recompute)")
        return target
    out = process_runs(runs, cfg)
    write_qc(out_dir, subj, out)
    if out["excluded"]:
        logger.info(f"{subj}: excluded, {out['n_windows']} kept windows < {cfg.min_windows}")
        return None
    path = save_subject(out_dir, subj, out)
    logger.info(f"{subj}: saved {out['n_windows']} windows from {len(runs)} runs -> {path}")
    return path
