"""Paper-mode loader behaviour: median/IQR normalisation, no window demeaning, legacy unchanged.

Synthetic .pt files only; real data is not read here.
"""
import numpy as np
import pytest
import torch

import OCD_symphony.data_preparation.make_hcp_dataset as mhd
from OCD_symphony.data_preparation.make_hcp_dataset import (
    _load_raw_file, dataset_dirs, make_hcp_dataset_from_pt,
)


class _AnySubject(dict):
    def __contains__(self, key):
        return True

    def __missing__(self, key):
        return dict(age=25.0, gender="M", condition="HC")


@pytest.fixture(autouse=True)
def _synthetic_demographics(monkeypatch):
    monkeypatch.setattr(mhd, "_demographics_table", lambda dataset: _AnySubject())


def _write_subject(path, subj, n_runs=3, offset=0.0, scale=50.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    fmri = torch.randn(1, n_runs, 450, 200, generator=g) * scale + offset
    fc = torch.randn(1, n_runs, 450, 450, generator=g).clamp(-1, 1)
    torch.save({"fmri_data": fmri, "fc_data": fc, "session": torch.zeros(1, n_runs)},
               path / f"{subj}.pt")


def test_dataset_dirs_selects_mode():
    paper = dataset_dirs("paper")
    legacy = dataset_dirs("legacy")
    assert paper["hcp_ya"].endswith("data/hcp_paper")
    assert paper["ocd"].endswith("data/OCD_paper")
    assert legacy["hcp_ya"].endswith("data/hcp")
    assert legacy["ocd"].endswith("OCDbaseline_fmriprep_MNI152NLin2009cAsym_res-2")
    with pytest.raises(ValueError):
        dataset_dirs("other")


def test_unknown_preprocessing_rejected(tmp_path):
    _write_subject(tmp_path, "900001", n_runs=1)
    with pytest.raises(ValueError):
        make_hcp_dataset_from_pt(path=str(tmp_path), label="gender", preprocessing="bogus")


def test_paper_normalisation_is_median_iqr(tmp_path):
    rng_offsets = [0.0, 300.0, -120.0, 40.0, 10.0, -5.0]
    for i, off in enumerate(rng_offsets):
        _write_subject(tmp_path, f"9100{i:02d}", n_runs=2, offset=off, seed=i)
    ds, _ = make_hcp_dataset_from_pt(path=str(tmp_path), label="gender", num_io_workers=2,
                                     preprocessing="paper", approx_quantile_samples=None)
    assert ds.preprocessing == "paper"
    x = ds.fmri_data.permute(1, 0, 2).reshape(450, -1)
    q = torch.quantile(x, torch.tensor([0.25, 0.5, 0.75]), dim=1)
    assert q[1].abs().max().item() < 0.02          # median ~ 0
    assert (q[2] - q[0] - 1).abs().max().item() < 0.02  # IQR ~ 1

def test_paper_gaussian_std_is_iqr_over_1349(tmp_path):
    # no sigma division: zero-offset Gaussian data has std ~ IQR/1.349 ~ 0.74 after scaling
    for i in range(4):
        _write_subject(tmp_path, f"9101{i:02d}", n_runs=2, offset=0.0, seed=10 + i)
    ds, _ = make_hcp_dataset_from_pt(path=str(tmp_path), label="gender", num_io_workers=2,
                                     preprocessing="paper")
    std = ds.fmri_data.std(dim=(0, 2))
    assert 0.65 < std.mean().item() < 0.85


def test_paper_keeps_window_offsets_for_ocd_legacy_removes_them(tmp_path):
    _write_subject(tmp_path, "900201", n_runs=1, offset=500.0, seed=3)
    fname = str(tmp_path / "900201.pt")
    paper = _load_raw_file(fname, dataset="ocd", preprocessing="paper")
    legacy = _load_raw_file(fname, dataset="ocd", preprocessing="legacy")
    # legacy demeans each window per ROI; paper keeps the ROI offset of the window
    assert legacy["fmri"].mean(dim=2).abs().max().item() < 1e-4
    assert paper["fmri"].mean(dim=2).mean().item() > 100


def test_legacy_normalisation_still_unit_sigma(tmp_path):
    for i in range(3):
        _write_subject(tmp_path, f"9200{i:02d}", n_runs=2, seed=i)
    ds, _ = make_hcp_dataset_from_pt(path=str(tmp_path), label="gender", num_io_workers=2,
                                     preprocessing="legacy")
    assert abs(ds.fmri_data.std().item() - 1.0) < 0.05
    assert ds.preprocessing == "legacy"
