"""Unit tests for the DAPT robustness fixes (no model training involved).

Run with:  conda run -n ocd-symphony python -m pytest tests/test_dapt_fixes.py -q
"""
import os
import time

import numpy as np
import pytest
import torch

from OCD_symphony.data_preparation.make_hcp_dataset import (
    _load_raw_file,
    make_hcp_dataset_from_pt,
)
from OCD_symphony.model.model_components import (
    TimePointMasker,
    TokenMasker1D,
    TokenMasker2D,
)


# ── Maskers ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("ratio", [0.0, 0.15, 0.5, 0.7])
def test_maskers_shapes_and_counts(ratio):
    B, R, T = 4, 450, 200
    x = torch.randn(B, R, T)

    m1 = TokenMasker1D(ratio)(x)
    m2 = TokenMasker2D(ratio)(x)
    m3 = TimePointMasker(ratio)(x)

    assert m1.shape == (B, T) and m2.shape == (B, R) and m3.shape == (B, R, T)
    for m in (m1, m2, m3):
        assert set(m.unique().tolist()) <= {0.0, 1.0}
    # exactly int(ratio * N) masked (zero) entries along the last dim, per row
    assert (m1 == 0).sum(-1).eq(int(ratio * T)).all()
    assert (m2 == 0).sum(-1).eq(int(ratio * R)).all()
    assert (m3 == 0).sum(-1).eq(int(ratio * T)).all()


def test_masker_ratio_zero_is_all_ones():
    x = torch.randn(2, 450, 200)
    assert TokenMasker1D(0)(x).eq(1).all()
    assert TokenMasker2D(0)(x).eq(1).all()
    assert TimePointMasker(0)(x).eq(1).all()


def test_time_point_masker_is_fast():
    x = torch.randn(8, 450, 200)
    masker = TimePointMasker(0.5)
    t0 = time.time()
    masker(x)
    assert time.time() - t0 < 1.0  # the old per-(b, r) python loop took far longer


def test_masks_are_random_across_calls():
    x = torch.randn(2, 450, 200)
    masker = TokenMasker2D(0.7)
    assert not torch.equal(masker(x), masker(x))


# ── Deterministic validation masks (fork_rng pattern used in dapt()) ─────────
def test_fork_rng_gives_identical_masks_and_preserves_train_stream():
    x = torch.randn(4, 450, 200)
    masker = TokenMasker2D(0.7)

    def val_masks():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1234)
            return masker(x)

    torch.manual_seed(0)
    a = val_masks()
    after_first = torch.rand(3)

    torch.manual_seed(0)
    b = val_masks()
    after_second = torch.rand(3)

    assert torch.equal(a, b)                      # same masks every "epoch"
    assert torch.equal(after_first, after_second)  # training RNG stream untouched


# ── Dataset validation ──────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _synthetic_demographics(monkeypatch):
    """The fixtures write synthetic subject IDs that are not in the demographics files.

    Answer every metadata lookup with a fixed record, so these tests keep exercising the
    run/subject validation logic. Real lookups still fail loudly for unknown subjects.
    """
    import OCD_symphony.data_preparation.make_hcp_dataset as mhd

    class _AnySubject(dict):
        def __contains__(self, key):
            return True

        def __missing__(self, key):
            return dict(age=25.0, gender="M", condition=None)

    monkeypatch.setattr(mhd, "_demographics_table", lambda dataset: _AnySubject())


def _write_subject(path, subj, n_runs=3, corrupt_runs=(), nan_fc_runs=(), scale=50.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    fmri = torch.randn(1, n_runs, 450, 200, generator=g) * scale
    fc = torch.randn(1, n_runs, 450, 450, generator=g).clamp(-1, 1)
    for r in corrupt_runs:
        fmri[0, r] *= 1e30
    for r in nan_fc_runs:
        fc[0, r, 0, 0] = float("nan")
    torch.save({"fmri_data": fmri, "fc_data": fc, "session": "x"}, path / f"{subj}.pt")


def test_load_raw_file_drops_bad_runs(tmp_path):
    _write_subject(tmp_path, "900001", n_runs=4, corrupt_runs=(1,), nan_fc_runs=(3,))
    rec = _load_raw_file(str(tmp_path / "900001.pt"), max_abs=1e4)
    assert not rec["excluded"]
    assert rec["n"] == 2 and rec["n_runs_total"] == 4
    assert sorted(i for i, _ in rec["dropped_runs"]) == [1, 3]
    assert rec["fmri"].shape == (2, 450, 200) and rec["fc"].shape == (2, 450, 450)


def test_load_raw_file_excludes_fully_corrupt_subject(tmp_path):
    _write_subject(tmp_path, "900002", n_runs=2, corrupt_runs=(0, 1))
    rec = _load_raw_file(str(tmp_path / "900002.pt"), max_abs=1e4)
    assert rec["excluded"] and "invalid" in rec["reason"]


def test_make_dataset_excludes_bad_subject_and_normalises(tmp_path):
    for i in range(5):
        _write_subject(tmp_path, f"90010{i}", n_runs=3, seed=i)
    _write_subject(tmp_path, "900199", n_runs=3, corrupt_runs=(0, 1, 2), seed=99)
    _write_subject(tmp_path, "900198", n_runs=3, nan_fc_runs=(2,), seed=98)

    ds, subjs = make_hcp_dataset_from_pt(
        path=str(tmp_path), label="gender", num_io_workers=2,
        exclusion_report_path=str(tmp_path / "report" / "excl.json"),
    )
    assert "900199" not in set(subjs)                      # fully corrupt -> excluded
    assert (np.array(subjs) == "900198").sum() == 2         # 1 NaN-FC run dropped
    assert len(ds.exclusion_report) == 2
    assert {e["subject"] for e in ds.exclusion_report} == {"900199", "900198"}
    # normalisation is not poisoned by the corrupt subject
    assert abs(ds.fmri_data.std().item() - 1.0) < 0.05
    assert ds.fmri_data.abs().max().item() < 200


def test_make_dataset_raises_when_all_invalid(tmp_path):
    _write_subject(tmp_path, "900300", n_runs=2, corrupt_runs=(0, 1))
    with pytest.raises(RuntimeError):
        make_hcp_dataset_from_pt(
            path=str(tmp_path), label="gender", num_io_workers=1,
            exclusion_report_path=str(tmp_path / "excl.json"),
        )
