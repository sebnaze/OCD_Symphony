"""Tests for loading reconstruction checkpoints (DAPT and Stage 3.2 classifier formats).

Only the checkpoint-resolution logic is exercised; no models are built.

Run with:  conda run -n ocd-symphony python -m pytest tests/test_reconstruction_checkpoints.py -q
"""
import json

import pytest
import torch

import OCD_symphony.reconstruction.dapt_reconstruction as dr


def _sec(val):
    return {"w": torch.full((2,), float(val))}


def _dapt_file(path, val=1.0):
    torch.save({"spatial": _sec(val), "temporal": _sec(val), "conv": _sec(val),
                "perceiver": _sec(val), "epoch": 3}, path)
    return str(path)


def _flat_cls_file(path, val=5.0, **extra):
    state = {"cls_token": torch.zeros(1), "perceiver.w": torch.full((2,), val),
             "classifier.w": torch.zeros(1)}
    state.update(extra)
    torch.save(state, path)
    return str(path)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "RESULTS_DIR", str(tmp_path / "results"))
    (tmp_path / "results" / "runs").mkdir(parents=True)
    return tmp_path / "results" / "runs"


def test_dapt_checkpoint_used_as_is(tmp_path, runs_dir):
    p = _dapt_file(tmp_path / "dapt.pth", val=1.0)
    enc, desc = dr.load_reconstruction_checkpoint(p, torch.device("cpu"))
    assert set(enc) == {"spatial", "temporal", "conv", "perceiver"}
    assert torch.equal(enc["perceiver"]["w"], torch.full((2,), 1.0))
    assert "DAPT checkpoint" in desc


def test_stage32_with_embedded_dapt_source(tmp_path, runs_dir):
    dapt = _dapt_file(tmp_path / "dapt.pth", val=1.0)
    cls = _flat_cls_file(tmp_path / "cls.pth", val=5.0, dapt_ckpt=dapt)
    enc, desc = dr.load_reconstruction_checkpoint(cls, torch.device("cpu"))
    # encoders come from the DAPT file, Perceiver from the Stage 3.2 file
    assert torch.equal(enc["spatial"]["w"], torch.full((2,), 1.0))
    assert torch.equal(enc["perceiver"]["w"], torch.full((2,), 5.0))
    assert "encoders from DAPT" in desc


def test_stage32_resolves_dapt_source_from_run_json(tmp_path, runs_dir):
    dapt = _dapt_file(tmp_path / "dapt.pth", val=1.0)
    cls = _flat_cls_file(tmp_path / "cls.pth", val=5.0)  # old file, no embedded source
    run = {"config": {"dapt_ckpt": dapt, "encoder_lr": 0.0},
           "history": {"ckpt_path": str(tmp_path / "cls.pth")}}
    (runs_dir / "run.json").write_text(json.dumps(run))
    enc, _ = dr.load_reconstruction_checkpoint(cls, torch.device("cpu"))
    assert torch.equal(enc["conv"]["w"], torch.full((2,), 1.0))


def test_stage32_with_stored_encoders_loads_sections(tmp_path, runs_dir):
    p = tmp_path / "stored.pth"
    torch.save({"cls_token": torch.zeros(1), "perceiver.w": torch.full((2,), 5.0),
                "spatial": _sec(7.0), "temporal": _sec(7.0), "conv": _sec(7.0),
                "perceiver": _sec(5.0), "encoder_lr": 1e-6}, p)
    enc, desc = dr.load_reconstruction_checkpoint(str(p), torch.device("cpu"))
    assert torch.equal(enc["spatial"]["w"], torch.full((2,), 7.0))
    assert "encoders stored" in desc


def test_encoder_lr_positive_without_stored_encoders_is_rejected(tmp_path, runs_dir):
    dapt = _dapt_file(tmp_path / "dapt.pth")
    cls = _flat_cls_file(tmp_path / "cls.pth", dapt_ckpt=dapt)
    # encoder_lr is only recorded in the run JSON for flat classifier files
    run = {"config": {"dapt_ckpt": dapt, "encoder_lr": 1e-6},
           "history": {"ckpt_path": str(tmp_path / "cls.pth")}}
    (runs_dir / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="encoders were not saved"):
        dr.load_reconstruction_checkpoint(cls, torch.device("cpu"))


def test_flat_checkpoint_without_dapt_source_is_rejected(tmp_path, runs_dir):
    cls = _flat_cls_file(tmp_path / "cls.pth")
    with pytest.raises(ValueError, match="no DAPT encoder source"):
        dr.load_reconstruction_checkpoint(cls, torch.device("cpu"))


def test_unrecognised_checkpoint_is_rejected(tmp_path, runs_dir):
    p = tmp_path / "junk.pth"
    torch.save({"foo": torch.zeros(1)}, p)
    with pytest.raises(ValueError, match="unrecognised checkpoint"):
        dr.load_reconstruction_checkpoint(str(p), torch.device("cpu"))


def test_missing_file_raises_file_not_found(tmp_path, runs_dir):
    with pytest.raises(FileNotFoundError):
        dr.load_reconstruction_checkpoint(str(tmp_path / "nope.pth"), torch.device("cpu"))
