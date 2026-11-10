"""CLI validation in parse_args and the explicit DAPT key in format_epoch_summary (Phase 7)."""
import pytest

from OCD_symphony.model.finetuning import format_epoch_summary, parse_args


def _parse(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["finetuning.py", *argv])
    return parse_args()


def test_label_must_belong_to_dataset(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--dataset", "hcp_ya", "--label", "condition")
    assert "not available for --dataset" in capsys.readouterr().err


def test_condition_label_is_valid_for_ocd(monkeypatch):
    args = _parse(monkeypatch, "--dataset", "ocd", "--label", "condition")
    assert args.label == "condition"


def test_dapt_with_ddp_is_rejected(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--mode", "dapt", "--ddp")
    assert "does not support --ddp" in capsys.readouterr().err


def test_recon_weight_out_of_range_is_rejected(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--recon_weight", "1.5")
    assert "must be in [0, 1]" in capsys.readouterr().err


def test_recon_weight_boundaries_are_accepted(monkeypatch):
    assert _parse(monkeypatch, "--recon_weight", "0.0").recon_weight == 0.0
    assert _parse(monkeypatch, "--recon_weight", "1.0").recon_weight == 1.0


def test_dapt_payload_is_identified_by_task_key():
    s = format_epoch_summary({"task": "dapt", "train_recon": 0.1, "val_recon": 0.2})
    assert "train recon 0.1000" in s
    assert "val recon 0.2000" in s
