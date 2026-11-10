"""Progress reporting: epoch summary text, loader load events, and the per-epoch callback."""
from types import SimpleNamespace

from OCD_symphony.model.finetuning import format_epoch_summary


def test_classification_summary_labels_train_accuracy_as_running():
    s = format_epoch_summary({"task": "classification", "train_acc": 0.6, "val_acc": 0.5})
    assert "train acc (running, train mode) 0.600" in s
    assert "val acc 0.500" in s


def test_regression_summary_shows_mae_and_r2():
    s = format_epoch_summary({"task": "regression", "train_mae": 3.14159,
                              "val_mae": 4.0, "val_r2": -0.12345})
    assert "train MAE 3.14" in s
    assert "val MAE 4.00" in s
    assert "val R² -0.123" in s


def test_dapt_summary_uses_reconstruction_loss():
    s = format_epoch_summary({"train_recon": 0.1, "val_recon": 0.2})
    assert "train recon 0.1000" in s
    assert "val recon 0.2000" in s


def test_nan_and_missing_values_render_as_dash():
    s = format_epoch_summary({"task": "classification", "train_acc": float("nan")})
    assert "train acc (running, train mode) —" in s
    assert "val acc —" in s


def test_loader_emits_one_load_event_per_file():
    from OCD_symphony.data_preparation.make_hcp_dataset import (
        DATASET_DIRS, make_hcp_dataset_from_pt,
    )
    events = []
    make_hcp_dataset_from_pt(
        path=DATASET_DIRS["hcp_ya"], label="gender", n_subjs=3,
        dataset="hcp_ya", num_io_workers=2, progress=events.append,
    )
    assert [e["stage"] for e in events] == ["load"] * 3
    assert [e["done"] for e in events] == [1, 2, 3]
    assert all(e["total"] == 3 for e in events)


def test_epoch_callback_receives_accuracy_payload():
    from OCD_symphony.model.finetuning import run_single_split
    events = []
    args = SimpleNamespace(
        dataset="hcp_ya", label="gender", n_subjs=10, n_epochs=1, batch_size=8,
        learning_rate=1e-4, backbone_lr=1e-5, freeze_perceiver=True, n_folds=5,
        fold_idx=None, test_frac=0.15, val_frac=0.15, seed=42, use_wandb=False,
        ddp=False, cv=False, pooling="mean", dropout=0.0, recon_weight=0.0,
        patience=0, encoder_lr=0.0, dapt_ckpt=None, mode="cls", max_abs=1e4,
        grad_clip=1.0, val_mask_seed=1234,
    )
    run_single_split(args, epoch_callback=lambda ep, m: events.append((ep, m)))
    assert len(events) == 1
    ep, payload = events[0]
    assert ep == 1 and payload["epoch"] == 1
    assert "train_acc" in payload and "val_acc" in payload
