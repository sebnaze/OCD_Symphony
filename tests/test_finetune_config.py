"""FinetuneConfig: defaults for namespaces missing fields, and the backbone_lr fallbacks."""
from types import SimpleNamespace

from OCD_symphony.model.finetuning import FinetuneConfig, get_experiment_tag


def test_bare_namespace_gets_historical_defaults():
    cfg = FinetuneConfig.from_args(SimpleNamespace())
    assert cfg.dataset == "hcp_ya"
    assert cfg.label == "gender"
    assert cfg.pooling == "mean"
    assert cfg.dropout == 0.3
    assert cfg.patience == 5
    assert cfg.freeze_perceiver is False
    assert cfg.dapt_weights == "1,1,1,1"
    assert cfg.current_mode == "single"
    assert cfg.current_fold_idx is None


def test_present_fields_override_defaults():
    cfg = FinetuneConfig.from_args(SimpleNamespace(pooling="attention", patience=0))
    assert cfg.pooling == "attention"
    assert cfg.patience == 0


def test_private_mode_attributes_map_to_public_fields():
    cfg = FinetuneConfig.from_args(SimpleNamespace(_current_mode="cv", _current_fold_idx=3))
    assert cfg.current_mode == "cv"
    assert cfg.current_fold_idx == 3


def test_config_is_immutable():
    cfg = FinetuneConfig()
    try:
        cfg.pooling = "flatten"
    except Exception:
        return
    raise AssertionError("FinetuneConfig should be frozen")


def test_tag_backbone_lr_fallback_is_1e5_when_absent():
    args = SimpleNamespace(label="gender", dataset="hcp_ya", freeze_perceiver=False,
                           learning_rate=1e-4)
    assert "_bblr1e-05" in get_experiment_tag(args, mode="single")


def test_tag_uses_explicit_backbone_lr():
    args = SimpleNamespace(label="gender", dataset="hcp_ya", freeze_perceiver=False,
                           learning_rate=1e-4, backbone_lr=5e-5)
    assert "_bblr5e-05" in get_experiment_tag(args, mode="single")
