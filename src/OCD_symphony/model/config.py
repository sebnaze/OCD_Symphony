"""Run configuration (``FinetuneConfig``) and DAPT option parsing.

Moved out of ``finetuning.py`` (Phase 6). ``finetuning.py`` re-exports every name.
"""
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np

from OCD_symphony.model.constants import LEARNING_RATE


_CONFIG_ATTR_SOURCE = {"current_mode": "_current_mode", "current_fold_idx": "_current_fold_idx"}


@dataclass(frozen=True)
class FinetuneConfig:
    """Typed view of the run options, with the defaults the training code has always used.

    Built from any namespace (argparse or the notebook's ``SimpleNamespace``); a field
    missing from the namespace takes its default here. ``backbone_lr=None`` keeps the two
    historical fallbacks: ``learning_rate * 0.1`` for training and ``1e-5`` for the
    experiment tag (argparse always sets it, so CLI runs are unaffected).
    """

    dataset: str = "hcp_ya"
    label: str = "gender"
    mode: str = "cls"
    learning_rate: float = LEARNING_RATE
    backbone_lr: Optional[float] = None
    freeze_perceiver: bool = False
    pooling: str = "mean"
    dropout: float = 0.3
    recon_weight: float = 0.0
    patience: int = 5
    encoder_lr: float = 0.0
    fold_idx: Optional[int] = 0
    dapt_ckpt: Optional[str] = None
    dapt_init_ckpt: Optional[str] = None
    # Stage 3.1 start from a DAPT checkpoint: encoders + Perceiver (mutually exclusive with dapt_init_ckpt).
    dapt_encoder_init: Optional[str] = None
    dapt_ckpt_name: Optional[str] = None
    dapt_mask_ratio: Optional[float] = None
    dapt_mask_ratio_spatial: float = 0.7
    dapt_mask_ratio_temporal: float = 0.7
    dapt_mask_ratio_context: float = 0.5
    dapt_weights: str = "1,1,1,1"
    # Stage 3.1 loss: "mse" (masked reconstruction only) or "vicreg" (MSE + VICReg on Perceiver latents).
    dapt_loss: str = "mse"
    vicreg_weight: float = 1.0
    grad_clip: float = 1.0
    val_mask_seed: int = 1234
    max_abs: float = 1e4
    current_mode: str = "single"
    current_fold_idx: Optional[int] = None
    # "sample": one classifier input per window. "subject": Perceiver latents are
    # mean-pooled over each subject's windows, and one vector per subject is classified.
    target_scope: str = "sample"
    # Label-permutation null: permute subject labels once before splitting (Stages 1/2/3.2).
    shuffle_labels: bool = False
    seed: int = 42
    # None falls back to ``seed``.
    label_perm_seed: Optional[int] = None

    @property
    def effective_perm_seed(self) -> int:
        """Seed of the label permutation (``label_perm_seed``, else ``seed``)."""
        return self.label_perm_seed if self.label_perm_seed is not None else self.seed

    @classmethod
    def from_args(cls, args):
        """Copy every field the namespace provides; the rest keep their defaults."""
        kwargs = {}
        for f in fields(cls):
            src = _CONFIG_ATTR_SOURCE.get(f.name, f.name)
            if hasattr(args, src):
                kwargs[f.name] = getattr(args, src)
        return cls(**kwargs)


def _resolve_dapt_mask_ratios(args):
    """Resolve per-encoder DAPT mask ratios -> (spatial, temporal, context).

    If the shared ``--dapt_mask_ratio`` is set it overrides all three;
    otherwise the per-encoder values are used (defaults 0.7 / 0.7 / 0.5).
    """
    cfg = FinetuneConfig.from_args(args)
    shared = cfg.dapt_mask_ratio
    if shared is not None:
        shared = float(shared)
        return shared, shared, shared
    sp = cfg.dapt_mask_ratio_spatial
    tp = cfg.dapt_mask_ratio_temporal
    cx = cfg.dapt_mask_ratio_context
    return float(sp), float(tp), float(cx)


DAPT_LOSSES = ("mse", "vicreg")


def _resolve_dapt_loss(args):
    """Resolve the Stage 3.1 loss -> (mode, vicreg_weight). mode is 'mse' or 'vicreg'."""
    cfg = FinetuneConfig.from_args(args)
    mode = str(cfg.dapt_loss or "mse").lower()
    if mode not in DAPT_LOSSES:
        raise ValueError(f"--dapt_loss must be one of {DAPT_LOSSES}, got {mode!r}")
    w = float(cfg.vicreg_weight)
    if not np.isfinite(w) or w <= 0:
        raise ValueError(f"--vicreg_weight must be a positive finite number, got {w!r}")
    return mode, w


def _parse_dapt_weights(args):
    """Parse the ``--dapt_weights`` string into 4 floats (sp,tp,conv,perceiver)."""
    raw = FinetuneConfig.from_args(args).dapt_weights
    try:
        vals = [float(x) for x in str(raw).split(",")]
    except ValueError:
        raise ValueError(f"--dapt_weights must be comma-separated floats, got {raw!r}")
    if len(vals) != 4:
        raise ValueError(
            "--dapt_weights must have exactly 4 values "
            f"(spatial,temporal,conv,perceiver), got {len(vals)}"
        )
    return vals
