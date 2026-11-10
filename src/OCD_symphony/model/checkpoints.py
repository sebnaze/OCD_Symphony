"""Pretrained encoder / Perceiver construction and checkpoint loading.

Moved out of ``finetuning.py`` (Phase 6). ``finetuning.py`` re-exports every name.
"""
import os

import torch

from OCD_symphony.utils import proj_dir
from OCD_symphony.model.constants import (
    DIM, NUM_LAYERS, NUM_HEADS, MLP_DIM, MAX_LEN_SPATIAL, NUM_GRADIENTS,
    DELTA, MAX_LEN_TEMP, TIME_STEPS, ROIs,
)
from OCD_symphony.model.model_components import *


def build_pretrained_models(args, device, mask_ratio=0.0, encoder_ckpt=None,
                            spatial_mask_ratio=None, temporal_mask_ratio=None):
    """Construct the three pretrained encoders + the Perceiver and load weights.

    Parameters
    ----------
    mask_ratio : float
        Fallback masking ratio for the token maskers. ``0.0`` reproduces the
        original fine-tuning behaviour (no masking); ``>0`` enables the
        masked-reconstruction path used by DAPT (Stage 3.1).
    encoder_ckpt : dict or None
        If given, a dict produced by the DAPT stage with keys ``spatial``,
        ``temporal``, ``conv`` and ``perceiver`` (each a ``state_dict``). When
        provided, these DAPT-adapted weights are used *instead of* the
        HCP-AGING checkpoints.
    spatial_mask_ratio : float or None
        Masking ratio for the spatial encoder's token masker. Falls back to
        ``mask_ratio`` when ``None``.
    temporal_mask_ratio : float or None
        Masking ratio for the temporal encoder's token masker. Falls back to
        ``mask_ratio`` when ``None``.

    Note
    ----
    The context (Conv1D) encoder has no internal masker -- its mask is passed
    externally to ``forward(mask=...)`` (see ``dapt()``).

    Returns
    -------
    (spatial_model, temporal_model, conv_model, perceiver)
    """
    sp_ratio = mask_ratio if spatial_mask_ratio is None else spatial_mask_ratio
    tp_ratio = mask_ratio if temporal_mask_ratio is None else temporal_mask_ratio

    # ── Temporal encoder ────────────────────────────────────────────────────
    base_transformer = TemporalTransformer(
        NUM_LAYERS, DIM, NUM_HEADS, MLP_DIM, TIME_STEPS, MAX_LEN_TEMP, ROIs
    ).to(device)
    masker = TokenMasker1D(mask_ratio=tp_ratio)
    temporal_model = MaskedTemporalTransformer(
        base_transformer, masker, roi_dim=ROIs, hidden_dim=256
    ).to(device)

    # ── Spatial encoder ─────────────────────────────────────────────────────
    base_transformer = SpatialTransformer(
        num_layers=NUM_LAYERS,
        dim=DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        time_steps=TIME_STEPS,
        max_len=MAX_LEN_SPATIAL,
        num_gradients=NUM_GRADIENTS,
        delta=DELTA,
        pos_enc_type="learnable",
    ).to(device)
    masker = TokenMasker2D(mask_ratio=sp_ratio)
    spatial_model = MaskedSpatialTransformer(
        base_transformer, masker, TIME_STEPS, hidden_dim=256
    ).to(device)

    # ── Context (Conv1D) encoder ────────────────────────────────────────────
    conv_model = MaskedConv1DTemporalModel(
        input_channels=1, signal_dim=64, decoder_hidden_dims=[32, 8]
    ).to(device)

    # ── Load encoder weights (DAPT-adapted if provided, else HCP-AGING) ──────
    if encoder_ckpt is not None:
        spatial_model.load_state_dict(encoder_ckpt["spatial"])
        temporal_model.load_state_dict(encoder_ckpt["temporal"])
        conv_model.load_state_dict(encoder_ckpt["conv"])
    else:
        temporal_model.load_state_dict(
            torch.load(
                os.path.join(
                    proj_dir, 'models',
                    f"temporal_transformer_pretrained_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
                ), map_location=device,
            )
        )
        spatial_model.load_state_dict(
            torch.load(
                os.path.join(
                    proj_dir, 'models',
                    f"spatial_transformer_pretrained_learnablePE_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
                ), map_location=device,
            )
        )
        conv_model.load_state_dict(
            torch.load(
                os.path.join(
                    proj_dir, 'models',
                    f"masked_conv1d_temporal_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
                ), map_location=device,
            )
        )

    # ── Perceiver ───────────────────────────────────────────────────────────
    perceiver = PerceiverIOModelROIWise(
        input_dim=64 * 3,
        latent_dim=DIM,
        num_rois=ROIs,
        num_heads=NUM_HEADS,
        time_steps=TIME_STEPS,
    ).to(device)
    if encoder_ckpt is not None:
        perceiver.load_state_dict(encoder_ckpt["perceiver"])
    else:
        perceiver.load_state_dict(
            torch.load(
                os.path.join(
                    proj_dir, 'models',
                    f"perceiver_fused_pretrained_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
                ), map_location=device,
            )
        )

    return spatial_model, temporal_model, conv_model, perceiver


def _load_perceiver_from_classifier_ckpt(ckpt_path, perceiver, device, is_main=True):
    """Load Perceiver weights from a finetuned PerceiverClassifier checkpoint.

    Classifier checkpoints store ``perceiver.*`` and ``classifier.*`` keys only,
    so the encoders are not restored here. Raises if the Perceiver keys do not
    match exactly.
    """
    state = torch.load(ckpt_path, map_location=device)
    prefix = "perceiver."
    perceiver_state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not perceiver_state:
        raise ValueError(f"No 'perceiver.*' keys found in {ckpt_path}; cannot initialise DAPT from it.")
    perceiver.load_state_dict(perceiver_state, strict=True)
    if is_main:
        print(f"[DAPT] Perceiver initialised from {ckpt_path} "
              f"({len(perceiver_state)} tensors); encoders use HCP-AGING init.")
