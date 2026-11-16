"""Stage 3.1 - Domain-Adaptive Continued Pretraining (DAPT).

Moved out of ``finetuning.py`` (Phase 6). Helpers that stay in ``finetuning`` are reached
through the module object (``_ft``) so ``importlib.reload(finetuning)`` remains effective.
"""
import os
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import wandb

from OCD_symphony.utils import proj_dir
from OCD_symphony.model import finetuning as _ft
from OCD_symphony.model.checkpoints import (
    build_pretrained_models, _load_perceiver_from_classifier_ckpt,
)
from OCD_symphony.model.config import (
    _parse_dapt_weights, _resolve_dapt_mask_ratios, _resolve_dapt_loss, FinetuneConfig,
)
from OCD_symphony.model.model_components import *


# VICReg coefficients (Bardes et al., 2022): invariance, variance, covariance, variance margin.
VICREG_INV, VICREG_VAR, VICREG_COV, VICREG_GAMMA = 25.0, 25.0, 1.0, 1.0


def _load_dapt_encoder_init(path, device):
    """Load a Stage 3.1 DAPT checkpoint dict (spatial/temporal/conv/perceiver state_dicts).

    Raises ``ValueError`` if the file is not a DAPT checkpoint (e.g. a classifier checkpoint,
    which belongs in ``dapt_init_ckpt``).
    """
    if not os.path.isfile(path):
        raise ValueError(f"DAPT init checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    required = ("spatial", "temporal", "conv", "perceiver")
    missing = [k for k in required if not isinstance(state, dict) or k not in state]
    if missing:
        hint = ""
        if isinstance(state, dict) and any(str(k).startswith("perceiver.") for k in state):
            hint = " It looks like a classifier checkpoint; use dapt_init_ckpt for it."
        raise ValueError(
            f"{path} is not a DAPT checkpoint: missing {missing}.{hint}"
        )
    return state


def vicreg_loss(z1, z2, gamma=VICREG_GAMMA, inv=VICREG_INV, var=VICREG_VAR, cov=VICREG_COV, eps=1e-4):
    """VICReg loss between two views of the same embeddings.

    ``z1`` and ``z2`` are flattened to ``(N, D)`` (N = batch x ROI samples). Returns
    ``(total, invariance, variance, covariance)``.

    * invariance: MSE between the two views.
    * variance:   hinge keeping each dimension's std above ``gamma`` (anti-collapse).
    * covariance: squared off-diagonal entries of the covariance matrix, / D (decorrelation).
    """
    z1 = z1.reshape(-1, z1.shape[-1])
    z2 = z2.reshape(-1, z2.shape[-1])
    n, d = z1.shape
    l_inv = F.mse_loss(z1, z2)

    def _var(z):
        std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
        return F.relu(gamma - std).mean()

    def _cov(z):
        zc = z - z.mean(dim=0)
        c = (zc.T @ zc) / max(n - 1, 1)
        off = c - torch.diag(torch.diag(c))
        return off.pow(2).sum() / d

    l_var = 0.5 * (_var(z1) + _var(z2))
    l_cov = 0.5 * (_cov(z1) + _cov(z2))
    total = inv * l_inv + var * l_var + cov * l_cov
    return total, l_inv, l_var, l_cov


def dapt(trainset, valset, args, local_rank=0, epoch_callback=None,
         exclusion_report=None):
    """Stage 3.1 — Domain-Adaptive Continued Pretraining (DAPT).

    Unfreezes *all* modules (spatial / temporal / conv encoders + their decoders
    and the Perceiver) and continues the self-supervised reconstruction
    objective on HCP-YA data.  No labels are used.

        loss = w_sp * L_spatial + w_tp * L_temporal
             + w_cv * L_conv    + w_pc * L_perceiver
             [+ w_vicreg * VICReg(Perceiver latents)   if --dapt_loss vicreg]

    The per-module terms are masked-reconstruction MSEs combined with the raw
    ``--dapt_weights`` (spatial,temporal,conv,perceiver). In ``vicreg`` mode the
    encoder->Perceiver path is run on a second, independently masked view; the
    VICReg invariance term compares the two views' Perceiver latents, and the
    variance/covariance terms regularise them. Reconstruction MSEs use view 1 only.

    Robustness features
    -------------------
    * **Per-batch guard** – a non-finite loss or gradient norm skips the
      optimiser/scheduler step (and aborts if >50% of an epoch is skipped).
    * **Gradient clipping** – ``--grad_clip`` (global norm, 0 disables).
    * **Deterministic validation** – the validation masks are drawn from a fixed
      seed (``--val_mask_seed``) inside ``torch.random.fork_rng`` so they are
      identical every epoch and do not disturb the training RNG stream.
    * **Per-term logging** – spatial / temporal / conv / perceiver losses are
      tracked separately for train and val (console, history, W&B).

    ``exclusion_report`` (optional) is the dataset-validation report produced by
    ``make_hcp_dataset_from_pt``; it is stored in ``history["excluded_data"]``.

    Returns the path to the best checkpoint (selected by validation
    reconstruction loss), or ``None`` on non-main ranks.
    """
    if FinetuneConfig.from_args(args).dapt_encoder_init and FinetuneConfig.from_args(args).dapt_init_ckpt:
        raise ValueError(
            "Set either dapt_encoder_init (DAPT checkpoint) or dapt_init_ckpt "
            "(classifier checkpoint), not both."
        )
    if args.ddp:
        raise NotImplementedError(
            "DAPT (--mode dapt) currently supports single-process training only; "
            "run without --ddp."
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = _ft.is_main_process()

    cfg = FinetuneConfig.from_args(args)
    sp_ratio, tp_ratio, cx_ratio = _resolve_dapt_mask_ratios(args)
    w_sp, w_tp, w_cv, w_pc = _parse_dapt_weights(args)
    loss_mode, vicreg_w = _resolve_dapt_loss(args)
    use_vicreg = loss_mode == "vicreg"
    if is_main:
        print(
            f"[DAPT] mask ratios — spatial={sp_ratio:.2f} "
            f"temporal={tp_ratio:.2f} context={cx_ratio:.2f}"
        )
        if use_vicreg:
            print(f"[DAPT] loss = MSE terms + {vicreg_w:g} * VICReg(perceiver latents)")

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=False,
    )

    # ── Build models with masking enabled ───────────────────────────────────
    # Init sources (mutually exclusive, checked at the top of dapt()):
    #   * dapt_encoder_init: a DAPT checkpoint -> encoders + Perceiver restored.
    #   * dapt_init_ckpt:    a classifier checkpoint -> Perceiver only (encoders HCP-AGING).
    #   * neither:           HCP-AGING init for everything.
    encoder_ckpt = None
    if cfg.dapt_encoder_init:
        encoder_ckpt = _load_dapt_encoder_init(cfg.dapt_encoder_init, device)
    try:
        spatial_model, temporal_model, conv_model, perceiver = build_pretrained_models(
            args, device, spatial_mask_ratio=sp_ratio, temporal_mask_ratio=tp_ratio,
            encoder_ckpt=encoder_ckpt,
        )
    except RuntimeError as exc:
        if encoder_ckpt is not None:
            raise RuntimeError(
                f"Could not load DAPT checkpoint {cfg.dapt_encoder_init} into the current "
                f"architecture (check --nL/--d/etc. match the source run): {exc}"
            ) from exc
        raise
    history_init_source = cfg.dapt_encoder_init or None
    if is_main and encoder_ckpt is not None:
        print(f"[DAPT] encoders+Perceiver initialised from {cfg.dapt_encoder_init}")

    # Classifier checkpoints only contain Perceiver weights, so encoders keep HCP-AGING init.
    init_ckpt_path = cfg.dapt_init_ckpt
    if init_ckpt_path:
        _load_perceiver_from_classifier_ckpt(init_ckpt_path, perceiver, device, is_main)

    # ── Unfreeze everything ─────────────────────────────────────────────────
    for model in (spatial_model, temporal_model, conv_model, perceiver):
        model.train(True)
        for param in model.parameters():
            param.requires_grad = True

    time_masker = TimePointMasker(mask_ratio=cx_ratio)

    trainable = [
        p for m in (spatial_model, temporal_model, conv_model, perceiver)
        for p in m.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(0.1 * args.n_epochs * len(train_loader)),
        args.n_epochs * len(train_loader),
    )

    def _forward_losses(fmri, fc, gm):
        """Return ((L_spatial, L_temporal, L_conv, L_perceiver), vic) for a batch.

        ``vic`` is ``None`` in MSE mode, else ``(total, inv, var, cov)`` of the VICReg
        term between two independently masked views of the Perceiver latents.
        """
        l_sp, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
        l_tp, _, _, _, time_embeds, _ = temporal_model(fmri)
        conv_mask = time_masker(fmri)
        preds, signal_context = conv_model(fmri, mask=conv_mask)
        l_cv = (
            ((preds - fmri) * (1 - conv_mask)).pow(2).sum()
            / (1 - conv_mask).sum().clamp(min=1)
        )
        recon, latents, _ = perceiver(roi_embeds, time_embeds, signal_context, roi_pos)
        l_pc = F.mse_loss(recon, fmri)
        terms = (l_sp, l_tp, l_cv, l_pc)
        if not use_vicreg:
            return terms, None

        # Second view: fresh masks for every masked encoder (spatial, temporal, conv/context).
        _, _, _, _, roi_embeds2, roi_pos2 = spatial_model(gm, fmri, fc)
        _, _, _, _, time_embeds2, _ = temporal_model(fmri)
        preds2, signal_context2 = conv_model(fmri, mask=time_masker(fmri))
        _, latents2, _ = perceiver(roi_embeds2, time_embeds2, signal_context2, roi_pos2)
        vic = vicreg_loss(latents, latents2)
        return terms, vic

    def _weighted_total(terms, vic=None):
        total = sum(w * t for w, t in zip(term_weights, terms))
        if vic is not None:
            total = total + vicreg_w * vic[0]
        return total

    term_names = ("spatial", "temporal", "conv", "perceiver")
    term_weights = (w_sp, w_tp, w_cv, w_pc)
    grad_clip = float(cfg.grad_clip or 0.0)
    val_mask_seed = int(cfg.val_mask_seed)
    # fork_rng expects CUDA device *indices*
    fork_devices = [device.index if device.index is not None else 0] \
        if device.type == "cuda" else []

    def _mean_square(ds, n=512):
        """Mean-square of the (normalised) fMRI of up to ``n`` evenly spaced samples."""
        if len(ds) == 0:
            return float("nan")
        idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
        return float(torch.stack([ds[int(i)][0] for i in idx]).float().pow(2).mean())

    # ── Run-start sanity check: train vs val signal power ───────────────────
    ms_train, ms_val = _mean_square(trainset), _mean_square(valset)
    if is_main:
        print(f"[DAPT] signal mean-square — train={ms_train:.3g} | val={ms_val:.3g}")
        ratio = max(ms_train, ms_val) / max(min(ms_train, ms_val), 1e-12)
        if not np.isfinite(ratio) or ratio > 100:
            print(
                "⚠ WARNING [DAPT]: train/val signal power differs by >100x — "
                "an outlier subject/run is probably contaminating one split. "
                "Validation loss will not be meaningful."
            )

    history = {"is_dapt": True, "train_recon": [], "val_recon": []}
    for name in term_names:
        history[f"train_{name}"] = []
        history[f"val_{name}"] = []
    history["grad_norm"] = []
    history["skipped_batches"] = []
    history["val_skipped_batches"] = []
    history["excluded_data"] = list(exclusion_report) if exclusion_report else []
    history["init_source"] = history_init_source
    # VICReg diagnostics (only in vicreg mode): total = weighted VICReg, then its parts.
    vic_names = ("total", "inv", "var", "cov")
    if use_vicreg:
        for name in vic_names:
            history[f"train_vicreg_{name}"] = []
            history[f"val_vicreg_{name}"] = []
        history["vicreg_weight"] = vicreg_w
    best_val_recon = float("inf")
    best_epoch = -1
    best_state = None
    t0 = time()

    for epoch in range(args.n_epochs):
        t1 = time()
        for model in (spatial_model, temporal_model, conv_model, perceiver):
            model.train(True)

        total = torch.zeros((), device=device)
        term_sums = [torch.zeros((), device=device) for _ in term_names]
        vic_sums = [torch.zeros((), device=device) for _ in vic_names]
        gnorm_sum = 0.0
        n_ok, n_skipped = 0, 0
        for fmri, fc, gm, _labels in train_loader:
            fmri, fc, gm = fmri.to(device), fc.to(device), gm.to(device)

            terms, vic = _forward_losses(fmri, fc, gm)
            loss = _weighted_total(terms, vic)

            # ── Guard 1: non-finite loss -> skip the whole step ────────────
            if not torch.isfinite(loss):
                n_skipped += 1
                if is_main and n_skipped <= 5:
                    bad = [n for n, t in zip(term_names, terms) if not torch.isfinite(t)]
                    if vic is not None and not torch.isfinite(vic[0]):
                        bad.append("vicreg")
                    print(f"[DAPT] ⚠ non-finite loss in term(s) {bad}; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # ── Gradient clipping (max_norm=inf only measures the norm) ────
            gnorm = torch.nn.utils.clip_grad_norm_(
                trainable, grad_clip if grad_clip > 0 else float("inf")
            )

            # ── Guard 2: non-finite gradient -> skip the optimiser step ────
            if not torch.isfinite(gnorm):
                n_skipped += 1
                if is_main and n_skipped <= 5:
                    print("[DAPT] ⚠ non-finite gradient norm; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.step()
            scheduler.step()

            total += loss.detach()
            for i, t in enumerate(terms):
                term_sums[i] += t.detach()
            if use_vicreg:
                for i, t in enumerate(vic):
                    vic_sums[i] += t.detach()
            gnorm_sum += float(gnorm)
            n_ok += 1

        if len(train_loader) > 0 and n_skipped > 0.5 * len(train_loader):
            raise RuntimeError(
                f"[DAPT] {n_skipped}/{len(train_loader)} batches skipped in epoch "
                f"{epoch+1} (non-finite loss/gradient). Aborting — inspect the data "
                "(see the exclusion report) and the learning rate."
            )

        denom = max(n_ok, 1)
        avg_train = total.item() / denom
        avg_train_terms = [s.item() / denom for s in term_sums]
        avg_train_vic = [s.item() / denom for s in vic_sums]
        avg_gnorm = gnorm_sum / denom

        # ── Validation (raw weighted reconstruction loss) ────────────────────
        # Deterministic masks: the RNG is forked and re-seeded so every epoch
        # sees identical val masks, and the training RNG stream is untouched.
        for model in (spatial_model, temporal_model, conv_model, perceiver):
            model.eval()
        val_total = torch.zeros((), device=device)
        val_term_sums = [torch.zeros((), device=device) for _ in term_names]
        val_vic_sums = [torch.zeros((), device=device) for _ in vic_names]
        val_ok, val_skipped = 0, 0
        with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(val_mask_seed)
            for fmri, fc, gm, _labels in val_loader:
                fmri, fc, gm = fmri.to(device), fc.to(device), gm.to(device)
                terms, vic = _forward_losses(fmri, fc, gm)
                vloss = _weighted_total(terms, vic)
                if not torch.isfinite(vloss):
                    val_skipped += 1
                    continue
                val_total += vloss
                for i, t in enumerate(terms):
                    val_term_sums[i] += t
                if use_vicreg:
                    for i, t in enumerate(vic):
                        val_vic_sums[i] += t
                val_ok += 1
        val_denom = max(val_ok, 1)
        avg_val = val_total.item() / val_denom if val_ok > 0 else float("nan")
        avg_val_terms = [s.item() / val_denom for s in val_term_sums]
        avg_val_vic = [s.item() / val_denom for s in val_vic_sums]

        history["train_recon"].append(avg_train)
        history["val_recon"].append(avg_val)
        for i, name in enumerate(term_names):
            history[f"train_{name}"].append(avg_train_terms[i])
            history[f"val_{name}"].append(avg_val_terms[i])
        if use_vicreg:
            for i, name in enumerate(vic_names):
                history[f"train_vicreg_{name}"].append(avg_train_vic[i])
                history[f"val_vicreg_{name}"].append(avg_val_vic[i])
        history["grad_norm"].append(avg_gnorm)
        history["skipped_batches"].append(n_skipped)
        history["val_skipped_batches"].append(val_skipped)

        if is_main:
            print(
                f"[DAPT Epoch {epoch+1}] Train Recon: {avg_train:.4f} | "
                f"Val Recon: {avg_val:.4f} | Time: {(time()-t1):.1f}s"
            )
            print(
                "   ↳ train " + " ".join(
                    f"{n[:2]}={v:.4f}" for n, v in zip(term_names, avg_train_terms)
                )
                + " | val " + " ".join(
                    f"{n[:2]}={v:.4f}" for n, v in zip(term_names, avg_val_terms)
                )
                + f" | gnorm={avg_gnorm:.3f} | skipped={n_skipped}"
                + (f" | val_skipped={val_skipped}" if val_skipped else "")
            )
            if use_vicreg:
                print(
                    "   ↳ vicreg train " + " ".join(
                        f"{n}={v:.4f}" for n, v in zip(vic_names[1:], avg_train_vic[1:])
                    )
                    + " | val " + " ".join(
                        f"{n}={v:.4f}" for n, v in zip(vic_names[1:], avg_val_vic[1:])
                    )
                    + f" | w={vicreg_w:g} | total train={avg_train_vic[0]:.4f} val={avg_val_vic[0]:.4f}"
                )
            if args.use_wandb:
                log_dict = {
                    "dapt_train_recon": avg_train,
                    "dapt_val_recon": avg_val,
                    "dapt/grad_norm": avg_gnorm,
                    "dapt/skipped_batches": n_skipped,
                    "dapt/val_skipped_batches": val_skipped,
                }
                for i, name in enumerate(term_names):
                    log_dict[f"dapt/train_{name}"] = avg_train_terms[i]
                    log_dict[f"dapt/val_{name}"] = avg_val_terms[i]
                if use_vicreg:
                    for i, name in enumerate(vic_names):
                        log_dict[f"dapt/train_vicreg_{name}"] = avg_train_vic[i]
                        log_dict[f"dapt/val_vicreg_{name}"] = avg_val_vic[i]
                wandb.log(log_dict, step=epoch + 1)

        if np.isfinite(avg_val) and avg_val < best_val_recon:
            best_val_recon = avg_val
            best_epoch = epoch + 1
            best_state = {
                # clone(): on CPU, .cpu() does not copy, so later training steps
                # would otherwise mutate the "best" checkpoint in place.
                "spatial": {k: v.detach().cpu().clone() for k, v in spatial_model.state_dict().items()},
                "temporal": {k: v.detach().cpu().clone() for k, v in temporal_model.state_dict().items()},
                "conv": {k: v.detach().cpu().clone() for k, v in conv_model.state_dict().items()},
                "perceiver": {k: v.detach().cpu().clone() for k, v in perceiver.state_dict().items()},
                "args": vars(args) if hasattr(args, "__dict__") else {},
                "epoch": best_epoch,
                "val_loss": best_val_recon,
            }

        if epoch_callback is not None and is_main:
            cb_payload = {
                "epoch": epoch + 1,
                "task": "dapt",
                "train_recon": avg_train,
                "val_recon": avg_val,
                "grad_norm": avg_gnorm,
                "skipped_batches": n_skipped,
            }
            for i, name in enumerate(term_names):
                cb_payload[f"train_{name}"] = avg_train_terms[i]
                cb_payload[f"val_{name}"] = avg_val_terms[i]
            if use_vicreg:
                for i, name in enumerate(vic_names):
                    cb_payload[f"train_vicreg_{name}"] = avg_train_vic[i]
                    cb_payload[f"val_vicreg_{name}"] = avg_val_vic[i]
            epoch_callback(epoch + 1, cb_payload)

    # ── Save best checkpoint ────────────────────────────────────────────────
    ckpt_path = None
    if is_main and best_state is not None:
        tag = _ft.get_experiment_tag(args, mode="dapt")
        ckpt_name = cfg.dapt_ckpt_name or f"dapt_{tag}.pth"
        ckpt_path = os.path.join(proj_dir, "models", ckpt_name)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(best_state, ckpt_path)
        history["ckpt_path"] = ckpt_path
        print(
            f"✅ DAPT complete. Best epoch {best_epoch} "
            f"(val recon {best_val_recon:.4f}) in {(time()-t0)/60:.1f} min."
        )
        print(f"[DAPT] Saved checkpoint to {ckpt_path}")
        _ft.save_run_history(history, None, None, args, tag)

    return ckpt_path, history
