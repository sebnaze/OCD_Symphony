import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from time import time, strftime
import json
import wandb

from OCD_symphony.utils import proj_dir
from OCD_symphony.data_preparation.make_hcp_dataset import (
    make_hcp_dataset_from_pt,
    subject_kfold,
    subject_train_val_test_split,
    make_dataloaders,
)
from OCD_symphony.model.model_components import *

# Level '1' filters out INFO logs, Level '2' filters out WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# DDP helpers -----------------------------------------------------------------
def ddp_setup():
    """Initialise the default distributed process group (NCCL backend).
    
    Reads RANK / WORLD_SIZE / LOCAL_RANK from environment (set by torchrun).
    Returns (rank, local_rank, world_size).  Falls back to single-GPU.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        print(f"[DDP] rank={rank}, local_rank={local_rank}, world_size={world_size}")
        return rank, local_rank, world_size
    return 0, 0, 1


def ddp_cleanup():
    """Clean up the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Return True only for rank-0 (the process that prints / logs / saves)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# === Device (constant – use local_rank argument at runtime) ===
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model parameters ===
DIM = 128
NUM_LAYERS = 6
NUM_HEADS = 4
MLP_DIM = 512
MAX_LEN_SPATIAL = 500
NUM_GRADIENTS = 10
DELTA = 0.5
MAX_LEN_TEMP = 5000

# === Training parameters ===
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# === Data parameters ===
TIME_STEPS = 200
ROIs = 450


def clean_gpu_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_run_history(history, test_acc, test_f1, args, tag):
    """Persist a run's per-epoch history and metrics to results/runs/<tag>_<id>.json.

    Enables offline re-visualization of a past run from the marimo notebook.
    """
    try:
        wandb_id = None
        if getattr(args, "use_wandb", False) and wandb.run is not None:
            wandb_id = getattr(wandb.run, "id", None)
        suffix = wandb_id if wandb_id else strftime("%Y%m%d_%H%M%S")

        out_dir = os.path.join(proj_dir, "results", "runs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{tag}_{suffix}.json")

        def _to_native(obj):
            if isinstance(obj, dict):
                return {k: _to_native(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_native(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        payload = {
            "meta": {
                "tag": tag,
                "wandb_id": wandb_id,
                "config": _to_native(vars(args)) if hasattr(args, "__dict__") else {},
            },
            "history": _to_native(history),
            "test_acc": None if test_acc is None else float(test_acc),
            "test_f1": None if test_f1 is None else float(test_f1),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[History] Saved run history to {out_path}")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[History] Failed to save run history: {exc}")


def get_experiment_tag(args, mode: str, fold_idx: int = None) -> str:
    """Construct descriptive experiment identifier string."""
    label_str = getattr(args, "label", "gender")
    stage_str = "stage1" if getattr(args, "freeze_perceiver", False) else "stage2"
    pool_str = getattr(args, "pooling", "mean")

    lr_str = f"lr{args.learning_rate:.0e}"
    if not getattr(args, "freeze_perceiver", False):
        bblr = getattr(args, "backbone_lr", 1e-5)
        lr_str += f"_bblr{bblr:.0e}"

    alpha = getattr(args, "recon_weight", 0.0)
    alpha_str = f"_a{alpha:.2f}" if alpha > 0.0 else ""

    if mode == "single":
        mode_str = "single"
    elif mode == "single_fold":
        f_idx = getattr(args, "fold_idx", 0) if fold_idx is None else fold_idx
        mode_str = f"fold{f_idx + 1}"
    elif mode == "cv":
        f_idx = 0 if fold_idx is None else fold_idx
        mode_str = f"cv_fold{f_idx + 1}"
    else:
        mode_str = str(mode)

    # Stage 3.1 — DAPT (reconstruction only) has its own tag shape
    if getattr(args, "mode", "cls") == "dapt":
        sp, tp, cx = _resolve_dapt_mask_ratios(args)
        lr = getattr(args, "learning_rate", LEARNING_RATE)
        return (
            f"{label_str}_dapt_mS{sp:.2f}_mT{tp:.2f}_mC{cx:.2f}"
            f"_lr{lr:.0e}_{mode_str}"
        )

    return f"{label_str}_{stage_str}_{pool_str}_{lr_str}{alpha_str}_{mode_str}"



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


def finetune(trainset, valset, testset, args, local_rank=0, epoch_callback=None):
    """
    Fine-tune the pretrained Perceiver classifier on the HCP dataset for gender classification.
    Uses Leave-Subject-Out cross-validation splits.

    When :attr:`args.ddp` is True the train DataLoader uses a
    ``DistributedSampler`` and the classifier is wrapped in
    ``DistributedDataParallel``.  Metrics are aggregated across all ranks with
    ``all_reduce`` / ``all_gather_object``.

    Parameters
    ----------
    trainset, valset, testset : Dataset or Subset
        Subject-level split datasets.
    args : argparse.Namespace
        Configuration.
    local_rank : int
        GPU index for this process.

    Returns
    -------
    (test_acc, test_f1) : tuple of float
        Accuracy and macro F1 on the held-out test set.
    """
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = is_main_process()

    # ── DataLoaders  (DistributedSampler for train in DDP mode) ────────────
    if args.ddp:
        train_sampler = DistributedSampler(
            trainset, shuffle=True,
            rank=dist.get_rank(), num_replicas=dist.get_world_size(),
        )
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size,
            sampler=train_sampler, num_workers=4,
            pin_memory=False, drop_last=False,
        )
    else:
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=False,
        )

    val_loader   = DataLoader(valset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=False)
    test_loader  = DataLoader(testset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=False)

    # ── Load pretrained models (HCP-AGING, or DAPT-adapted HCP-YA weights) ──
    dapt_ckpt = None
    dapt_ckpt_path = getattr(args, "dapt_ckpt", None)
    if dapt_ckpt_path:
        dapt_ckpt = torch.load(dapt_ckpt_path, map_location=device)
        if is_main:
            print(f"[Stage 3.2] Loading DAPT-adapted weights from {dapt_ckpt_path}")

    spatial_model, temporal_model, conv_model, perceiver = build_pretrained_models(
        args, device, mask_ratio=0.0, encoder_ckpt=dapt_ckpt
    )

    # Encoders act as feature extractors: frozen by default. Passing
    # --encoder_lr > 0 unfreezes them at a very low LR (Stage 3.2 option).
    encoder_lr = getattr(args, "encoder_lr", 0.0)
    train_encoders = encoder_lr > 0.0
    for model in [spatial_model, temporal_model, conv_model]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = train_encoders

    freeze_perceiver = getattr(args, "freeze_perceiver", False)
    for param in perceiver.parameters():
        param.requires_grad = not freeze_perceiver

    pooling = getattr(args, "pooling", "mean")
    dropout = getattr(args, "dropout", 0.3)
    classifier = PerceiverClassifier(
        perceiver,
        latent_dim=DIM,
        num_classes=2,
        pooling=pooling,
        dropout=dropout,
    ).to(device)

    # ── Wrap in DDP when distributed ────────────────────────────────────────
    if args.ddp:
        classifier = DDP(classifier, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer and Loss ──────────────────────────────────────────────────
    if freeze_perceiver:
        trainable_params = [p for p in classifier.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.learning_rate, weight_decay=1e-2
        )
    else:
        # Differential learning rates for Stage 2 / Stage 3.2
        raw_classifier = classifier.module if args.ddp else classifier
        backbone_params = [p for p in raw_classifier.perceiver.parameters() if p.requires_grad]
        head_params = [
            p for n, p in raw_classifier.named_parameters()
            if p.requires_grad and not n.startswith("perceiver.")
        ]
        backbone_lr = getattr(args, "backbone_lr", args.learning_rate * 0.1)

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": 1e-2},
            {"params": head_params, "lr": args.learning_rate, "weight_decay": 1e-2},
        ]

        if train_encoders:
            encoder_params = [
                p for model in (spatial_model, temporal_model, conv_model)
                for p in model.parameters() if p.requires_grad
            ]
            param_groups.insert(
                0, {"params": encoder_params, "lr": encoder_lr, "weight_decay": 1e-2}
            )
            if is_main:
                print(
                    f"[Stage 3.2] Encoders unfrozen (lr={encoder_lr:.1e}) | "
                    f"Perceiver backbone (lr={backbone_lr:.1e}) | "
                    f"Head (lr={args.learning_rate:.1e})"
                )
        elif is_main:
            print(f"[Stage 2] Perceiver backbone unfrozen (lr={backbone_lr:.1e}) | Head (lr={args.learning_rate:.1e})")

        optimizer = torch.optim.AdamW(param_groups)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(0.1 * args.n_epochs * len(train_loader)),
        args.n_epochs * len(train_loader),
    )

    # Inverse-frequency class weights to penalize collapsing into majority class
    if isinstance(trainset, Subset):
        train_labels_tensor = trainset.dataset.encoded_labels[trainset.indices]
    else:
        train_labels_tensor = trainset.encoded_labels

    class_counts = torch.bincount(train_labels_tensor)
    class_weights = (len(train_labels_tensor) / (len(class_counts) * class_counts.float())).to(device)
    if is_main:
        print(f"[Class Balance] counts: {class_counts.tolist()} | weights: {[round(w, 4) for w in class_weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Training Loop ───────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_acc": [],
        "val_f1": [],
    }
    t0 = time()
    for epoch in range(args.n_epochs):
        t1 = time()

        if args.ddp:
            train_sampler.set_epoch(epoch)

        classifier.train()
        for model in (spatial_model, temporal_model, conv_model):
            model.train(train_encoders)
        total_loss, total_loss_cls, total_loss_recon = 0.0, 0.0, 0.0
        correct = 0
        all_preds, all_labels = [], []

        alpha = getattr(args, "recon_weight", 0.0)

        for fmri, fc, gm, labels in train_loader:
            fmri, fc, gm, labels = (
                fmri.to(device),
                fc.to(device),
                gm.to(device),
                labels.to(device),
            )

            if train_encoders:
                # Encoders are being fine-tuned: keep the graph for gradients.
                _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
                _, _, _, _, time_embeds, _ = temporal_model(fmri)
                _, signal_context = conv_model(fmri, mask=None)
            else:
                with torch.no_grad():
                    _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
                    _, _, _, _, time_embeds, _ = temporal_model(fmri)
                    _, signal_context = conv_model(fmri, mask=None)

            logits, recon = classifier(roi_embeds, time_embeds, signal_context, roi_pos)
            loss_cls = criterion(logits, labels)
            if alpha > 0.0:
                # Scale MSE by baseline (~1.40) so both losses sit on approximately a [0, 1] scale
                loss_recon = F.mse_loss(recon, fmri) / 1.40
                loss = (1.0 - alpha) * loss_cls + alpha * loss_recon
                total_loss_recon += loss_recon.item()
            else:
                loss = loss_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            total_loss_cls += loss_cls.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        total = len(all_labels)

        # ── Aggregate training metrics across all DDP processes ─────────────
        if args.ddp:
            correct_t = torch.tensor(correct, device=device)
            total_t   = torch.tensor(total,   device=device)
            dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)
            correct = correct_t.item()
            total   = total_t.item()

            _gl = [None] * dist.get_world_size()
            dist.all_gather_object(_gl, all_preds)
            all_preds = [x for r in _gl for x in r]
            dist.all_gather_object(_gl, all_labels)
            all_labels = [x for r in _gl for x in r]

        train_acc = correct / total if total > 0 else 0.0
        train_f1 = (f1_score(all_labels, all_preds, average="macro")
                    if len(set(all_labels)) > 1 else 0.0)

        num_batches = len(train_loader) if len(train_loader) > 0 else 1
        avg_train_loss = total_loss / num_batches
        avg_train_loss_cls = total_loss_cls / num_batches
        avg_train_loss_recon = total_loss_recon / num_batches if alpha > 0.0 else 0.0

        if is_main:
            loss_str = f"Train Loss: {avg_train_loss:.4f}"
            if alpha > 0.0:
                loss_str += f" (Cls: {avg_train_loss_cls:.4f}, Recon: {avg_train_loss_recon:.4f})"
            print(
                f"[Epoch {epoch+1}] {loss_str} | "
                f"Acc: {train_acc:.4f} | F1: {train_f1:.4f}"
            )
            if args.use_wandb:
                wandb_log_dict = {
                    "train_loss": avg_train_loss,
                    "train_loss_cls": avg_train_loss_cls,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                }
                if alpha > 0.0:
                    wandb_log_dict["train_loss_recon"] = avg_train_loss_recon
                wandb.log(wandb_log_dict, step=epoch + 1)

        # ── Validation ──────────────────────────────────────────────────────
        classifier.eval()
        for model in (spatial_model, temporal_model, conv_model):
            model.eval()
        val_correct, val_total_v = 0, 0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for fmri, fc, gm, labels in val_loader:
                fmri, fc, gm, labels = (
                    fmri.to(device),
                    fc.to(device),
                    gm.to(device),
                    labels.to(device),
                )

                _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
                _, _, _, _, time_embeds, _ = temporal_model(fmri)
                _, signal_context = conv_model(fmri, mask=None)

                logits, recon = classifier(roi_embeds, time_embeds, signal_context, roi_pos)
                preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total_v += labels.size(0)
                val_preds.extend(preds.cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        # Aggregate val metrics
        if args.ddp:
            correct_t = torch.tensor(val_correct, device=device)
            total_t   = torch.tensor(val_total_v, device=device)
            dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)
            val_correct = correct_t.item()
            val_total_v = total_t.item()

            _gv = [None] * dist.get_world_size()
            dist.all_gather_object(_gv, val_preds)
            val_preds = [x for r in _gv for x in r]
            dist.all_gather_object(_gv, val_labels_list)
            val_labels_list = [x for r in _gv for x in r]

        val_acc = val_correct / val_total_v if val_total_v > 0 else 0.0
        val_f1 = (f1_score(val_labels_list, val_preds, average="macro")
                  if len(set(val_labels_list)) > 1 else 0.0)

        if is_main:
            print(
                f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f} | Time: {(time()-t1):.1f}s"
            )
            if args.use_wandb:
                wandb.log({"val_acc": val_acc, "val_f1": val_f1}, step=epoch + 1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_state = (
                    classifier.module.state_dict() if args.ddp else classifier.state_dict()
                )
                exp_mode = getattr(args, "_current_mode", "single")
                current_fold = getattr(args, "_current_fold_idx", None)
                exp_tag = get_experiment_tag(args, mode=exp_mode, fold_idx=current_fold)

                wandb_hash = None
                if args.use_wandb and wandb.run is not None:
                    wandb_hash = getattr(wandb.run, 'id', None)

                hash_str = f"_{wandb_hash}" if wandb_hash else ""
                save_filename = f"Fine_tuned_perceiver_classifier_HCP_YA_{exp_tag}{hash_str}.pth"

                save_path = os.path.join(proj_dir, 'models', save_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model_state, save_path)

        train_loss = avg_train_loss
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if epoch_callback is not None and is_main:
            epoch_callback(epoch + 1, {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_acc': val_acc,
                'val_f1': val_f1,
            })

    # ── Final evaluation on held-out test set ───────────────────────────────
    classifier.eval()
    for model in (spatial_model, temporal_model, conv_model):
        model.eval()
    test_correct, test_total_t = 0, 0
    test_preds, test_labels_list = [], []

    with torch.no_grad():
        for fmri, fc, gm, labels in test_loader:
            fmri, fc, gm, labels = (
                fmri.to(device),
                fc.to(device),
                gm.to(device),
                labels.to(device),
            )

            _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
            _, _, _, _, time_embeds, _ = temporal_model(fmri)
            _, signal_context = conv_model(fmri, mask=None)

            logits, recon = classifier(roi_embeds, time_embeds, signal_context, roi_pos)
            preds = logits.argmax(dim=1)

            test_correct += (preds == labels).sum().item()
            test_total_t += labels.size(0)
            test_preds.extend(preds.cpu().tolist())
            test_labels_list.extend(labels.cpu().tolist())

    # Aggregate test metrics
    if args.ddp:
        correct_t = torch.tensor(test_correct, device=device)
        total_t   = torch.tensor(test_total_t, device=device)
        dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_t,   op=dist.ReduceOp.SUM)
        test_correct = correct_t.item()
        test_total_t = total_t.item()

        _gt = [None] * dist.get_world_size()
        dist.all_gather_object(_gt, test_preds)
        test_preds = [x for r in _gt for x in r]
        dist.all_gather_object(_gt, test_labels_list)
        test_labels_list = [x for r in _gt for x in r]

    test_acc = test_correct / test_total_t if test_total_t > 0 else 0.0
    test_f1 = (f1_score(test_labels_list, test_preds, average="macro")
               if len(set(test_labels_list)) > 1 else 0.0)

    if is_main:
        print(
            f"✅ Fine-tuning Complete.\n"
            f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | "
            f"Total Time: {(time()-t0)/60/60:.1f} hours."
        )
        if args.use_wandb:
            wandb.log({"test_acc": test_acc, "test_f1": test_f1})

        exp_mode = getattr(args, "_current_mode", "single")
        current_fold = getattr(args, "_current_fold_idx", None)
        history_tag = get_experiment_tag(args, mode=exp_mode, fold_idx=current_fold)
        save_run_history(history, test_acc, test_f1, args, history_tag)

    return test_acc, test_f1, history


def _resolve_dapt_mask_ratios(args):
    """Resolve per-encoder DAPT mask ratios -> (spatial, temporal, context).

    If the shared ``--dapt_mask_ratio`` is set it overrides all three;
    otherwise the per-encoder values are used (defaults 0.7 / 0.7 / 0.5).
    """
    shared = getattr(args, "dapt_mask_ratio", None)
    if shared is not None:
        shared = float(shared)
        return shared, shared, shared
    sp = getattr(args, "dapt_mask_ratio_spatial", 0.7)
    tp = getattr(args, "dapt_mask_ratio_temporal", 0.7)
    cx = getattr(args, "dapt_mask_ratio_context", 0.5)
    return float(sp), float(tp), float(cx)


def _parse_dapt_weights(args):
    """Parse the ``--dapt_weights`` string into 4 floats (sp,tp,conv,perceiver)."""
    raw = getattr(args, "dapt_weights", "1,1,1,1")
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


def dapt(trainset, valset, args, local_rank=0, epoch_callback=None,
         exclusion_report=None):
    """Stage 3.1 — Domain-Adaptive Continued Pretraining (DAPT).

    Unfreezes *all* modules (spatial / temporal / conv encoders + their decoders
    and the Perceiver) and continues the self-supervised reconstruction
    objective on HCP-YA data.  No labels are used.

        loss = w_sp * L_spatial + w_tp * L_temporal
             + w_cv * L_conv    + w_pc * L_perceiver

    The per-module terms are masked-reconstruction MSEs combined with the raw
    ``--dapt_weights`` (spatial,temporal,conv,perceiver).

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
    if args.ddp:
        raise NotImplementedError(
            "DAPT (--mode dapt) currently supports single-process training only; "
            "run without --ddp."
        )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = is_main_process()

    sp_ratio, tp_ratio, cx_ratio = _resolve_dapt_mask_ratios(args)
    w_sp, w_tp, w_cv, w_pc = _parse_dapt_weights(args)
    if is_main:
        print(
            f"[DAPT] mask ratios — spatial={sp_ratio:.2f} "
            f"temporal={tp_ratio:.2f} context={cx_ratio:.2f}"
        )

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=False,
    )

    # ── Build models (HCP-AGING init) with masking enabled ──────────────────
    spatial_model, temporal_model, conv_model, perceiver = build_pretrained_models(
        args, device, spatial_mask_ratio=sp_ratio, temporal_mask_ratio=tp_ratio
    )

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
        """Return (L_spatial, L_temporal, L_conv, L_perceiver) for a batch."""
        l_sp, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
        l_tp, _, _, _, time_embeds, _ = temporal_model(fmri)
        conv_mask = time_masker(fmri)
        preds, signal_context = conv_model(fmri, mask=conv_mask)
        l_cv = (
            ((preds - fmri) * (1 - conv_mask)).pow(2).sum()
            / (1 - conv_mask).sum().clamp(min=1)
        )
        recon, _, _ = perceiver(roi_embeds, time_embeds, signal_context, roi_pos)
        l_pc = F.mse_loss(recon, fmri)
        return l_sp, l_tp, l_cv, l_pc

    term_names = ("spatial", "temporal", "conv", "perceiver")
    term_weights = (w_sp, w_tp, w_cv, w_pc)
    grad_clip = float(getattr(args, "grad_clip", 1.0) or 0.0)
    val_mask_seed = int(getattr(args, "val_mask_seed", 1234))
    # fork_rng expects CUDA device *indices*
    fork_devices = [device.index if device.index is not None else 0] \
        if device.type == "cuda" else []

    def _weighted_total(terms):
        return sum(w * t for w, t in zip(term_weights, terms))

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
        gnorm_sum = 0.0
        n_ok, n_skipped = 0, 0
        for fmri, fc, gm, _labels in train_loader:
            fmri, fc, gm = fmri.to(device), fc.to(device), gm.to(device)

            terms = _forward_losses(fmri, fc, gm)
            loss = _weighted_total(terms)

            # ── Guard 1: non-finite loss -> skip the whole step ────────────
            if not torch.isfinite(loss):
                n_skipped += 1
                if is_main and n_skipped <= 5:
                    bad = [n for n, t in zip(term_names, terms) if not torch.isfinite(t)]
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
        avg_gnorm = gnorm_sum / denom

        # ── Validation (raw weighted reconstruction loss) ────────────────────
        # Deterministic masks: the RNG is forked and re-seeded so every epoch
        # sees identical val masks, and the training RNG stream is untouched.
        for model in (spatial_model, temporal_model, conv_model, perceiver):
            model.eval()
        val_total = torch.zeros((), device=device)
        val_term_sums = [torch.zeros((), device=device) for _ in term_names]
        val_ok, val_skipped = 0, 0
        with torch.no_grad(), torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(val_mask_seed)
            for fmri, fc, gm, _labels in val_loader:
                fmri, fc, gm = fmri.to(device), fc.to(device), gm.to(device)
                terms = _forward_losses(fmri, fc, gm)
                vloss = _weighted_total(terms)
                if not torch.isfinite(vloss):
                    val_skipped += 1
                    continue
                val_total += vloss
                for i, t in enumerate(terms):
                    val_term_sums[i] += t
                val_ok += 1
        val_denom = max(val_ok, 1)
        avg_val = val_total.item() / val_denom if val_ok > 0 else float("nan")
        avg_val_terms = [s.item() / val_denom for s in val_term_sums]

        history["train_recon"].append(avg_train)
        history["val_recon"].append(avg_val)
        for i, name in enumerate(term_names):
            history[f"train_{name}"].append(avg_train_terms[i])
            history[f"val_{name}"].append(avg_val_terms[i])
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
                wandb.log(log_dict, step=epoch + 1)

        if np.isfinite(avg_val) and avg_val < best_val_recon:
            best_val_recon = avg_val
            best_epoch = epoch + 1
            best_state = {
                "spatial": {k: v.detach().cpu() for k, v in spatial_model.state_dict().items()},
                "temporal": {k: v.detach().cpu() for k, v in temporal_model.state_dict().items()},
                "conv": {k: v.detach().cpu() for k, v in conv_model.state_dict().items()},
                "perceiver": {k: v.detach().cpu() for k, v in perceiver.state_dict().items()},
                "args": vars(args) if hasattr(args, "__dict__") else {},
                "epoch": best_epoch,
                "val_loss": best_val_recon,
            }

        if epoch_callback is not None and is_main:
            cb_payload = {
                "epoch": epoch + 1,
                "train_recon": avg_train,
                "val_recon": avg_val,
                "grad_norm": avg_gnorm,
                "skipped_batches": n_skipped,
            }
            for i, name in enumerate(term_names):
                cb_payload[f"train_{name}"] = avg_train_terms[i]
                cb_payload[f"val_{name}"] = avg_val_terms[i]
            epoch_callback(epoch + 1, cb_payload)

    # ── Save best checkpoint ────────────────────────────────────────────────
    ckpt_path = None
    if is_main and best_state is not None:
        tag = get_experiment_tag(args, mode="dapt")
        ckpt_name = getattr(args, "dapt_ckpt_name", None) or f"dapt_HCP_YA_{tag}.pth"
        ckpt_path = os.path.join(proj_dir, "models", ckpt_name)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(best_state, ckpt_path)
        history["ckpt_path"] = ckpt_path
        print(
            f"✅ DAPT complete. Best epoch {best_epoch} "
            f"(val recon {best_val_recon:.4f}) in {(time()-t0)/60:.1f} min."
        )
        print(f"[DAPT] Saved checkpoint to {ckpt_path}")
        save_run_history(history, None, None, args, tag)

    return ckpt_path, history


def run_dapt(args, local_rank=0, epoch_callback=None):
    """Stage 3.1 wrapper: build the dataset, split it, and run DAPT.

    DAPT is self-supervised (labels are ignored). It trains on the train
    subjects and monitors reconstruction on the val subjects. The test subjects
    are deliberately held out so that Stage 3.2 reports a clean,
    non-transductive result.

    Returns
    -------
    (ckpt_path, history)
        ``ckpt_path`` is the saved Stage-3.1 checkpoint (or ``None`` on
        non-main ranks); ``history`` is the recon-loss history dict.
    """
    is_main = is_main_process()
    t0 = time()

    dataset, subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, "data/hcp"),
        label=args.label,
        n_subjs=args.n_subjs,
        max_abs=getattr(args, "max_abs", 1e4),
    )
    if is_main:
        print(
            f"Dataset loaded in {time()-t0:.1f}s ({(time()-t0)/60:.1f} min) | "
            f"{len(np.unique(subjs))} subjects | {len(dataset)} samples"
        )

    trainset, valset, _testset = subject_train_val_test_split(
        dataset, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed,
    )

    # DAPT trains on the train subjects only; the val subjects are the held-out
    # monitor set (used for checkpoint selection) and the test subjects stay
    # untouched so Stage 3.2 reports a clean, non-transductive result.
    dapt_trainset = trainset

    args._current_mode = "dapt"
    args._current_fold_idx = None
    tag = get_experiment_tag(args, mode="dapt")

    if is_main:
        print(f"[DAPT] train={len(dapt_trainset)} | val(monitor)={len(valset)} samples")

    if args.use_wandb and is_main:
        wandb.init(
            project="OCD_Symphony",
            name=f"DAPT_HCP_YA_{tag}",
            config=vars(args) if hasattr(args, "__dict__") else None,
        )

    ckpt_path, history = dapt(
        dapt_trainset, valset, args,
        local_rank=local_rank, epoch_callback=epoch_callback,
        exclusion_report=getattr(dataset, "exclusion_report", None),
    )

    if args.use_wandb and is_main:
        wandb.finish()

    return ckpt_path, history


def run_single_split(args, local_rank=0, epoch_callback=None):
    """Single deterministic train/val/test split using subject_train_val_test_split."""
    is_main = is_main_process()
    t0 = time()

    dataset, subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, "data/hcp"),
        label=args.label,
        n_subjs=args.n_subjs,
        max_abs=getattr(args, "max_abs", 1e4),
    )
    if is_main:
        print(
            f"Dataset loaded in {time()-t0:.1f}s ({(time()-t0)/60:.1f} min) | "
            f"{len(np.unique(subjs))} subjects | {len(dataset)} samples"
        )

    trainset, valset, testset = subject_train_val_test_split(
        dataset, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed,
    )

    args._current_mode = "single"
    args._current_fold_idx = None
    tag = get_experiment_tag(args, mode="single")

    if args.use_wandb and is_main:
        wandb.init(
            project="OCD_Symphony",
            name=f"Finetune_HCP_{tag}",
            config=vars(args) if hasattr(args, "__dict__") else None,
        )

    test_acc, test_f1, history = finetune(trainset, valset, testset, args, local_rank=local_rank, epoch_callback=epoch_callback)

    if args.use_wandb and is_main:
        wandb.finish()

    return test_acc, test_f1, history


def split_fold_data(dataset, trainset, testset, fold_idx, args):
    """
    Split a cross-validation fold trainset into train and validation sets
    using subject-level StratifiedGroupKFold without data leakage.
    """
    ds = dataset
    train_sample_indices = list(trainset.indices)
    train_subjs = ds.subjs[train_sample_indices]

    unique_train_subjs = np.unique(train_subjs)
    train_labels_subj = []
    for s in unique_train_subjs:
        mask = train_subjs == s
        lbls = ds.encoded_labels.numpy()[train_sample_indices][mask]
        counts = np.bincount(lbls)
        train_labels_subj.append(np.argmax(counts))
    train_labels_subj = np.array(train_labels_subj)

    n_inner = max(2, round(1.0 / args.val_frac))
    inner_sgkf = StratifiedGroupKFold(
        n_splits=n_inner, shuffle=True, random_state=args.seed + fold_idx
    )
    inner_splits = list(
        inner_sgkf.split(unique_train_subjs, train_labels_subj, groups=unique_train_subjs)
    )
    _, val_subj_idx = inner_splits[-1]
    train_subj_idx = np.setdiff1d(
        np.arange(len(unique_train_subjs)), val_subj_idx
    )

    val_subjs = unique_train_subjs[val_subj_idx]
    new_train_subjs = unique_train_subjs[train_subj_idx]

    for s in val_subjs:
        assert s not in new_train_subjs, f"Subject leakage detected: {s}"

    val_sample_indices = np.where(np.isin(ds.subjs, val_subjs))[0].tolist()
    new_train_sample_indices = np.where(np.isin(ds.subjs, new_train_subjs))[0].tolist()

    new_trainset = Subset(ds, new_train_sample_indices)
    valset = Subset(ds, val_sample_indices)

    return new_trainset, valset, testset


def run_single_fold(args, local_rank=0, epoch_callback=None):
    """Run fine-tuning for a single specified CV fold."""
    is_main = is_main_process()
    t0 = time()

    if args.fold_idx < 0 or args.fold_idx >= args.n_folds:
        raise ValueError(
            f"--fold_idx must be between 0 and {args.n_folds - 1}, got {args.fold_idx}"
        )

    dataset, subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, "data/hcp"),
        label=args.label,
        n_subjs=args.n_subjs,
        max_abs=getattr(args, "max_abs", 1e4),
    )
    if is_main:
        print(
            f"Dataset loaded in {time()-t0:.1f}s ({(time()-t0)/60:.1f} min) | "
            f"{len(np.unique(subjs))} subjects | {len(dataset)} samples"
        )

    folds = list(subject_kfold(dataset, n_splits=args.n_folds, seed=args.seed))
    trainset, testset = folds[args.fold_idx]

    if is_main:
        print("\n" + "=" * 60)
        print(f"  Running Fold {args.fold_idx + 1} / {args.n_folds} (index: {args.fold_idx})")
        print("=" * 60)

    new_trainset, valset, testset = split_fold_data(
        dataset, trainset, testset, args.fold_idx, args
    )

    if is_main:
        print(
            f"train={len(new_trainset)} | val={len(valset)} | test={len(testset)} samples"
        )

    args._current_mode = "single_fold"
    args._current_fold_idx = args.fold_idx
    tag = get_experiment_tag(args, mode="single_fold", fold_idx=args.fold_idx)

    if args.use_wandb and is_main:
        wandb.init(
            project="OCD_Symphony",
            name=f"Finetune_HCP_{tag}",
            reinit=True,
            config=vars(args) if hasattr(args, "__dict__") else None,
        )

    test_acc, test_f1, history = finetune(
        new_trainset, valset, testset, args,
        local_rank=local_rank,
        epoch_callback=epoch_callback
    )

    if args.use_wandb and is_main:
        wandb.finish()

    return test_acc, test_f1, history


def run_cross_validation(args, local_rank=0, epoch_callback=None):
    """Full K-fold leave-subject-out cross-validation using subject_kfold."""
    is_main = is_main_process()
    t0 = time()

    dataset, subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, "data/hcp"),
        label=args.label,
        n_subjs=args.n_subjs,
        max_abs=getattr(args, "max_abs", 1e4),
    )
    if is_main:
        print(
            f"Dataset loaded in {time()-t0:.1f}s ({(time()-t0)/60:.1f} min) | "
            f"{len(np.unique(subjs))} subjects | {len(dataset)} samples"
        )

    folds = list(subject_kfold(dataset, n_splits=args.n_folds, seed=args.seed))

    all_accs = []
    all_f1s = []
    all_histories = []

    for fold_idx, (trainset, testset) in enumerate(folds):
        if is_main:
            print("\n" + "=" * 60)
            print(f"  Fold {fold_idx + 1} / {args.n_folds}")
            print("=" * 60)

        new_trainset, valset, testset = split_fold_data(
            dataset, trainset, testset, fold_idx, args
        )

        if is_main:
            print(
                f"train={len(new_trainset)} | val={len(valset)} | test={len(testset)} samples"
            )

        args._current_mode = "cv"
        args._current_fold_idx = fold_idx
        tag = get_experiment_tag(args, mode="cv", fold_idx=fold_idx)

        if args.use_wandb and is_main:
            wandb.init(
                project="OCD_Symphony",
                name=f"Finetune_HCP_{tag}",
                reinit=True,
                config=vars(args) if hasattr(args, "__dict__") else None,
            )

        def fold_epoch_callback(ep, metrics):
            if epoch_callback is not None:
                m_copy = dict(metrics)
                m_copy["fold"] = fold_idx + 1
                m_copy["n_folds"] = args.n_folds
                epoch_callback(ep, m_copy)

        test_acc, test_f1, history = finetune(
            new_trainset, valset, testset, args,
            local_rank=local_rank,
            epoch_callback=fold_epoch_callback if epoch_callback else None
        )
        all_accs.append(test_acc)
        all_f1s.append(test_f1)
        all_histories.append(history)

        if args.use_wandb and is_main:
            wandb.finish()

    mean_acc = sum(all_accs) / len(all_accs) if all_accs else 0.0
    std_acc = (sum((a - mean_acc) ** 2 for a in all_accs) / len(all_accs)) ** 0.5 if all_accs else 0.0
    mean_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    std_f1 = (sum((f - mean_f1) ** 2 for f in all_f1s) / len(all_f1s)) ** 0.5 if all_f1s else 0.0

    if is_main:
        print("\n" + "=" * 60)
        print(f"  Cross-Validation Results  ({args.n_folds} folds)")
        print("=" * 60)
        for i, (acc, f1) in enumerate(zip(all_accs, all_f1s)):
            print(f"  Fold {i+1}:  Test Acc = {acc:.4f}  |  Test F1 = {f1:.4f}")
        print("  " + "─" * 45)
        print(
            f"  Mean ± Std:  Acc = {mean_acc:.4f} ± {std_acc:.4f}  |  "
            f"F1 = {mean_f1:.4f} ± {std_f1:.4f}"
        )
        print("=" * 60)

    cv_summary = {
        "is_cv": True,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "fold_accs": all_accs,
        "fold_f1s": all_f1s,
        "histories": all_histories,
    }

    if is_main:
        save_run_history(
            cv_summary, mean_acc, mean_f1, args,
            get_experiment_tag(args, mode="cv"),
        )

    return mean_acc, mean_f1, cv_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Perceiver Classifier on HCP dataset (new API)"
    )
    parser.add_argument('--label', type=str, default='gender')
    parser.add_argument('--n_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument(
        '--n_subjs', type=int, default=None,
        help='Total number of subjects to load (None = all)'
    )
    parser.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of LSO cross-validation folds (for CV mode)'
    )
    parser.add_argument(
        '--fold_idx', type=int, default=None,
        help='Run only a specific fold (0-indexed) of K-fold CV. Overrides full --cv loop.'
    )
    parser.add_argument(
        '--test_frac', type=float, default=0.15,
        help='Fraction of subjects reserved for test (single-split mode)'
    )
    parser.add_argument(
        '--val_frac', type=float, default=0.15,
        help='Fraction of subjects reserved for validation (single-split mode)'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument(
        '--use_wandb', action='store_true',
        help='Whether to log training with Weights & Biases'
    )
    parser.add_argument(
        '--cv', action='store_true',
        help='Run full K-fold cross-validation (uses subject_kfold)'
    )
    parser.add_argument(
        '--pooling', type=str, choices=['mean', 'attention', 'flatten'], default='mean',
        help="Pooling strategy for ROI latents ('mean', 'attention', or 'flatten')"
    )
    parser.add_argument(
        '--dropout', type=float, default=0.3,
        help='Dropout rate in classification head (default: 0.3)'
    )
    parser.add_argument(
        '--recon_weight', type=float, default=0.0,
        help='Convex loss balance alpha in [0.0, 1.0]: (1-alpha)*cls + alpha*recon (0.0=100%% cls, 0.5=50/50, 1.0=100%% recon)'
    )
    parser.add_argument(
        '--freeze_perceiver', action='store_true',
        help='If set, freeze Perceiver backbone (Stage 1). Default False (Stage 2: unfreeze Perceiver).'
    )
    parser.add_argument(
        '--backbone_lr', type=float, default=1e-5,
        help='Learning rate for Perceiver backbone in Stage 2 (default: 1e-5)'
    )
    parser.add_argument(
        '--mode', type=str, choices=['cls', 'dapt'], default='cls',
        help="Training mode: 'cls' = supervised classification fine-tuning "
             "(Stages 1-3.2); 'dapt' = Stage 3.1 domain-adaptive continued "
             "pretraining (reconstruction only, all modules unfrozen)."
    )
    parser.add_argument(
        '--dapt_ckpt', type=str, default=None,
        help='Path to a Stage-3.1 DAPT checkpoint. If set, the encoders and '
             'Perceiver are initialised from it instead of the HCP-AGING '
             'checkpoints (Stage 3.2).'
    )
    parser.add_argument(
        '--dapt_mask_ratio', type=float, default=None,
        help='Shared DAPT masking ratio. If set, it overrides all three '
             'per-encoder ratios below (default: None).'
    )
    parser.add_argument(
        '--dapt_mask_ratio_spatial', type=float, default=0.7,
        help='DAPT masking ratio for the spatial encoder (default: 0.7).'
    )
    parser.add_argument(
        '--dapt_mask_ratio_temporal', type=float, default=0.7,
        help='DAPT masking ratio for the temporal encoder (default: 0.7).'
    )
    parser.add_argument(
        '--dapt_mask_ratio_context', type=float, default=0.5,
        help='DAPT masking ratio for the context (Conv1D) encoder (default: 0.5).'
    )
    parser.add_argument(
        '--dapt_weights', type=str, default='1,1,1,1',
        help='Comma-separated reconstruction-loss weights for DAPT in the order '
             'spatial,temporal,conv,perceiver (default: 1,1,1,1).'
    )
    parser.add_argument(
        '--dapt_ckpt_name', type=str, default=None,
        help='Optional filename override for the saved DAPT checkpoint.'
    )
    parser.add_argument(
        '--grad_clip', type=float, default=1.0,
        help='Global gradient-norm clipping threshold for DAPT (0 disables clipping).'
    )
    parser.add_argument(
        '--max_abs', type=float, default=1e4,
        help='Data validation: fMRI runs with |value| above this (or non-finite '
             'fMRI/FC) are dropped; subjects with no valid run are excluded.'
    )
    parser.add_argument(
        '--val_mask_seed', type=int, default=1234,
        help='Seed for the (deterministic) DAPT validation masks.'
    )
    parser.add_argument(
        '--encoder_lr', type=float, default=0.0,
        help='Learning rate for the spatial/temporal/conv encoders (Stage 3.2). '
             '0.0 (default) keeps them frozen; >0 unfreezes them at this low LR.'
    )
    parser.add_argument(
        '--ddp', action='store_true',
        help='Enable DistributedDataParallel (multi-GPU). Launch with torchrun.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    import numpy as np

    args = parse_args()

    # Initialise DDP if requested (reads RANK / WORLD_SIZE from env)
    local_rank = 0
    if args.ddp:
        _, local_rank, _ = ddp_setup()

    if args.mode == 'dapt':
        run_dapt(args, local_rank=local_rank)
    elif args.fold_idx is not None:
        run_single_fold(args, local_rank=local_rank)
    elif args.cv:
        run_cross_validation(args, local_rank=local_rank)
    else:
        run_single_split(args, local_rank=local_rank)

    if args.ddp:
        ddp_cleanup()
