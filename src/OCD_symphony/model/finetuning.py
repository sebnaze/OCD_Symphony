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
from sklearn.metrics import f1_score, balanced_accuracy_score
from time import time, strftime
import json
import wandb

from OCD_symphony.utils import proj_dir
from OCD_symphony.data_preparation.make_hcp_dataset import (
    make_dataloaders,
    DATASET_LABELS,
    DATASET_TAGS,
)
from OCD_symphony.model.model_components import *
from OCD_symphony.model.constants import (
    BATCH_SIZE,
    DELTA,
    DIM,
    LEARNING_RATE,
    MAX_LEN_SPATIAL,
    MAX_LEN_TEMP,
    MLP_DIM,
    NUM_EPOCHS,
    NUM_GRADIENTS,
    NUM_HEADS,
    NUM_LAYERS,
    ROIs,
    TIME_STEPS,
)
from OCD_symphony.model.config import (
    FinetuneConfig, _CONFIG_ATTR_SOURCE, _parse_dapt_weights, _resolve_dapt_mask_ratios,
)
from OCD_symphony.model.checkpoints import (
    build_pretrained_models, _load_perceiver_from_classifier_ckpt,
)
from OCD_symphony.model.metrics import (
    _EarlyStopper,
    _regression_metrics,
    _root_indices,
    _subject_ids_and_labels,
    subject_accuracy,
    subject_regression_mae,
)


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
    cfg = FinetuneConfig.from_args(args)
    label_str = cfg.label
    dataset_str = cfg.dataset
    # Dataset prefix (HCP_YA / OCD): keeps checkpoints and run files of the two datasets apart.
    prefix = f"{DATASET_TAGS.get(dataset_str, dataset_str.upper())}_"
    stage_str = "stage1" if cfg.freeze_perceiver else "stage2"
    pool_str = cfg.pooling

    lr_str = f"lr{args.learning_rate:.0e}"
    if not cfg.freeze_perceiver:
        bblr = cfg.backbone_lr if cfg.backbone_lr is not None else 1e-5
        lr_str += f"_bblr{bblr:.0e}"

    alpha = cfg.recon_weight
    alpha_str = f"_a{alpha:.2f}" if alpha > 0.0 else ""

    if mode == "single":
        mode_str = "single"
    elif mode == "single_fold":
        f_idx = cfg.fold_idx if fold_idx is None else fold_idx
        mode_str = f"fold{f_idx + 1}"
    elif mode == "cv":
        f_idx = 0 if fold_idx is None else fold_idx
        mode_str = f"cv_fold{f_idx + 1}"
    else:
        mode_str = str(mode)

    # Stage 3.1 — DAPT (reconstruction only) has its own tag shape
    if cfg.mode == "dapt":
        sp, tp, cx = _resolve_dapt_mask_ratios(args)
        lr = cfg.learning_rate
        return (
            f"{prefix}{label_str}_dapt_mS{sp:.2f}_mT{tp:.2f}_mC{cx:.2f}"
            f"_lr{lr:.0e}_{mode_str}"
        )

    return f"{prefix}{label_str}_{stage_str}_{pool_str}_{lr_str}{alpha_str}_{mode_str}"


class _RecordingSampler(torch.utils.data.Sampler):
    """Wraps a sampler and remembers the index order it yielded in the last epoch.

    Batches are produced in sampler order, so batch k covers ``order[pos:pos+bs]``.
    Used to map training batches back to subjects.
    """

    def __init__(self, base):
        self.base = base
        self.order = []

    def __iter__(self):
        self.order = list(self.base)
        return iter(self.order)

    def __len__(self):
        return len(self.base)


def _gather_list(local, ddp):
    """All-gather a Python list across DDP ranks (identity when not distributed)."""
    if not ddp:
        return list(local)
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, list(local))
    return [x for r in out for x in r]


def _unwrap(model, ddp):
    """Return the underlying module of a DDP-wrapped model (identity when not DDP)."""
    return model.module if ddp else model


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

    cfg = FinetuneConfig.from_args(args)

    # ── Task: classification (gender / condition) or regression (age) ───────
    _tpos, root_ds = _root_indices(trainset)
    is_regression = getattr(root_ds, "task", "classification") == "regression"
    target_mu, target_sd = 0.0, 1.0
    if is_regression:
        # Standardise the age target with training-split statistics only.
        _t = root_ds.targets[torch.as_tensor(_tpos)].double()
        target_mu = float(_t.mean())
        target_sd = float(_t.std()) if _t.numel() > 1 else 0.0
        if not target_sd > 0:
            target_sd = 1.0
        if is_main:
            print(f"[Regression] age target: train mean={target_mu:.2f}, sd={target_sd:.2f}")

    # ── DataLoaders  (DistributedSampler for train in DDP mode) ────────────
    if args.ddp:
        train_sampler = DistributedSampler(
            trainset, shuffle=True,
            rank=dist.get_rank(), num_replicas=dist.get_world_size(),
        )
        train_order = _RecordingSampler(train_sampler)
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size,
            sampler=train_order, num_workers=4,
            pin_memory=False, drop_last=False,
        )
    else:
        train_order = _RecordingSampler(torch.utils.data.RandomSampler(trainset))
        train_loader = DataLoader(
            trainset, batch_size=args.batch_size,
            sampler=train_order, num_workers=4, pin_memory=True, drop_last=False,
        )

    val_loader   = DataLoader(valset,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=False)
    test_loader  = DataLoader(testset,  batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=False)

    # ── Load pretrained models (HCP-AGING, or DAPT-adapted HCP-YA weights) ──
    dapt_ckpt = None
    dapt_ckpt_path = cfg.dapt_ckpt
    if dapt_ckpt_path:
        dapt_ckpt = torch.load(dapt_ckpt_path, map_location=device)
        if is_main:
            print(f"[Stage 3.2] Loading DAPT-adapted weights from {dapt_ckpt_path}")

    spatial_model, temporal_model, conv_model, perceiver = build_pretrained_models(
        args, device, mask_ratio=0.0, encoder_ckpt=dapt_ckpt
    )

    # Encoders act as feature extractors: frozen by default. Passing
    # --encoder_lr > 0 unfreezes them at a very low LR (Stage 3.2 option).
    encoder_lr = cfg.encoder_lr
    train_encoders = encoder_lr > 0.0
    for model in [spatial_model, temporal_model, conv_model]:
        model.eval()
        for param in model.parameters():
            param.requires_grad = train_encoders

    freeze_perceiver = cfg.freeze_perceiver
    for param in perceiver.parameters():
        param.requires_grad = not freeze_perceiver

    pooling = cfg.pooling
    dropout = cfg.dropout
    classifier = PerceiverClassifier(
        perceiver,
        latent_dim=DIM,
        num_classes=1 if is_regression else 2,
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
        raw_classifier = _unwrap(classifier, args.ddp)
        backbone_params = [p for p in raw_classifier.perceiver.parameters() if p.requires_grad]
        head_params = [
            p for n, p in raw_classifier.named_parameters()
            if p.requires_grad and not n.startswith("perceiver.")
        ]
        backbone_lr = cfg.backbone_lr if cfg.backbone_lr is not None else args.learning_rate * 0.1

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
    criterion = None
    if not is_regression:
        if isinstance(trainset, Subset):
            train_labels_tensor = trainset.dataset.encoded_labels[trainset.indices]
        else:
            train_labels_tensor = trainset.encoded_labels

        class_counts = torch.bincount(train_labels_tensor)
        class_weights = (len(train_labels_tensor) / (len(class_counts) * class_counts.float())).to(device)
        if is_main:
            print(f"[Class Balance] counts: {class_counts.tolist()} | weights: {[round(w, 4) for w in class_weights.tolist()]}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)

    def _task_loss(logits, labels):
        """Training loss: weighted CE (classification) or MSE on the standardised age."""
        if is_regression:
            return F.mse_loss(logits.squeeze(-1), (labels - target_mu) / target_sd)
        return criterion(logits, labels)

    def _to_pred(logits):
        """Predictions: class index (classification) or age in years (regression)."""
        if is_regression:
            return logits.squeeze(-1).detach() * target_sd + target_mu
        return logits.argmax(dim=1)

    def _encode(fmri, fc, gm):
        """Run the three frozen-or-tuned encoders; returns (roi_embeds, roi_pos, time_embeds, signal_context)."""
        _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
        _, _, _, _, time_embeds, _ = temporal_model(fmri)
        _, signal_context = conv_model(fmri, mask=None)
        return roi_embeds, roi_pos, time_embeds, signal_context

    def _run_eval(loader, dataset, name, with_loss=False):
        """Evaluation pass (no grad) over ``loader``. Returns per-sample lists and local counts.

        ``with_loss`` adds the unweighted task loss (the early-stopping metric).
        Batches must follow ``dataset`` order (``shuffle=False``); this is checked.
        """
        out = {"correct": 0, "total": 0, "preds": [], "labels": [], "probs": [],
               "subj": [], "loss_sum": 0.0, "n": 0}
        pos = 0
        with torch.no_grad():
            for fmri, fc, gm, labels in loader:
                fmri, fc, gm, labels = (
                    fmri.to(device),
                    fc.to(device),
                    gm.to(device),
                    labels.to(device),
                )
                roi_embeds, roi_pos, time_embeds, signal_context = _encode(fmri, fc, gm)
                logits, _recon = classifier(roi_embeds, time_embeds, signal_context, roi_pos)
                preds = _to_pred(logits)

                if with_loss:
                    if is_regression:
                        out["loss_sum"] += F.mse_loss(
                            logits.squeeze(-1), (labels - target_mu) / target_sd, reduction="sum",
                        ).item()
                    else:
                        out["loss_sum"] += F.cross_entropy(logits, labels, reduction="sum").item()
                    out["n"] += labels.size(0)
                if not is_regression:
                    out["correct"] += (preds == labels).sum().item()
                out["total"] += labels.size(0)
                out["preds"].extend(preds.cpu().tolist())
                out["labels"].extend(labels.cpu().tolist())
                _pos = np.arange(pos, pos + labels.size(0))
                pos += labels.size(0)
                _subj, _enc = _subject_ids_and_labels(dataset, _pos)
                if not np.array_equal(_enc, labels.cpu().numpy()):
                    raise RuntimeError(f"{name} batch does not match the dataset order")
                out["subj"].extend(_subj.tolist())
                out["probs"].append(torch.softmax(logits, dim=1).cpu().numpy())
        return out

    def _ddp_reduce(correct, total, preds, labels, loss=None):
        """Sum counts and gather per-sample lists across ranks (identity when not DDP).

        ``loss`` is an optional ``(sum, count)`` pair, reduced in float32 as before.
        Collectives run in the fixed order: counts, preds, labels, [loss].
        """
        if args.ddp:
            correct_t = torch.tensor(correct, device=device)
            total_t = torch.tensor(total, device=device)
            dist.all_reduce(correct_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_t, op=dist.ReduceOp.SUM)
            correct, total = correct_t.item(), total_t.item()
            preds = _gather_list(preds, True)
            labels = _gather_list(labels, True)
            if loss is not None:
                vl_t = torch.tensor([loss[0], float(loss[1])], device=device)
                dist.all_reduce(vl_t, op=dist.ReduceOp.SUM)
                loss = (vl_t[0].item(), vl_t[1].item())
        return correct, total, preds, labels, loss

    # ── Training Loop ───────────────────────────────────────────────────────
    stopper = _EarlyStopper(patience=cfg.patience)
    best_state = None      # CPU copy of the best-validation-loss weights
    stopped_epoch = None
    history = {
        "task": "regression" if is_regression else "classification",
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "train_subj_acc": [],
        "train_mae": [],
        "train_r2": [],
        "train_subj_mae": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_bal_acc": [],
        "val_subj_acc": [],
        "val_mae": [],
        "val_r2": [],
        "val_subj_mae": [],
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
        # Per-sample probabilities and subject IDs of this epoch's training batches.
        train_probs, train_subj, train_pos = [], [], 0

        alpha = cfg.recon_weight

        for fmri, fc, gm, labels in train_loader:
            fmri, fc, gm, labels = (
                fmri.to(device),
                fc.to(device),
                gm.to(device),
                labels.to(device),
            )

            # Encoders keep the autograd graph only when they are being fine-tuned.
            with torch.set_grad_enabled(train_encoders):
                roi_embeds, roi_pos, time_embeds, signal_context = _encode(fmri, fc, gm)

            logits, recon = classifier(roi_embeds, time_embeds, signal_context, roi_pos)
            loss_cls = _task_loss(logits, labels)
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
            preds = _to_pred(logits)
            if not is_regression:
                correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            # Batch k covers the sampler positions [train_pos, train_pos + bs).
            _bs = labels.size(0)
            _pos = train_order.order[train_pos:train_pos + _bs]
            train_pos += _bs
            _subj, _enc = _subject_ids_and_labels(trainset, _pos)
            if not np.array_equal(_enc, labels.cpu().numpy()):
                raise RuntimeError("training batch does not match the recorded sampler order")
            train_subj.extend(_subj.tolist())
            train_probs.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())

        total = len(all_labels)

        # ── Aggregate training metrics across all DDP processes ─────────────
        correct, total, all_preds, all_labels, _ = _ddp_reduce(
            correct, total, all_preds, all_labels)

        train_subj_acc = float("nan")
        train_acc = train_f1 = float("nan")
        train_mae = train_r2 = train_subj_mae = float("nan")
        if is_regression:
            _tr = _regression_metrics(all_labels, all_preds)
            train_mae, train_r2 = _tr["mae"], _tr["r2"]
            train_subj_mae, _ = subject_regression_mae(
                all_preds, all_labels, np.asarray(_gather_list(train_subj, args.ddp)),
            )
        else:
            train_subj_acc, _ = subject_accuracy(
                np.concatenate(_gather_list(train_probs, args.ddp)) if train_probs else np.zeros((0, 2)),
                np.asarray(all_labels),
                np.asarray(_gather_list(train_subj, args.ddp)),
            )
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
            if is_regression:
                metric_str = f"MAE: {train_mae:.3f} | R2: {train_r2:.4f}"
            else:
                metric_str = f"Acc: {train_acc:.4f} | F1: {train_f1:.4f}"
            print(f"[Epoch {epoch+1}] {loss_str} | {metric_str}")
            if args.use_wandb:
                wandb_log_dict = {
                    "train_loss": avg_train_loss,
                    "train_loss_cls": avg_train_loss_cls,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                    "train_subj_acc": train_subj_acc,
                    "train_mae": train_mae,
                    "train_r2": train_r2,
                    "train_subj_mae": train_subj_mae,
                }
                if alpha > 0.0:
                    wandb_log_dict["train_loss_recon"] = avg_train_loss_recon
                wandb.log(wandb_log_dict, step=epoch + 1)

        # ── Validation ──────────────────────────────────────────────────────
        classifier.eval()
        for model in (spatial_model, temporal_model, conv_model):
            model.eval()
        ev = _run_eval(val_loader, valset, "validation", with_loss=True)
        val_correct, val_total_v, val_preds, val_labels_list, (val_loss_sum, val_loss_n) = _ddp_reduce(
            ev["correct"], ev["total"], ev["preds"], ev["labels"],
            loss=(ev["loss_sum"], ev["n"]),
        )
        val_probs, val_subj = ev["probs"], ev["subj"]

        val_subj_acc = float("nan")
        val_acc = val_f1 = val_bal_acc = float("nan")
        val_mae = val_r2 = val_subj_mae = float("nan")
        val_loss = val_loss_sum / val_loss_n if val_loss_n > 0 else float("nan")
        if is_regression:
            _vr = _regression_metrics(val_labels_list, val_preds)
            val_mae, val_r2 = _vr["mae"], _vr["r2"]
            val_subj_mae, val_subj_n = subject_regression_mae(
                val_preds, val_labels_list, np.asarray(_gather_list(val_subj, args.ddp)),
            )
        else:
            val_subj_acc, val_subj_n = subject_accuracy(
                np.concatenate(_gather_list(val_probs, args.ddp)) if val_probs else np.zeros((0, 2)),
                np.asarray(val_labels_list),
                np.asarray(_gather_list(val_subj, args.ddp)),
            )
            val_acc = val_correct / val_total_v if val_total_v > 0 else 0.0
            val_f1 = (f1_score(val_labels_list, val_preds, average="macro")
                      if len(set(val_labels_list)) > 1 else 0.0)
            val_bal_acc = (balanced_accuracy_score(val_labels_list, val_preds)
                           if len(set(val_labels_list)) > 1 else 0.0)

        if is_main:
            if is_regression:
                print(
                    f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.3f} | "
                    f"Val R2: {val_r2:.4f} | Time: {(time()-t1):.1f}s"
                )
                print(
                    f"[Epoch {epoch+1}] Subj MAE: train {train_subj_mae:.3f} | val {val_subj_mae:.3f} "
                    f"(val subjects: {val_subj_n})"
                )
            else:
                print(
                    f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Val Bal Acc: {val_bal_acc:.4f} | Val F1: {val_f1:.4f} | "
                    f"Time: {(time()-t1):.1f}s"
                )
                print(
                    f"[Epoch {epoch+1}] Subj Acc: train {train_subj_acc:.4f} | val {val_subj_acc:.4f} "
                    f"(val subjects: {val_subj_n})"
                )
            if args.use_wandb:
                wandb.log({"val_loss": val_loss, "val_acc": val_acc,
                           "val_bal_acc": val_bal_acc, "val_f1": val_f1,
                           "train_subj_acc": train_subj_acc, "val_subj_acc": val_subj_acc,
                           "val_mae": val_mae, "val_r2": val_r2,
                           "train_subj_mae": train_subj_mae, "val_subj_mae": val_subj_mae},
                          step=epoch + 1)

        # ── Best-model tracking (on all ranks: val_loss is already reduced) ────
        improved = stopper.update(val_loss, epoch + 1)
        if improved:
            raw_model = _unwrap(classifier, args.ddp)
            best_state = {k: v.detach().cpu().clone() for k, v in raw_model.state_dict().items()}
            if dapt_ckpt_path:
                # Record the DAPT source so the Stage 3.2 checkpoint can be reconstructed
                # without the run JSON (its encoders are the frozen DAPT encoders).
                best_state["dapt_ckpt"] = str(dapt_ckpt_path)
            if train_encoders:
                # Encoders were fine-tuned, so they must be stored with the checkpoint.
                # Add DAPT-format sections (spatial/temporal/conv/perceiver) next to the
                # flat classifier keys so the file is loadable as an encoder checkpoint.
                best_state["spatial"] = {k: v.detach().cpu().clone() for k, v in spatial_model.state_dict().items()}
                best_state["temporal"] = {k: v.detach().cpu().clone() for k, v in temporal_model.state_dict().items()}
                best_state["conv"] = {k: v.detach().cpu().clone() for k, v in conv_model.state_dict().items()}
                best_state["perceiver"] = {k: v.detach().cpu().clone() for k, v in raw_model.perceiver.state_dict().items()}
                best_state["encoder_lr"] = encoder_lr
            if is_main:
                exp_mode = cfg.current_mode
                current_fold = cfg.current_fold_idx
                exp_tag = get_experiment_tag(args, mode=exp_mode, fold_idx=current_fold)

                wandb_hash = None
                if args.use_wandb and wandb.run is not None:
                    wandb_hash = getattr(wandb.run, 'id', None)

                hash_str = f"_{wandb_hash}" if wandb_hash else ""
                save_filename = f"Fine_tuned_perceiver_classifier_{exp_tag}{hash_str}.pth"

                save_path = os.path.join(proj_dir, 'models', save_filename)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)
                history["ckpt_path"] = save_path

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_bal_acc'].append(val_bal_acc)
        history['train_subj_acc'].append(train_subj_acc)
        history['val_subj_acc'].append(val_subj_acc)
        history['train_mae'].append(train_mae)
        history['train_r2'].append(train_r2)
        history['train_subj_mae'].append(train_subj_mae)
        history['val_mae'].append(val_mae)
        history['val_r2'].append(val_r2)
        history['val_subj_mae'].append(val_subj_mae)
        stopped_epoch = epoch + 1

        if epoch_callback is not None and is_main:
            epoch_callback(epoch + 1, {
                'epoch': epoch + 1,
                'task': "regression" if is_regression else "classification",
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_r2': train_r2,
                'val_r2': val_r2,
            })

        if stopper.should_stop:
            if is_main:
                print(
                    f"[Early stopping] No val-loss improvement for {stopper.patience} epochs. "
                    f"Stopping after epoch {epoch+1}; best epoch {stopper.best_epoch} "
                    f"(val loss {stopper.best_loss:.4f})."
                )
            break

    # ── Restore best-validation-loss weights before the final evaluation ────
    if best_state is not None:
        raw_model = _unwrap(classifier, args.ddp)
        # best_state also holds checkpoint metadata (dapt_ckpt path, encoder sections,
        # encoder_lr); only the classifier tensors belong to load_state_dict.
        raw_model.load_state_dict({k: v for k, v in best_state.items() if torch.is_tensor(v)})
    history["best_epoch"] = stopper.best_epoch
    history["stopped_epoch"] = stopped_epoch
    history["patience"] = stopper.patience

    # ── Final evaluation on held-out test set ───────────────────────────────
    classifier.eval()
    for model in (spatial_model, temporal_model, conv_model):
        model.eval()
    ev = _run_eval(test_loader, testset, "test")
    test_correct, test_total_t, test_preds, test_labels_list, _ = _ddp_reduce(
        ev["correct"], ev["total"], ev["preds"], ev["labels"],
    )
    test_probs, test_subj = ev["probs"], ev["subj"]

    if is_regression:
        # Age regression: no accuracy / F1. Test metrics are stored in the history.
        test_acc, test_f1 = None, None
        _te = _regression_metrics(test_labels_list, test_preds)
        test_subj_mae, test_subj_n = subject_regression_mae(
            test_preds, test_labels_list, np.asarray(_gather_list(test_subj, args.ddp)),
        )
        history["test_mae"] = _te["mae"]
        history["test_r2"] = _te["r2"]
        history["test_r"] = _te["r"]
        history["test_subj_mae"] = test_subj_mae
        history["test_subj_n"] = test_subj_n
        history["test_subj_acc"] = float("nan")
    else:
        test_acc = test_correct / test_total_t if test_total_t > 0 else 0.0
        test_f1 = (f1_score(test_labels_list, test_preds, average="macro")
                   if len(set(test_labels_list)) > 1 else 0.0)
        test_subj_acc, test_subj_n = subject_accuracy(
            np.concatenate(_gather_list(test_probs, args.ddp)) if test_probs else np.zeros((0, 2)),
            np.asarray(test_labels_list),
            np.asarray(_gather_list(test_subj, args.ddp)),
        )
        history["test_subj_acc"] = test_subj_acc
        history["test_subj_n"] = test_subj_n

    if is_main:
        if is_regression:
            print(
                f"✅ Fine-tuning Complete.\n"
                f"Test MAE: {_te['mae']:.3f} | Test R2: {_te['r2']:.4f} | "
                f"Total Time: {(time()-t0)/60/60:.1f} hours."
            )
            print(f"Subj Test MAE: {test_subj_mae:.3f} (n={test_subj_n} subjects)")
            if args.use_wandb:
                wandb.log({"test_mae": _te["mae"], "test_r2": _te["r2"],
                           "test_subj_mae": test_subj_mae})
        else:
            print(
                f"✅ Fine-tuning Complete.\n"
                f"Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | "
                f"Total Time: {(time()-t0)/60/60:.1f} hours."
            )
            print(f"Subj Test Acc: {test_subj_acc:.4f} (n={test_subj_n} subjects)")
            if args.use_wandb:
                wandb.log({"test_acc": test_acc, "test_f1": test_f1,
                           "test_subj_acc": test_subj_acc})

        exp_mode = cfg.current_mode
        current_fold = cfg.current_fold_idx
        history_tag = get_experiment_tag(args, mode=exp_mode, fold_idx=current_fold)
        save_run_history(history, test_acc, test_f1, args, history_tag)

    return test_acc, test_f1, history


def format_epoch_summary(payload):
    """One-line summary of an epoch payload, for the progress-bar subtitle.

    Classification: running train accuracy (train mode, dropout active) and validation
    accuracy. Regression: train/validation MAE and validation R². DAPT: train/validation
    reconstruction loss. Missing or NaN values are shown as an em dash.
    """
    def _fmt(value, nd):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "—"
        return "—" if v != v else f"{v:.{nd}f}"

    if payload.get("task") == "dapt" or "train_recon" in payload:
        return (f"train recon {_fmt(payload.get('train_recon'), 4)} | "
                f"val recon {_fmt(payload.get('val_recon'), 4)}")
    if payload.get("task") == "regression":
        return (f"train MAE {_fmt(payload.get('train_mae'), 2)} | "
                f"val MAE {_fmt(payload.get('val_mae'), 2)} | "
                f"val R² {_fmt(payload.get('val_r2'), 3)}")
    return (f"train acc (running, train mode) {_fmt(payload.get('train_acc'), 3)} | "
            f"val acc {_fmt(payload.get('val_acc'), 3)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Perceiver Classifier on HCP dataset (new API)"
    )
    parser.add_argument(
        '--dataset', type=str, choices=['hcp_ya', 'ocd'], default='hcp_ya',
        help="Training dataset: 'hcp_ya' (labels gender, age) or 'ocd' (labels condition, gender, age). "
             "OCD always uses all subjects."
    )
    parser.add_argument(
        '--label', type=str, default='gender',
        help="Target: gender, age (regression) or condition (OCD only)."
    )
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
        '--patience', type=int, default=5,
        help='Early stopping: stop after this many epochs without a new best validation loss; '
             'the best-validation-loss weights are used for the test evaluation (0 = disabled).'
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
        '--dapt_init_ckpt', type=str, default=None,
        help='Path to a finetuned classifier checkpoint (Stage 1/2). If set, '
             'Stage 3.1 DAPT initialises the Perceiver from it; encoders stay '
             'at HCP-AGING init (default: None = HCP-AGING baseline).'
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
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _validate_args(parser, args):
    """Reject option combinations that would otherwise fail deep inside a run."""
    available = DATASET_LABELS.get(args.dataset, ())
    if args.label not in available:
        parser.error(f"--label {args.label!r} is not available for --dataset {args.dataset!r}; "
                     f"choose one of {list(available)}")
    if args.mode == "dapt" and args.ddp:
        parser.error("--mode dapt does not support --ddp (run DAPT on a single process)")
    if not 0.0 <= float(args.recon_weight) <= 1.0:
        parser.error(f"--recon_weight must be in [0, 1], got {args.recon_weight}")


# Experiment wrappers live in experiment.py (Phase 4); DAPT training lives in dapt.py (Phase 6).
# Both are resolved lazily to avoid circular imports, so
# ``from OCD_symphony.model.finetuning import run_single_split`` keeps working.
_LAZY_ATTRS = {
    **{n: "OCD_symphony.model.experiment" for n in (
        "load_training_dataset", "split_fold_data", "run_dapt", "run_single_split",
        "run_single_fold", "run_cross_validation",
    )},
    "dapt": "OCD_symphony.model.dapt",
}


def __getattr__(name):
    module_path = _LAZY_ATTRS.get(name)
    if module_path is not None:
        import importlib
        return getattr(importlib.import_module(module_path), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

if __name__ == '__main__':
    from OCD_symphony.model.experiment import (
        run_dapt, run_single_fold, run_cross_validation, run_single_split,
    )
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
