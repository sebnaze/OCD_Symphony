import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from time import time
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


def finetune(trainset, valset, testset, args, local_rank=0):
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

    # ── Load pretrained models ──────────────────────────────────────────────
    base_transformer = TemporalTransformer(
        NUM_LAYERS, DIM, NUM_HEADS, MLP_DIM, TIME_STEPS, MAX_LEN_TEMP, ROIs
    ).to(device)
    masker = TokenMasker1D(mask_ratio=0)
    temporal_model = MaskedTemporalTransformer(
        base_transformer, masker, roi_dim=ROIs, hidden_dim=256
    ).to(device)

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
    masker = TokenMasker2D(mask_ratio=0)
    spatial_model = MaskedSpatialTransformer(
        base_transformer, masker, TIME_STEPS, hidden_dim=256
    ).to(device)

    masker = TimePointMasker(mask_ratio=0)
    conv_model = MaskedConv1DTemporalModel(
        input_channels=1, signal_dim=64, decoder_hidden_dims=[32, 8]
    ).to(device)

    # ── Load state dicts ────────────────────────────────────────────────────
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

    temporal_model.eval()
    spatial_model.eval()
    conv_model.eval()

    for model in [spatial_model, temporal_model, conv_model]:
        for param in model.parameters():
            param.requires_grad = False

    # ── Load Pretrained Perceiver ───────────────────────────────────────────
    perceiver = PerceiverIOModelROIWise(
        input_dim=64 * 3,
        latent_dim=DIM,
        num_rois=ROIs,
        num_heads=NUM_HEADS,
        time_steps=TIME_STEPS,
    ).to(device)
    perceiver.load_state_dict(
        torch.load(
            os.path.join(
                proj_dir, 'models',
                f"perceiver_fused_pretrained_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
            ), map_location=device,
        )
    )
    classifier = PerceiverClassifier(perceiver, latent_dim=DIM, num_classes=2).to(device)

    # ── Wrap in DDP when distributed ────────────────────────────────────────
    if args.ddp:
        classifier = DDP(classifier, device_ids=[local_rank], find_unused_parameters=False)

    # ── Optimizer and Loss ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=args.learning_rate, weight_decay=1e-2
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(0.1 * args.n_epochs * len(train_loader)),
        args.n_epochs * len(train_loader),
    )
    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ───────────────────────────────────────────────────────
    best_val_acc = 0.0
    t0 = time()
    for epoch in range(args.n_epochs):
        t1 = time()

        if args.ddp:
            train_sampler.set_epoch(epoch)

        classifier.train()
        total_loss, correct = 0.0, 0
        all_preds, all_labels = [], []

        for fmri, fc, gm, labels in train_loader:
            fmri, fc, gm, labels = (
                fmri.to(device),
                fc.to(device),
                gm.to(device),
                labels.to(device),
            )

            with torch.no_grad():
                _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
                _, _, _, _, time_embeds, _ = temporal_model(fmri)
                _, signal_context = conv_model(fmri, mask=None)

            logits, recon = classifier(roi_embeds, time_embeds, signal_context, roi_pos)
            loss = criterion(logits, labels) + F.mse_loss(recon, fmri)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
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

        if is_main:
            print(
                f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | "
                f"Acc: {train_acc:.4f} | F1: {train_f1:.4f}"
            )
            if args.use_wandb:
                wandb.log({
                    "train_loss": total_loss,
                    "train_acc": train_acc,
                    "train_f1": train_f1,
                }, step=epoch + 1)

        # ── Validation ──────────────────────────────────────────────────────
        classifier.eval()
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
                torch.save(
                    model_state,
                    os.path.join(
                        proj_dir, 'models',
                        f"Fine_tuned_perceiver_classifier_gender_HCP_YA_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
                    ),
                )

    # ── Final evaluation on held-out test set ───────────────────────────────
    classifier.eval()
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

    return test_acc, test_f1


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
        '--ddp', action='store_true',
        help='Enable DistributedDataParallel (multi-GPU). Launch with torchrun.'
    )
    return parser.parse_args()


def run_single_split(args, local_rank=0):
    """Single deterministic train/val/test split using subject_train_val_test_split."""
    is_main = is_main_process()
    t0 = time()

    dataset, subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, 'data/hcp'),
        label=args.label,
        n_subjs=args.n_subjs,
    )
    if is_main:
        print(
            f"Dataset loaded in {time()-t0:.1f}s ({(time()-t0)/60:.1f} min) | "
            f"{len(np.unique(subjs))} subjects | {len(dataset)} samples"
        )

    trainset, valset, testset = subject_train_val_test_split(
        dataset, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed,
    )

    if args.use_wandb and is_main:
        wandb.init(
            project="OCD_Symphony",
            name=f"Finetuning_HCP_{args.label}_single",
        )

    test_acc, test_f1 = finetune(trainset, valset, testset, args, local_rank=local_rank)

    if args.use_wandb and is_main:
        wandb.finish()

    return test_acc, test_f1


def split_fold_data(dataset, trainset, testset, fold_idx, args):
    """
    Split a cross-validation fold's trainset into train and validation sets
    using subject-level StratifiedGroupKFold without data leakage.
    """
    ds = dataset  # the original FMRIDataset

    # Determine training subjects and their labels
    train_sample_indices = list(trainset.indices)
    train_subjs = ds.subjs[train_sample_indices]

    # Get unique training subjects with consensus labels (majority vote per subject)
    unique_train_subjs = np.unique(train_subjs)
    train_labels_subj = []
    for s in unique_train_subjs:
        mask = train_subjs == s
        lbls = ds.encoded_labels.numpy()[train_sample_indices][mask]
        counts = np.bincount(lbls)
        train_labels_subj.append(np.argmax(counts))
    train_labels_subj = np.array(train_labels_subj)

    # StratifiedGroupKFold on training subjects for val hold-out
    n_inner = max(2, round(1.0 / args.val_frac))
    inner_sgkf = StratifiedGroupKFold(
        n_splits=n_inner, shuffle=True, random_state=args.seed + fold_idx
    )
    inner_splits = list(
        inner_sgkf.split(unique_train_subjs, train_labels_subj, groups=unique_train_subjs)
    )
    # Last fold → validation
    _, val_subj_idx = inner_splits[-1]
    train_subj_idx = np.setdiff1d(
        np.arange(len(unique_train_subjs)), val_subj_idx
    )

    val_subjs = unique_train_subjs[val_subj_idx]
    new_train_subjs = unique_train_subjs[train_subj_idx]

    # Verify no leakage (assertion raises if overlap detected)
    for s in val_subjs:
        assert s not in new_train_subjs, f"Subject leakage detected: {s}"

    # Map subject IDs to sample indices
    val_sample_indices = np.where(np.isin(ds.subjs, val_subjs))[0].tolist()
    new_train_sample_indices = np.where(np.isin(ds.subjs, new_train_subjs))[0].tolist()

    new_trainset = Subset(ds, new_train_sample_indices)
    valset = Subset(ds, val_sample_indices)

    return new_trainset, valset, testset


def run_single_fold(args, local_rank=0):
    """Run fine-tuning for a single specified CV fold."""
    is_main = is_main_process()
    t0 = time()

    if args.fold_idx < 0 or args.fold_idx >= args.n_folds:
        raise ValueError(
            f"--fold_idx must be between 0 and {args.n_folds - 1}, got {args.fold_idx}"
        )

    dataset, subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, 'data/hcp'),
        label=args.label,
        n_subjs=args.n_subjs,
    )
    if is_main:
        print(
            f"Dataset loaded in {time()-t0:.1f}s ({(time()-t0)/60:.1f} min) | "
            f"{len(np.unique(subjs))} subjects | {len(dataset)} samples"
        )

    folds = list(subject_kfold(dataset, n_splits=args.n_folds, seed=args.seed))
    trainset, testset = folds[args.fold_idx]

    if is_main:
        print(f"\n{'='*60}")
        print(f"  Running Fold {args.fold_idx + 1} / {args.n_folds} (index: {args.fold_idx})")
        print(f"{'='*60}")

    new_trainset, valset, testset = split_fold_data(
        dataset, trainset, testset, args.fold_idx, args
    )

    if is_main:
        print(
            f"train={len(new_trainset)} | val={len(valset)} | test={len(testset)} samples"
        )

    if args.use_wandb and is_main:
        wandb.init(
            project="OCD_Symphony",
            name=f"Finetuning_HCP_{args.label}_fold{args.fold_idx}",
            reinit=True,
        )

    test_acc, test_f1 = finetune(new_trainset, valset, testset, args, local_rank=local_rank)

    if args.use_wandb and is_main:
        wandb.finish()

    return test_acc, test_f1


def run_cross_validation(args, local_rank=0):
    """
    Full K-fold leave-subject-out cross-validation using subject_kfold.
    The val fold from each CV split is used as the test set here.
    """
    is_main = is_main_process()
    t0 = time()

    dataset, subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, 'data/hcp'),
        label=args.label,
        n_subjs=args.n_subjs,
    )
    if is_main:
        print(
            f"Dataset loaded in {time()-t0:.1f}s ({(time()-t0)/60:.1f} min) | "
            f"{len(np.unique(subjs))} subjects | {len(dataset)} samples"
        )

    folds = subject_kfold(dataset, n_splits=args.n_folds, seed=args.seed)

    all_accs = []
    all_f1s  = []

    for fold_idx, (trainset, testset) in enumerate(folds):
        if is_main:
            print(f"\n{'='*60}")
            print(f"  Fold {fold_idx + 1} / {args.n_folds}")
            print(f"{'='*60}")

        new_trainset, valset, testset = split_fold_data(
            dataset, trainset, testset, fold_idx, args
        )

        if is_main:
            print(
                f"train={len(new_trainset)} | val={len(valset)} | test={len(testset)} samples"
            )

        if args.use_wandb and is_main:
            wandb.init(
                project="OCD_Symphony",
                name=f"Finetuning_HCP_{args.label}_fold{fold_idx}",
                reinit=True,
            )

        test_acc, test_f1 = finetune(new_trainset, valset, testset, args, local_rank=local_rank)
        all_accs.append(test_acc)
        all_f1s.append(test_f1)

        if args.use_wandb and is_main:
            wandb.finish()

    # ── Aggregate CV results (main process only) ────────────────────────────
    if is_main:
        print(f"\n{'='*60}")
        print(f"  Cross-Validation Results  ({args.n_folds} folds)")
        print(f"{'='*60}")
        for i, (acc, f1) in enumerate(zip(all_accs, all_f1s)):
            print(f"  Fold {i+1}:  Test Acc = {acc:.4f}  |  Test F1 = {f1:.4f}")
        print(f"  {'─'*45}")
        mean_acc = sum(all_accs) / len(all_accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in all_accs) / len(all_accs)) ** 0.5
        mean_f1 = sum(all_f1s) / len(all_f1s)
        std_f1 = (sum((f - mean_f1) ** 2 for f in all_f1s) / len(all_f1s)) ** 0.5
        print(
            f"  Mean ± Std:  Acc = {mean_acc:.4f} ± {std_acc:.4f}  |  "
            f"F1 = {mean_f1:.4f} ± {std_f1:.4f}"
        )
        print(f"{'='*60}")

    return all_accs, all_f1s


if __name__ == '__main__':
    import numpy as np

    args = parse_args()

    # Initialise DDP if requested (reads RANK / WORLD_SIZE from env)
    local_rank = 0
    if args.ddp:
        _, local_rank, _ = ddp_setup()

    if args.fold_idx is not None:
        run_single_fold(args, local_rank=local_rank)
    elif args.cv:
        run_cross_validation(args, local_rank=local_rank)
    else:
        run_single_split(args, local_rank=local_rank)

    if args.ddp:
        ddp_cleanup()
