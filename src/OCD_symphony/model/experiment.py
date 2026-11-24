"""Experiment wrappers: dataset loading, splits, and the single / fold / CV / DAPT runs.

Moved out of ``finetuning.py`` (Phase 4). The training functions (``finetune``, ``dapt``)
stay in ``finetuning.py`` and are reached through the module object (``_ft``), so
``importlib.reload(finetuning)`` remains effective for them. ``finetuning.py`` re-exports
the names below lazily (module ``__getattr__``) to avoid a circular import.
"""
import numpy as np
import wandb
from time import time
from torch.utils.data import Subset

from OCD_symphony.model import finetuning as _ft
from OCD_symphony.model.config import FinetuneConfig
from OCD_symphony.data_preparation.make_hcp_dataset import (
    make_hcp_dataset_from_pt,
    split_kfold,
    split_train_val_test,
    split_inner_train_val,
    subject_overlap,
    apply_label_permutation,
    dataset_dirs,
)


def _protocol(args):
    """Split protocol from args: 'subject' (default, valid) or 'sample' (leaky diagnostic).

    Subject-level target scope always uses subject folds: one row per subject with
    sample-level folds would put the same subject in train and test.
    """
    if getattr(args, "target_scope", "sample") == "subject":
        return "subject"
    return getattr(args, "split_protocol", "subject") or "subject"


def _report_overlap(args, subsets, names, is_main):
    """Print how many subjects appear in more than one partition (0 for subject protocol)."""
    if not is_main:
        return
    n = subject_overlap(subsets)
    tag = "" if _protocol(args) == "subject" else "  [sample protocol: LEAKY]"
    detail = ", ".join(f"{nm}={len(s)}" for nm, s in zip(names, subsets))
    print(f"[splits] protocol={_protocol(args)} | subjects shared across partitions: {n}{tag} | {detail}")



def load_training_dataset(args, progress=None):
    """Load the dataset selected by ``args.dataset`` ('hcp_ya' or 'ocd') as an FMRIDataset.

    The OCD cohort always uses all subjects (no subject-count scope).
    ``progress`` receives ``{"stage": "load", "done", "total"}`` per file (display only).
    """
    dataset_name = getattr(args, "dataset", "hcp_ya")
    n_subjs = None if dataset_name == "ocd" else args.n_subjs
    preprocessing = getattr(args, "preprocessing", "paper")  # "legacy" routes to the old files
    return make_hcp_dataset_from_pt(
        path=dataset_dirs(preprocessing)[dataset_name],
        label=args.label,
        n_subjs=n_subjs,
        max_abs=getattr(args, "max_abs", 1e4),
        dataset=dataset_name,
        preprocessing=preprocessing,
        progress=progress,
    )


def _load_and_report(args, progress, is_main):
    """Load the dataset and print the one-line summary on the main process."""
    t0 = time()
    dataset, subjs = load_training_dataset(args, progress=progress)
    if is_main:
        print(
            f"Dataset loaded in {time()-t0:.1f}s ({(time()-t0)/60:.1f} min) | "
            f"{len(np.unique(subjs))} subjects | {len(dataset)} samples"
        )
    return dataset, subjs


def _apply_label_shuffle(args, dataset, is_main):
    """Label-permutation null: permute subject labels once, before any split (splits unchanged)."""
    cfg = FinetuneConfig.from_args(args)
    if not cfg.shuffle_labels:
        return None
    info = apply_label_permutation(dataset, cfg.effective_perm_seed)
    if is_main:
        print(f"[Label permutation] labels shuffled across {info['n_subjects']} subjects "
              f"(perm seed={info['seed']}); encoded_labels and splits unchanged")
    return info


def _wandb_start(args, name, is_main, reinit=False):
    """Start a W&B run on the main process when ``args.use_wandb`` is set."""
    if args.use_wandb and is_main:
        kwargs = {
            "project": "OCD_Symphony",
            "name": name,
            "config": vars(args) if hasattr(args, "__dict__") else None,
        }
        if reinit:
            kwargs["reinit"] = True
        wandb.init(**kwargs)


def _wandb_finish(args, is_main):
    if args.use_wandb and is_main:
        wandb.finish()


def split_fold_data(dataset, trainset, testset, fold_idx, args):
    """
    Split a cross-validation fold trainset into train and validation sets.

    Subject protocol: StratifiedGroupKFold over subjects (no subject in both sets).
    Sample protocol: StratifiedKFold over windows (subjects may appear in both sets).
    """
    protocol = _protocol(args)
    trainset_inner, valset = split_inner_train_val(
        dataset, list(trainset.indices), args.val_frac,
        seed=args.seed + fold_idx, protocol=protocol,
    )
    if protocol == "subject":
        leaked = np.intersect1d(
            np.unique(dataset.subjs[list(valset.indices)]),
            np.unique(dataset.subjs[list(trainset_inner.indices)]),
        )
        if leaked.size:
            raise AssertionError(f"Subject leakage detected: {leaked.tolist()}")
    return trainset_inner, valset, testset


def run_dapt(args, local_rank=0, epoch_callback=None, progress=None):
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
    is_main = _ft.is_main_process()
    if FinetuneConfig.from_args(args).shuffle_labels:
        raise ValueError("shuffle_labels does not apply to Stage 3.1 DAPT (label-free)")
    dataset, _subjs = _load_and_report(args, progress, is_main)

    trainset, valset, _testset = split_train_val_test(
        dataset, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed,
        protocol=_protocol(args),
    )
    _report_overlap(args, [trainset, valset, _testset], ["train", "val", "test"], is_main)

    # DAPT trains on the train subjects only; the val subjects are the held-out
    # monitor set (used for checkpoint selection) and the test subjects stay
    # untouched so Stage 3.2 reports a clean, non-transductive result.
    dapt_trainset = trainset

    args._current_mode = "dapt"
    args._current_fold_idx = None
    tag = _ft.get_experiment_tag(args, mode="dapt")

    if is_main:
        print(f"[DAPT] train={len(dapt_trainset)} | val(monitor)={len(valset)} samples")

    _wandb_start(args, f"DAPT_{tag}", is_main)
    try:
        ckpt_path, history = _ft.dapt(
            dapt_trainset, valset, args,
            local_rank=local_rank, epoch_callback=epoch_callback,
            exclusion_report=getattr(dataset, "exclusion_report", None),
        )
    finally:
        _wandb_finish(args, is_main)

    return ckpt_path, history


def run_single_split(args, local_rank=0, epoch_callback=None, progress=None):
    """Single deterministic train/val/test split using subject_train_val_test_split."""
    is_main = _ft.is_main_process()
    dataset, _subjs = _load_and_report(args, progress, is_main)
    _apply_label_shuffle(args, dataset, is_main)

    trainset, valset, testset = split_train_val_test(
        dataset, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed,
        protocol=_protocol(args),
    )
    _report_overlap(args, [trainset, valset, testset], ["train", "val", "test"], is_main)

    args._current_mode = "single"
    args._current_fold_idx = None
    tag = _ft.get_experiment_tag(args, mode="single")

    _wandb_start(args, f"Finetune_{tag}", is_main)
    try:
        test_acc, test_f1, history = _ft.finetune(
            trainset, valset, testset, args, local_rank=local_rank, epoch_callback=epoch_callback)
    finally:
        _wandb_finish(args, is_main)

    return test_acc, test_f1, history


def run_single_fold(args, local_rank=0, epoch_callback=None, progress=None):
    """Run fine-tuning for a single specified CV fold."""
    is_main = _ft.is_main_process()

    if args.fold_idx < 0 or args.fold_idx >= args.n_folds:
        raise ValueError(
            f"--fold_idx must be between 0 and {args.n_folds - 1}, got {args.fold_idx}"
        )

    dataset, _subjs = _load_and_report(args, progress, is_main)
    _apply_label_shuffle(args, dataset, is_main)

    folds = list(split_kfold(dataset, n_splits=args.n_folds, seed=args.seed, protocol=_protocol(args)))
    trainset, testset = folds[args.fold_idx]

    if is_main:
        print("\n" + "=" * 60)
        print(f"  Running Fold {args.fold_idx + 1} / {args.n_folds} (index: {args.fold_idx})")
        print("=" * 60)

    new_trainset, valset, testset = split_fold_data(
        dataset, trainset, testset, args.fold_idx, args
    )
    _report_overlap(args, [new_trainset, valset, testset], ["train", "val", "test"], is_main)

    if is_main:
        print(
            f"train={len(new_trainset)} | val={len(valset)} | test={len(testset)} samples"
        )

    args._current_mode = "single_fold"
    args._current_fold_idx = args.fold_idx
    tag = _ft.get_experiment_tag(args, mode="single_fold", fold_idx=args.fold_idx)

    _wandb_start(args, f"Finetune_{tag}", is_main, reinit=True)
    try:
        test_acc, test_f1, history = _ft.finetune(
            new_trainset, valset, testset, args,
            local_rank=local_rank,
            epoch_callback=epoch_callback
        )
    finally:
        _wandb_finish(args, is_main)

    return test_acc, test_f1, history


def run_cross_validation(args, local_rank=0, epoch_callback=None, progress=None):
    """Full K-fold leave-subject-out cross-validation using subject_kfold."""
    is_main = _ft.is_main_process()
    dataset, _subjs = _load_and_report(args, progress, is_main)
    _apply_label_shuffle(args, dataset, is_main)

    folds = list(split_kfold(dataset, n_splits=args.n_folds, seed=args.seed, protocol=_protocol(args)))

    all_accs = []
    all_f1s = []
    all_histories = []
    cv_task = "classification"

    for fold_idx, (trainset, testset) in enumerate(folds):
        if is_main:
            print("\n" + "=" * 60)
            print(f"  Fold {fold_idx + 1} / {args.n_folds}")
            print("=" * 60)

        new_trainset, valset, testset = split_fold_data(
            dataset, trainset, testset, fold_idx, args
        )
        _report_overlap(args, [new_trainset, valset, testset], ["train", "val", "test"], is_main)

        if is_main:
            print(
                f"train={len(new_trainset)} | val={len(valset)} | test={len(testset)} samples"
            )

        args._current_mode = "cv"
        args._current_fold_idx = fold_idx
        tag = _ft.get_experiment_tag(args, mode="cv", fold_idx=fold_idx)

        _wandb_start(args, f"Finetune_{tag}", is_main, reinit=True)

        def fold_epoch_callback(ep, metrics):
            if epoch_callback is not None:
                m_copy = dict(metrics)
                m_copy["fold"] = fold_idx + 1
                m_copy["n_folds"] = args.n_folds
                epoch_callback(ep, m_copy)

        try:
            test_acc, test_f1, history = _ft.finetune(
                new_trainset, valset, testset, args,
                local_rank=local_rank,
                epoch_callback=fold_epoch_callback if epoch_callback else None
            )
        finally:
            _wandb_finish(args, is_main)
        if history.get("task") == "regression":
            cv_task = "regression"
            all_accs.append(history["test_mae"])
            all_f1s.append(history["test_r2"])
        else:
            all_accs.append(test_acc)
            all_f1s.append(test_f1)
        all_histories.append(history)

    mean_acc = sum(all_accs) / len(all_accs) if all_accs else 0.0
    std_acc = (sum((a - mean_acc) ** 2 for a in all_accs) / len(all_accs)) ** 0.5 if all_accs else 0.0
    mean_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0.0
    std_f1 = (sum((f - mean_f1) ** 2 for f in all_f1s) / len(all_f1s)) ** 0.5 if all_f1s else 0.0

    m1, m2 = ("MAE", "R2") if cv_task == "regression" else ("Acc", "F1")
    if is_main:
        print("\n" + "=" * 60)
        print(f"  Cross-Validation Results  ({args.n_folds} folds)")
        print("=" * 60)
        for i, (acc, f1) in enumerate(zip(all_accs, all_f1s)):
            print(f"  Fold {i+1}:  Test {m1} = {acc:.4f}  |  Test {m2} = {f1:.4f}")
        print("  " + "─" * 45)
        print(
            f"  Mean ± Std:  {m1} = {mean_acc:.4f} ± {std_acc:.4f}  |  "
            f"{m2} = {mean_f1:.4f} ± {std_f1:.4f}"
        )
        print("=" * 60)

    cv_summary = {
        "is_cv": True,
        "task": cv_task,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "fold_accs": all_accs,
        "fold_f1s": all_f1s,
        "histories": all_histories,
    }

    if is_main:
        _ft.save_run_history(
            cv_summary, mean_acc, mean_f1, args,
            _ft.get_experiment_tag(args, mode="cv"),
        )

    return mean_acc, mean_f1, cv_summary
