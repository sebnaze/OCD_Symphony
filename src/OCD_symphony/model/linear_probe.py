"""Frozen-feature logistic-regression probe.

Extracts Perceiver latents from frozen (HCP-AGING or DAPT) encoders, reduces them
with one of four pooling modes, and fits a class-balanced logistic regression on
the train split. C is selected on validation log-loss; the test split is evaluated
once at the selected C.

Pooling modes (latents are (windows, 450 ROIs, 128 features)):
  * ``feature``: mean over ROIs -> 128 features (whole-brain summary).
  * ``roi``:     mean over features -> 450 features (one value per ROI).
  * ``network``: mean over ROIs within each network -> 8 x 128 = 1024 features.
  * ``pca``:     no averaging; all 57,600 values reduced to ``PCA_COMPONENTS``
                 principal components, PCA fitted on each fit's train split only.

The question this answers: is the label linearly decodable from the frozen
representation? If the probe is at the majority baseline, fine-tuning sweeps
are unlikely to help.
"""

import json
import os
from time import strftime

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from OCD_symphony.utils import proj_dir
from OCD_symphony.data_preparation.make_hcp_dataset import (
    DATASET_DIRS,
    make_hcp_dataset_from_pt,
    subject_kfold,
    subject_train_val_test_split,
)
from OCD_symphony.model.finetuning import (
    _regression_metrics,
    build_pretrained_models,
    split_fold_data,
)
from sklearn.linear_model import Ridge

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
RIDGE_ALPHA_GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
POOLING_MODES = ("feature", "roi", "network", "pca")
DEFAULT_POOLING = "pca"
PCA_COMPONENTS = 256


def load_network_labels():
    """Per-ROI network labels (450,) as an array of strings, in ROI order."""
    from OCD_symphony.data_validation.validate_preprocessing import load_RSNs

    rs_nets, _ = load_RSNs()
    return np.asarray(rs_nets)


def pool_latents(latents, mode, roi_networks=None):
    """Pool Perceiver latents (B, R, D) according to ``mode``.

    ``roi_networks`` (length R, strings) is required for ``network`` mode. Networks
    are ordered by first appearance in the label list. ``pca`` returns the full
    flattened (B, R*D) tensor; PCA itself is applied later, inside the fit.
    """
    if mode not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}, got {mode!r}")
    B, R, D = latents.shape
    if mode == "feature":
        return latents.mean(dim=1)                       # (B, D)
    if mode == "roi":
        return latents.mean(dim=2)                       # (B, R)
    if mode == "pca":
        return latents.reshape(B, R * D)                 # (B, R*D)
    # network
    if roi_networks is None or len(roi_networks) != R:
        raise ValueError("network pooling needs one label per ROI")
    names = list(dict.fromkeys(list(roi_networks)))      # order of first appearance
    per_net = []
    for name in names:
        mask = torch.as_tensor(np.asarray(roi_networks) == name, device=latents.device)
        per_net.append(latents[:, mask, :].mean(dim=1))  # (B, D)
    return torch.stack(per_net, dim=1).reshape(B, len(names) * D)  # (B, Nnet*D)


@torch.no_grad()
def extract_pooled_latents(subset, models, device, pooling=DEFAULT_POOLING,
                           roi_networks=None, batch_size=8, progress=None):
    """Return (X, y): pooled Perceiver latents (N, F) as float32, and labels (N,).

    ``progress``, if given, receives ``{"stage": "extract", "total": n}`` once, then
    ``{"stage": "extract", "done": i, "total": n}`` after each batch.
    """
    spatial_model, temporal_model, conv_model, perceiver = models
    for m in (spatial_model, temporal_model, conv_model, perceiver):
        m.eval()

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)
    n_batches = len(loader)
    _emit(progress, stage="extract", total=n_batches)
    feats, labels = [], []
    for i, (fmri, fc, gm, y) in enumerate(loader, start=1):
        fmri, fc, gm = fmri.to(device), fc.to(device), gm.to(device)
        _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
        _, _, _, _, time_embeds, _ = temporal_model(fmri)
        _, signal_context = conv_model(fmri, mask=None)
        _, latents, _ = perceiver(roi_embeds, time_embeds, signal_context, roi_pos)  # (B, R, D)
        pooled = pool_latents(latents, pooling, roi_networks)
        feats.append(pooled.float().cpu().numpy())
        labels.append(y.numpy())
        _emit(progress, stage="extract", done=i, total=n_batches)
    return np.concatenate(feats), np.concatenate(labels)


def _emit(progress, **event):
    """Forward a progress event if a callback was given; no-op otherwise."""
    if progress is not None:
        progress(event)


def _split_metrics(clf, X, y):
    pred = clf.predict(X)
    return {
        "n": int(len(y)),
        "acc": float(np.mean(pred == y)),
        "balanced_acc": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pred_counts": np.bincount(pred, minlength=2).tolist(),
        "true_counts": np.bincount(y, minlength=2).tolist(),
    }


def _reduce_features(X_tr, X_va, X_te, pca_components, seed):
    """Return transformed (Xtr, Xva, Xte) and the effective PCA component count.

    Without PCA: standardize on train. With PCA: fit PCA on the centered train
    matrix (PCA centers internally), project val and test, then standardize the
    components on train. The count is clamped to ``n_train - 1``.
    """
    if pca_components is None:
        scaler = StandardScaler().fit(X_tr)
        return scaler.transform(X_tr), scaler.transform(X_va), scaler.transform(X_te), None

    k = int(min(pca_components, X_tr.shape[0] - 1, X_tr.shape[1]))
    pca = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(X_tr)
    Ztr, Zva, Zte = pca.transform(X_tr), pca.transform(X_va), pca.transform(X_te)
    scaler = StandardScaler().fit(Ztr)
    return scaler.transform(Ztr), scaler.transform(Zva), scaler.transform(Zte), k


def fit_logreg_probe(X_tr, y_tr, X_va, y_va, X_te, y_te, c_grid=C_GRID, seed=42,
                     pca_components=None, on_step=None):
    """Reduce features (standardize, or PCA + standardize), sweep C on validation
    log-loss, evaluate test once at the selected C.

    ``on_step(done, total)``, if given, is called after each C value.
    """
    Xtr, Xva, Xte, k_eff = _reduce_features(X_tr, X_va, X_te, pca_components, seed)

    grid, best = [], None
    for step, C in enumerate(c_grid, start=1):
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=seed)
        clf.fit(Xtr, y_tr)
        val_ll = float(log_loss(y_va, clf.predict_proba(Xva), labels=[0, 1]))
        grid.append({"C": float(C), "val_log_loss": val_ll,
                     "val_balanced_acc": float(balanced_accuracy_score(y_va, clf.predict(Xva)))})
        if best is None or val_ll < best[0]:
            best = (val_ll, C, clf)
        if on_step is not None:
            on_step(step, len(c_grid))

    _, best_C, best_clf = best
    majority = int(np.bincount(y_tr, minlength=2).argmax())
    return {
        "best_C": float(best_C),
        "c_grid": grid,
        "n_features_input": int(X_tr.shape[1]),
        "n_features_model": int(Xtr.shape[1]),
        "pca_components_effective": None if k_eff is None else int(k_eff),
        "train": _split_metrics(best_clf, Xtr, y_tr),
        "val": _split_metrics(best_clf, Xva, y_va),
        "test": _split_metrics(best_clf, Xte, y_te),
        "majority_baseline": {
            "majority_class": majority,
            "val_acc": float(np.mean(y_va == majority)),
            "test_acc": float(np.mean(y_te == majority)),
        },
    }


def fit_ridge_probe(X_tr, y_tr, X_va, y_va, X_te, y_te, alpha_grid=RIDGE_ALPHA_GRID,
                    seed=42, pca_components=None, on_step=None):
    """Regression probe (age): ridge regression. Alpha is selected on validation MSE.

    Same feature reduction as the classification probe (standardize, or PCA + standardize).
    Returns MAE, R² and Pearson r per split.
    """
    Xtr, Xva, Xte, k_eff = _reduce_features(X_tr, X_va, X_te, pca_components, seed)

    grid, best = [], None
    for step, alpha in enumerate(alpha_grid, start=1):
        model = Ridge(alpha=alpha).fit(Xtr, y_tr)
        pred_va = model.predict(Xva)
        val_mse = float(np.mean((pred_va - y_va) ** 2))
        grid.append({"alpha": float(alpha), "val_mse": val_mse,
                     "val_mae": float(np.mean(np.abs(pred_va - y_va)))})
        if best is None or val_mse < best[0]:
            best = (val_mse, alpha, model)
        if on_step is not None:
            on_step(step, len(alpha_grid))

    _, best_alpha, best_model = best
    return {
        "task": "regression",
        "best_alpha": float(best_alpha),
        "alpha_grid": grid,
        "n_features_input": int(X_tr.shape[1]),
        "n_features_model": int(Xtr.shape[1]),
        "pca_components_effective": None if k_eff is None else int(k_eff),
        "train": _regression_split_metrics(best_model, Xtr, y_tr),
        "val": _regression_split_metrics(best_model, Xva, y_va),
        "test": _regression_split_metrics(best_model, Xte, y_te),
        "mean_baseline": {
            "train_mean": float(np.mean(y_tr)),
            "test_mae": float(np.mean(np.abs(y_te - np.mean(y_tr)))),
        },
    }


def _regression_split_metrics(model, X, y):
    pred = model.predict(X)
    m = _regression_metrics(y, pred)
    return {"n": int(len(y)), "mae": m["mae"], "r2": m["r2"], "r": m["r"],
            "mean_pred": float(np.mean(pred)), "mean_true": float(np.mean(y))}


def _save_probe_result(payload, tag):
    out_dir = os.path.join(proj_dir, "results", "probes")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}_{strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return out_path


def _extract_all(dataset, models, device, pooling, roi_networks, progress=None):
    """Extract features once for every dataset index (row i == dataset[i])."""
    return extract_pooled_latents(
        Subset(dataset, list(range(len(dataset)))), models, device,
        pooling=pooling, roi_networks=roi_networks, progress=progress,
    )


def _summarize_folds(folds, keys=("balanced_acc", "acc")):
    """Mean and sample SD of per-fold test metrics."""
    out = {}
    for key in keys:
        vals = np.array([f["result"]["test"][key] for f in folds], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "sd": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "per_fold": vals.tolist(),
        }
    return out


def run_logreg_probe(args, progress=None):
    """Run the frozen-feature logistic-regression probe for the given config.

    ``args.probe_eval``:
      * ``"split"`` (default): one subject-level train/val/test split.
      * ``"cv"``: subject-level K-fold CV (``subject_kfold``, same folds as fine-tuning
        CV). Inside each outer fold, C (and the PCA fit, if used) uses only the inner
        train split from ``split_fold_data``; the outer test fold is scored once.

    ``args.pooling``: one of ``feature``, ``roi``, ``network``, ``pca`` (default ``pca``).

    ``progress``: optional callable receiving event dicts (display only, no effect on
    results): ``{"stage": "extract", "total"}``, ``{"stage": "extract", "done", "total"}``,
    and ``{"stage": "split" | "fold", "fold"?, "done", "total"}`` after each C value.

    Encoders run once over all windows; per-fold results index those features.
    Returns (payload, saved_json_path).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_mode = getattr(args, "probe_eval", "split")
    if eval_mode not in ("split", "cv"):
        raise ValueError(f"probe_eval must be 'split' or 'cv', got {eval_mode!r}")
    pooling = getattr(args, "pooling", DEFAULT_POOLING)
    if pooling not in POOLING_MODES:
        raise ValueError(f"pooling must be one of {POOLING_MODES}, got {pooling!r}")
    pca_components = PCA_COMPONENTS if pooling == "pca" else None
    roi_networks = load_network_labels() if pooling == "network" else None

    dataset_name = getattr(args, "dataset", "hcp_ya")
    n_subjs = None if dataset_name == "ocd" else args.n_subjs  # OCD: always all subjects
    dataset, subjs = make_hcp_dataset_from_pt(
        path=DATASET_DIRS[dataset_name],
        label=args.label,
        n_subjs=n_subjs,
        max_abs=getattr(args, "max_abs", 1e4),
        dataset=dataset_name,
    )
    is_regression = dataset.task == "regression"
    fit_probe = fit_ridge_probe if is_regression else fit_logreg_probe
    probe_kind = "ridge" if is_regression else "logreg"
    metric_keys = ("mae", "r2") if is_regression else ("balanced_acc", "acc")

    dapt_ckpt_path = getattr(args, "dapt_ckpt", None)
    encoder_ckpt = torch.load(dapt_ckpt_path, map_location=device) if dapt_ckpt_path else None
    init = "dapt" if encoder_ckpt is not None else "hcp_aging"
    models = build_pretrained_models(args, device, mask_ratio=0.0, encoder_ckpt=encoder_ckpt)
    X_all, y_all = _extract_all(dataset, models, device, pooling, roi_networks, progress=progress)
    subj_all = np.asarray(dataset.subjs)

    meta = {
        "probe": f"{probe_kind}_pooled_latents",
        "task": dataset.task,
        "dataset": dataset_name,
        "probe_eval": eval_mode,
        "pooling": pooling,
        "pca_components_requested": pca_components,
        "label": args.label,
        "init": init,
        "dapt_ckpt": dapt_ckpt_path,
        "n_subjs": n_subjs,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "val_frac": args.val_frac,
        "n_subjects_total": int(len(np.unique(subjs))),
        "class_weight": None if is_regression else "balanced",
    }

    if eval_mode == "split":
        trainset, valset, testset = subject_train_val_test_split(
            dataset, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed,
        )
        meta["n_subjects"] = {
            "train": int(len(np.unique(subj_all[trainset.indices]))),
            "val": int(len(np.unique(subj_all[valset.indices]))),
            "test": int(len(np.unique(subj_all[testset.indices]))),
            "total": int(len(np.unique(subjs))),
        }
        result = fit_probe(
            X_all[trainset.indices], y_all[trainset.indices],
            X_all[valset.indices], y_all[valset.indices],
            X_all[testset.indices], y_all[testset.indices],
            seed=args.seed, pca_components=pca_components,
            on_step=lambda done, total: _emit(progress, stage="split", done=done, total=total),
        )
        payload = {"meta": meta, "result": result}
        tag = f"{probe_kind}_{dataset_name}_{args.label}_{init}_{pooling}_split"
    else:
        n_folds = int(getattr(args, "n_folds", 5))
        meta["n_folds"] = n_folds
        meta["n_subjects"] = {"total": int(len(np.unique(subjs)))}
        folds = []
        for fold_idx, (trainset, testset) in enumerate(
            subject_kfold(dataset, n_splits=n_folds, seed=args.seed)
        ):
            new_train, valset, testset = split_fold_data(dataset, trainset, testset, fold_idx, args)
            res = fit_probe(
                X_all[new_train.indices], y_all[new_train.indices],
                X_all[valset.indices], y_all[valset.indices],
                X_all[testset.indices], y_all[testset.indices],
                seed=args.seed, pca_components=pca_components,
                on_step=lambda done, total, f=fold_idx: _emit(
                    progress, stage="fold", fold=f, n_folds=n_folds, done=done, total=total),
            )
            folds.append({
                "fold": fold_idx,
                "n_train_subjects": int(len(np.unique(subj_all[new_train.indices]))),
                "n_val_subjects": int(len(np.unique(subj_all[valset.indices]))),
                "n_test_subjects": int(len(np.unique(subj_all[testset.indices]))),
                "result": res,
            })
        payload = {
            "meta": meta,
            "summary": _summarize_folds(folds, keys=metric_keys),
            "folds": folds,
        }
        tag = f"{probe_kind}_{dataset_name}_{args.label}_{init}_{pooling}_cv{n_folds}"

    out_path = _save_probe_result(payload, tag)
    return payload, out_path
