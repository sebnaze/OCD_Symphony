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
from OCD_symphony.model.metrics import _subject_ids_and_labels
from OCD_symphony.data_preparation.make_hcp_dataset import (
    dataset_dirs,
    apply_label_permutation,
    make_hcp_dataset_from_pt,
    split_kfold,
    split_train_val_test,
    subject_overlap,
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
# "all" runs every mode in POOLING_ALL_ORDER sequentially, from one extraction pass.
ALL_POOLING = "all"
POOLING_ALL_ORDER = ("pca", "feature", "roi", "network")
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


def _pooled_batches(subset, models, device, modes, roi_networks=None, batch_size=8, progress=None):
    """Yield ({mode: pooled features (B, F) float32 numpy}, labels (B,)) for each batch, in order.

    The encoders and Perceiver run once per batch; every requested pooling mode is derived
    from the same latents.
    """
    spatial_model, temporal_model, conv_model, perceiver = models
    for m in (spatial_model, temporal_model, conv_model, perceiver):
        m.eval()

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)
    n_batches = len(loader)
    _emit(progress, stage="extract", total=n_batches)
    for i, (fmri, fc, gm, y) in enumerate(loader, start=1):
        with torch.no_grad():  # a generator body does not inherit the caller's no_grad
            fmri, fc, gm = fmri.to(device), fc.to(device), gm.to(device)
            _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
            _, _, _, _, time_embeds, _ = temporal_model(fmri)
            _, signal_context = conv_model(fmri, mask=None)
            _, latents, _ = perceiver(roi_embeds, time_embeds, signal_context, roi_pos)  # (B, R, D)
            pooled = {m: pool_latents(latents, m, roi_networks).float().cpu().numpy() for m in modes}
        yield pooled, y.numpy()
        _emit(progress, stage="extract", done=i, total=n_batches)


@torch.no_grad()
def extract_pooled_latents_multi(subset, models, device, modes, roi_networks=None,
                                 batch_size=8, progress=None):
    """Return {mode: (X, y)}: pooled Perceiver latents (N, F) float32 per mode, and labels (N,).

    Single extraction pass: the encoders and Perceiver run once for all ``modes``.
    ``progress``, if given, receives ``{"stage": "extract", "total": n}`` once, then
    ``{"stage": "extract", "done": i, "total": n}`` after each batch.
    """
    modes = tuple(modes)
    feats = {m: [] for m in modes}
    labels = []
    for pooled, y in _pooled_batches(subset, models, device, modes, roi_networks,
                                     batch_size, progress):
        for m in modes:
            feats[m].append(pooled[m])
        labels.append(y)
    y_all = np.concatenate(labels)
    return {m: (np.concatenate(feats[m]), y_all) for m in modes}


def extract_pooled_latents(subset, models, device, pooling=DEFAULT_POOLING,
                           roi_networks=None, batch_size=8, progress=None):
    """Return (X, y) for one pooling mode. See ``extract_pooled_latents_multi``."""
    X, y = extract_pooled_latents_multi(subset, models, device, (pooling,), roi_networks,
                                        batch_size, progress)[pooling]
    return X, y


def extract_subject_pooled_latents_multi(subset, models, device, modes, roi_networks=None,
                                         batch_size=8, progress=None):
    """Return (subject_ids, {mode: X}, y): one row per subject per mode, the mean of its windows'
    pooled latents.

    Pooling modes are linear in the latents, so averaging the pooled features over windows
    equals pooling the window-averaged latents. Sums are accumulated batch by batch, so the
    per-window matrix is never held in memory. ``subset`` is scanned in order; the subject of
    each window comes from the dataset.
    """
    modes = tuple(modes)
    subj_win, y_win = _subject_ids_and_labels(subset, np.arange(len(subset)))
    uniq, first, inv = np.unique(subj_win, return_index=True, return_inverse=True)
    counts = np.bincount(inv).astype(np.float64)
    sums, pos = {}, 0
    for pooled, y_batch in _pooled_batches(subset, models, device, modes, roi_networks,
                                           batch_size, progress):
        for m in modes:
            if m not in sums:
                sums[m] = np.zeros((len(uniq), pooled[m].shape[1]), dtype=np.float64)
            np.add.at(sums[m], inv[pos:pos + len(y_batch)], pooled[m])
        pos += len(y_batch)
    X = {m: (sums[m] / counts[:, None]).astype(np.float32) for m in modes}
    return uniq, X, y_win[first]


def extract_subject_pooled_latents(subset, models, device, pooling=DEFAULT_POOLING,
                                   roi_networks=None, batch_size=8, progress=None):
    """Return (subject_ids, X, y) for one pooling mode. See ``extract_subject_pooled_latents_multi``."""
    uniq, Xs, y = extract_subject_pooled_latents_multi(subset, models, device, (pooling,),
                                                       roi_networks, batch_size, progress)
    return uniq, Xs[pooling], y



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


def _summarize_null(runs, keys, is_cv):
    """Mean and sample SD over label permutations of the test metrics.

    Split: each run's ``result.test[key]``. CV: each run's fold-mean of ``folds[i].result.test[key]``.
    """
    out = {}
    for key in keys:
        if is_cv:
            vals = [float(np.mean([f["result"]["test"][key] for f in r["folds"]])) for r in runs]
        else:
            vals = [float(r["result"]["test"][key]) for r in runs]
        arr = np.asarray(vals, dtype=float)
        out[key] = {
            "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "per_permutation": arr.tolist(),
        }
    return out


def _permuted_row_labels(dataset, row_subj, subject_scope):
    """Labels of the feature rows after a label permutation (``labels_tensor`` is permuted).

    Window scope: one label per window, in dataset order. Subject scope: one label per subject
    (first window), ordered like ``row_subj``.
    """
    if not subject_scope:
        return dataset.labels_tensor.numpy()
    subjs = np.asarray(dataset.subjs)
    uniq, first = np.unique(subjs, return_index=True)
    by_subj = dict(zip(uniq.tolist(), dataset.labels_tensor.numpy()[first].tolist()))
    return np.asarray([by_subj[s] for s in np.asarray(row_subj).tolist()], dtype=np.int64)


def _extract_features(dataset, models, device, modes, roi_networks, subject_scope, progress=None):
    """Extract features once for every dataset index, for all ``modes``.

    Returns {mode: (row_subj, X_all, y_all)}. Window scope: row i == dataset[i]. Subject
    scope: one row per subject, ``row_subj`` holds the subject IDs.
    """
    subset = Subset(dataset, list(range(len(dataset))))
    if subject_scope:
        if len(modes) == 1:
            row_subj, X, y = extract_subject_pooled_latents(
                subset, models, device, modes[0], roi_networks, progress=progress)
            return {modes[0]: (row_subj, X, y)}
        row_subj, Xs, y = extract_subject_pooled_latents_multi(
            subset, models, device, modes, roi_networks, progress=progress)
        return {m: (row_subj, Xs[m], y) for m in modes}

    win_subj = np.asarray(dataset.subjs)
    if len(modes) == 1:
        X, y = extract_pooled_latents(subset, models, device, modes[0], roi_networks, progress=progress)
        return {modes[0]: (win_subj, X, y)}
    feats = extract_pooled_latents_multi(subset, models, device, modes, roi_networks, progress=progress)
    return {m: (win_subj, feats[m][0], feats[m][1]) for m in modes}


def _rows_for_subjects(row_subj, X_all, y_all, subjects):
    """Select the rows of ``X_all`` for the given subject IDs (order preserved)."""
    pos = {s: i for i, s in enumerate(row_subj)}
    sel = np.asarray([pos[s] for s in subjects], dtype=np.int64)
    return X_all[sel], y_all[sel]


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


def _prepare_probe(args, progress=None):
    """Load the dataset and encoders, and extract features once for every pooling mode needed.

    Returns a context dict shared by all modes (see ``run_logreg_probe``).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_mode = getattr(args, "probe_eval", "split")
    if eval_mode not in ("split", "cv"):
        raise ValueError(f"probe_eval must be 'split' or 'cv', got {eval_mode!r}")
    pooling = getattr(args, "pooling", DEFAULT_POOLING)
    # A list/tuple selects several modes (fitted in the given order from one extraction pass).
    multi = isinstance(pooling, (list, tuple))
    if multi:
        modes = tuple(dict.fromkeys(pooling))
        if not modes or any(m not in POOLING_MODES for m in modes):
            raise ValueError(f"pooling list must contain only {POOLING_MODES}, got {pooling!r}")
    else:
        if pooling != ALL_POOLING and pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {POOLING_MODES + (ALL_POOLING,)}, got {pooling!r}")
        modes = POOLING_ALL_ORDER if pooling == ALL_POOLING else (pooling,)
    # Label-permutation null: n_permutations N > 0 runs N permuted evaluations (seeds
    # seed0..seed0+N-1). shuffle_labels alone means a single permuted run (N = 1).
    n_perm = int(getattr(args, "n_permutations", 0) or 0)
    if n_perm < 0:
        raise ValueError(f"n_permutations must be >= 0, got {n_perm}")
    if n_perm == 0 and getattr(args, "shuffle_labels", False):
        n_perm = 1
    lps = getattr(args, "label_perm_seed", None)
    seed0 = int(lps) if lps is not None else int(args.seed)
    n_folds = int(getattr(args, "n_folds", 5))
    roi_networks = load_network_labels() if "network" in modes else None

    dataset_name = getattr(args, "dataset", "hcp_ya")
    subject_scope = getattr(args, "target_scope", "sample") == "subject"
    # Subject-level scope: one row per subject, so only subject-level folds are meaningful.
    protocol = "subject" if subject_scope else (getattr(args, "split_protocol", "subject") or "subject")
    n_subjs = None if dataset_name == "ocd" else args.n_subjs  # OCD: always all subjects
    preprocessing = getattr(args, "preprocessing", "paper")  # "legacy" routes to the old files
    dataset, subjs = make_hcp_dataset_from_pt(
        path=dataset_dirs(preprocessing)[dataset_name],
        label=args.label,
        n_subjs=n_subjs,
        max_abs=getattr(args, "max_abs", 1e4),
        dataset=dataset_name,
        preprocessing=preprocessing,
    )
    is_regression = dataset.task == "regression"
    fit_probe = fit_ridge_probe if is_regression else fit_logreg_probe
    probe_kind = "ridge" if is_regression else "logreg"
    metric_keys = ("mae", "r2") if is_regression else ("balanced_acc", "acc")

    dapt_ckpt_path = getattr(args, "dapt_ckpt", None)
    encoder_ckpt = torch.load(dapt_ckpt_path, map_location=device) if dapt_ckpt_path else None
    init = "dapt" if encoder_ckpt is not None else "hcp_aging"
    models = build_pretrained_models(args, device, mask_ratio=0.0, encoder_ckpt=encoder_ckpt)
    # Encoders and Perceiver run once over all windows; every mode is pooled from the same latents.
    features = _extract_features(dataset, models, device, modes, roi_networks, subject_scope, progress)

    return {
        "args": args, "progress": progress, "pooling_all": multi or pooling == ALL_POOLING, "modes": modes,
        "dataset": dataset, "dataset_name": dataset_name, "subjs": subjs,
        "win_subj": np.asarray(dataset.subjs), "subject_scope": subject_scope, "protocol": protocol,
        "eval_mode": eval_mode, "n_folds": n_folds, "n_perm": n_perm, "seed0": seed0,
        "metric_keys": metric_keys, "fit_probe": fit_probe, "probe_kind": probe_kind,
        "init": init, "dapt_ckpt_path": dapt_ckpt_path, "n_subjs": n_subjs, "features": features,
    }


def run_logreg_probe(args, progress=None):
    """Run the frozen-feature logistic-regression probe for the given config.

    ``args.probe_eval``:
      * ``"split"`` (default): one subject-level train/val/test split.
      * ``"cv"``: subject-level K-fold CV (``subject_kfold``, same folds as fine-tuning
        CV). Inside each outer fold, C (and the PCA fit, if used) uses only the inner
        train split from ``split_fold_data``; the outer test fold is scored once.

    ``args.pooling``: one of ``feature``, ``roi``, ``network``, ``pca`` (default ``pca``), ``all``,
    or a list/tuple of those modes. ``all`` and lists extract latents once and fit every requested
    mode sequentially (in the given order); they return a list of (payload, saved_json_path), one
    per mode. A single mode string returns one (payload, saved_json_path).

    ``args.target_scope``: ``"sample"`` (default) classifies each window. ``"subject"`` averages
    each subject's pooled latents over its windows and classifies one row per subject; folds are
    subject-level regardless of ``split_protocol``.

    ``progress``: optional callable receiving event dicts (display only, no effect on
    results): ``{"stage": "extract", "total"}``, ``{"stage": "extract", "done", "total"}``,
    and ``{"stage": "split" | "fold", "mode", "fold"?, "done", "total"}`` after each C value.
    """
    ctx = _prepare_probe(args, progress)
    results = [_run_mode(ctx, mode) for mode in ctx["modes"]]
    return results if ctx["pooling_all"] else results[0]


def _run_mode(ctx, mode):
    """Fit and evaluate one pooling mode on its extracted features. Returns (payload, saved_path).

    Encoders are not run here; ``ctx["features"][mode]`` holds the rows for this mode.
    """
    args, progress = ctx["args"], ctx["progress"]
    dataset, dataset_name = ctx["dataset"], ctx["dataset_name"]
    subjs, win_subj = ctx["subjs"], ctx["win_subj"]
    subject_scope, protocol = ctx["subject_scope"], ctx["protocol"]
    eval_mode, n_folds, n_perm, seed0 = ctx["eval_mode"], ctx["n_folds"], ctx["n_perm"], ctx["seed0"]
    metric_keys, fit_probe, probe_kind = ctx["metric_keys"], ctx["fit_probe"], ctx["probe_kind"]
    init, dapt_ckpt_path, n_subjs = ctx["init"], ctx["dapt_ckpt_path"], ctx["n_subjs"]
    row_subj, X_all, y_all = ctx["features"][mode]
    is_regression = dataset.task == "regression"
    pooling = mode
    pca_components = PCA_COMPONENTS if pooling == "pca" else None

    def _rows(subset, y_src=None):
        """(X, y) rows of a window subset: its windows, or its subjects under subject scope.

        ``y_src`` (row-aligned labels) overrides ``y_all``; used by the label-permutation null.
        """
        y = y_all if y_src is None else y_src
        if not subject_scope:
            return X_all[subset.indices], y[subset.indices]
        return _rows_for_subjects(row_subj, X_all, y, np.unique(win_subj[list(subset.indices)]))

    def _n_subjects(subset):
        return int(len(np.unique(win_subj[list(subset.indices)])))

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
        "split_protocol": protocol,
        "target_scope": "subject" if subject_scope else "sample",
    }

    def _split_or_folds(y_src, emit):
        """Evaluate the fixed subject splits with row labels ``y_src``; features are reused."""
        if eval_mode == "split":
            trainset, valset, testset = split_train_val_test(
                dataset, test_frac=args.test_frac, val_frac=args.val_frac, seed=args.seed,
                protocol=protocol,
            )
            Xtr, ytr = _rows(trainset, y_src)
            Xva, yva = _rows(valset, y_src)
            Xte, yte = _rows(testset, y_src)
            result = fit_probe(
                Xtr, ytr, Xva, yva, Xte, yte,
                seed=args.seed, pca_components=pca_components,
                on_step=lambda done, total: emit(stage="split", done=done, total=total),
            )
            return {"subsets": (trainset, valset, testset), "result": result}
        folds = []
        for fold_idx, (trainset, testset) in enumerate(
            split_kfold(dataset, n_splits=n_folds, seed=args.seed, protocol=protocol)
        ):
            new_train, valset, testset = split_fold_data(dataset, trainset, testset, fold_idx, args)
            Xtr, ytr = _rows(new_train, y_src)
            Xva, yva = _rows(valset, y_src)
            Xte, yte = _rows(testset, y_src)
            res = fit_probe(
                Xtr, ytr, Xva, yva, Xte, yte,
                seed=args.seed, pca_components=pca_components,
                on_step=lambda done, total, f=fold_idx: emit(
                    stage="fold", fold=f, n_folds=n_folds, done=done, total=total),
            )
            folds.append({
                "fold": fold_idx,
                "n_train_subjects": _n_subjects(new_train),
                "n_val_subjects": _n_subjects(valset),
                "n_test_subjects": _n_subjects(testset),
                "subject_overlap": subject_overlap([new_train, valset, testset]),
                "result": res,
            })
        return folds

    if n_perm == 0:
        out = _split_or_folds(y_all, lambda **ev: _emit(progress, mode=mode, **ev))
        if eval_mode == "split":
            trainset, valset, testset = out["subsets"]
            meta["subject_overlap"] = subject_overlap([trainset, valset, testset])
            meta["n_subjects"] = {
                "train": _n_subjects(trainset),
                "val": _n_subjects(valset),
                "test": _n_subjects(testset),
                "total": int(len(np.unique(subjs))),
            }
            payload = {"meta": meta, "result": out["result"]}
            tag = f"{probe_kind}_{dataset_name}_{args.label}_{init}_{pooling}_split"
        else:
            meta["n_folds"] = n_folds
            meta["n_subjects"] = {"total": int(len(np.unique(subjs)))}
            payload = {
                "meta": meta,
                "summary": _summarize_folds(out, keys=metric_keys),
                "folds": out,
            }
            tag = f"{probe_kind}_{dataset_name}_{args.label}_{init}_{pooling}_cv{n_folds}"
    else:
        # Label-permutation null: features are label-independent, so they are reused. Each
        # permutation permutes subject labels, then re-runs the same splits. The real labels
        # are restored afterwards: all modes share this dataset object.
        runs = []
        saved_labels = dataset.labels_tensor
        saved_targets = dataset.targets if is_regression else None
        try:
            for i in range(n_perm):
                perm_seed = seed0 + i
                apply_label_permutation(dataset, perm_seed)
                y_perm = _permuted_row_labels(dataset, row_subj, subject_scope)
                # Progress is not emitted per permutation (it would overflow a single bar).
                out = _split_or_folds(y_perm, lambda **_ev: None)
                if eval_mode == "split":
                    runs.append({"permutation": i, "seed": perm_seed, "result": out["result"]})
                else:
                    runs.append({"permutation": i, "seed": perm_seed, "folds": out,
                                 "summary": _summarize_folds(out, keys=metric_keys)})
        finally:
            dataset.labels_tensor = saved_labels
            if is_regression:
                dataset.targets = saved_targets
        meta["label_permutation"] = {"n_permutations": n_perm, "seed0": seed0,
                                     "seeds": [seed0 + i for i in range(n_perm)]}
        meta["n_folds"] = n_folds if eval_mode == "cv" else None
        payload = {
            "meta": meta,
            "null_runs": runs,
            "null_summary": _summarize_null(runs, metric_keys, is_cv=eval_mode == "cv"),
        }
        split_part = "split" if eval_mode == "split" else f"cv{n_folds}"
        tag = f"{probe_kind}_{dataset_name}_{args.label}_{init}_{pooling}_{split_part}_null{n_perm}_s{seed0}"

    if protocol == "sample":
        tag += "_sample"
    if subject_scope:
        tag += "_subject"
    out_path = _save_probe_result(payload, tag)
    return payload, out_path

