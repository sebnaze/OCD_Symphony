"""Unit tests for early stopping (_EarlyStopper) and the logistic-regression probe.

No dataset or checkpoint needed.

Run with:  conda run -n ocd-symphony python -m pytest tests/test_early_stopping_and_probe.py -q
"""
import numpy as np
import torch
from types import SimpleNamespace
import pytest

from OCD_symphony.model.finetuning import _EarlyStopper
from OCD_symphony.model.linear_probe import fit_logreg_probe


# ── Early stopper ───────────────────────────────────────────────────────────
def test_stopper_tracks_best_and_resets_counter():
    s = _EarlyStopper(patience=2)
    assert s.update(1.0, 1) is True
    assert s.update(0.8, 2) is True       # new best resets counter
    assert s.update(0.9, 3) is False
    assert not s.should_stop              # 1 epoch without improvement, patience 2
    assert s.update(0.85, 4) is False
    assert s.should_stop                  # 2 epochs without improvement
    assert s.best_epoch == 2
    assert s.best_loss == pytest.approx(0.8)


def test_stopper_patience_zero_never_stops():
    s = _EarlyStopper(patience=0)
    s.update(0.5, 1)
    for e in range(2, 50):
        s.update(1.0, e)
    assert not s.should_stop
    assert s.best_epoch == 1


def test_stopper_non_finite_loss_is_not_an_improvement():
    s = _EarlyStopper(patience=2)
    assert s.update(float("nan"), 1) is False
    assert s.best_epoch is None
    assert s.update(0.7, 2) is True
    assert s.update(float("nan"), 3) is False
    assert s.update(float("inf"), 4) is False
    assert s.should_stop


# ── Logistic-regression probe ───────────────────────────────────────────────
def _split(X, y, rng, n_tr=200, n_va=60):
    idx = rng.permutation(len(y))
    tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def test_probe_recovers_separable_signal():
    rng = np.random.default_rng(0)
    n, d = 400, 16
    y = rng.integers(0, 2, n)
    X = rng.normal(size=(n, d)) + 4.0 * (y[:, None] - 0.5) * np.eye(d)[0]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _split(X, y, rng)

    res = fit_logreg_probe(Xtr, ytr, Xva, yva, Xte, yte)
    assert res["test"]["balanced_acc"] > 0.85
    assert res["majority_baseline"]["majority_class"] in (0, 1)
    assert len(res["c_grid"]) == 6
    assert res["best_C"] in [g["C"] for g in res["c_grid"]]


def test_probe_random_labels_near_chance():
    rng = np.random.default_rng(1)
    n, d = 400, 16
    X = rng.normal(size=(n, d))
    y = rng.integers(0, 2, n)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _split(X, y, rng)

    res = fit_logreg_probe(Xtr, ytr, Xva, yva, Xte, yte)
    assert 0.35 < res["test"]["balanced_acc"] < 0.65


def test_probe_metrics_fields_present():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(120, 4))
    y = rng.integers(0, 2, 120)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = _split(X, y, rng, n_tr=60, n_va=30)
    res = fit_logreg_probe(Xtr, ytr, Xva, yva, Xte, yte, c_grid=(1.0,))
    for split in ("train", "val", "test"):
        for key in ("n", "acc", "balanced_acc", "macro_f1", "pred_counts", "true_counts"):
            assert key in res[split]
    assert res["test"]["n"] == len(yte)
    assert sum(res["test"]["pred_counts"]) == len(yte)


# ── Subject-level CV folds used by the probe (no encoders needed) ───────────
class _StubLabelEncoder:
    classes_ = np.array(["0", "1"])


class _StubDataset:
    """Minimal stand-in for FMRIDataset: only the attributes the splitters read."""

    def __init__(self, n_subjects=40, windows_per_subject=5, seed=0):
        rng = np.random.default_rng(seed)
        self.subjs = np.repeat(np.arange(n_subjects), windows_per_subject)
        subj_labels = rng.integers(0, 2, n_subjects)
        self.encoded_labels = torch.from_numpy(np.repeat(subj_labels, windows_per_subject)).long()
        self.label_encoder = _StubLabelEncoder()
        self.label = "gender"

    def __len__(self):
        return len(self.subjs)


def test_subject_kfold_is_subject_disjoint_and_covers_all_subjects():
    from types import SimpleNamespace
    from OCD_symphony.data_preparation.make_hcp_dataset import subject_kfold
    from OCD_symphony.model.finetuning import split_fold_data

    ds = _StubDataset()
    folds = list(subject_kfold(ds, n_splits=5, seed=42))
    assert len(folds) == 5

    tested = []
    args = SimpleNamespace(val_frac=0.15, seed=42)
    for fold_idx, (trainset, testset) in enumerate(folds):
        train_subj = set(ds.subjs[list(trainset.indices)])
        test_subj = set(ds.subjs[list(testset.indices)])
        assert train_subj.isdisjoint(test_subj)
        tested.extend(test_subj)

        new_train, valset, testset2 = split_fold_data(ds, trainset, testset, fold_idx, args)
        new_train_subj = set(ds.subjs[list(new_train.indices)])
        val_subj = set(ds.subjs[list(valset.indices)])
        assert new_train_subj.isdisjoint(val_subj)
        assert new_train_subj.isdisjoint(test_subj)
        assert val_subj.isdisjoint(test_subj)

    # Every subject is held out as test exactly once across folds.
    assert sorted(tested) == sorted(set(ds.subjs.tolist()))


def test_probe_eval_default_is_split():
    from types import SimpleNamespace
    import OCD_symphony.model.linear_probe as lp

    # Unsupported modes are rejected before any data is loaded.
    with pytest.raises(ValueError):
        lp.run_logreg_probe(SimpleNamespace(probe_eval="bogus", label="gender"))


# ── Pooling modes for the probe ──────────────────────────────────────────────
def _fake_networks():
    # 450 ROIs in 4 uneven groups, ordered by first appearance.
    names = ["sub_cort"] * 50 + ["Vis"] * 150 + ["Default"] * 200 + ["Cont"] * 50
    return np.array(names)


def test_pool_latents_shapes_for_all_modes():
    from OCD_symphony.model.linear_probe import pool_latents

    lat = torch.randn(3, 450, 128)
    nets = _fake_networks()
    assert pool_latents(lat, "feature").shape == (3, 128)
    assert pool_latents(lat, "roi").shape == (3, 450)
    assert pool_latents(lat, "network", nets).shape == (3, 4 * 128)
    assert pool_latents(lat, "pca").shape == (3, 450 * 128)


def test_network_pooling_is_mean_over_each_networks_rois():
    from OCD_symphony.model.linear_probe import pool_latents

    lat = torch.randn(2, 450, 128)
    nets = _fake_networks()
    out = pool_latents(lat, "network", nets).reshape(2, 4, 128)
    # Networks are ordered by first appearance: sub_cort, Vis, Default, Cont.
    vis_mask = torch.as_tensor(nets == "Vis")
    assert torch.allclose(out[:, 1, :], lat[:, vis_mask, :].mean(dim=1), atol=1e-6)


def test_roi_pooling_is_mean_over_features():
    from OCD_symphony.model.linear_probe import pool_latents

    lat = torch.randn(2, 450, 128)
    assert torch.allclose(pool_latents(lat, "roi"), lat.mean(dim=2), atol=1e-6)


def test_network_pooling_requires_one_label_per_roi():
    from OCD_symphony.model.linear_probe import pool_latents

    with pytest.raises(ValueError):
        pool_latents(torch.randn(1, 450, 128), "network", np.array(["Vis"] * 10))


def test_fit_with_pca_clamps_components_and_runs():
    from OCD_symphony.model.linear_probe import fit_logreg_probe

    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 300)).astype(np.float32)
    y = rng.integers(0, 2, 120)
    res = fit_logreg_probe(X[:60], y[:60], X[60:90], y[60:90], X[90:], y[90:],
                           c_grid=(1.0,), pca_components=256)
    # n_train - 1 = 59 is the binding limit.
    assert res["pca_components_effective"] == 59
    assert res["n_features_model"] == 59
    assert res["n_features_input"] == 300
    # Without PCA, features pass through unchanged.
    res2 = fit_logreg_probe(X[:60], y[:60], X[60:90], y[60:90], X[90:], y[90:], c_grid=(1.0,))
    assert res2["pca_components_effective"] is None
    assert res2["n_features_model"] == 300


def test_network_labels_have_450_rois_and_8_networks():
    from OCD_symphony.model.linear_probe import load_network_labels

    labels = load_network_labels()
    assert labels.shape == (450,)
    assert len(set(labels.tolist())) == 8


# ── Progress callback (display only) ─────────────────────────────────────────
def test_fit_on_step_fires_once_per_c_and_does_not_change_results():
    from OCD_symphony.model.linear_probe import fit_logreg_probe, C_GRID

    rng = np.random.default_rng(5)
    X = rng.normal(size=(150, 12)).astype(np.float32)
    y = rng.integers(0, 2, 150)
    args = (X[:60], y[:60], X[60:100], y[60:100], X[100:], y[100:])

    events = []
    with_cb = fit_logreg_probe(*args, on_step=lambda d, t: events.append((d, t)))
    without_cb = fit_logreg_probe(*args)

    assert events == [(i, len(C_GRID)) for i in range(1, len(C_GRID) + 1)]
    assert with_cb["best_C"] == without_cb["best_C"]
    assert with_cb["test"] == without_cb["test"]


def test_emit_is_noop_without_callback_and_forwards_with_one():
    from OCD_symphony.model.linear_probe import _emit

    _emit(None, stage="extract", total=3)  # must not raise
    got = []
    _emit(got.append, stage="extract", done=1, total=3)
    assert got == [{"stage": "extract", "done": 1, "total": 3}]
