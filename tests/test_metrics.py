"""Unit tests for OCD_symphony.model.metrics (regression metrics, subject MAE, re-exports)."""
import math

import numpy as np
import pytest

from OCD_symphony.model import metrics
from OCD_symphony.model import finetuning


def test_regression_metrics_perfect_prediction():
    m = metrics._regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert m["mae"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(1.0)
    assert m["r"] == pytest.approx(1.0)


def test_regression_metrics_known_values():
    t, p = [0.0, 2.0, 4.0], [1.0, 2.0, 3.0]
    m = metrics._regression_metrics(t, p)
    assert m["mae"] == pytest.approx(2.0 / 3.0)
    # ss_res = 1+0+1 = 2, ss_tot = 4+0+4 = 8
    assert m["r2"] == pytest.approx(1.0 - 2.0 / 8.0)


def test_regression_metrics_constant_target_is_nan_for_r2_and_r():
    m = metrics._regression_metrics([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])
    assert m["mae"] == pytest.approx(2.0 / 3.0)
    assert math.isnan(m["r2"])
    assert math.isnan(m["r"])


def test_regression_metrics_empty_is_nan():
    m = metrics._regression_metrics([], [])
    assert all(math.isnan(v) for v in m.values())


def test_subject_regression_mae_averages_windows_per_subject():
    # subject "a": windows predict 9 and 11 -> mean 10, target 12 -> error 2
    # subject "b": single window predicts 20, target 20 -> error 0
    preds = [9.0, 11.0, 20.0]
    targets = [12.0, 12.0, 20.0]
    subj = ["a", "a", "b"]
    mae, n = metrics.subject_regression_mae(preds, targets, subj)
    assert n == 2
    assert mae == pytest.approx((2.0 + 0.0) / 2.0)


def test_subject_regression_mae_empty_is_nan_zero_subjects():
    mae, n = metrics.subject_regression_mae([], [], [])
    assert math.isnan(mae)
    assert n == 0


def test_subject_regression_mae_is_not_window_weighted():
    # subject "a" has 3 windows, subject "b" has 1; averaging per subject, not per window.
    preds = [10.0, 10.0, 10.0, 0.0]
    targets = [10.0, 10.0, 10.0, 4.0]
    subj = ["a", "a", "a", "b"]
    mae, n = metrics.subject_regression_mae(preds, targets, subj)
    assert n == 2
    assert mae == pytest.approx((0.0 + 4.0) / 2.0)


def test_reexports_point_to_moved_objects():
    for name in ["_regression_metrics", "subject_regression_mae", "subject_accuracy",
                 "_EarlyStopper", "_subject_ids_and_labels", "_root_indices"]:
        assert getattr(finetuning, name) is getattr(metrics, name), name


def test_early_stopper_stops_after_patience_non_finite_counts():
    s = metrics._EarlyStopper(patience=2)
    assert s.update(1.0, 1) is True
    assert s.update(float("nan"), 2) is False
    assert not s.should_stop
    assert s.update(float("inf"), 3) is False
    assert s.should_stop
    assert s.best_epoch == 1
    assert s.best_loss == pytest.approx(1.0)
