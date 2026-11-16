"""Unit tests for subject-level reconstructed-vs-input FC (no checkpoint, no data needed).

Run with:  conda run -n ocd-symphony python -m pytest tests/test_recon_fc.py -q
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from OCD_symphony.reconstruction.fc_metrics import (
    EdgeAcrossSubjectsAccumulator,
    FCAccumulator,
    across_subject_edge_r,
    aggregate,
    edge_across_subjects_summary,
    edge_stats,
    network_blocks,
    network_membership,
    rank_rois,
    roi_profile_r,
    subject_fc_metrics,
    summarise_fc,
    upper_vector,
    window_fc,
)


R = 12
NETS = ["A"] * 4 + ["B"] * 5 + ["C"] * 3


def _random_fc(seed, r=R):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((r, 60))
    return np.corrcoef(x)


def test_window_fc_matches_corrcoef_and_handles_constant_rows():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 5, 40))
    x[1, 2] = 7.0  # constant ROI -> NaN row/column
    fc = window_fc(x)
    assert fc.shape == (3, 5, 5)
    np.testing.assert_allclose(fc[0], np.corrcoef(x[0]), atol=1e-10)
    assert np.isnan(fc[1, 2, :]).all() and np.isnan(fc[1, :, 2]).all()


def test_fisher_mean_matches_manual_atanh_average():
    windows = np.stack([_random_fc(s) for s in range(4)])
    acc = FCAccumulator(R)
    acc.add(windows[:2])
    acc.add(windows[2:])
    expected = np.tanh(np.arctanh(np.clip(windows, -0.999999, 0.999999)).mean(axis=0))
    np.testing.assert_allclose(acc.mean(), expected, atol=1e-10)
    assert acc.n_windows == 4


def test_identity_reconstruction_gives_perfect_agreement():
    S = _random_fc(1)
    membership, _ = network_membership(NETS)
    m = subject_fc_metrics(S, S.copy(), membership)
    assert m["edge"]["r"] == pytest.approx(1.0)
    assert m["edge"]["mae"] == pytest.approx(0.0)
    assert m["edge"]["bias"] == pytest.approx(0.0)
    np.testing.assert_allclose(m["profile"], 1.0, atol=1e-10)
    np.testing.assert_allclose(m["net_rec"], m["net_in"])


def test_diagonal_is_ignored_in_edges_profiles_and_network_blocks():
    S_in = _random_fc(2)
    S_rec = _random_fc(3)
    S_rec_diag = S_rec.copy()
    np.fill_diagonal(S_rec_diag, 5.0)  # corrupt only the diagonal
    membership, _ = network_membership(NETS)
    a = subject_fc_metrics(S_in, S_rec, membership)
    b = subject_fc_metrics(S_in, S_rec_diag, membership)
    assert a["edge"] == b["edge"]
    np.testing.assert_allclose(a["profile"], b["profile"])
    np.testing.assert_allclose(a["net_rec"], b["net_rec"])


def test_network_blocks_match_bruteforce_loop():
    S = _random_fc(4)
    membership, names = network_membership(NETS)
    got = network_blocks(S, membership)
    labels = np.asarray(NETS)
    expected = np.zeros((len(names), len(names)))
    for a, na in enumerate(names):
        for b, nb in enumerate(names):
            vals = [S[i, j] for i in range(R) for j in range(R)
                    if i != j and labels[i] == na and labels[j] == nb]
            expected[a, b] = np.mean(vals)
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_roi_profile_excludes_self_and_is_nan_for_constant_profile():
    S_in = _random_fc(5)
    S_rec = S_in.copy()
    S_rec[3, :] = 0.2  # constant off-diagonal row -> undefined profile for ROI 3
    S_rec[:, 3] = 0.2
    S_rec[3, 3] = 1.0
    p = roi_profile_r(S_in, S_rec)
    assert np.isnan(p[3])
    assert np.isfinite(np.delete(p, 3)).all()


def test_edge_stats_ignore_nan_edges():
    S_in = _random_fc(6)
    S_rec = S_in.copy()
    S_rec[0, 1] = S_rec[1, 0] = np.nan
    st = edge_stats(S_in, S_rec)
    assert st["n_edges"] == R * (R - 1) // 2 - 1
    assert st["mae"] == pytest.approx(0.0)


def test_rank_rois_best_worst_with_nan_and_ties():
    profile = np.array([0.5, np.nan, 0.9, 0.1, 0.9, -0.2])
    best, worst = rank_rois(profile, k=2)
    assert list(best) == [2, 4]        # tie at 0.9 broken by index
    assert list(worst) == [5, 3]
    assert 1 not in best and 1 not in worst


def test_aggregate_handles_nan_and_empty():
    agg = aggregate([1.0, 2.0, np.nan, 3.0])
    assert agg["n"] == 3 and agg["mean"] == pytest.approx(2.0)
    assert agg["median"] == pytest.approx(2.0)
    empty = aggregate([np.nan])
    assert empty["n"] == 0 and np.isnan(empty["mean"])


def test_summarise_fc_shapes_and_diff_sign():
    membership, names = network_membership(NETS)
    Ss = [_random_fc(s) for s in range(5)]
    Rs = [_random_fc(s + 100) for s in range(5)]
    metrics = [subject_fc_metrics(a, b, membership) for a, b in zip(Ss, Rs)]
    out = summarise_fc(
        [f"s{i}" for i in range(5)],
        [m["edge"] for m in metrics],
        np.stack([m["profile"] for m in metrics]),
        np.stack([m["net_in"] for m in metrics]),
        np.stack([m["net_rec"] for m in metrics]),
        names, k=3,
    )
    assert out["roi_profile"].shape == (5, R)
    assert out["net_diff"].shape == (len(names), len(names))
    np.testing.assert_allclose(out["net_diff"], out["net_rec"] - out["net_in"], atol=1e-12)
    assert out["edge_summary"]["r"]["n"] == 5
    assert len(out["roi_best"]) == 3 and len(out["roi_worst"]) == 3


def test_plot_functions_return_figures():
    import matplotlib.pyplot as plt

    from OCD_symphony.reconstruction.fc_plots import plot_fc_matrices, plot_fc_summary

    membership, names = network_membership(NETS)
    S_in, S_rec = _random_fc(7), _random_fc(8)
    node_names = [f"roi{i}" for i in range(R)]
    m = subject_fc_metrics(S_in, S_rec, membership)
    fc = summarise_fc(["s0"], [m["edge"]], m["profile"][None], m["net_in"][None],
                      m["net_rec"][None], names, k=2)
    acc = EdgeAcrossSubjectsAccumulator(upper_vector(S_in).size)
    for s in range(4):
        acc.add(upper_vector(_random_fc(20 + s)), upper_vector(_random_fc(40 + s)))
    fc["edge_across_subjects"] = edge_across_subjects_summary(acc)
    fig1 = plot_fc_matrices(S_in, S_rec, NETS)
    fig2 = plot_fc_summary(fc, NETS, node_names)
    assert fig1 is not None and fig2 is not None
    plt.close("all")


# ── Subject-level (specificity) measure: per-edge r across subjects ──────────

def _subject_stack(n_subj, n_edges, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_subj, n_edges))


def test_across_subject_r_identity_is_one():
    A = _subject_stack(10, 30, 0)
    r = across_subject_edge_r(A, A.copy())
    np.testing.assert_allclose(r, 1.0, atol=1e-10)


def test_across_subject_r_removes_constant_offset():
    # A constant per-edge offset is a group-level error; it must not reduce subject-level r.
    A = _subject_stack(10, 30, 1)
    r = across_subject_edge_r(A, A + 0.18)
    np.testing.assert_allclose(r, 1.0, atol=1e-10)


def test_across_subject_r_low_when_only_group_mean_is_reproduced():
    # Reconstruction = group mean FC for every subject: no individual differences tracked.
    # Group-level r (across edges) stays high; subject-level r is undefined (zero variance).
    rng = np.random.default_rng(2)
    group_mean = rng.standard_normal(40) * 0.5
    A = group_mean + 0.05 * rng.standard_normal((12, 40))  # subject deviations
    B = np.tile(group_mean, (12, 1))
    r_subject = across_subject_edge_r(A, B)
    assert np.isnan(r_subject).all()
    # Group-level agreement on the same vectors is high.
    assert float(np.corrcoef(A.mean(axis=0), B.mean(axis=0))[0, 1]) > 0.99


def test_streaming_accumulator_matches_reference():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((15, 50)) + 10.0  # large offset exercises the shift
    B = 0.6 * A + 0.8 * rng.standard_normal((15, 50)) - 3.0
    acc = EdgeAcrossSubjectsAccumulator(50)
    for a, b in zip(A, B):
        acc.add(a, b)
    assert acc.n_subjects == 15
    np.testing.assert_allclose(acc.r_per_edge(), across_subject_edge_r(A, B), atol=1e-9)


def test_accumulator_ignores_nan_and_needs_three_subjects():
    A = _subject_stack(4, 3, 4)
    B = A.copy()
    A[:, 0] = np.nan          # edge 0 never valid
    A[2:, 1] = np.nan        # edge 1 has only 2 valid subjects
    acc = EdgeAcrossSubjectsAccumulator(3)
    for a, b in zip(A, B):
        acc.add(a, b)
    r = acc.r_per_edge()
    assert np.isnan(r[0]) and np.isnan(r[1])
    assert r[2] == pytest.approx(1.0)


def test_accumulator_rejects_wrong_size():
    acc = EdgeAcrossSubjectsAccumulator(5)
    with pytest.raises(ValueError):
        acc.add(np.zeros(4), np.zeros(5))


def test_edge_across_subjects_summary_reports_counts():
    A = _subject_stack(8, 20, 5)
    acc = EdgeAcrossSubjectsAccumulator(20)
    for a in A:
        acc.add(a, a)
    out = edge_across_subjects_summary(acc)
    assert out["n_subjects"] == 8 and out["n_edges"] == 20 and out["n_valid_edges"] == 20
    assert out["stats"]["median"] == pytest.approx(1.0)
    assert out["r_per_edge"].dtype == np.float32

