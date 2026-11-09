"""Unit tests for the DAPT reconstruction metrics and trace plots (no checkpoint needed).

Run with:  conda run -n ocd-symphony python -m pytest tests/test_recon_visualization.py -q
"""
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from sklearn.metrics import r2_score

from OCD_symphony.data_validation.validate_preprocessing import (
    compute_r2,
    compute_roi_correlation,
    load_RSNs,
    plot_recon_barplot,
    plot_recon_histogram,
    plot_recon_summary,
    plot_recon_timeseries,
    select_rois_for_display,
)


def _make_pair(N=6, R=450, T=200, noise=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    orig = torch.randn(N, R, T, generator=g)
    recon = orig + noise * torch.randn(N, R, T, generator=g)
    return orig, recon


def test_compute_r2_matches_flattened_sklearn():
    orig, recon = _make_pair(N=3, R=5, T=20)
    r2 = compute_r2(orig, recon)
    assert r2.shape == (5,)
    for i in range(5):
        expected = r2_score(orig[:, i, :].numpy().ravel(), recon[:, i, :].numpy().ravel())
        assert r2[i] == pytest.approx(expected)


def test_compute_r2_perfect_reconstruction_is_one():
    orig, _ = _make_pair()
    assert np.allclose(compute_r2(orig, orig), 1.0)


def test_compute_r2_accepts_grad_tensors_and_arrays():
    orig, recon = _make_pair(N=2, R=4, T=10)
    out_tensor = compute_r2(orig, recon.requires_grad_())
    out_array = compute_r2(orig.numpy(), recon.detach().numpy())
    assert np.allclose(out_tensor, out_array)


def test_roi_correlation_is_pearson_and_nan_for_constant_roi():
    orig, recon = _make_pair(N=2, R=3, T=50)
    recon[0, 1, :] = 1.0  # constant ROI -> undefined correlation
    c = compute_roi_correlation(orig, recon)
    assert c.shape == (2, 3)
    assert c[0, 0] == pytest.approx(np.corrcoef(orig[0, 0], recon[0, 0])[0, 1], abs=1e-6)
    assert np.isnan(c[0, 1])


def test_select_rois_strategies_return_expected_sizes():
    rng = np.random.default_rng(0)
    r2 = rng.uniform(0, 1, size=450)
    assert len(select_rois_for_display(r2, 10, "best")) == 10
    assert len(select_rois_for_display(r2, 10, "worst")) == 10
    assert len(select_rois_for_display(r2, 10, "median")) == 10
    best = select_rois_for_display(r2, 5, "best")
    worst = select_rois_for_display(r2, 5, "worst")
    assert r2[best].min() > r2[worst].max()


def test_network_selection_gives_one_roi_per_network():
    rs_nets, names = load_RSNs()
    idx = select_rois_for_display(np.zeros(450), n_rois=50, strategy="network")
    chosen = list(np.asarray(rs_nets)[idx])
    assert len(chosen) == len(set(chosen)) == len(names)


def test_load_rsns_labels_have_no_quotes():
    rs_nets, rs = load_RSNs()
    assert len(rs_nets) == 450
    assert all("'" not in str(n) for n in rs_nets)
    assert all("'" not in n for n in rs)


def test_plot_timeseries_returns_figure_with_requested_rois():
    orig, recon = _make_pair(N=2, R=450, T=200)
    fig = plot_recon_timeseries(orig, recon, sample_idx=1, roi_indices=[0, 10, 200], t_window=(0, 50))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_histogram_and_barplot_return_figures():
    orig, recon = _make_pair(N=4, R=450, T=200)
    subjs = np.array(["a", "a", "b", "c"])
    fig = plot_recon_histogram(orig, recon, subjs)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    rs_nets, _ = load_RSNs()
    fig2 = plot_recon_barplot(compute_r2(orig, recon), rs_nets.values)
    assert isinstance(fig2, plt.Figure)
    plt.close(fig2)


def test_plot_recon_summary_returns_compact_figure():
    orig, recon = _make_pair(N=4, R=450, T=200)
    subjs = np.array(["a", "a", "b", "c"])
    fig = plot_recon_summary(
        compute_r2(orig, recon), compute_roi_correlation(orig, recon), subjs
    )
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 5
    assert fig.get_size_inches()[0] <= 10
    plt.close(fig)


def test_streaming_r2_matches_compute_r2():
    from OCD_symphony.reconstruction.dapt_reconstruction import _add_r2_sums, _r2_from_sums

    orig, recon = _make_pair(N=5, R=7, T=30)
    o, r = orig.numpy(), recon.numpy()
    sums = None
    for i in range(0, 5, 2):  # batches of samples, as produced per subject
        sums = _add_r2_sums(sums, o[i:i + 2], r[i:i + 2])
    assert np.allclose(_r2_from_sums(sums), compute_r2(orig, recon), atol=1e-10)


def test_select_representative_subjects_best_worst_and_median():
    from OCD_symphony.reconstruction.dapt_reconstruction import select_representative_subjects

    scores = {f"s{i:02d}": i / 100 for i in range(20)}
    scores["s99"] = float("nan")  # ignored
    sel = select_representative_subjects(scores, k=6)
    assert len(sel) == len(set(sel)) == 6
    assert sel[0] == "s19" and sel[-1] == "s00"  # best first, worst last
    assert "s99" not in sel
    middle = sel[1:-1]
    assert all(abs(scores[s] - 0.095) <= 0.06 for s in middle)  # near median 0.095
    # fewer valid subjects than k -> all returned, best first
    small = select_representative_subjects({"a": 0.1, "b": 0.5, "c": float("nan")}, k=6)
    assert small == ["b", "a"]


def test_baseline_models_load_original_pretrained_weights():
    import os
    from OCD_symphony.reconstruction.dapt_reconstruction import get_dapt_models
    from OCD_symphony.utils import proj_dir

    files = [
        "temporal_transformer_pretrained_nL6_d128_h4_mlp512.pth",
        "spatial_transformer_pretrained_learnablePE_nL6_d128_h4_mlp512.pth",
        "masked_conv1d_temporal_nL6_d128_h4_mlp512.pth",
        "perceiver_fused_pretrained_nL6_d128_h4_mlp512.pth",
    ]
    if not all(os.path.isfile(os.path.join(proj_dir, "models", f)) for f in files):
        pytest.skip("baseline weight files not present")
    models = get_dapt_models(None, torch.device("cpu"))
    assert len(models) == 4
    assert all(not m.training for m in models)


def test_reconstruct_subject_stats_accepts_baseline_none():
    from OCD_symphony.reconstruction import dapt_reconstruction as d

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(d, "get_dapt_models", lambda ckpt, device: (_ for _ in ()).throw(RuntimeError("sentinel")))
        with pytest.raises(RuntimeError, match="sentinel"):
            d.reconstruct_subject_stats(None, ["100001"], np.ones(450), np.zeros(450), device=torch.device("cpu"))


def test_reconstruct_subject_stats_missing_checkpoint_raises():
    from OCD_symphony.reconstruction.dapt_reconstruction import reconstruct_subject_stats

    with pytest.raises(FileNotFoundError):
        reconstruct_subject_stats("/nonexistent/ckpt.pth", ["100001"], np.ones(450), np.zeros(450))


def test_subject_subset_is_nested_and_reproducible():
    from OCD_symphony.reconstruction.dapt_reconstruction import select_test_subject_subset

    subjs = [f"{100000 + i}" for i in range(72)]
    fracs = [0.1, 0.2, 0.5, 1.0]
    picks = {f: select_test_subject_subset(subjs, frac=f, seed=0) for f in fracs}
    assert len(picks[0.1]) == 7 and len(picks[1.0]) == 72
    for small, large in zip(fracs, fracs[1:]):
        assert set(picks[small]) <= set(picks[large])
    assert select_test_subject_subset(subjs, frac=0.2, seed=0) == picks[0.2]


def test_estimate_is_positive_for_both_devices():
    from OCD_symphony.reconstruction.dapt_reconstruction import estimate_seconds_per_subject

    assert estimate_seconds_per_subject(torch.device("cuda")) > 0
    assert estimate_seconds_per_subject(torch.device("cpu")) > estimate_seconds_per_subject(torch.device("cuda"))
