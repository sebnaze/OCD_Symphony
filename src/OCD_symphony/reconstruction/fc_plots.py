"""Figures for the reconstructed-vs-input subject FC comparison (see ``fc_metrics``)."""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from OCD_symphony.reconstruction.fc_metrics import network_membership


def _network_order(rs_nets):
    """ROI permutation that groups ROIs by network (order of first appearance)."""
    labels = np.asarray(rs_nets).astype(str)
    names = list(dict.fromkeys(labels.tolist()))
    order = np.concatenate([np.where(labels == n)[0] for n in names])
    return order, names


def _network_colors(names):
    cmap = plt.get_cmap("tab20")
    return {n: cmap(i % 20) for i, n in enumerate(names)}


def plot_fc_matrices(S_in, S_rec, rs_nets, figsize=(11, 3.8)):
    """Input FC, reconstructed FC and their difference, ROIs ordered by network.

    The diagonal is masked. Returns the Figure.
    """
    order, _ = _network_order(rs_nets)
    S_in = np.asarray(S_in, dtype=np.float64)[np.ix_(order, order)]
    S_rec = np.asarray(S_rec, dtype=np.float64)[np.ix_(order, order)]
    panels = [("Input FC", S_in, -1.0, 1.0),
              ("Reconstructed FC", S_rec, -1.0, 1.0)]
    diff = S_rec - S_in
    finite = np.abs(diff[np.isfinite(diff)])
    lim = float(finite.max()) if finite.size else 1.0
    panels.append(("Difference (rec - input)", diff, -lim, lim))

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (title, M, vmin, vmax) in zip(axes, panels):
        X = np.array(M, dtype=np.float64, copy=True)
        np.fill_diagonal(X, np.nan)
        im = ax.imshow(np.ma.masked_invalid(X), cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout()
    return fig


def plot_fc_summary(fc, rs_nets, node_names, figsize=(10, 7.5)):
    """Compact FC summary with group-level and subject-level reconstruction side by side.

    Top row:
      (a) GROUP-LEVEL: per-subject edge r across edges (captures the shared mean FC pattern),
          with the group bias and MAE.
      (b) SUBJECT-LEVEL (specificity): per-edge r across subjects after removing each edge's
          group mean (captures individual differences).
    Bottom row: best/worst ROIs and network-block difference.

    ``fc`` is the dict returned by ``summarise_fc`` (``reconstruction.fc_metrics``) with the
    ``edge_across_subjects`` entry added by ``reconstruct_subject_stats``.
    Returns the Figure.
    """
    node_names = np.asarray(node_names)
    rs_nets = np.asarray(rs_nets).astype(str)
    _, net_names = network_membership(rs_nets)
    colors = _network_colors(net_names)
    edge = fc["edge"]
    e_sum = fc["edge_summary"]
    across = fc.get("edge_across_subjects")

    with plt.rc_context({"font.size": 8}):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.15])

        ax = fig.add_subplot(gs[0, 0])
        r = edge["r"][np.isfinite(edge["r"])]
        if r.size:
            ax.hist(r, bins=20, color="C0", edgecolor="white")
        ax.set_xlabel("Per-subject edge r (reconstructed vs input, across edges)")
        ax.set_ylabel("subjects")
        ax.set_title("(a) GROUP-LEVEL reconstruction\n"
                     "shared mean FC pattern across edges", fontsize=8, loc="left")
        ax.text(0.98, 0.97,
                f"median r {e_sum['r']['median']:.3f}\n"
                f"bias {e_sum['bias']['mean']:+.3f}\n"
                f"MAE {e_sum['mae']['mean']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))

        ax = fig.add_subplot(gs[0, 1])
        if across is not None and across["stats"]["n"] > 0:
            r_edge = np.asarray(across["r_per_edge"], dtype=np.float64)
            r_edge = r_edge[np.isfinite(r_edge)]
            ax.hist(r_edge, bins=40, color="C2", edgecolor="white")
            ax.axvline(across["stats"]["median"], color="k", lw=0.9, ls="--")
            ax.text(0.02, 0.97,
                    f"median {across['stats']['median']:.3f}\n"
                    f"mean {across['stats']['mean']:.3f}\n"
                    f"valid edges {across['n_valid_edges']}/{across['n_edges']}\n"
                    f"subjects {across['n_subjects']}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        else:
            ax.text(0.5, 0.5, "not available", transform=ax.transAxes,
                    ha="center", va="center")
        ax.set_xlabel("Per-edge r across subjects (group mean removed)")
        ax.set_ylabel("edges")
        ax.set_title("(b) SUBJECT-LEVEL reconstruction (specificity)\n"
                     "tracks individual differences between subjects", fontsize=8, loc="left")

        ax = fig.add_subplot(gs[1, 0])
        best, worst = fc["roi_best"], fc["roi_worst"]
        idx = list(best) + list(worst)
        vals = np.asarray(fc["roi_profile_mean"])[idx]
        labels = [f"{node_names[i]} ({rs_nets[i]})" for i in idx]
        bar_colors = [colors[rs_nets[i]] for i in idx]
        y = np.arange(len(idx))[::-1]
        ax.barh(y, vals, color=bar_colors)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6)
        ax.axhline(len(best) - 0.5, color="k", lw=0.8, ls="--")
        ax.set_xlabel("FC-profile r (mean over subjects)")
        ax.set_title(f"(c) Best {len(best)} (top) / worst {len(worst)} (bottom) ROIs",
                     fontsize=8, loc="left")

        ax = fig.add_subplot(gs[1, 1])
        D = np.asarray(fc["net_diff"], dtype=np.float64)
        lim = float(np.nanmax(np.abs(D))) if np.isfinite(D).any() else 1.0
        im = ax.imshow(np.ma.masked_invalid(D), cmap="RdBu_r", vmin=-lim, vmax=lim)
        K = D.shape[0]
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(fc["net_names"], rotation=60, ha="right", fontsize=6)
        ax.set_yticklabels(fc["net_names"], fontsize=6)
        if K <= 12:
            for i in range(K):
                for j in range(K):
                    if np.isfinite(D[i, j]):
                        ax.text(j, i, f"{D[i, j]:+.2f}", ha="center", va="center", fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title("(d) Network FC difference (rec - input)", fontsize=8, loc="left")

        for a in fig.axes:
            a.spines["top"].set_visible(False)
            a.spines["right"].set_visible(False)
        fig.tight_layout()
    return fig
