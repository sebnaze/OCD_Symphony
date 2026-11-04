import matplotlib
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
import seaborn as sbn
import scipy
from sklearn import metrics
import torch

import OCD_symphony
from OCD_symphony.utils import proj_dir, today, get_BrainSymphony_atlas_labels

#plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif", "font.sans-serif": "Helvetica","font.size":10})

def _to_numpy(x) -> np.ndarray:
    """Detach a torch tensor (any device / grad state) or pass arrays through."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def load_RSNs() -> tuple:
    """ Load resting-state networks for each node corresponding to the Schaefer400_TianS3 parcellation.

    Returns the per-ROI network labels (quotes stripped) and the unique network names in order of appearance.
    """
    rs_nets = pd.read_csv(os.path.join(proj_dir, 'parcellations', 'networks.txt'), header=None, sep=',')
    rs_nets = rs_nets[0].astype(str).str.strip().str.strip("'\"")
    rs = list(rs_nets.unique())
    return rs_nets, rs


def compute_r2(orig_data, recon_data) -> np.ndarray:
    """ Returns the ROI-wise R² between original and reconstructed data.

    For each ROI, R² is computed over *all* subjects and time points, i.e. the
    fraction of variance of that ROI's BOLD signal explained by the reconstruction
    (``1 - SS_res / SS_tot``). Inputs are (N, R, T) tensors or arrays.
    """
    orig = _to_numpy(orig_data)
    recon = _to_numpy(recon_data)
    n_rois = orig.shape[1]
    r2 = np.zeros((n_rois,))
    for i in range(n_rois):
        r2[i] = metrics.r2_score(orig[:, i, :].ravel(), recon[:, i, :].ravel())
    return r2


def compute_roi_correlation(orig_data, recon_data) -> np.ndarray:
    """ Per-sample, per-ROI Pearson correlation along time, vectorised. Returns (N, R). """
    orig = _to_numpy(orig_data).astype(np.float64)
    recon = _to_numpy(recon_data).astype(np.float64)
    o = orig - orig.mean(axis=-1, keepdims=True)
    r = recon - recon.mean(axis=-1, keepdims=True)
    denom = np.sqrt((o ** 2).sum(-1) * (r ** 2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 0, (o * r).sum(-1) / denom, np.nan)


def plot_recon_barplot(r2:np.ndarray, rs_nets:np.ndarray):
    """ Bar plot of mean ROI-wise R² per resting-state network. Returns the Figure. """
    df_ = pd.DataFrame({'r2': np.asarray(r2), 'network': np.asarray(rs_nets)})

    fig, ax = plt.subplots(figsize=(4, 3))
    sbn.barplot(data=df_, x='network', y='r2', ax=ax, errorbar='sd')

    ax.tick_params(axis='x', rotation=35)
    ax.set_xlabel("RSN")
    ax.set_ylabel("$R^2$")
    fig.tight_layout()
    return fig


def plot_recon_histogram(original_data, recon_data, subjs):
    """ Distribution of ROI-wise R² (overall), and per-subject/network/ROI Pearson R.

    Returns the matplotlib Figure.
    """
    rs_nets, _ = load_RSNs()
    _, node_names = get_BrainSymphony_atlas_labels()
    orig = _to_numpy(original_data)
    recon = _to_numpy(recon_data)
    subjs = np.asarray(subjs)

    r2_score = compute_r2(orig, recon)
    corr = compute_roi_correlation(orig, recon)  # (N, R)

    n_samples, n_rois = corr.shape
    df_subjs_recon = pd.DataFrame({
        'R': corr.ravel(),
        'subj': np.repeat(subjs, n_rois),
        'network': np.tile(np.asarray(rs_nets), n_samples),
        'roi': np.tile(np.asarray(node_names), n_samples),
    }).dropna(subset=['R'])

    fig = plt.figure(figsize=(10, 30))
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[1, 12], figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    sbn.histplot(pd.DataFrame({'r2': r2_score}), x='r2', bins=20, stat='probability', ax=ax1)
    ax1.set_xlabel('ROI-wise $R^2$')

    ax2 = fig.add_subplot(gs[0, 1])
    sbn.kdeplot(data=df_subjs_recon, x='R', hue='network', ax=ax2, common_norm=True)

    ax3 = fig.add_subplot(gs[0, 2])
    sbn.kdeplot(data=df_subjs_recon, x='R', hue='subj', ax=ax3, legend=False, alpha=0.25, common_norm=True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = fig.add_subplot(gs[1, 0:3])
    sbn.barplot(data=df_subjs_recon, x='R', y='roi', hue='network', ax=ax4, legend=False, errorbar='sd')
    ax4.tick_params(axis='y', labelsize=5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def plot_recon_summary(orig, recon, subjs, r2=None, corr=None, figsize=(9, 6.5)):
    """ Compact overview of reconstruction quality over all samples. Returns the Figure.

    Panels: (a) histogram of ROI-wise R², (b) mean R² per resting-state network,
    (c) KDE of per-ROI Pearson R by network, (d) KDE of Pearson R per subject,
    (e) ROI-wise R² for every ROI ordered by network (bars coloured by network).
    ``r2`` (R,) and ``corr`` (N, R) may be passed in to avoid recomputation.
    """
    rs_nets, _ = load_RSNs()
    rs_nets = np.asarray(rs_nets)
    subjs = np.asarray(subjs)
    if r2 is None:
        r2 = compute_r2(orig, recon)
    if corr is None:
        corr = compute_roi_correlation(orig, recon)
    r2 = np.asarray(r2)
    n_samples, n_rois = corr.shape

    df_r2 = pd.DataFrame({'r2': r2, 'network': rs_nets, 'roi_idx': np.arange(n_rois)})
    df_corr = pd.DataFrame({
        'R': corr.ravel(),
        'subj': np.repeat(subjs, n_rois),
        'network': np.tile(rs_nets, n_samples),
    }).dropna(subset=['R'])

    with plt.rc_context({'font.size': 8}):
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(nrows=2, ncols=6, height_ratios=[1, 0.9], figure=fig)

        ax = fig.add_subplot(gs[0, 0:2])
        sbn.histplot(df_r2, x='r2', bins=20, stat='probability', ax=ax)
        ax.set_xlabel('ROI-wise $R^2$')
        ax.set_title('(a) $R^2$ distribution', fontsize=8, loc='left')

        ax = fig.add_subplot(gs[0, 2:4])
        sbn.barplot(data=df_r2, x='network', y='r2', ax=ax, errorbar='sd', color='C0')
        ax.tick_params(axis='x', rotation=45, labelsize=7)
        ax.set_xlabel('')
        ax.set_ylabel('$R^2$')
        ax.set_title('(b) $R^2$ per RSN', fontsize=8, loc='left')

        ax = fig.add_subplot(gs[0, 4:6])
        sbn.kdeplot(data=df_corr, x='R', hue='network', ax=ax, common_norm=True, legend=False, lw=1)
        ax.set_xlabel('Pearson R')
        ax.set_title('(c) R by network', fontsize=8, loc='left')

        ax = fig.add_subplot(gs[1, 0:2])
        n_subj = len(np.unique(subjs))
        sbn.kdeplot(data=df_corr, x='R', hue='subj', ax=ax, legend=False, alpha=0.25, common_norm=True, lw=0.8)
        ax.set_xlabel('Pearson R')
        ax.set_title(f'(d) R per subject (n={n_subj})', fontsize=8, loc='left')

        ax = fig.add_subplot(gs[1, 2:6])
        order = np.argsort(pd.Categorical(rs_nets, categories=list(pd.unique(rs_nets))).codes, kind='stable')
        nets_sorted = rs_nets[order]
        palette = dict(zip(pd.unique(rs_nets), sbn.color_palette('tab20', len(pd.unique(rs_nets)))))
        ax.bar(np.arange(n_rois), r2[order], width=1.0, color=[palette[n] for n in nets_sorted])
        bounds = [i for i in range(1, n_rois) if nets_sorted[i] != nets_sorted[i - 1]]
        starts = [0] + bounds
        ends = bounds + [n_rois]
        ax.set_xticks([(s + e) / 2 for s, e in zip(starts, ends)])
        ax.set_xticklabels([nets_sorted[s] for s in starts], rotation=60, ha='right', fontsize=6)
        ax.set_xlim(-0.5, n_rois - 0.5)
        ax.set_ylabel('$R^2$')
        ax.set_title('(e) ROI-wise $R^2$ (ordered by network)', fontsize=8, loc='left')

        for a in fig.axes:
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
        fig.tight_layout()
    return fig


def select_rois_for_display(r2: np.ndarray, n_rois: int = 10, strategy: str = "network") -> np.ndarray:
    """ Deterministically pick ROI indices to display.

    strategy:
      - "network": first ROI of each RSN (up to n_rois networks), ordered by network
      - "best":    the n_rois ROIs with the highest R²
      - "worst":   the n_rois ROIs with the lowest R²
      - "median":  n_rois ROIs centred on the median R²
    """
    r2 = np.asarray(r2)
    if strategy == "best":
        return np.argsort(-r2)[:n_rois]
    if strategy == "worst":
        return np.argsort(r2)[:n_rois]
    if strategy == "median":
        order = np.argsort(r2)
        mid = len(order) // 2
        start = max(0, mid - n_rois // 2)
        return order[start:start + n_rois]
    # default: one representative ROI per network
    rs_nets, _ = load_RSNs()
    picks, seen = [], set()
    for idx, net in enumerate(np.asarray(rs_nets)):
        if net not in seen:
            seen.add(net)
            picks.append(idx)
        if len(picks) == n_rois:
            break
    return np.asarray(picks)


def plot_recon_timeseries(orig, recon, sample_idx=0, roi_indices=None, t_window=None, n_rois=10):
    """ Overlay original (black) and reconstructed (red dashed) BOLD for selected ROIs.

    orig/recon: (N, R, T) tensors or arrays. roi_indices defaults to one ROI per network.
    t_window: optional (start, stop) time-point range for readability.
    Returns the matplotlib Figure.
    """
    ori = _to_numpy(orig)[sample_idx]   # (R, T)
    rec = _to_numpy(recon)[sample_idx]  # (R, T)
    _, node_names = get_BrainSymphony_atlas_labels()

    if roi_indices is None:
        roi_indices = select_rois_for_display(compute_r2(orig, recon), n_rois=n_rois)
    roi_indices = list(roi_indices)
    t0, t1 = (0, ori.shape[-1]) if t_window is None else t_window
    t = np.arange(t0, t1)

    with plt.rc_context({'font.size': 8}):
        fig, axes = plt.subplots(len(roi_indices), 1, figsize=(6, 0.9 * len(roi_indices) + 0.8),
                                 sharex=True, squeeze=False)
        for i, idx in enumerate(roi_indices):
            ax = axes[i, 0]
            ax.plot(t, ori[idx, t0:t1], color='k', lw=1, label='original')
            ax.plot(t, rec[idx, t0:t1], '--', color='r', lw=1, label='reconstructed')
            r = np.corrcoef(ori[idx], rec[idx])[0, 1]
            ax.set_title(f"{node_names[idx]}   R={r:.3f}", fontsize=8, loc='left')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        axes[0, 0].legend(loc='upper right', fontsize=7, frameon=False)
        axes[-1, 0].set_xlabel('TR')
        fig.tight_layout()
    return fig