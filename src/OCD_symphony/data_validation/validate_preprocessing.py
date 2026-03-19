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

def load_RSNs() -> tuple:
    """ Load resting-state networks for each node corresponding to the Schaefer400_TianS3 parcellation """ 
    rs_nets = pd.read_csv(os.path.join(proj_dir, 'parcellations', 'networks.txt'), header=None, sep=',')
    rs_nets = rs_nets[0]
    rs = rs_nets.unique()
    rs = [s[1:-1] for s in rs]
    return rs_nets, rs


def compute_r2(orig_data:torch.Tensor, recon_data:torch.Tensor) -> np.ndarray:
    """ Returns the ROI-wise R-square between the original data and the reconstructed data """
    n_rois = orig_data.shape[1]
    r2 = np.zeros((n_rois,))
    for i in range(n_rois):
        r2[i] = metrics.r2_score(orig_data[:,i,:], recon_data[:,i,:])
    return r2


def plot_recon_barplot(r2:np.ndarray, rs_nets:np.ndarray) -> None:
    """ Show bar plot of network-wise reconstruction error (RSN-wise) """ 
    df_ = pd.DataFrame({'r2':r2, 'network':rs_nets})

    plt.figure(figsize=(3,3))
    ax = sbn.barplot(data=df_, x='network', y='r2')

    plt.xticks(rotation=35)

    plt.xlabel("RSN")
    plt.ylabel("$R^2$")


def plot_recon_histogram(original_data, recon_data, subjs) -> None:
    # get RSNs correpsondign to ROIs 
    rs_nets, rs = OCD_symphony.data_validation.load_RSNs()
    node_ids, node_names = get_BrainSymphony_atlas_labels()

    r2_score = OCD_symphony.data_validation.compute_r2(original_data, recon_data)

    # Overall
    plt.figure(figsize=(10,30))
    gs = GridSpec(nrows=2, ncols=3, height_ratios=[1,12])    
    ax1 = plt.subplot(gs[0,0])
    sbn.histplot(pd.DataFrame({'r2':r2_score}), bins=20, kde=False, stat='probability', ax=ax1)

    # Subject specific
    subjs_recon = []
    for subj in np.unique(subjs):
        subj_inds, = np.where(subjs==subj)
        for sample_idx in subj_inds:
            for roi_idx in np.arange(original_data.shape[1]):
                R = np.corrcoef(original_data[sample_idx,roi_idx,:], recon_data[sample_idx,roi_idx,:])[0,1]
                subj_recon = {'R':R, 'subj':subj, 'network': rs_nets[roi_idx], 'roi': node_names[roi_idx]}
                subjs_recon.append(subj_recon)

    df_subjs_recon = pd.DataFrame(subjs_recon)
    
    ax2 = plt.subplot(gs[0,1])
    sbn.kdeplot(data=df_subjs_recon, x='R', hue='network', ax=ax2, common_norm=True)
    
    ax3 = plt.subplot(gs[0,2])
    sbn.kdeplot(data=df_subjs_recon, x='R', hue='subj', ax=ax3, legend=False, alpha=0.25, common_norm=True)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax4 = plt.subplot(gs[1,0:3])
    sbn.barplot(data=df_subjs_recon, x='R', y='roi', hue='network', ax=ax4, legend=False, errorbar='sd')
    ax4.tick_params(axis='y', labelsize=5)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    plt.tight_layout()


def plot_recon_timeseries(orig, recon, sample_idx=0) -> None:
    ori = orig[sample_idx,:,:]
    rec = recon[sample_idx,:,:]

    nrois = 10

    plt.rcParams.update({'font.size':8})
    node_ids, node_names = get_BrainSymphony_atlas_labels()

    plt.figure(figsize=(6,10))
    for i, idx in enumerate(np.random.randint(0,450,nrois)):
        roi_name = node_names[idx]
        ax = plt.subplot(nrois, 1, i+1)
        plt.plot(ori[idx,:], color='k', label='original')
        plt.plot(rec[idx,:], '--', color='r', label='reconstructed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if i != (nrois-1):
            ax.spines['bottom'].set_visible(False)
            ax.set_xticks([])
        plt.title(f"{roi_name}:    R={np.corrcoef(ori[idx,:], rec[idx,:])[0,1]:.3f}", fontsize=8)
    plt.tight_layout()