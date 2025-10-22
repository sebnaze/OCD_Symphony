import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sbn
import scipy
import sklearn
import torch

import OCD_symphony
from OCD_symphony.utils import proj_dir, today

#plt.rcParams.update({"text.usetex": False, "font.family": "sans-serif", "font.sans-serif": "Helvetica","font.size":10})

def load_RSNs() -> tuple:
    """ Load resting-state networks for each node corresponding to the Schaefer400_TianS3 parcellation """ 
    rs_nets = pd.read_csv(os.path.join(proj_dir, 'parcellations', 'networks.txt'), header=None, sep=',')
    rs_nets = rs_nets[0]
    rs = rs_nets.unique()
    rs = [s[1:-1] for s in rs]
    return rs_nets, rs


def compute_r2(orig_data:torch.Tensor, recon_data:torch.Tensor) -> np.array:
    """ Returns the ROI-wise R-square between the original data and the reconstructed data """
    n_rois = orig_data.shape[1]
    r2 = np.zeros((n_rois,))
    for i in range(n_rois):
        r2[i] = sklearn.metrics.r2_score(orig_data[:,i,:], recon_data[:,i,:])
    return r2


def plot_recon_barplot(r2:np.array, rs_nets:np.array) -> None:
    """ Show bar plot of network-wise reconstruction error (RSN-wise) """ 
    df_ = pd.DataFrame({'r2':r2, 'network':rs_nets})

    plt.figure(figsize=(3,3))
    ax = sbn.barplot(data=df_, x='network', y='r2')

    plt.xticks(rotation=35)

    plt.xlabel("RSN")
    plt.ylabel("$R^2$")


