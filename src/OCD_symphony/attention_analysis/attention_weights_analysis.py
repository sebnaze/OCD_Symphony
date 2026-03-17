import json
import matplotlib
from matplotlib import pyplot as plt
import nilearn
import nilearn.image
import nilearn.plotting
import numpy as np
from numpy.typing import NDArray
import os
import pickle
import scipy

import OCD_symphony
from OCD_symphony.utils import proj_dir, today


def load_attn_scores():
    """ Load saved attention score T tests and Niftii image.
    
    Return
    ------
        attn_heads: list of length 2 (2 heads), each heads is a tuple (TtestResult, Nifti1Image) 
    """
    with open(os.path.join(proj_dir, 'results', 'attention_heads_20251202.pkl'), 'rb') as f:
        attn_heads = pickle.load(f)
    return attn_heads 


def compute_weights_ttest(all_perceiver_attn:NDArray, labels:NDArray, head:int=0):
    """ Compute independent t-test for each brain regions' attention weight""" 
    hc_inds, = np.where(labels==0)
    ocd_inds, = np.where(labels==1)

    hc_attn = all_perceiver_attn[hc_inds, head, :, :].squeeze()
    ocd_attn = all_perceiver_attn[ocd_inds, head, :, :].squeeze()
    
    # t-test using Welch method (not assuming equal variance)
    #res = scipy.stats.ttest_ind(hc_attn, ocd_attn, axis=0, equal_var=False) # use permutation=100 if n_samples < 30
    
    # Parallelize across brain regions (axis=1 in the 2D attention matrix)
    n_regions = hc_attn.shape[1]
    results = Parallel(n_jobs=-1)(
        delayed(scipy.stats.ttest_ind)(
            hc_attn[:, i], ocd_attn[:, i],  equal_var=False
        )
        for i in range(n_regions)
    )

    # Stack results into the expected format
    t_stats = np.array([r[0] for r in results])
    p_vals = np.array([r[1] for r in results])
    
    return res


def compute_significant_pval(res, alpha=0.01):
    """ Compute significant brain region's p-values """
    sigp = np.zeros((450,450))
    row,col = np.where(res[1]<alpha)
    sigp[row,col] = (1-res[1])[row,col]
    return sigp, (row,col)


def plot_significant_pvals(sigp):
    """ Plot significant weights' p-values """
    plt.figure(figsize=(4,4))
    plt.contourf(sigp, cmap='Grays')
    plt.colorbar()
    plt.title('Attention weights (p<0.01)', fontsize=10)


def compute_significant_tstat(res, row, col):
    sigt = np.zeros((450,450))
    sigt[row,col] = res[0][row,col]
    return sigt


def plot_significant_tstat(sigt):
    plt.figure(figsize=(4,4))
    plt.imshow(sigt, cmap='RdBu', vmin=-np.abs(sigt).max(), vmax=np.abs(sigt).max())
    plt.colorbar()
    plt.title("T stat (p<0.01)", fontsize=10)


def create_attention_brain_image(sigvals:NDArray, roi_inds:NDArray):
    """ Create a brain image from parcellated atlas with regions colorcoded by significance magnitude.
    
    Parameters
    ----------
        sigvals: np.array
            Array of significant values

        roi_inds: np.array
            Indices of the mapping from our Schaefer400_TianS3 to BrainSymphony's. 

    Returns
    -------
        attn_img: NIFTII image
            Formatted brain image with significant regions colorcoded. 
    """
    # use t-stat or p-value as threshold (I think t-stat is better)
    thr = 3*np.std(np.abs(sigvals).sum(axis=0))
    inds, = np.where(np.abs(sigvals).sum(axis=0)>thr)
    metric = np.abs(sigvals).sum(axis=0)

    # load atlas img and id/labels pairs
    with open(os.path.join(proj_dir, 'parcellations', 'atlas_config.json'), 'r') as f:
        atlas_config = json.load(f)
    atlas = nilearn.image.load_img(os.path.join(proj_dir, 'parcellations', 'schaefer400_tianS3MNI_lps_mni.nii.gz'))

    # re-org ROI indices: ctx_L, sub-ctx_L, ctx_R, sub-ctx_R
    reverse_roi_inds = np.concatenate([np.arange(50,250), np.arange(25,50), np.arange(250,450), np.arange(0, 25)])

    # get significant regions names and ids    
    node_names = np.array(atlas_config['schaefer400_tianS3']['node_names'])[reverse_roi_inds[inds]]
    node_ids = np.array(atlas_config['schaefer400_tianS3']['node_ids'])[reverse_roi_inds[inds]]

    # create attention brain img
    attn_map = np.zeros(atlas.get_fdata().shape) # type: ignore
    for i,node in enumerate(node_ids):
        x,y,z = np.where(atlas.get_fdata()==node) # type: ignore
        attn_map[x,y,z] = metric[roi_inds[node-1]]
        print(f"Node index {node:4} \t {node_names[i]:30} \t : \t {metric[roi_inds[node-1]]:.2f} \t Metric index = {roi_inds[node-1]}")

    attn_img = nilearn.image.new_img_like(atlas, attn_map)
    return attn_img


def plot_attention_brain_image(attn_img):
    nilearn.plotting.plot_stat_map(attn_img, cmap='RdBu_r', cut_coords=[-44, 50, 4])


def create_brain_image_from(vals:NDArray, inds:NDArray, roi_inds:NDArray) :
    """ Create a brain image from parcellated atlas with regions colorcoded by significance magnitude.
    
    Parameters
    ----------
        vals: np.array
            Vector array of values used to fill brain image

        inds: np.array
            Vector array of indices from the original brain symphony space

        roi_inds: np.array
            Vector array of indices of the mapping from our Schaefer400_TianS3 to BrainSymphony's. 

    Returns
    -------
        attn_img: NIFTII image
            Formatted brain image with regions colorcoded. 
    """
    

    # load atlas img and id/labels pairs
    with open(os.path.join(proj_dir, 'parcellations', 'atlas_config.json'), 'r') as f:
        atlas_config = json.load(f)
    atlas = nilearn.image.load_img(os.path.join(proj_dir, 'parcellations', 'schaefer400_tianS3MNI_lps_mni.nii.gz'))

    # re-org ROI indices from Brain Symphony to Schaefer: sub-ctx_R, sub-ctx_L, ctx_L, ctx_R --> ctx_L, sub-ctx_L, ctx_R, sub-ctx_R
    reverse_roi_inds = np.concatenate([np.arange(50,250), np.arange(25,50), np.arange(250,450), np.arange(0, 25)])

    # get significant regions names and ids from Brain Symph to Schaefer    
    node_names = np.array(atlas_config['schaefer400_tianS3']['node_names'])[reverse_roi_inds[inds]]
    node_ids = np.array(atlas_config['schaefer400_tianS3']['node_ids'])[reverse_roi_inds[inds]]

    # create attention brain img
    attn_map = np.zeros(atlas.get_fdata().shape) # type: ignore
    for i,node in enumerate(node_ids):
        x,y,z = np.where(atlas.get_fdata()==node) # type: ignore
        attn_map[x,y,z] = vals[i]
        #print(f"Node index {node:4} \t {node_names[i]:30} \t : \t {vals[roi_inds[node-1]]:.2f} \t Metric index = {roi_inds[node-1]}")

    attn_img = nilearn.image.new_img_like(atlas, attn_map)
    return attn_img, node_names
