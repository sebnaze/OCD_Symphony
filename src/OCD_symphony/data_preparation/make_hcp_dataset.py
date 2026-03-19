# Make OCD dataset from pre-saved patients' tensor files  
import dask
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
from dask.delayed import delayed
import numpy as np
import OCD_symphony
from OCD_symphony.utils import proj_dir
import os
import pandas as pd
import scipy
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, random_split
from typing import Optional, Dict, Tuple

class FMRIDataset(IterableDataset):
    """
    Iterable dataset that returns a tuple of available modalities: (fmri_data, fc_data, G_tensor) for each sample.
    """
    def __init__(self, fnames, W, B) -> None:
        self.fnames = fnames
        self.W = torch.from_numpy(W).float()
        self.B = torch.from_numpy(B).float()

    def __iter__(self):
        for fname in self.fnames:
            samples = load_hcp_from_pt(fname)
            samples = preprocess_samples(samples, self.W, self.B)
            for i in range(samples['fmri_data'].shape[0]):
                yield (samples['fmri_data'][i], samples['fc_data'][i], samples['G_tensor'][i])
    

def load_hcp_from_pt(fpath: str) -> Dict[str, torch.Tensor]:
    """
    Load dictionnary files for each subject and combine them
    """

    tnsr = torch.load(fpath, map_location='cpu', weights_only=True)

    all_tensors = {key:torch.Tensor() for key in tnsr.keys()}
    all_tensors['subj'] = [] # type: ignore
    
    # ad-hoc extraction of subject id from file name
    subj = os.path.basename(fpath)[:-3] 
    n_samples = tnsr['fmri_data'].shape[0]*tnsr['fmri_data'].shape[1]    
    all_tensors['subj'].append(np.repeat(subj, n_samples))        
            
    # reshaping first dim of data to n_samples
    for key in tnsr.keys():  # fmri_data, fc_data, session
        if key in ['fmri_data', 'fc_data', 'G_tensor']:
            shp = tnsr[key].shape
            all_tensors[key] = torch.cat([all_tensors[key], tnsr[key].reshape(-1,shp[-2], shp[-1])], dim=0)

    
    for key in all_tensors.keys():
        if type(all_tensors[key])!=torch.Tensor:
            all_tensors[key] = np.array(all_tensors[key]).reshape(-1,) # type: ignore
        #print(f"all_tensors[{key}].shape : \t{all_tensors[key].shape}")

    return all_tensors


def preprocess_samples(samples: Dict[str, torch.Tensor], W, B) -> Dict[str, torch.Tensor]:
    """
    Preprocess samples (e.g. reshape, re-index, normalize).
    """
    # reshape to (n_samples, n_rois, n_timesteps) and (n_samples, n_rois, n_n_rois) and (n_samples, n_rois, n_Gradients)
    samples['fmri_data'] = samples['fmri_data'].reshape(-1,450,200)
    samples['fc_data'] = samples['fc_data'].reshape(-1,450,450)
    samples['G_tensor'] = torch.randn(samples['fmri_data'].shape[0], 450, 10)

    # re-org ROI indices: sub-ctx_R, sub-ctx_L, ctx_L, ctx_R
    roi_inds = torch.from_numpy(np.concatenate([np.arange(425,450), np.arange(200,225), np.arange(0,200), np.arange(225, 425)]))
    samples['fmri_data'] = samples['fmri_data'][:,roi_inds,:]
    samples['fc_data'] = samples['fc_data'][:,roi_inds,:][:,:,roi_inds]

    # fmri_data shape: (n_samples, 450, 200)
    # W and B shape: (450,)
    # We unsqueeze W and B to align with the ROI dimension (dim 1)
    samples['fmri_data'] = samples['fmri_data'] * W.reshape(1, -1, 1) + B.reshape(1, -1, 1)
    return samples


def get_fmri_stats(sample : Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, int]:
    tensor = sample['fmri_data'] # shape (n_samples, 450, 200)
    if tensor.shape[-1] != 200 or tensor.shape[1] != 450:
        return (np.zeros((450,)), np.zeros((450,)), 0)  # Return zeros if shape is unexpected
    
    # Sum along the time axis (dim -1)
    roi_sum = tensor.sum(dim=(0,2)).numpy()        
    roi_sum_sq = (tensor**2).sum(dim=(0,2)).numpy() 
    n_points = tensor.shape[0]*tensor.shape[2]  # total number of time points per ROI across all samples                    
    return (roi_sum, roi_sum_sq, n_points, sample['subj']) # type: ignore



def get_fmri_raw_points(sample: Dict[str, torch.Tensor]):
    tensor = sample['fmri_data'] # (n_samples, 450, 200)
    if tensor.shape[1] != 450 or tensor.shape[2] != 200:
        return []
    
    # Reshape to (450, n_samples * 200) and transpose 
    # so each row in our future dataframe is one timepoint for all 450 ROIs
    flattened = tensor.permute(0, 2, 1).reshape(-1, 450).numpy()
    return (flattened, sample['subj']) # type: ignore


def get_fmri_subjs(sample: Dict[str, torch.Tensor]):
    return sample['subj'] # type: ignore

def make_hcp_dataset_from_pt(path: str, include: str = '', n_subjs=None) -> FMRIDataset:
    fnames = sorted([os.path.join(path, fname) for fname in os.listdir(path) if (".pt" in fname and include in fname)])
    
    if n_subjs is not None:
        fnames = fnames[:n_subjs] # For testing, limit to first x files.
    
    b = db.from_sequence(fnames, npartitions=32).map(load_hcp_from_pt) # type: ignore
    
    #--------------------------------------------#
    #       OLD APPROACH: Mean and STD           #
    #--------------------------------------------#
    #stats = b.map(get_fmri_stats)
    #def combine_stats(a,b):
    #    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], np.concatenate([a[3], b[3]]))
        
    #initial_stats = (np.zeros((450,)), np.zeros((450,)), 0, np.array([]))
    #total_sum, total_sum_sq, total_n, subjs = stats.fold(combine_stats, initial=initial_stats).compute()
    #print(f"Number of samples: {subjs.shape}, unique subjects: {len(np.unique(subjs))}, avg samples per subject: {len(subjs)/len(np.unique(subjs))}") # type: ignore
    
    #mu = total_sum / total_n # type: ignore
    #sigma_sq = (total_sum_sq / total_n) - (mu ** 2) # type: ignore
    #sigma = np.sqrt(sigma_sq)
    

    #--------------------------------------------#
    #  NEW APPROACH: Mean, Std, Median and IQR   #
    #--------------------------------------------#
        
    # 1. Convert to Dask DataFrame (keeping your existing setup)
    raw_data = b.map(get_fmri_raw_points)
    fmri_bag = raw_data.map(lambda x: x[0])
    ddf = fmri_bag.flatten().to_dataframe(columns=[f'roi_{i}' for i in range(450)])

    # 2. Extract subjects (this is a separate Bag operation, but we can compute it together)
    subjs_bag = raw_data.map(lambda x: x[1])
    
    # Subjs reduction
    #subjs_lazy = subjs_bag.fold(np.concatenate, initial=np.array([]))
    subjs_lazy = subjs_bag.flatten().to_dataframe(columns=['subj'])
    
    
    # 3. Define all required aggregations lazily
    # Quantiles for Robust Scaling
    q_lazy = ddf.quantile([0.25, 0.5, 0.75])

    # 4. TRICK: To get Mean/Std of the Robust data in the same pass, 
    # we calculate Mean/Std of the RAW data and transform them algebraically.
    mean_lazy = ddf.mean()
    std_lazy = ddf.std()

    # 5. Execute everything in ONE pass
    q, subjs, raw_mu, raw_std = dask.compute(q_lazy, subjs_lazy, mean_lazy, std_lazy)
    subjs = np.array(subjs).squeeze() # type: ignore

    # 6. Extract constants from results
    median = q.loc[0.5].values
    iqr = (q.loc[0.75] - q.loc[0.25]).values
    iqr = np.where(iqr == 0, 1e-8, iqr)

    # 7. Algebraic Shift: 
    # Instead of a second pass for 'mu_robust', calculate what they would be:
    # If x_robust = (x - median) / iqr, then:
    mu_robust = (raw_mu.values - median) / iqr
    sigma_robust = raw_std.values / iqr
    sigma_robust = np.where(sigma_robust == 0, 1e-8, sigma_robust)

    # 8. Collapse constants for the linear transform: x_final = x*W + B
    denom = iqr * sigma_robust
    W = 1.0 / denom
    B = (-median / denom) - (mu_robust / sigma_robust)

    return FMRIDataset(fnames, W, B), subjs # type: ignore


def split_dataset(
    ds: Dataset,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Deterministic split into train/val/test according to fractions.
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    n = len(ds) # type: ignore
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=g) # type: ignore


def make_dataloaders(
    ds: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader objects for train, validation and test sets
    """ 
    train_ds, val_ds, test_ds = split_dataset(ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    return train_loader, val_loader, test_loader

# Optional: custom collate if your samples contain variable-length sequences.
# For now, default PyTorch collate is fine for fixed-shape tensors.
def identity_collate(batch):
    return torch.utils.data._utils.collate.default_collate(batch) # type: ignore



def load_hcp_age_gender(subjs):
    """ Load HCP demographics file and return age and gender lists for each sample.
    
    Arguments:
    ==========
        subjs: list of subject IDs corresponding to the samples in the dataset (e.g. ['100307', '100408', ...])
    
    """
    hcp_demographics = pd.read_csv(os.path.join(proj_dir, "data/HCP_demographic_data.csv"))
    hcp_gender = []
    hcp_age = []
    for subj in subjs:
        subj_row = hcp_demographics[hcp_demographics.Subject.astype(str)==subj]
        if len(subj_row):
            hcp_gender.append(subj_row.iloc[0]['Gender'])
            hcp_age.append(subj_row.iloc[0]['Age'])
        else:
            print(f"HCP {subj} not in demographic file!?!")
    return hcp_age, hcp_gender



if __name__ == "__main__":
    # Example usage (adjust path and keys):
    d = make_hcp_dataset_from_pt(os.path.join(proj_dir, 'data/hcp'))
    train_loader, val_loader, test_loader = make_dataloaders(d, batch_size=16)
    for batch in train_loader:
        out = batch  # (fmri,) or (fmri, fc, G)
        print(f"Length = {len(out)}")
        print(f"shapes = {[o.shape for o in out]}")
        break
    pass
