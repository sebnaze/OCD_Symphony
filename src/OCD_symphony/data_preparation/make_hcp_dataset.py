# Make OCD dataset from pre-saved patients' tensor files  
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
    def __init__(self, fnames, mu, sigma) -> None:
        self.fnames = fnames
        self.mu = torch.from_numpy(mu).float()
        self.sigma = torch.from_numpy(sigma).float()

    def __iter__(self):
        for fname in self.fnames:
            samples = load_hcp_from_pt(fname)
            samples = preprocess_samples(samples, self.mu, self.sigma)
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


def preprocess_samples(samples: Dict[str, torch.Tensor], mu, sigma) -> Dict[str, torch.Tensor]:
    """
    Preprocess samples (e.g. reshape, re-index, normalize).
    """
    # reshape to (n_samples, n_rois, n_timesteps) and (n_samples, n_rois, n_timesteps)
    samples['fmri_data'] = samples['fmri_data'].reshape(-1,450,200)
    samples['fc_data'] = samples['fc_data'].reshape(-1,450,450)
    samples['G_tensor'] = torch.randn(samples['fmri_data'].shape[0], 450, 10)

    # re-org ROI indices: sub-ctx_R, sub-ctx_L, ctx_L, ctx_R
    roi_inds = torch.from_numpy(np.concatenate([np.arange(425,450), np.arange(200,225), np.arange(0,200), np.arange(225, 425)]))
    samples['fmri_data'] = samples['fmri_data'][:,roi_inds,:]
    samples['fc_data'] = samples['fc_data'][:,roi_inds,:][:,:,roi_inds]

    # normalize fMRI timeseries ROI-wise
    for i in np.arange(samples['fmri_data'].shape[1]):
        m = samples['fmri_data'][:,i,:].median()
        iqr = scipy.stats.iqr(samples['fmri_data'][:,i,:])
        samples['fmri_data'][:,i,:] = (samples['fmri_data'][:,i,:] - m) / (iqr + 1e-8)

    # z-score fMRI timeseries across entire dataset
    samples['fmri_data'] = (samples['fmri_data'] - mu[np.newaxis, :, np.newaxis]) / (sigma[np.newaxis, :, np.newaxis] + 1e-8)
    return samples


def get_fmri_stats(sample : Dict[str, torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, int]:
    tensor = sample['fmri_data'] # shape (n_samples, 450, 200)
    if tensor.shape[-1] != 200 or tensor.shape[1] != 450:
        return (np.zeros((450,)), np.zeros((450,)), 0)  # Return zeros if shape is unexpected
    
    # Sum along the time axis (dim -1)
    roi_sum = tensor.sum(dim=(0,2)).numpy()        
    roi_sum_sq = (tensor**2).sum(dim=(0,2)).numpy() 
    n_points = tensor.shape[0]*tensor.shape[2]  # total number of time points across all samples                    
    return (roi_sum, roi_sum_sq, n_points, sample['subj']) # type: ignore


def make_hcp_dataset_from_pt(path: str, include: str = '') -> FMRIDataset:
    fnames = sorted([os.path.join(path, fname) for fname in os.listdir(path) if (".pt" in fname and include in fname)])
    #fnames = fnames[:100]  # For testing, limit to first x files. Remove this line for full dataset.
    b = db.from_sequence(fnames, npartitions=32).map(load_hcp_from_pt) # type: ignore
    stats = b.map(get_fmri_stats)

    def combine_stats(a,b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], np.concatenate([a[3], b[3]]))
        
    initial_stats = (np.zeros((450,)), np.zeros((450,)), 0, np.array([]))
    total_sum, total_sum_sq, total_n, subjs = stats.fold(combine_stats, initial=initial_stats).compute()
    print(f"Number of samples: {subjs.shape}, unique subjects: {len(np.unique(subjs))}, avg samples per subject: {len(subjs)/len(np.unique(subjs))}") # type: ignore
    
    mu = total_sum / total_n # type: ignore
    sigma_sq = (total_sum_sq / total_n) - (mu ** 2) # type: ignore
    sigma = np.sqrt(sigma_sq)
    
    #d = b.map(lambda x: preprocess_samples(x, mu, sigma)).compute()

    return FMRIDataset(fnames, mu, sigma), subjs # type: ignore


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
