# Make OCD dataset from pre-saved patients' tensor files  

import numpy as np
import OCD_symphony
from OCD_symphony.utils import proj_dir
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Dict, Tuple

class FMRIDataset(Dataset):
    """
    Simple dataset that returns a tuple of available modalities:
    - (fmri, fc, G) if all are provided
    - (fmri, fc)    if fmri and fc are provided
    - (fmri,)       if only fmri is provided
    """
    def __init__(
        self,
        fmri_data: torch.Tensor,
        fc_data: Optional[torch.Tensor] = None,
        G_tensor: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None 
        ) -> None:
        
        assert isinstance(fmri_data, torch.Tensor), "fmri_data must be a torch.Tensor"
        self.fmri_data = fmri_data
        self.fc_data = fc_data
        self.G_tensor = G_tensor
        self.labels = labels
        n = fmri_data.shape[0]
        if fc_data is not None:
            assert fc_data.shape[0] == n, "fc_data and fmri_data must have same first dim"
        if G_tensor is not None:
            assert G_tensor.shape[0] == n, "G_tensor and fmri_data must have same first dim"
        if labels is not None:
            assert labels.shape[0] == n, "labels and fmri_data must have same first dim"

    def __len__(self) -> int:
        return self.fmri_data.shape[0]

    def __getitem__(self, idx: int):
        x = [self.fmri_data[idx]]
        if self.fc_data is not None:
            x.append(self.fc_data[idx])
        if self.G_tensor is not None:
            x.append(self.G_tensor[idx])
        if self.labels is not None:
            x.append(self.labels[idx])
        return tuple(x)

def load_from_pt(path: str, include: str = '') -> Dict[str, torch.Tensor]:
    """
    Load dictionnary files for each subject and combine them
    """
    fnames = sorted([fname for fname in os.listdir(path) if (".pt" in fname and include in fname)])
    tensors_list = [torch.load(os.path.join(path, fname)) for fname in fnames]
    
    all_tensors = {key:torch.Tensor() for key in tensors_list[0].keys()}
    all_tensors['subj'] = []

    for i,t in enumerate(tensors_list):

        # ad-hoc extraction of subject id from file name
        subj = fnames[i][:-3] 
        n_samples = t['fmri_data'].shape[0]*t['fmri_data'].shape[1]
        all_tensors['subj'].append(np.repeat(subj, n_samples))
            
        # reshaping first dim of data to n_samples
        for key in t.keys():  # fmri_data, fc_data, session
            if key in ['fmri_data', 'fc_data', 'G_tensor']:
                shp = t[key].shape
                all_tensors[key] = torch.cat([all_tensors[key], t[key].reshape(-1,shp[-2], shp[-1])], dim=0)
    
    for key in all_tensors.keys():
        if type(all_tensors[key])!=torch.Tensor:
            all_tensors[key] = np.array(all_tensors[key]).reshape(-1,)
        print(f"all_tensors[{key}].shape : \t{all_tensors[key].shape}")

    return all_tensors

def make_dataset_from_pt(path: str, include: str = '') -> FMRIDataset:
    d = load_from_pt(path, include)
    return FMRIDataset(
        fmri_data=d["fmri_data"],
        fc_data=d.get("fc_data", None),
        G_tensor=d.get("G_tensor", None),
        labels=d.get("labels", None)
    ), d['subj']

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
    n = len(ds)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_test = n - n_train - n_val
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n_test], generator=g)

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
    return torch.utils.data._utils.collate.default_collate(batch)


if __name__ == "__main__":
    # Example usage (adjust path and keys):
    d = make_dataset_from_pt(os.path.join(proj_dir, 'data'))
    train_loader, val_loader, test_loader = make_dataloaders(d, batch_size=16)
    for batch in train_loader:
        out = batch  # (fmri,) or (fmri, fc, G)
        print(f"Length = {len(out)}")
        print(f"shapes = {[o.shape for o in out]}")
        break
    pass
