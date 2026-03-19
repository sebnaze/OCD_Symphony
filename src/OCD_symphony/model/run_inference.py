import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from OCD_symphony.utils import proj_dir, today
from OCD_symphony.model.model_components import *


def run_inference(dataset, output_file, n_samples):
    BATCH_SIZE = 16

    #dataset = FMRIDataset(fmri_data, fc_data, G_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle = False)

    # === Load pretrained models ===

    DIM = 128
    NUM_LAYERS = 6
    NUM_HEADS = 4
    MLP_DIM = 512
    TIME_STEPS = 200
    MAX_LEN_TEMP = 5000
    TIME_STEPS = 200
    MAX_LEN_SPATIAL = 500
    NUM_GRADIENTS = 10
    DELTA = 0.5
    LEARNING_RATE = 1e-4
    ROIs = 450



    base_transformer = TemporalTransformer(NUM_LAYERS, DIM, NUM_HEADS, MLP_DIM, TIME_STEPS, MAX_LEN_TEMP, ROIs).to(device)
    masker = TokenMasker1D(mask_ratio=0)
    temporal_model = MaskedTemporalTransformer(base_transformer, masker, roi_dim = 450, hidden_dim=256).to(device)


    base_transformer = SpatialTransformer(
        num_layers=NUM_LAYERS,
        dim=DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        time_steps=TIME_STEPS,
        max_len=MAX_LEN_SPATIAL,
        num_gradients=NUM_GRADIENTS,
        delta=DELTA,
        pos_enc_type="learnable"
    ).to(device)
    masker = TokenMasker2D(mask_ratio=0)
    spatial_model = MaskedSpatialTransformer(base_transformer, masker, TIME_STEPS,  hidden_dim=256).to(device)


    masker = TimePointMasker(mask_ratio=0)
    conv_model = MaskedConv1DTemporalModel(input_channels=1, signal_dim=64, decoder_hidden_dims=[32, 8]).to(device)


    temporal_model.load_state_dict(torch.load(os.path.join(proj_dir, 'models', f"temporal_transformer_pretrained_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth")))
    spatial_model.load_state_dict(torch.load(os.path.join(proj_dir, 'models', f"spatial_transformer_pretrained_learnablePE_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth")))
    conv_model.load_state_dict(torch.load(os.path.join(proj_dir, 'models', f"masked_conv1d_temporal_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth")))


    perceiver = PerceiverIOModelROIWise(input_dim=64*3, latent_dim=128, num_rois=450, num_heads=2, time_steps=200, dropout=0.0).to(device)


    perceiver.load_state_dict(torch.load(os.path.join(proj_dir, 'models', f"perceiver_fused_pretrained_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth")))



    temporal_model.eval()
    spatial_model.eval()
    conv_model.eval()
    perceiver.eval()


    # Create memory-mapped storage
    mmap_arrays = {
        'all_recon_temp': np.memmap(f'{output_file}.recon_temp.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, ROIs, TIME_STEPS)),
        "all_recon_spat": np.memmap(f'{output_file}.recon_spat.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, ROIs, TIME_STEPS)),   
        "all_recon_percv": np.memmap(f'{output_file}.recon_percv.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, ROIs, TIME_STEPS)),   
        "all_temporal_attn_first": np.memmap(f'{output_file}.temporal_attn_first.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, 4, TIME_STEPS, TIME_STEPS)),
        "all_temporal_attn_last": np.memmap(f'{output_file}.temporal_attn_last.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, 4, TIME_STEPS, TIME_STEPS)),   
        "all_spatial_attn_first": np.memmap(f'{output_file}.spatial_attn_first.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, 4, ROIs, ROIs)),   
        "all_spatial_attn_last": np.memmap(f'{output_file}.spatial_attn_last.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, 4, ROIs, ROIs)),   
        "all_perceiver_attn": np.memmap(f'{output_file}.perceiver_attn.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, 2, ROIs, ROIs)),   
        "all_roi_embeddings": np.memmap(f'{output_file}.roi_embeddings.npy', dtype=np.float32, 
                                    mode='w+', shape=(n_samples, ROIs, DIM)) 
    }

    idx = 0
    for fmri, fc, gm in data_loader:
        fmri = fmri.to(device)
        fc = fc.to(device)
        gm = gm.to(device)
        gm, fmri, fc = gm.float(), fmri.float(), fc.float()



        with torch.no_grad():
            _, recon_spat, _, spatial_attn, roi_embeds,  roi_pos = spatial_model(gm, fmri, fc)         # (B, R, D)
            _, recon_temp, mask_temp, temporal_attn, time_embeds, _ = temporal_model(fmri)       # (B, T, D)
            _, signal_context = conv_model(fmri, mask=None)            # (B, R, T, d_signal)
            recon_percv, latents, percv_attn = perceiver(roi_embeds, time_embeds, signal_context, roi_pos)

            # Saving to RAM for immediate analysis and freeing VRAM. 
            # For larger datasets, consider saving to disk instead using memmap
            mmap_arrays['all_recon_temp'][idx:idx+BATCH_SIZE] = recon_temp.detach().cpu().numpy()
            mmap_arrays['all_recon_spat'][idx:idx+BATCH_SIZE] = recon_spat.detach().cpu().numpy()
            mmap_arrays['all_recon_percv'][idx:idx+BATCH_SIZE] = recon_percv.detach().cpu().numpy()
            mmap_arrays['all_roi_embeddings'][idx:idx+BATCH_SIZE] = latents.detach().cpu().numpy()
            mmap_arrays['all_spatial_attn_first'][idx:idx+BATCH_SIZE] = spatial_attn[0].detach().cpu().numpy()
            mmap_arrays['all_spatial_attn_last'][idx:idx+BATCH_SIZE] = spatial_attn[-1].detach().cpu().numpy()
            mmap_arrays['all_temporal_attn_first'][idx:idx+BATCH_SIZE] = temporal_attn[0].detach().cpu().numpy()
            mmap_arrays['all_temporal_attn_last'][idx:idx+BATCH_SIZE] = temporal_attn[-1].detach().cpu().numpy()
            mmap_arrays['all_perceiver_attn'][idx:idx+BATCH_SIZE] = percv_attn[-1].detach().cpu().numpy()
            idx += BATCH_SIZE
        
    # Flush to disk
    for arr in mmap_arrays.values():
        arr.flush()
    
    return  mmap_arrays



def load_HCP_fmri_data(dataset) -> torch.Tensor:
    """
    Load fMRI data from the whole dataset and return as a tensor. Be careful of memory consumption, only use for data validation on sub-set of HCP dataset.
    """
    data_loader = DataLoader(dataset, batch_size=16, shuffle = False)
    fmri_data = torch.Tensor()
    for fmri, _, _ in data_loader:
        fmri_data = torch.cat([fmri_data, fmri], dim=0)
    return fmri_data
