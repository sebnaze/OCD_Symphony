import argparse
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score
from time import time
import wandb

from OCD_symphony.utils import proj_dir, today
from OCD_symphony.data_preparation.make_hcp_dataset import FMRIDataset, make_hcp_dataset_from_pt
from OCD_symphony.model.model_components import *

#class FMRIDataset(Dataset):
#    def __init__(self, fmri_data, fc_data, G_tensor, gender_labels):
#        self.fmri_data = fmri_data
#        self.fc_data = fc_data
#        self.G_tensor = G_tensor
#        self.gender = gender_labels  # shape: (N,) or (N, 1)

#    def __getitem__(self, idx):
#        return (
#            self.fmri_data[idx],
#            self.fc_data[idx],
#            self.G_tensor[idx],
#            self.gender[idx],
#        )

#    def __len__(self):
#        return len(self.fmri_data)


# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Model parameters ===
DIM = 128
NUM_LAYERS = 6
NUM_HEADS = 4
MLP_DIM = 512
MAX_LEN_SPATIAL = 500
NUM_GRADIENTS = 10
DELTA = 0.5
MAX_LEN_TEMP = 5000

# === Training parameters ===
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# === Data parameters ===
TIME_STEPS = 200
ROIs = 450 


def finetune(trainset, testset, args):
    """
    Fine-tune the pretrained Perceiver classifier on the HCP dataset for gender classification.
    """
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle = True)
    val_loader = DataLoader(testset, batch_size=args.batch_size, shuffle = False)


    # === Load pretrained models ===
    base_transformer = TemporalTransformer(NUM_LAYERS, DIM, NUM_HEADS, MLP_DIM, TIME_STEPS, MAX_LEN_TEMP, ROIs).to(device)
    masker = TokenMasker1D(mask_ratio=0)
    temporal_model = MaskedTemporalTransformer(base_transformer, masker, roi_dim=ROIs, hidden_dim=256).to(device)

    base_transformer = SpatialTransformer(
        num_layers=NUM_LAYERS,
        dim=DIM,
        num_heads=NUM_HEADS,
        mlp_dim=MLP_DIM,
        time_steps=TIME_STEPS,
        max_len=MAX_LEN_SPATIAL,
        num_gradients=NUM_GRADIENTS,
        delta=DELTA,
        pos_enc_type="learnable",
    ).to(device)
    masker = TokenMasker2D(mask_ratio=0)
    spatial_model = MaskedSpatialTransformer(
        base_transformer, masker, TIME_STEPS, hidden_dim=256
    ).to(device)

    masker = TimePointMasker(mask_ratio=0)
    conv_model = MaskedConv1DTemporalModel(
        input_channels=1, signal_dim=64, decoder_hidden_dims=[32, 8]
    ).to(device)

    # Load state dicts
    temporal_model.load_state_dict(
        torch.load(
            os.path.join(proj_dir, 'models',
                         f"temporal_transformer_pretrained_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
            )
        )
    )
    spatial_model.load_state_dict(
        torch.load(
            os.path.join(proj_dir, 'models',
                f"spatial_transformer_pretrained_learnablePE_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
            )
        )
    )
    conv_model.load_state_dict(
        torch.load(
            os.path.join(proj_dir, 'models',
                f"masked_conv1d_temporal_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
            )
        )
    )

    temporal_model.eval()
    spatial_model.eval()
    conv_model.eval()

    for model in [spatial_model, temporal_model, conv_model]:
        for param in model.parameters():
            param.requires_grad = False

    # === Load Pretrained Perceiver ===
    perceiver = PerceiverIOModelROIWise(
        input_dim=64 * 3,
        latent_dim=DIM,
        num_rois=ROIs,
        num_heads=NUM_HEADS,
        time_steps=TIME_STEPS,
    ).to(device)
    perceiver.load_state_dict(
        torch.load(
            os.path.join(proj_dir, 'models',
                f"perceiver_fused_pretrained_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
            )
        )
    )
    classifier = PerceiverClassifier(perceiver, latent_dim=DIM, num_classes=2).to(device)

    # === Optimizer and Loss ===
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, int(0.1 * NUM_EPOCHS * len(train_loader)), NUM_EPOCHS * len(train_loader)
    )
    criterion = nn.CrossEntropyLoss()

    # === Training Loop ===
    best_val_acc = 0.0

    for epoch in range(args.n_epochs):
        classifier.train()
        total_loss, correct, total = 0, 0, 0
        all_preds, all_labels = [], []

        for fmri, fc, gm, labels in train_loader:
            fmri, fc, gm, labels = (
                fmri.to(device),
                fc.to(device),
                gm.to(device),
                labels.to(device),
            )

            with torch.no_grad():
                _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
                _, _, _, _, time_embeds, _ = temporal_model(fmri)
                _, signal_context = conv_model(fmri, mask=None)

            logits, recon = classifier(roi_embeds, time_embeds, signal_context, roi_pos)
            loss = criterion(logits, labels) + F.mse_loss(recon, fmri)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        train_acc = correct / total
        train_f1 = f1_score(all_labels, all_preds, average="macro")
        print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        if args.use_wandb:
            wandb.log({"train_loss": total_loss, "train_acc": train_acc, 'train_f1':train_f1}, epoch=epoch+1)

        # === Validation ===
        classifier.eval()
        val_correct, val_total = 0, 0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for fmri, fc, gm, labels in val_loader:
                fmri, fc, gm, labels = (
                    fmri.to(device),
                    fc.to(device),
                    gm.to(device),
                    labels.to(device),
                )

                _, _, _, _, roi_embeds, roi_pos = spatial_model(gm, fmri, fc)
                _, _, _, _, time_embeds, _ = temporal_model(fmri)
                _, signal_context = conv_model(fmri, mask=None)

                logits, recon = classifier(roi_embeds, time_embeds, signal_context, roi_pos)
                preds = logits.argmax(dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(preds.cpu().tolist())
                val_labels_list.extend(labels.cpu().tolist())

        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels_list, val_preds, average="macro")
        print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        if args.use_wandb:
            wandb.log({"val_acc": train_acc, 'val_f1':train_f1}, epoch=epoch+1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                classifier.state_dict(),
                os.path.join(proj_dir, 'models',
                    f"Fine_tuned_perceiver_classifier_gender_HCP_YA_nL{NUM_LAYERS}_d{DIM}_h{NUM_HEADS}_mlp{MLP_DIM}.pth"
                )
            )

    print("✅ Fine-tuning Complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Perceiver Classifier on HCP dataset")
    parser.add_argument('--label', type=str, default='gender')
    parser.add_argument('--n_epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE)
    parser.add_argument('--n_train_subjs', type=int, default=100, help='Number of subjects to use for training')
    parser.add_argument('--n_test_subjs', type=int, default=20, help='Number of subjects to use for testing')
    parser.add_argument('--use_wandb', action='store_true', help='Whether to log training with Weights & Biases')
    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    t0 = time()

    # Create train and test sets
    trainset, train_subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, 'data/hcp'), label=args.label, n_subjs=args.n_train_subjs, start_idx=0
    )
    testset, test_subjs = make_hcp_dataset_from_pt(
        path=os.path.join(proj_dir, 'data/hcp'), label=args.label, n_subjs=args.n_test_subjs, start_idx=args.n_train_subjs
    )
    print(f"Lazy Loading of dataset done in {time()-t0:.1f}s, i.e. {(time()-t0)/60:.1f} minutes.")

    if args.use_wandb:
            wandb.init(project="OCD_Symphony", name="Finetuning_HCP_gender")

    finetune(trainset, testset, args)

    if args.use_wandb:
            wandb.finish()

