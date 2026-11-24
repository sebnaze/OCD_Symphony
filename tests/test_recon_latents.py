"""Tests for latent collection in the reconstruction pass (no checkpoint, no data needed).

Encoders, the Perceiver and the subject loader are replaced by small fakes with the same call
signatures, so the wiring is tested without GPU models or raw files.

Run with:  conda run -n ocd-symphony python -m pytest tests/test_recon_latents.py -q
"""
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from OCD_symphony.reconstruction import dapt_reconstruction as dr

R, T, D = 6, 8, 4


def _fake_models():
    def spatial(gm, fmri, fc):
        roi_embeds = fmri[:, :, :D].clone()          # (B, R, D)
        return None, None, None, None, roi_embeds, None

    def temporal(fmri):
        time_embeds = fmri.transpose(1, 2)[..., :D].clone()  # (B, T, D)
        return None, None, None, None, time_embeds, None

    def conv(fmri, mask=None):
        return None, fmri.mean(dim=(1, 2))[:, None]

    def perceiver(roi_embeds, time_embeds, signal_context, roi_pos):
        # Time-varying so per-ROI correlations with the input are defined.
        recon = roi_embeds.sum(dim=-1, keepdim=True) + torch.arange(T, dtype=torch.float32)
        return recon, roi_embeds, None                # latents (B, R, D)

    return spatial, temporal, conv, perceiver


def _fake_loader(sid_list, W, B, max_abs=1e4, dataset="hcp_ya", **_):
    """One subject, 4 windows, deterministic from the subject id."""
    sid = sid_list[0]
    g = torch.Generator().manual_seed(sum(ord(c) for c in sid))
    n = 4
    fmri = torch.randn(n, R, T, generator=g)
    fc = torch.tanh(torch.randn(n, R, R, generator=g))
    gm = torch.zeros(n, R)
    labels = torch.zeros(n, dtype=torch.long)
    return TensorDataset(fmri, fc, gm, labels)


@pytest.fixture
def fake_env(monkeypatch):
    spatial, temporal, conv, perceiver = _fake_models()
    models = (spatial, temporal, conv, perceiver)
    monkeypatch.setattr(dr, "get_dapt_models", lambda ckpt, device: models)
    monkeypatch.setattr(dr, "load_subjects_normalised", _fake_loader)
    monkeypatch.setattr(dr, "load_RSNs", lambda: (np.array(["A", "A", "B", "B", "C", "C"]), None))
    return models


def test_forward_subject_default_yields_three_items(fake_env):
    batches = list(dr._forward_subject(fake_env, "s1", None, None, 1e4, 2, torch.device("cpu")))
    assert all(len(b) == 3 for b in batches)


def test_forward_subject_return_latents_yields_flat_latents(fake_env):
    batches = list(dr._forward_subject(fake_env, "s1", None, None, 1e4, 2,
                                       torch.device("cpu"), return_latents=True))
    lat = np.concatenate([b[3] for b in batches])
    assert lat.shape == (4, R * D)
    assert lat.dtype == np.float32


def test_cap_windows_keeps_first_rows():
    x = np.arange(20).reshape(10, 2)
    assert dr._cap_windows(x, None).shape == (10, 2)
    assert dr._cap_windows(x, 3).shape == (3, 2)
    assert np.array_equal(dr._cap_windows(x, 3), x[:3])


def test_reconstruct_default_has_no_geometry(fake_env):
    res = dr.reconstruct_subject_stats(None, ["s1", "s2", "s3"], None, None, keep_k=2,
                                       device=torch.device("cpu"), batch_size=2)
    assert res["latent_geometry"] is None
    assert res["latent_geometry_random"] is None


def test_reconstruct_geometry_uses_same_forward_windows(fake_env):
    res = dr.reconstruct_subject_stats(None, ["s1", "s2", "s3"], None, None, keep_k=2,
                                       device=torch.device("cpu"), batch_size=2, geometry=True)
    g = res["latent_geometry"]
    assert g["n_windows"] == 12 and g["n_subjects"] == 3
    assert g["n_features"] == R * D
    assert res["latent_geometry_random"] is None


def test_reconstruct_geometry_window_cap(fake_env):
    res = dr.reconstruct_subject_stats(None, ["s1", "s2", "s3"], None, None, keep_k=2,
                                       device=torch.device("cpu"), batch_size=2, geometry=True,
                                       max_windows_per_subject=2)
    assert res["latent_geometry"]["n_windows"] == 6


def test_baseline_runs_second_pass_with_random_perceiver(fake_env, monkeypatch):
    calls = []

    def fake_random(models, device, seed=0):
        calls.append(seed)
        spatial, temporal, conv, _ = models
        # Different latents from the trained fake: a scaled perceiver.
        return spatial, temporal, conv, (lambda re, te, sc, rp: (
            re.sum(-1, keepdim=True) + torch.arange(T, dtype=torch.float32), re * -3.0, None))

    monkeypatch.setattr(dr, "random_perceiver_models", fake_random)
    steps = []
    res = dr.reconstruct_subject_stats(None, ["s1", "s2", "s3"], None, None, keep_k=2,
                                       device=torch.device("cpu"), batch_size=2,
                                       geometry_baseline=True, seed=7,
                                       progress=lambda d, t: steps.append((d, t)))
    assert calls == [7]
    assert res["latent_geometry"] is not None          # baseline implies geometry
    assert res["latent_geometry_random"]["n_windows"] == 12
    # Progress: 3 subjects (pass 1) + 2 representatives (pass 2) + 3 subjects (baseline pass).
    assert steps[-1] == (8, 8)


def test_random_perceiver_models_is_seeded_and_eval():
    # Real architecture: constructor only, encoders are not used.
    a = dr.random_perceiver_models((None, None, None, None), torch.device("cpu"), seed=3)[3]
    b = dr.random_perceiver_models((None, None, None, None), torch.device("cpu"), seed=3)[3]
    c = dr.random_perceiver_models((None, None, None, None), torch.device("cpu"), seed=4)[3]
    sa, sb, sc = a.state_dict(), b.state_dict(), c.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k])
    assert any(not torch.equal(sa[k], sc[k]) for k in sa)
    assert not a.training
