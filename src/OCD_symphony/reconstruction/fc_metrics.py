"""Subject-level FC of reconstructed BOLD compared with the input FC (pure NumPy).

Conventions
-----------
* Window FC is ``np.corrcoef`` of each (R, T) window, as in ``data_preparation``.
* Subject FC = tanh(mean over windows of atanh(clip(FC_window))), a Fisher-z average.
* The diagonal (self-correlation) is excluded from every statistic and network block.
* Subject FC values are correlations (r units). Differences are ``reconstructed - input``.
"""
import warnings
from typing import Dict, List, Sequence, Tuple

import numpy as np

FISHER_CLIP = 0.999999


def _pearson(a, b) -> float:
    """Pearson r over entries finite in both arrays; NaN if undefined."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok] - a[ok].mean()
    b = b[ok] - b[ok].mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def window_fc(x) -> np.ndarray:
    """Pearson correlation matrices of each window. ``x``: (n, R, T) -> (n, R, R), float64."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected (n, R, T) windows, got shape {x.shape}")
    xc = x - x.mean(axis=-1, keepdims=True)
    norm = np.sqrt((xc ** 2).sum(-1))                       # (n, R)
    denom = norm[:, :, None] * norm[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        fc = np.einsum("nrt,nst->nrs", xc, xc) / denom
    return np.where(denom > 0, fc, np.nan)


class FCAccumulator:
    """Running Fisher-z average of window FC matrices for one subject."""

    def __init__(self, n_roi: int):
        self.z_sum = np.zeros((n_roi, n_roi), dtype=np.float64)
        self.count = np.zeros((n_roi, n_roi), dtype=np.float64)
        self.n_windows = 0

    def add(self, fc_windows) -> None:
        fc = np.asarray(fc_windows, dtype=np.float64)
        z = np.arctanh(np.clip(fc, -FISHER_CLIP, FISHER_CLIP))
        ok = np.isfinite(z)
        self.z_sum += np.where(ok, z, 0.0).sum(axis=0)
        self.count += ok.sum(axis=0)
        self.n_windows += fc.shape[0]

    def mean(self) -> np.ndarray:
        """Subject FC in r units: tanh(mean z). NaN where no window was valid."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.tanh(self.z_sum / self.count)


def network_membership(roi_networks) -> Tuple[np.ndarray, List[str]]:
    """One-hot (R, K) ROI-to-network matrix; networks ordered by first appearance."""
    labels = np.asarray(roi_networks).astype(str)
    names = list(dict.fromkeys(labels.tolist()))
    M = np.zeros((len(labels), len(names)), dtype=np.float64)
    for k, name in enumerate(names):
        M[labels == name, k] = 1.0
    return M, names


def network_blocks(S, membership: np.ndarray) -> np.ndarray:
    """(K, K) mean FC within and between networks, diagonal excluded from every block."""
    S = np.asarray(S, dtype=np.float64)
    R = S.shape[0]
    off = ~np.eye(R, dtype=bool)
    valid = np.isfinite(S) & off
    sums = membership.T @ np.where(valid, S, 0.0) @ membership
    counts = membership.T @ valid.astype(np.float64) @ membership
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def edge_stats(S_in, S_rec) -> Dict[str, float]:
    """Edge-wise agreement between two subject FC matrices (upper triangle, no diagonal)."""
    R = S_in.shape[0]
    iu = np.triu_indices(R, k=1)
    a = np.asarray(S_in, dtype=np.float64)[iu]
    b = np.asarray(S_rec, dtype=np.float64)[iu]
    ok = np.isfinite(a) & np.isfinite(b)
    d = b[ok] - a[ok]
    if d.size == 0:
        nan = float("nan")
        return {"r": nan, "mae": nan, "bias": nan, "rmse": nan, "n_edges": 0}
    return {
        "r": _pearson(a, b),
        "mae": float(np.abs(d).mean()),
        "bias": float(d.mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "n_edges": int(ok.sum()),
    }


def roi_profile_r(S_in, S_rec) -> np.ndarray:
    """Per ROI, Pearson r between its input and reconstructed FC rows (diagonal excluded)."""
    S_in = np.asarray(S_in, dtype=np.float64)
    S_rec = np.asarray(S_rec, dtype=np.float64)
    R = S_in.shape[0]
    out = np.full(R, np.nan)
    for i in range(R):
        keep = np.arange(R) != i
        out[i] = _pearson(S_in[i, keep], S_rec[i, keep])
    return out


def subject_fc_metrics(S_in, S_rec, membership: np.ndarray) -> Dict[str, object]:
    """All per-subject FC quantities used by the summary."""
    if membership.shape[0] != S_in.shape[0]:
        raise ValueError(
            f"network labels ({membership.shape[0]}) do not match ROI count ({S_in.shape[0]})")
    return {
        "edge": edge_stats(S_in, S_rec),
        "profile": roi_profile_r(S_in, S_rec),
        "net_in": network_blocks(S_in, membership),
        "net_rec": network_blocks(S_rec, membership),
    }


def upper_vector(S) -> np.ndarray:
    """Upper-triangle entries (diagonal excluded) of an (R, R) matrix, as a 1-D vector."""
    S = np.asarray(S, dtype=np.float64)
    iu = np.triu_indices(S.shape[0], k=1)
    return S[iu]


class EdgeAcrossSubjectsAccumulator:
    """Streaming per-edge Pearson r across subjects (each edge's group mean removed).

    Group-mean removal is implicit: Pearson r is invariant to subtracting a constant from
    each variable, so the centred cross-products are computed from sums. Each edge keeps
    its own shift (the first finite value seen) to limit cancellation. No N x E matrix
    is stored; memory is a few float64 vectors of length E.

    ``add`` takes the upper-triangle vectors of one subject's input and reconstructed FC.
    ``r_per_edge`` returns (E,) r values, NaN where fewer than 3 subjects are valid or the
    input or reconstruction has zero between-subject variance on that edge.
    """

    MIN_SUBJECTS = 3

    def __init__(self, n_edges: int):
        self.n_edges = int(n_edges)
        self.n_subjects = 0
        self.shift_a = None
        self.shift_b = None
        self.n = np.zeros(self.n_edges, dtype=np.float64)
        self.sa = np.zeros(self.n_edges, dtype=np.float64)
        self.sb = np.zeros(self.n_edges, dtype=np.float64)
        self.saa = np.zeros(self.n_edges, dtype=np.float64)
        self.sbb = np.zeros(self.n_edges, dtype=np.float64)
        self.sab = np.zeros(self.n_edges, dtype=np.float64)

    def add(self, a, b) -> None:
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        if a.size != self.n_edges or b.size != self.n_edges:
            raise ValueError(f"expected {self.n_edges} edges, got {a.size} and {b.size}")
        ok = np.isfinite(a) & np.isfinite(b)
        if self.shift_a is None:
            self.shift_a = np.where(ok, a, 0.0)
            self.shift_b = np.where(ok, b, 0.0)
        da = np.where(ok, a - self.shift_a, 0.0)
        db = np.where(ok, b - self.shift_b, 0.0)
        self.n += ok
        self.sa += da
        self.sb += db
        self.saa += da * da
        self.sbb += db * db
        self.sab += da * db
        self.n_subjects += 1

    def r_per_edge(self) -> np.ndarray:
        n = self.n
        with np.errstate(divide="ignore", invalid="ignore"):
            cov = self.sab - self.sa * self.sb / n
            va = self.saa - self.sa * self.sa / n
            vb = self.sbb - self.sb * self.sb / n
            r = cov / np.sqrt(va * vb)
        valid = (n >= self.MIN_SUBJECTS) & (va > 0) & (vb > 0) & np.isfinite(r)
        return np.where(valid, np.clip(r, -1.0, 1.0), np.nan)


def across_subject_edge_r(A, B) -> np.ndarray:
    """Reference (non-streaming) per-edge r across subjects. ``A``, ``B``: (N, E).

    Each edge's group mean is subtracted explicitly, then r is computed per column.
    Used for validation of ``EdgeAcrossSubjectsAccumulator``.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    out = np.full(A.shape[1], np.nan)
    for e in range(A.shape[1]):
        a, b = A[:, e], B[:, e]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < EdgeAcrossSubjectsAccumulator.MIN_SUBJECTS:
            continue
        out[e] = _pearson(a[ok] - a[ok].mean(), b[ok] - b[ok].mean())
    return out


def edge_across_subjects_summary(acc: EdgeAcrossSubjectsAccumulator) -> Dict[str, object]:
    """Subject-level (specificity) summary of an accumulator: r_per_edge plus its statistics."""
    r_edge = acc.r_per_edge()
    finite = r_edge[np.isfinite(r_edge)]
    return {
        "r_per_edge": r_edge.astype(np.float32),
        "n_subjects": int(acc.n_subjects),
        "n_edges": int(acc.n_edges),
        "n_valid_edges": int(finite.size),
        "stats": aggregate(finite),
    }


def rank_rois(profile, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Indices of the ``k`` highest (best) and ``k`` lowest (worst) profile values.

    NaN values are ignored. Ties are broken by ROI index, so the result is deterministic.
    If fewer than ``2k`` ROIs are valid, best and worst can overlap.
    """
    p = np.asarray(profile, dtype=np.float64)
    idx = np.arange(p.size)
    ok = np.isfinite(p)
    i_ok, v = idx[ok], p[ok]
    best = i_ok[np.lexsort((i_ok, -v))][:k]
    worst = i_ok[np.lexsort((i_ok, v))][:k]
    return best, worst


def aggregate(values) -> Dict[str, float]:
    """NaN-aware summary statistics across subjects."""
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "median": nan, "std": nan, "min": nan, "max": nan,
                "q25": nan, "q75": nan}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
    }


def _nanmean_axis0(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, axis=0)


def summarise_fc(subject_ids: Sequence[str], edges: List[Dict[str, float]],
                 profiles: np.ndarray, net_in: np.ndarray, net_rec: np.ndarray,
                 net_names: List[str], k: int = 10) -> Dict[str, object]:
    """Collect per-subject FC results and the cross-subject summary.

    ``profiles`` (N, R); ``net_in`` / ``net_rec`` (N, K, K).
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    net_in = np.asarray(net_in, dtype=np.float64)
    net_rec = np.asarray(net_rec, dtype=np.float64)
    per_subject = {m: np.asarray([e[m] for e in edges], dtype=np.float64)
                   for m in ("r", "mae", "bias", "rmse")}
    profile_mean = _nanmean_axis0(profiles)
    best, worst = rank_rois(profile_mean, k=k)
    return {
        "subjects": list(subject_ids),
        "edge": per_subject,
        "edge_summary": {m: aggregate(v) for m, v in per_subject.items()},
        "roi_profile": profiles,
        "roi_profile_mean": profile_mean,
        "roi_best": best,
        "roi_worst": worst,
        "net_names": list(net_names),
        "net_in": _nanmean_axis0(net_in),
        "net_rec": _nanmean_axis0(net_rec),
        "net_diff": _nanmean_axis0(net_rec - net_in),
    }
