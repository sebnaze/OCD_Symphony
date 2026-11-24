# Paper-matched ROI-level fMRI cleaning (BrainSymphony preprint S1.1).
#
# Order inside one run (full run, flagged frames included, paper order):
#   detrend (linear, per ROI) -> band-pass (zero-phase Butterworth)
#   -> motion regression (6 rigid parameters) -> framewise displacement flagging.
# Flagged frames are then linearly interpolated, the run is cut into 200-TR windows and
# windows with too many flagged frames are discarded.
#
# Parameters not stated by the paper (filter order/type, motion parameter set) are fixed
# here as documented defaults. WM/CSF regression is not performed (no segmentation for HCP).

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal

# Power et al. (2012): rotations are converted to arc length on a sphere of this radius (mm).
POWER_HEAD_RADIUS_MM = 50.0

@dataclass(frozen=True)
class CleaningConfig:
    fd_threshold: float = 0.5       # mm; frames with FD > threshold are flagged
    window_size: int = 200          # TR per window
    max_flagged_per_window: int = 5  # windows with more flagged frames are discarded
    min_windows: int = 4            # N: subject excluded if fewer kept windows remain (HCP-YA)
    band_low: float = 0.008         # Hz
    band_high: float = 0.09         # Hz
    filter_order: int = 2           # Butterworth order (applied forward-backward, zero phase)

    def as_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Elementary operations
# ---------------------------------------------------------------------------

def detrend_linear(ts: np.ndarray) -> np.ndarray:
    """Remove a least-squares linear trend from each ROI. ``ts`` is (R, T)."""
    ts = np.asarray(ts, dtype=np.float64)
    return signal.detrend(ts, axis=-1, type="linear")


def bandpass(ts: np.ndarray, tr: float, low: float = 0.008, high: float = 0.09,
             order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth band-pass per ROI. ``ts`` is (R, T), ``tr`` in seconds."""
    ts = np.asarray(ts, dtype=np.float64)
    fs = 1.0 / float(tr)
    nyq = fs / 2.0
    if not (0 < low < high < nyq):
        raise ValueError(f"band ({low}, {high}) Hz invalid for TR={tr}s (Nyquist {nyq:.3f} Hz)")
    sos = signal.butter(order, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, ts, axis=-1)


def regress_out(ts: np.ndarray, confounds: np.ndarray) -> np.ndarray:
    """Least-squares removal of ``confounds`` (T, K) (plus an intercept) from each ROI.

    ``ts`` is (R, T). Returns the residuals with the same shape.
    """
    ts = np.asarray(ts, dtype=np.float64)
    conf = np.asarray(confounds, dtype=np.float64)
    if conf.ndim == 1:
        conf = conf[:, None]
    if conf.shape[0] != ts.shape[-1]:
        raise ValueError(f"confounds have {conf.shape[0]} rows, series has {ts.shape[-1]} TR")
    X = np.column_stack([np.ones(conf.shape[0]), conf])
    beta, *_ = np.linalg.lstsq(X, ts.T, rcond=None)
    return ts - (X @ beta).T


def framewise_displacement_power(rigid: np.ndarray, radius_mm: float = POWER_HEAD_RADIUS_MM
                                 ) -> np.ndarray:
    """Power et al. (2012) framewise displacement from six rigid-body parameters.

    ``rigid`` is (T, 6): three rotation columns and three translation columns. The first three
    columns are treated as rotations and multiplied by ``radius_mm``; the last three are
    translations in mm. FD = sum |diff| over the six parameters. FD[0] is 0.

    ``radius_mm`` must match the unit of the rotation columns. For radians use the head radius
    (50 mm). HCP-YA ``Movement_Regressors.txt`` rotation columns are on a mm-equivalent scale:
    their L2 displacement matches HCP's ``Movement_RelativeRMS.txt`` (Pearson r = 0.993) only
    with radius 1, while radius 50 gives mean FD of about 2 mm and r = 0.62.
    """
    rigid = np.asarray(rigid, dtype=np.float64)
    if rigid.ndim != 2 or rigid.shape[1] != 6:
        raise ValueError(f"expected (T, 6) rigid parameters, got {rigid.shape}")
    d = np.diff(rigid, axis=0, prepend=rigid[:1])
    rot_mm = radius_mm * np.abs(d[:, :3])
    trans_mm = np.abs(d[:, 3:])
    fd = rot_mm.sum(axis=1) + trans_mm.sum(axis=1)
    fd[0] = 0.0
    return fd


def flag_frames(fd: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Boolean mask of frames whose framewise displacement exceeds ``threshold`` (mm)."""
    fd = np.asarray(fd, dtype=np.float64)
    flags = fd > threshold
    flags[~np.isfinite(fd)] = True  # unknown displacement is treated as flagged
    return flags


def interpolate_flagged(ts: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Replace flagged frames by linear interpolation between unflagged neighbours.

    ``ts`` is (R, T); ``flags`` is (T,) boolean. Frames at the run edges take the nearest
    unflagged value. Raises if every frame is flagged.
    """
    ts = np.asarray(ts, dtype=np.float64)
    flags = np.asarray(flags, dtype=bool)
    good = np.flatnonzero(~flags)
    if good.size == 0:
        raise ValueError("all frames are flagged; nothing to interpolate from")
    if good.size == ts.shape[-1]:
        return ts.copy()
    t = np.arange(ts.shape[-1])
    out = np.empty_like(ts)
    for r in range(ts.shape[0]):
        out[r] = np.interp(t, good, ts[r, good])
    return out


# ---------------------------------------------------------------------------
# Run-level and window-level pipeline
# ---------------------------------------------------------------------------

@dataclass
class CleanedRun:
    series: np.ndarray        # (R, T) cleaned and interpolated series (full run)
    flags: np.ndarray         # (T,) bool, frames flagged by FD
    fd: np.ndarray            # (T,) framewise displacement (mm)


def clean_run(ts: np.ndarray, tr: float, confounds: Optional[np.ndarray], fd: np.ndarray,
              cfg: CleaningConfig = CleaningConfig()) -> CleanedRun:
    """Detrend -> band-pass -> motion regression -> FD flagging on one full run.

    ``confounds`` is (T, K) motion regressors (may be None to skip regression).
    Flagged frames are kept in every filtering step (paper order). The returned
    ``series`` is interpolated at flagged frames.
    """
    ts = np.asarray(ts, dtype=np.float64)
    if not np.isfinite(ts).all():
        raise ValueError("non-finite values in raw ROI series")
    fd = np.asarray(fd, dtype=np.float64)
    if fd.shape[0] != ts.shape[-1]:
        raise ValueError(f"FD has {fd.shape[0]} frames, series has {ts.shape[-1]}")

    x = detrend_linear(ts)
    x = bandpass(x, tr, cfg.band_low, cfg.band_high, cfg.filter_order)
    if confounds is not None:
        x = regress_out(x, confounds)
    flags = flag_frames(fd, cfg.fd_threshold)
    if flags.any() and not flags.all():
        x = interpolate_flagged(x, flags)
    # If every frame is flagged no window can survive window_run; keep the series as is.
    return CleanedRun(series=x, flags=flags, fd=fd)


def window_run(run: CleanedRun, cfg: CleaningConfig = CleaningConfig()
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut a cleaned run into non-overlapping windows and discard flagged ones.

    Returns ``(windows, n_flagged, starts)`` where ``windows`` is (n_kept, R, W),
    ``n_flagged`` (n_kept,) gives the number of FD-flagged frames of each kept window and
    ``starts`` (n_kept,) the first TR of each kept window. Windows with more than
    ``cfg.max_flagged_per_window`` flagged frames are dropped. Incomplete trailing frames
    (T mod W) are dropped as in the legacy pipeline.
    """
    W = cfg.window_size
    R, T = run.series.shape
    n_full = T // W
    keep_w, keep_nf, keep_s = [], [], []
    for k in range(n_full):
        s = k * W
        nf = int(run.flags[s:s + W].sum())
        if nf > cfg.max_flagged_per_window:
            continue
        keep_w.append(run.series[:, s:s + W])
        keep_nf.append(nf)
        keep_s.append(s)
    if not keep_w:
        return (np.zeros((0, R, W), dtype=np.float64), np.zeros(0, dtype=int),
                np.zeros(0, dtype=int))
    return np.stack(keep_w), np.asarray(keep_nf, dtype=int), np.asarray(keep_s, dtype=int)


def window_fc(windows: np.ndarray) -> np.ndarray:
    """Pearson correlation matrices (n, R, R) of windows (n, R, W)."""
    return np.array([np.corrcoef(w) for w in windows]) if len(windows) else \
        np.zeros((0, windows.shape[1], windows.shape[1]))


def subject_passes(n_windows_kept: int, cfg: CleaningConfig = CleaningConfig()) -> bool:
    """Subject is kept only if it retains at least ``cfg.min_windows`` windows."""
    return n_windows_kept >= cfg.min_windows


def process_runs(runs: Sequence[Dict], cfg: CleaningConfig = CleaningConfig()) -> Dict:
    """Clean and window a list of runs of one subject.

    Each run dict needs: ``ts`` (R, T), ``tr`` (s), ``fd`` (T,), optional ``confounds`` (T, K),
    and ``session`` (run index). Returns a dict with stacked ``fmri`` (n, R, W), ``fc``
    (n, R, R), ``session`` (n,), ``n_flagged`` (n,), ``window_start`` (n,) and per-run
    ``run_stats``. ``excluded`` is True if fewer than ``cfg.min_windows`` windows survive.
    """
    fmri, fc, sess, nflag, starts, stats = [], [], [], [], [], []
    for run in runs:
        cr = clean_run(run["ts"], run["tr"], run.get("confounds"), run["fd"], cfg)
        windows, nf, st = window_run(cr, cfg)
        stats.append(dict(session=int(run["session"]), n_frames=int(cr.series.shape[1]),
                          n_flagged=int(cr.flags.sum()), n_windows_kept=int(len(windows)),
                          n_windows_total=int(cr.series.shape[1] // cfg.window_size)))
        if len(windows) == 0:
            continue
        fmri.append(windows)
        fc.append(window_fc(windows))
        sess.append(np.full(len(windows), int(run["session"]), dtype=np.int64))
        nflag.append(nf)
        starts.append(st)
    n_kept = int(sum(len(w) for w in fmri))
    out = dict(run_stats=stats, n_windows=n_kept, excluded=not subject_passes(n_kept, cfg))
    if n_kept:
        out.update(
            fmri=np.concatenate(fmri, axis=0),
            fc=np.concatenate(fc, axis=0),
            session=np.concatenate(sess),
            n_flagged=np.concatenate(nflag),
            window_start=np.concatenate(starts),
        )
    return out


def cleaning_metadata(cfg: CleaningConfig, tr: float, sources: Dict) -> Dict:
    """Metadata written next to every paper-mode output folder."""
    return dict(
        preprocessing="paper",
        steps=["detrend_linear", "bandpass_butterworth_zero_phase", "motion_regression_6rigid",
               "fd_flagging", "linear_interpolation_flagged", "window_200TR_discard"],
        not_applied=["wm_csf_regression (no segmentation for HCP-YA)",
                     "per-window demeaning (legacy OCD only)",
                     "sigma_robust division (legacy only)"],
        tr_seconds=float(tr),
        cleaning=cfg.as_dict(),
        motion_parameters="6 rigid, no derivatives",
        sources=sources,
    )
