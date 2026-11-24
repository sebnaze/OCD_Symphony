import numpy as np
import pytest

from OCD_symphony.data_preparation.fmri_cleaning import (
    CleaningConfig, bandpass, clean_run, detrend_linear, flag_frames,
    framewise_displacement_power, interpolate_flagged, process_runs, regress_out,
    window_run,
)

TR = 0.72


def _t(n):
    return np.arange(n) * TR


def test_detrend_removes_linear_trend_exactly():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((3, 400))
    trend = np.outer([1.0, -2.0, 0.5], np.linspace(-3, 7, 400))
    out = detrend_linear(base + trend)
    np.testing.assert_allclose(out, detrend_linear(base), atol=1e-9)
    # residual has no linear component left
    t = np.arange(400)
    slope = np.polyfit(t, out[0], 1)[0]
    assert abs(slope) < 1e-9


def test_bandpass_keeps_in_band_and_removes_out_of_band():
    n = 1200
    t = _t(n)
    in_band = np.sin(2 * np.pi * 0.03 * t)      # 0.03 Hz, inside 0.008-0.09
    too_fast = np.sin(2 * np.pi * 0.25 * t)     # above 0.09 Hz
    too_slow = np.sin(2 * np.pi * 0.001 * t)    # below 0.008 Hz
    out = bandpass(np.vstack([in_band, too_fast, too_slow]), TR)
    mid = slice(200, -200)  # ignore edge transients
    assert np.std(out[0, mid]) == pytest.approx(np.std(in_band[mid]), rel=0.05)
    assert np.std(out[1, mid]) < 0.02
    assert np.std(out[2, mid]) < 0.05


def test_bandpass_is_zero_phase():
    n = 1200
    t = _t(n)
    x = np.sin(2 * np.pi * 0.03 * t)
    y = bandpass(x[None], TR)[0]
    mid = slice(200, -200)
    # peak of the cross-correlation lies at zero lag
    lags = np.arange(-20, 21)
    cc = [np.dot(x[mid], np.roll(y, l)[mid]) for l in lags]
    assert lags[int(np.argmax(cc))] == 0


def test_bandpass_rejects_band_above_nyquist():
    with pytest.raises(ValueError):
        bandpass(np.zeros((1, 100)), tr=2.0, high=0.3)


def test_regress_out_removes_motion_component():
    rng = np.random.default_rng(1)
    T = 500
    motion = rng.standard_normal((T, 6))
    noise = rng.standard_normal((2, T)) * 0.1
    ts = noise + np.outer([1.0, 2.0], motion[:, 0]) + np.outer([0.5, -1.0], motion[:, 3])
    res = regress_out(ts, motion)
    assert res.shape == ts.shape
    # residuals orthogonal to motion regressors
    for r in res:
        corr = np.abs(np.corrcoef(r, motion[:, 0])[0, 1])
        assert corr < 0.05
    assert np.std(res) == pytest.approx(0.1, rel=0.2)


def test_fd_power_on_synthetic_motion_trace():
    T = 10
    rigid = np.zeros((T, 6))
    rigid[5, 3] = 1.0              # 1 mm translation in x at frame 5
    rigid[7, 0] = 0.02             # 0.02 rad rotation about x at frame 7 -> 1 mm on 50 mm radius
    fd = framewise_displacement_power(rigid)
    assert fd[0] == 0
    assert fd[5] == pytest.approx(1.0)
    assert fd[6] == pytest.approx(1.0)  # jump back to zero at frame 6
    assert fd[7] == pytest.approx(1.0)
    assert fd[8] == pytest.approx(1.0)
    assert fd[9] == 0


def test_fd_radius_scales_rotation_only():
    rigid = np.zeros((5, 6))
    rigid[1, 0] = 0.02   # rotation only
    rigid[3, 3] = 0.5    # translation only (frame 3 vs 2 has no rotation change)
    fd50 = framewise_displacement_power(rigid, radius_mm=50.0)
    fd1 = framewise_displacement_power(rigid, radius_mm=1.0)
    assert fd50[1] == pytest.approx(1.0)
    assert fd1[1] == pytest.approx(0.02)
    assert fd50[3] == pytest.approx(0.5)
    assert fd1[3] == pytest.approx(0.5)


def test_fd_rejects_wrong_shape():
    with pytest.raises(ValueError):
        framewise_displacement_power(np.zeros((10, 12)))


def test_flag_frames_threshold_is_strict():
    fd = np.array([0.0, 0.5, 0.51, 0.2, np.nan])
    np.testing.assert_array_equal(flag_frames(fd, 0.5), [False, False, True, False, True])


def test_interpolate_flagged_recovers_linear_signal():
    T = 100
    truth = np.linspace(0, 10, T)
    ts = np.vstack([truth, 2 * truth])
    flags = np.zeros(T, bool)
    flags[[10, 11, 50, 90]] = True
    corrupted = ts.copy()
    corrupted[:, flags] = 1e6
    out = interpolate_flagged(corrupted, flags)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, ts, atol=1e-9)


def test_interpolate_flagged_requires_some_good_frame():
    with pytest.raises(ValueError):
        interpolate_flagged(np.zeros((1, 5)), np.ones(5, bool))


def _run(T=1000, R=4, seed=0, fd=None):
    rng = np.random.default_rng(seed)
    ts = rng.standard_normal((R, T)) + 100.0
    if fd is None:
        fd = np.zeros(T)
    return ts, fd


def test_clean_run_has_no_nan_and_flags_counted():
    ts, fd = _run()
    fd = fd.copy()
    fd[[300, 301, 555]] = 2.0
    cr = clean_run(ts, TR, None, fd)
    assert np.isfinite(cr.series).all()
    assert cr.flags.sum() == 3
    assert cr.series.shape == ts.shape


def test_window_discard_rule_drops_more_than_five_flagged():
    T = 1000
    cfg = CleaningConfig()
    ts = np.zeros((2, T))
    fd = np.zeros(T)
    # window 0 (frames 0-199): 5 flagged -> kept; window 1 (200-399): 6 flagged -> dropped
    fd[0:200:40] = 1.0          # 5 flagged frames
    fd[200:200 + 6] = 1.0       # 6 flagged frames
    cr = clean_run(ts, TR, None, fd, cfg)
    windows, nf, starts = window_run(cr, cfg)
    assert 0 in starts
    assert 200 not in starts
    assert nf[list(starts).index(0)] == 5
    assert windows.shape[1:] == (2, 200)
    assert len(windows) == 4  # 5 windows minus the dropped one


def test_window_drops_incomplete_trailing_frames():
    ts = np.zeros((1, 450))
    cr = clean_run(ts, TR, None, np.zeros(450))
    windows, _, starts = window_run(cr)
    assert len(windows) == 2
    assert list(starts) == [0, 200]


def test_subject_excluded_below_min_windows():
    cfg = CleaningConfig(min_windows=5)
    runs = [dict(ts=np.random.default_rng(i).standard_normal((3, 400)), tr=TR,
                 fd=np.zeros(400), session=i) for i in range(2)]
    out = process_runs(runs, cfg)
    assert out["n_windows"] == 4
    assert out["excluded"] is True


def test_process_runs_shapes_and_sessions():
    runs = [dict(ts=np.random.default_rng(i).standard_normal((3, 600)), tr=TR,
                 fd=np.zeros(600), session=i) for i in range(3)]
    out = process_runs(runs)
    assert out["excluded"] is False
    assert out["fmri"].shape == (9, 3, 200)
    assert out["fc"].shape == (9, 3, 3)
    assert set(out["session"].tolist()) == {0, 1, 2}
    # windows never cross run boundaries: each run contributes exactly 3 windows
    assert [int((out["session"] == s).sum()) for s in range(3)] == [3, 3, 3]
