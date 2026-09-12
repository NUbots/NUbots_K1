#!/usr/bin/env python3
#
# MIT License
#
# Copyright (c) 2026 NUbots
#
# This file is part of the NUbots codebase.
# See https://github.com/NUbots/NUbots for further info.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
"""
Fit NUSim's odometry error model (NUSim mujoco/config/odometry.yaml) to a recording of the K1's
odometry against motion capture ground truth.

Record with the data/odometry_mocap role: walk the robot around under the OptiTrack rig with the
keyboard for a few minutes, forwards, sideways and turning, at the speeds it walks in games. Then

    ./b nbs odometry_fit recordings/odometry_mocap/<recording>.nbs

compares the controller's odometry (BoosterOdometry, rt/odometer_state) with the robot's mocap pose
(RobotPoseGroundTruth, from localisation::Mocap) and prints the model's parameters as odometry.yaml.

The model, per planar body axis (forward x, sideways y, yaw), is v_est = scale * v_true + bias +
white noise, with the bias a Gauss-Markov process (standard deviation b, time constant tau). So:

  - scale is the ratio of the odometry's displacement to the true displacement, in the body frame;
  - after correcting the scale, the odometry's error over a window of T seconds has variance
        floor + s^2 T + 2 b^2 tau^2 (T / tau - 1 + exp(-T / tau)),
    which is fitted over a range of window lengths for the white noise density s, b and tau (the
    floor soaks up the mocap's own noise).

The mocap rigid body need not be aligned with the robot: the heading offset between its axes and the
robot's is fitted, and so is any constant time offset between the two streams. The recording is split
wherever the odometry resets (on startup and mode changes), the mocap loses the robot, or either stream
has a gap. Windows in which the robot barely moves are left out, as a standing robot's odometry does
not drift like a walking one's; their drift is reported separately.

`--self-test` runs the fit on odometry simulated with NUSim's model and known parameters instead.
"""

import math

import numpy as np

from utility.nbs import LinearDecoder

AXES = ("x", "y", "yaw")
UNITS = {"density": ("m/sqrt(s)", "m/sqrt(s)", "rad/sqrt(s)"), "bias": ("m/s", "m/s", "rad/s")}


def register(command):
    command.description = "Fit NUSim's odometry error model to a recording of odometry against motion capture"
    command.add_argument("files", metavar="files", nargs="*", help="The nbs recordings to fit")
    command.add_argument(
        "--rigid-body-id",
        type=int,
        default=None,
        help="Only use times the mocap tracked this rigid body (Mocap.yaml robot_rigid_body_id); default: any",
    )
    command.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
        help="Window lengths (s) over which the odometry error growth is measured",
    )
    command.add_argument(
        "--scale-window", type=float, default=2.0, help="Window length (s) the scale factors are fitted over"
    )
    command.add_argument(
        "--min-speed",
        type=float,
        default=0.03,
        help="Windows in which the robot moves slower than this (m/s) are treated as standing",
    )
    command.add_argument(
        "--max-time-offset", type=float, default=0.3, help="Largest time offset (s) searched between the streams"
    )
    command.add_argument("--output", help="Also write the fitted parameters to this odometry.yaml")
    command.add_argument(
        "--self-test",
        action="store_true",
        help="Fit odometry simulated with NUSim's model and known parameters instead of a recording",
    )


# --------------------------------------------------------------------------------------------------
# Loading


def _planar_from_iso3(m):
    """(x, y, yaw) of the frame whose pose in the field is the iso3 m (columns x, y, z, t)."""
    return m.t.x, m.t.y, math.atan2(m.x.y, m.x.x)


def load(files, rigid_body_id):
    """Odometry and ground truth series from the recordings, as (times, poses) arrays, and the times
    of mode changes (which reset the controller's odometry)."""
    odometry, truth, tracking, modes = [], [], [], []
    types = [
        "message.booster.BoosterOdometry",
        "message.booster.BoosterModeState",
        "message.localisation.RobotPoseGroundTruth",
        "message.input.MotionCapture",
    ]
    for packet in LinearDecoder(*files, types=types, show_progress=True):
        t = packet.emit_timestamp * 1e-6
        name = packet.type.name
        if name == "message.booster.BoosterOdometry":
            odometry.append((t, packet.msg.x, packet.msg.y, packet.msg.theta))
        elif name == "message.booster.BoosterModeState":
            modes.append((t, int(packet.msg.mode)))
        elif name == "message.localisation.RobotPoseGroundTruth":
            truth.append((t, *_planar_from_iso3(packet.msg.Hft)))
        elif rigid_body_id is not None:
            bodies = [b for b in packet.msg.rigid_bodies if b.id == rigid_body_id]
            tracking.append((t, bool(bodies) and bodies[0].tracking_valid))

    if not odometry:
        raise SystemExit("No message.booster.BoosterOdometry in the recording")
    if not truth:
        raise SystemExit(
            "No message.localisation.RobotPoseGroundTruth in the recording: run input::NatNet and "
            "localisation::Mocap (the data/odometry_mocap role) under the mocap rig"
        )
    odometry = np.array(sorted(odometry))
    truth = np.array(sorted(truth))

    # Drop ground truth the mocap was not tracking the robot for
    if tracking:
        tracking = np.array(sorted(tracking))
        idx = np.clip(np.searchsorted(tracking[:, 0], truth[:, 0]), 0, len(tracking) - 1)
        truth = truth[tracking[idx, 1] > 0.5]

    modes.sort()
    resets = [t for (t, mode), (_, previous) in zip(modes[1:], modes) if mode != previous]
    return odometry, truth, np.array(resets)


def segments(odometry, truth, time_offset=0.0, resets=(), max_gap=0.25, reset_jump=0.25):
    """Split into continuous stretches, each (times, odometry poses, true poses at the same times).

    Splits at the reset times, wherever the odometry jumps (a reset too) or either stream has a gap,
    and resamples the ground truth at the odometry's times, unwrapping both headings.
    """
    t_odo, t_gt = odometry[:, 0], truth[:, 0] + time_offset

    breaks = np.zeros(len(t_odo), dtype=bool)
    step = np.hypot(np.diff(odometry[:, 1]), np.diff(odometry[:, 2]))
    turn = np.abs(np.angle(np.exp(1j * np.diff(odometry[:, 3]))))
    breaks[1:] = (np.diff(t_odo) > max_gap) | (step > reset_jump) | (turn > 2 * reset_jump)
    # The first odometry sample after each mode change (the reset is not always a visible jump)
    after = np.searchsorted(t_odo, np.asarray(resets))
    breaks[after[(after > 0) & (after < len(t_odo))]] = True

    # Ground truth near each odometry sample: the gap between the truth samples around it
    idx = np.searchsorted(t_gt, t_odo)
    ok = (idx > 0) & (idx < len(t_gt))
    gap = np.full(len(t_odo), np.inf)
    gap[ok] = t_gt[idx[ok]] - t_gt[idx[ok] - 1]
    covered = gap <= max_gap

    out = []
    start = 0
    for end in list(np.flatnonzero(breaks)) + [len(t_odo)]:
        mask = covered[start:end]
        # Within a stretch, further split wherever the ground truth drops out
        run_start = None
        for i, good in enumerate(list(mask) + [False]):
            if good and run_start is None:
                run_start = i
            elif not good and run_start is not None:
                a, b = start + run_start, start + i
                if b - a >= 10:
                    t = t_odo[a:b]
                    odo = odometry[a:b, 1:].copy()
                    odo[:, 2] = np.unwrap(odo[:, 2])
                    yaw_gt = np.unwrap(truth[:, 3])
                    gt = np.stack(
                        [np.interp(t, t_gt, truth[:, 1]), np.interp(t, t_gt, truth[:, 2]), np.interp(t, t_gt, yaw_gt)],
                        axis=1,
                    )
                    out.append((t, odo, gt))
                run_start = None
        start = end
    return out


# --------------------------------------------------------------------------------------------------
# Fitting


def _rotate(v, h):
    """Rotate the rows (x, y) of v by the angles h."""
    c, s = np.cos(h), np.sin(h)
    return np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], axis=1)


def _odometer(pose, heading_offset=0.0):
    """Cumulative per-axis distance: each step in the body frame at its midpoint heading, summed.

    Returns an (n, 3) array whose rows are the running sums of the body-frame x and y steps and the
    heading change. The difference over a window is the integral of each axis' body velocity over it,
    so it carries exactly the model's per-axis velocity error (a pose difference would also carry the
    heading error turning the path, and mix the axes whenever the robot turns).
    """
    mid = pose[:-1, 2] + 0.5 * np.diff(pose[:, 2]) - heading_offset
    steps = np.column_stack([_rotate(np.diff(pose[:, :2], axis=0), -mid), np.diff(pose[:, 2])])
    return np.vstack([np.zeros(3), np.cumsum(steps, axis=0)])


def _window_bounds(t, window, stride):
    starts = np.arange(t[0], t[-1] - window, stride)
    i0 = np.searchsorted(t, starts)
    i1 = np.searchsorted(t, starts + window)
    keep = (i1 < len(t)) & (np.abs(t[np.minimum(i1, len(t) - 1)] - t[i0] - window) < 0.1 * window + 0.05)
    return i0[keep], i1[keep]


def increments(segs, window, stride, heading_offset=0.0):
    """Per-axis odometer distances over windows of `window` seconds, starting every `stride` seconds.

    Returns (odometry, truth) arrays of (x, y, yaw) distances (see _odometer). The odometry's body frame
    is its own heading; the truth's is the mocap body's heading less heading_offset, the rotation of
    the mocap body's axes from the robot's.
    """
    odo_inc, gt_inc = [], []
    for t, odo, gt in segs:
        if t[-1] - t[0] < window:
            continue
        i0, i1 = _window_bounds(t, window, stride)
        c_odo, c_gt = _odometer(odo), _odometer(gt, heading_offset)
        odo_inc.append(c_odo[i1] - c_odo[i0])
        gt_inc.append(c_gt[i1] - c_gt[i0])
    if not odo_inc:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(odo_inc), np.concatenate(gt_inc)


def fit_time_offset(odometry, truth, resets, max_offset):
    """The time offset of the ground truth that best matches the heading changes over 0.5 s."""
    best = (np.inf, 0.0)
    for offset in np.arange(-max_offset, max_offset + 1e-9, 0.01):
        o, g = increments(segments(odometry, truth, offset, resets), 0.5, 0.25)
        if len(o) > 20:
            err = np.mean((o[:, 2] - g[:, 2]) ** 2)
            best = min(best, (err, offset))
    return best[1]


def fit_heading_offset(segs, window, min_motion):
    """Rotation of the mocap body's axes from the robot's, fitted jointly with the x and y scales.

    Minimises the odometry's x and y distances' squared error from the scaled true distances: the
    scales differ per axis, so the rotation cannot be fitted on its own.
    """
    o, _ = increments(segs, window, window / 2)

    def residual(offset):
        _, g = increments(segs, window, window / 2, offset)
        moving = np.hypot(g[:, 0], g[:, 1]) > min_motion
        o_m, g_m = o[moving, :2], g[moving, :2]
        s = np.sum(o_m * g_m, axis=0) / np.sum(g_m**2, axis=0)
        # Negative scales fit the offset half a turn away just as well; the scales are positive
        return np.sum((o_m - s * g_m) ** 2) / max(len(o_m), 1) if np.all(s > 0) else np.inf

    grid = np.radians(np.arange(-180.0, 180.0, 2.0))
    best = grid[np.argmin([residual(h) for h in grid])]
    fine = best + np.radians(np.arange(-2.0, 2.0, 0.05))
    return float(fine[np.argmin([residual(h) for h in fine])])


def fit_scale(o, g, axis, min_motion):
    """Least squares ratio of the odometry's displacement to the truth's, or None without motion."""
    moving = np.abs(g[:, axis]) > min_motion
    if moving.sum() < 10:
        return None
    return float(np.sum(o[moving, axis] * g[moving, axis]) / np.sum(g[moving, axis] ** 2))


def _gauss_markov(T, tau):
    """Pose error variance growth of a unit Gauss-Markov velocity bias over a window of T seconds."""
    return 2 * tau**2 * (T / tau - 1 + np.exp(-T / tau))


def _nnls(A, v):
    """Non-negative least squares by enumerating the active set (A has few columns)."""
    n = A.shape[1]
    best = (np.inf, np.zeros(n))
    for mask in range(1, 2**n):
        cols = [i for i in range(n) if mask >> i & 1]
        x, *_ = np.linalg.lstsq(A[:, cols], v, rcond=None)
        if np.all(x >= 0):
            full = np.zeros(n)
            full[cols] = x
            r = np.sum((A @ full - v) ** 2)
            if r < best[0]:
                best = (r, full)
    return best


def fit_noise(windows, variances):
    """Fit floor + s^2 T + b^2 g_tau(T) to the variance at each window length T, in relative error.

    Returns (s, b, tau, floor, relative rms error).
    """
    T, v = np.asarray(windows), np.asarray(variances)
    best = None
    for tau in np.geomspace(0.5, 200.0, 60):
        # Divide each row by the variance, so each window length counts by its relative error
        A = np.stack([np.ones_like(T), T, _gauss_markov(T, tau)], axis=1) / v[:, None]
        r, (floor, s2, b2) = _nnls(A, np.ones_like(v))
        if best is None or r < best[0]:
            best = (r, math.sqrt(s2), math.sqrt(b2), tau, floor)
    r, s, b, tau, floor = best
    return s, b, tau, floor, math.sqrt(r / len(v))


def fit(segs, windows, scale_window, min_speed):
    """Fit the model to the segments. Returns the parameters and the statistics behind them."""
    report = {}
    min_motion = min_speed * scale_window

    # Heading offset of the mocap body, then the scale per axis, over windows with motion
    _, g = increments(segs, scale_window, scale_window / 2)
    if (np.hypot(g[:, 0], g[:, 1]) > min_motion).sum() < 10:
        raise SystemExit("The robot barely moved in the recording: walk it around under the mocap")
    heading_offset = fit_heading_offset(segs, scale_window, min_motion)
    o, g = increments(segs, scale_window, scale_window / 2, heading_offset)
    scale = []
    for axis in range(3):
        s = fit_scale(o, g, axis, min_motion if axis < 2 else min_speed * scale_window)
        if s is None:
            print(f"  Not enough {AXES[axis]} motion to fit its scale; leaving it at 1")
            s = 1.0
        elif s <= 0:
            raise SystemExit(
                f"The odometry's {AXES[axis]} runs opposite to the mocap's (scale {s:.2f}): check Mocap.cpp's axes"
            )
        scale.append(s)

    # Error growth with window length, over windows in which the robot walks
    density, bias, tau, growth = [], [], [], {a: [] for a in AXES}
    standing_drift = {}
    used_windows = []
    for T in windows:
        o, g = increments(segs, T, T / 2, heading_offset)
        walking = np.hypot(g[:, 0], g[:, 1]) / T > min_speed
        standing = np.hypot(g[:, 0], g[:, 1]) / T < min_speed / 3
        if walking.sum() < 10:
            print(f"  Too few walking {T:g} s windows ({walking.sum()}); skipping that window length")
            continue
        used_windows.append(T)
        e = o - np.asarray(scale) * g
        for a, name in enumerate(AXES):
            growth[name].append((float(np.var(e[walking, a])), float(np.mean(e[walking, a])), int(walking.sum())))
        if standing.sum() >= 10:
            standing_drift[T] = np.std(o[standing], axis=0)
    if len(used_windows) < 3:
        raise SystemExit("Need at least three window lengths with enough walking to fit the noise")

    for a, name in enumerate(AXES):
        s, b, t, floor, rms = fit_noise(used_windows, [v for v, _, _ in growth[name]])
        density.append(s)
        bias.append(b)
        tau.append(t)
        report[name] = {"floor": floor, "relative_rms": rms}

    report.update(heading_offset=heading_offset, windows=used_windows, growth=growth, standing_drift=standing_drift)
    params = {"scale": scale, "velocity_noise_density": density, "bias_sigma": bias, "bias_time_constant": tau}
    return params, report


# --------------------------------------------------------------------------------------------------
# NUSim's model, for the self-test


def simulate(params, duration, rng, dt=0.02, mocap_dt=1 / 120, mocap_noise=0.0005, heading_offset=0.3):
    """A walk with odometry from NUSim's model (OdometryModel.cpp) and mocap of a rotated body."""
    n = int(duration / dt)
    # Commands held for a few seconds each, including standing still, like walking with a keyboard
    cmd = np.zeros((n, 3))
    i = 0
    while i < n:
        hold = int(rng.uniform(2, 6) / dt)
        cmd[i : i + hold] = rng.choice([0, 1], p=[0.2, 0.8]) * rng.uniform([-0.1, -0.15, -0.6], [0.35, 0.15, 0.6])
        i += hold
    scale, s, b, tau = (np.asarray(params[k]) for k in params)
    truth, est = np.zeros((n, 3)), np.zeros((n, 3))
    bias = b * rng.standard_normal(3)
    for k in range(1, n):
        a = np.exp(-dt / tau)
        bias = a * bias + b * np.sqrt(1 - a * a) * rng.standard_normal(3)
        v = scale * cmd[k] + bias + s / np.sqrt(dt) * rng.standard_normal(3)
        for pose, vel in ((truth, cmd[k]), (est, v)):
            h = pose[k - 1, 2] + 0.5 * vel[2] * dt
            pose[k] = pose[k - 1] + [
                (np.cos(h) * vel[0] - np.sin(h) * vel[1]) * dt,
                (np.sin(h) * vel[0] + np.cos(h) * vel[1]) * dt,
                vel[2] * dt,
            ]
    t = np.arange(n) * dt
    odometry = np.column_stack([t, est])
    t_gt = np.arange(0, t[-1], mocap_dt)
    gt = np.column_stack(
        [
            t_gt,
            np.interp(t_gt, t, truth[:, 0]) + mocap_noise * rng.standard_normal(len(t_gt)),
            np.interp(t_gt, t, truth[:, 1]) + mocap_noise * rng.standard_normal(len(t_gt)),
            np.interp(t_gt, t, truth[:, 2]) + heading_offset,
        ]
    )
    return odometry, gt


# --------------------------------------------------------------------------------------------------


def _yaml(params):
    def row(v, fmt):
        return "[" + ", ".join(fmt.format(x) for x in v) + "]"

    return (
        "enabled: true\n"
        f"scale: {row(params['scale'], '{:.4f}')}\n"
        f"velocity_noise_density: {row(params['velocity_noise_density'], '{:.4g}')}\n"
        f"bias_sigma: {row(params['bias_sigma'], '{:.4g}')}\n"
        f"bias_time_constant: {row(params['bias_time_constant'], '{:.3g}')}\n"
        "seed: 0\n"
    )


def _print_report(params, report):
    print(f"\nMocap body heading offset from the robot's: {math.degrees(report['heading_offset']):+.1f} deg")
    print("\nOdometry error over walking windows (after the scale), and the model's fit:")
    for name in AXES:
        print(
            f"  {name}: "
            + ", ".join(
                f"{T:g} s: {math.sqrt(v):.4f} (n={n})"
                for T, (v, _, n) in zip(report["windows"], report["growth"][name])
            )
        )
        means = [m for _, m, _ in report["growth"][name]]
        print(
            f"     mean error {', '.join(f'{m:+.4f}' for m in means)}; "
            f"fit relative rms {report[name]['relative_rms']:.2f}, floor {math.sqrt(report[name]['floor']):.4f}"
        )
    if report["standing_drift"]:
        print("\nDrift while standing (std of the odometry's own motion over the window):")
        for T, d in report["standing_drift"].items():
            print(f"  {T:g} s: x {d[0]:.4f} m, y {d[1]:.4f} m, yaw {d[2]:.4f} rad")
        print("  NUSim's model drifts a standing robot like a walking one; if these are near zero, the real")
        print("  odometry does not, and the fitted noise overstates standing drift.")
    print("\nFitted parameters:")
    print(f"  scale:                  {', '.join(f'{v:.4f}' for v in params['scale'])}")
    for key, unit in (("velocity_noise_density", "density"), ("bias_sigma", "bias")):
        print(f"  {key + ':':<24}{', '.join(f'{v:.4g} {u}' for v, u in zip(params[key], UNITS[unit]))}")
    print(f"  bias_time_constant:     {', '.join(f'{v:.3g} s' for v in params['bias_time_constant'])}")


def run(files, rigid_body_id, windows, scale_window, min_speed, max_time_offset, output, self_test, **kwargs):
    if self_test:
        truth_params = {
            "scale": [1.08, 0.9, 0.95],
            "velocity_noise_density": [0.02, 0.015, 0.03],
            "bias_sigma": [0.015, 0.01, 0.02],
            "bias_time_constant": [8.0, 8.0, 8.0],
        }
        odometry, truth = simulate(truth_params, 1800.0, np.random.default_rng(1))
        resets = np.array([])
        print("Self-test: fitting 30 minutes of odometry simulated with NUSim's model")
    else:
        if not files:
            raise SystemExit("Give the nbs recordings to fit, or --self-test")
        odometry, truth, resets = load(files, rigid_body_id)

    offset = fit_time_offset(odometry, truth, resets, max_time_offset)
    segs = segments(odometry, truth, offset, resets)
    duration = sum(t[-1] - t[0] for t, _, _ in segs)
    walked = sum(np.sum(np.hypot(*np.diff(gt[:, :2], axis=0).T)) for _, _, gt in segs)
    print(
        f"{len(segs)} continuous stretches, {duration:.0f} s, {walked:.1f} m walked; mocap time offset {offset:+.2f} s"
    )

    params, report = fit(segs, windows, scale_window, min_speed)
    _print_report(params, report)
    if self_test:
        print("\nSimulated with:")
        for key, value in truth_params.items():
            print(f"  {key + ':':<24}{', '.join(f'{v:.4g}' for v in value)}")

    text = _yaml(params)
    print("\nFor NUSim mujoco/config/odometry.yaml:\n\n" + text)
    if output:
        with open(output, "w") as f:
            f.write(text)
        print(f"Written to {output}")
