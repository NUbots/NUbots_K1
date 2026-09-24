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
Play a recording through field localisation, first FieldLocalisationNLopt and then FieldLocalisationSRIF, and plot
both estimates against the recording's ground truth.

    ./b localisation playback recordings/<recording>.nbs

runs the playback/localisation_nlopt and playback/localisation_srif roles on the recording (each writes its estimates
to a csv), then plots the torso's field pose (x, y, yaw) from both next to the ground truth, with their errors, into
recordings/<recording>_localisation.png. The csvs are copied to recordings/localisation_playback/.

Ground truth is read from the recording itself:
  - message.localisation.RobotPoseGroundTruth (Hft): the real robot under motion capture (localisation::Mocap)
  - message.booster.NUSimRobotGroundTruth (Hst): NUSim (input::NUSimGroundTruth). NUSim's world is the field turned
    half way round (our goal is at -x in NUSim, +x in {f}); --nusim-field-yaw changes that.
"""

import math
import os
import shutil
import subprocess

from termcolor import cprint

import b
from utility.dockerise import run_on_docker

METHODS = ("nlopt", "srif")
COLOURS = {"ground truth": "black", "nlopt": "tab:blue", "srif": "tab:orange"}


@run_on_docker
def register(command):
    command.description = (
        "Play a recording through NLopt then SRIF field localisation and plot them against ground truth"
    )
    command.add_argument("file", help="The nbs recording to play back")
    command.add_argument(
        "--config-hostname",
        default="nusim",
        help="Hostname the roles load their config as, which picks the field (config/nusim/ for NUSim). "
        "Use the robot's hostname, or docker for the defaults, for real robot recordings",
    )
    command.add_argument(
        "--nusim-field-yaw",
        type=float,
        default=180.0,
        help="Rotation (deg) from NUSim's world to the NUbots field frame, for NUSimRobotGroundTruth",
    )
    command.add_argument(
        "--plot-only", action="store_true", help="Plot the csvs of an earlier run instead of playing back again"
    )


def resolve(path, build_dir):
    """The recording as seen from inside the container: relative to the cwd, else to the build directory, where
    recordings made with `./b run` land."""
    for candidate in (os.path.abspath(path), os.path.join(build_dir, path)):
        if os.path.isfile(candidate):
            return candidate
    raise SystemExit(f"Recording not found: {path}")


def play(method, nbs, build_dir, hostname):
    binary = os.path.join("bin", "playback", f"localisation_{method}")
    if not os.path.isfile(os.path.join(build_dir, binary)):
        raise SystemExit(f"{binary} is not built: run ./b configure and ./b build")
    cprint(f"Playing {os.path.basename(nbs)} through {method.upper()}", "cyan", attrs=["bold"])
    env = dict(os.environ, ROBOT_HOSTNAME=hostname)
    if subprocess.run([binary, nbs], cwd=build_dir, env=env).returncode != 0:
        raise SystemExit(f"{binary} failed")


def load_estimates(path):
    import numpy as np

    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    # Seconds since the epoch, like the ground truth
    return {"t": data["t_ns"] * 1e-9, "x": data["x"], "y": data["y"], "yaw": data["yaw"]}


def load_ground_truth(nbs, nusim_field_yaw):
    import numpy as np

    from utility.nbs import LinearDecoder
    from utility.nbs.protobuf_types import MessageTypes

    wanted = ["message.localisation.RobotPoseGroundTruth", "message.booster.NUSimRobotGroundTruth"]
    types = [t for t in wanted if t in MessageTypes]
    c, s = math.cos(math.radians(nusim_field_yaw)), math.sin(math.radians(nusim_field_yaw))

    rows = []
    for packet in LinearDecoder(nbs, types=types, show_progress=True):
        t = packet.index_timestamp * 1e-9
        if packet.type.name == "message.localisation.RobotPoseGroundTruth":
            m = packet.msg.Hft
            rows.append((t, m.t.x, m.t.y, math.atan2(m.x.y, m.x.x)))
        else:
            # Hft = Rz(nusim_field_yaw) * Hst
            m = packet.msg.Hst
            yaw = math.atan2(m.x.y, m.x.x) + math.radians(nusim_field_yaw)
            rows.append((t, c * m.t.x - s * m.t.y, s * m.t.x + c * m.t.y, math.atan2(math.sin(yaw), math.cos(yaw))))

    if not rows:
        cprint(
            "No ground truth in the recording (RobotPoseGroundTruth or NUSimRobotGroundTruth): plotting the estimates only",
            "yellow",
        )
        return None
    return np.array(sorted(rows))


def errors(estimate, truth):
    """Position and heading error of each estimate against the ground truth interpolated to its time."""
    import numpy as np

    t = estimate["t"]
    inside = (t >= truth[0, 0]) & (t <= truth[-1, 0])
    t = t[inside]
    x = np.interp(t, truth[:, 0], truth[:, 1])
    y = np.interp(t, truth[:, 0], truth[:, 2])
    yaw = np.interp(t, truth[:, 0], np.unwrap(truth[:, 3]))
    position = np.hypot(estimate["x"][inside] - x, estimate["y"][inside] - y)
    heading = np.abs(np.angle(np.exp(1j * (estimate["yaw"][inside] - yaw))))
    return t, position, heading


def plot(nbs, estimates, truth, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    t0 = min([e["t"][0] for e in estimates.values() if len(e["t"])] + ([truth[0, 0]] if truth is not None else []))

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(5, 2, width_ratios=[1, 1.4])
    ax_xy = fig.add_subplot(grid[:, 0])
    ax_x, ax_y, ax_yaw, ax_pos, ax_head = (fig.add_subplot(grid[i, 1]) for i in range(5))
    for ax in (ax_y, ax_yaw, ax_pos, ax_head):
        ax.sharex(ax_x)

    series = ([("ground truth", truth[:, 0], truth[:, 1], truth[:, 2], truth[:, 3])] if truth is not None else []) + [
        (m, e["t"], e["x"], e["y"], e["yaw"]) for m, e in estimates.items()
    ]
    for name, t, x, y, yaw in series:
        style = dict(color=COLOURS[name], label=name.upper() if name != "ground truth" else name)
        line = dict(linewidth=2.0) if name == "ground truth" else dict(linewidth=1.0, alpha=0.85)
        ax_xy.plot(x, y, **style, **line)
        ax_x.plot(t - t0, x, **style, **line)
        ax_y.plot(t - t0, y, **style, **line)
        if name == "ground truth":
            # Break the line where the heading wraps, rather than drawing across the plot
            yaw = np.where(np.abs(np.diff(yaw, prepend=yaw[0])) > math.pi, np.nan, yaw)
            ax_yaw.plot(t - t0, np.degrees(yaw), **style, **line)
        else:
            ax_yaw.plot(t - t0, np.degrees(yaw), **style, marker=".", markersize=2, linestyle="none")

    summary = []
    if truth is not None:
        for name, e in estimates.items():
            if not len(e["t"]):
                continue
            t, position, heading = errors(e, truth)
            if not len(t):
                continue
            ax_pos.plot(t - t0, position, color=COLOURS[name], linewidth=1.0, label=name.upper())
            ax_head.plot(t - t0, np.degrees(heading), color=COLOURS[name], linewidth=1.0, label=name.upper())
            summary.append(
                f"{name.upper()}: position RMSE {np.sqrt(np.mean(position ** 2)):.3f} m, "
                f"heading RMSE {np.degrees(np.sqrt(np.mean(heading ** 2))):.1f} deg"
            )
    ax_pos.set_ylabel("position error [m]")
    ax_head.set_ylabel("heading error [deg]")
    ax_head.set_xlabel("time [s]")

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("field x [m]")
    ax_xy.set_ylabel("field y [m]")
    ax_xy.set_title("torso in the field")
    ax_xy.legend()
    ax_x.set_ylabel("x [m]")
    ax_y.set_ylabel("y [m]")
    ax_yaw.set_ylabel("yaw [deg]")
    for ax in (ax_xy, ax_x, ax_y, ax_yaw, ax_pos, ax_head):
        ax.grid(True, alpha=0.3)
    if ax_pos.lines:
        ax_pos.legend(fontsize="small")
    fig.suptitle(os.path.basename(nbs) + ("\n" + "    ".join(summary) if summary else ""))
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return summary


@run_on_docker
def run(file, config_hostname, nusim_field_yaw, plot_only, **kwargs):
    build_dir = os.path.realpath(os.path.join(b.project_dir, "..", "build"))
    nbs = resolve(file, build_dir)
    name = os.path.splitext(os.path.basename(nbs))[0]

    recordings = os.path.join(b.project_dir, "recordings")
    csv_dir = os.path.join(recordings, "localisation_playback")
    os.makedirs(csv_dir, exist_ok=True)

    estimates = {}
    for method in METHODS:
        csv = os.path.join(csv_dir, f"{name}_localisation_{method}.csv")
        if not plot_only:
            play(method, nbs, build_dir, config_hostname)
            # The role writes into the build directory (LocalisationPlayback.yaml output_directory), which in the
            # container is usually the project's recordings/ already
            written = os.path.join(build_dir, "recordings", "localisation_playback", os.path.basename(csv))
            if not (os.path.exists(csv) and os.path.samefile(written, csv)):
                shutil.copyfile(written, csv)
        estimates[method] = load_estimates(csv)
        count = len(estimates[method]["t"])
        cprint(f"{method.upper()}: {count} estimates ({csv})", "green")

    truth = load_ground_truth(nbs, nusim_field_yaw)
    output = os.path.join(recordings, f"{name}_localisation.png")
    for line in plot(nbs, estimates, truth, output):
        cprint(line, "cyan")
    cprint(f"Plot saved to {output}", "green", attrs=["bold"])
