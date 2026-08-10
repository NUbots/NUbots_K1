#!/usr/bin/env python3
"""Split a K1WalkPolicy capture into moving / frozen segments before reporting anything.

Why this exists: the first hardware walk log is 81.5% robot-standing-still. Segmented by
leg joint velocity it splits into 150 moving ticks and 663 frozen ticks, including one
continuous 540-tick (10.8 s) stretch in which the leg joints move by 0.00038 rad — the
encoder quantisation floor. Statistics over the whole file describe a policy shouting at a
robot that is not listening, and they are not the same statistics:

    metric                whole log    moving ticks only
    mean |action|           0.790            0.538
    fraction |a| > 0.99     0.469            0.149
    L/R knee commanded  -0.036 / +0.823  +0.098 / +0.572

Every claim about a capture should cite a segment, not a file.

Input: any text stream containing the TRACE lines K1WalkPolicy emits, one per policy tick:

    WALKOBS <tick> mode=<K1Mode int> t=<seconds> dt=<seconds> <79 floats>

Usage:
    python3 tools/analysis/segment_walk_log.py capture.log
    python3 tools/analysis/segment_walk_log.py capture.log --min-run 25 --csv obs.csv
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re
import sys

import numpy as np

# 79-obs contract (NUSim docs/OBS_ACTION_CONTRACT.md). The 82-obs contract prefixes three
# base-linear-velocity floats; --obs-dim 82 shifts every slice by 3.
LAYOUT_79 = {
    "gyro": (0, 3),
    "gravity": (3, 6),
    "command": (6, 9),
    "q_rel": (9, 31),
    "dq": (31, 53),
    "last_action": (53, 75),
    "phase": (75, 79),
}
LEG_SLOTS = list(range(10, 22))  # JointIndexK1: both legs
MODE_NAMES = {0: "DAMP", 1: "PREP", 2: "WALK", 3: "CUSTOM", 4: "SOCCER", -1: "unknown"}

LINE = re.compile(r"WALKOBS\s+(\d+)\s+mode=(-?\d+)\s+t=([0-9.eE+-]+)\s+dt=([0-9.eE+-]+)\s+(.*)")


def parse(stream) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ticks, modes, times, dts, obs = [], [], [], [], []
    width = None
    for line in stream:
        m = LINE.search(line)
        if m is None:
            continue
        values = np.fromstring(m.group(5), sep=" ")
        if width is None:
            width = values.size
        elif values.size != width:
            continue  # truncated line (log rotation, interleaved writer)
        ticks.append(int(m.group(1)))
        modes.append(int(m.group(2)))
        times.append(float(m.group(3)))
        dts.append(float(m.group(4)))
        obs.append(values)
    if not obs:
        raise SystemExit("no WALKOBS lines found — is log_level TRACE on K1WalkPolicy?")
    return (
        np.array(ticks),
        np.array(modes),
        np.array(times),
        np.array(dts),
        np.vstack(obs),
    )


def layout_for(dim: int) -> dict[str, tuple[int, int]]:
    if dim == 79:
        return LAYOUT_79
    if dim == 82:
        shifted = {k: (a + 3, b + 3) for k, (a, b) in LAYOUT_79.items()}
        shifted["linvel"] = (0, 3)
        return shifted
    raise SystemExit(f"unsupported observation width {dim}; expected 79 or 82")


def runs_of(mask: np.ndarray, min_run: int):
    """Yield (start, stop, value) for maximal runs of a boolean array."""
    if mask.size == 0:
        return
    edges = np.flatnonzero(np.diff(mask.astype(int))) + 1
    bounds = np.concatenate(([0], edges, [mask.size]))
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a >= min_run:
            yield int(a), int(b), bool(mask[a])


def report(name: str, obs: np.ndarray, lay: dict, dts: np.ndarray, n: int) -> None:
    act = obs[:, slice(*lay["last_action"])]
    dq = obs[:, slice(*lay["dq"])][:, LEG_SLOTS]
    q_rel = obs[:, slice(*lay["q_rel"])]
    cmd = obs[:, slice(*lay["command"])]
    print(f"  {name:<10} n={n:5d}  duration={dts.sum():6.2f}s  rate={n / max(dts.sum(), 1e-9):5.1f} Hz")
    print(
        f"      |a|: mean {np.abs(act).mean():.3f}  frac>0.99 {np.mean(np.abs(act) > 0.99):.3f}"
        f"   leg max|dq| mean {np.abs(dq).max(axis=1).mean():.4f}"
    )
    print(
        f"      knee q-rel L/R {q_rel[:, 13].mean():+.3f} / {q_rel[:, 19].mean():+.3f}"
        f"   ankle pitch q-rel L/R {q_rel[:, 14].mean():+.3f} / {q_rel[:, 20].mean():+.3f}"
    )
    print(f"      command mean [{cmd[:, 0].mean():+.3f} {cmd[:, 1].mean():+.3f} {cmd[:, 2].mean():+.3f}]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", type=pathlib.Path, nargs="?", help="capture file (default: stdin)")
    parser.add_argument(
        "--move-threshold",
        type=float,
        default=0.05,
        help="rad/s; a tick counts as MOVING when max|dq| over the 12 leg joints exceeds this",
    )
    parser.add_argument("--min-run", type=int, default=10, help="minimum ticks for a segment to be listed")
    parser.add_argument("--csv", type=pathlib.Path, default=None, help="dump the parsed observations")
    args = parser.parse_args()

    stream = args.log.open() if args.log else sys.stdin
    ticks, modes, times, dts, obs = parse(stream)
    lay = layout_for(obs.shape[1])

    print(f"{obs.shape[0]} ticks, observation width {obs.shape[1]}")
    gaps = np.flatnonzero(np.diff(ticks) != 1)
    if gaps.size:
        print(f"WARNING: {gaps.size} tick discontinuities (dropped ticks or a policy restart)")
    print(f"loop period: mean {dts.mean() * 1e3:.2f} ms  p95 {np.percentile(dts, 95) * 1e3:.2f} ms  max {dts.max() * 1e3:.2f} ms")
    for mode in sorted(set(modes.tolist())):
        m = modes == mode
        print(f"mode {MODE_NAMES.get(mode, mode)}: {m.sum()} ticks ({m.mean():.1%})")

    dq_leg = obs[:, slice(*lay["dq"])][:, LEG_SLOTS]
    moving = np.abs(dq_leg).max(axis=1) > args.move_threshold
    print(f"\nMOVING {moving.sum()} ticks ({moving.mean():.1%}), FROZEN {(~moving).sum()} ticks")

    print("\nsegments:")
    for a, b, is_moving in runs_of(moving, args.min_run):
        label = "MOVING" if is_moving else "FROZEN"
        print(f"  [{a:5d}:{b:5d}] {label} {b - a:5d} ticks  {dts[a:b].sum():6.2f}s")

    print("\naggregates (cite these, never the whole file):")
    report("WHOLE", obs, lay, dts, obs.shape[0])
    if moving.any():
        report("MOVING", obs[moving], lay, dts[moving], int(moving.sum()))
    if (~moving).any():
        report("FROZEN", obs[~moving], lay, dts[~moving], int((~moving).sum()))

    if args.csv is not None:
        names = []
        for field, (a, b) in sorted(lay.items(), key=lambda kv: kv[1][0]):
            names += [f"{field}{i}" for i in range(b - a)]
        with args.csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["tick", "mode", "t", "dt", "moving", *names])
            for i in range(obs.shape[0]):
                writer.writerow([ticks[i], modes[i], times[i], dts[i], int(moving[i]), *obs[i].tolist()])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
