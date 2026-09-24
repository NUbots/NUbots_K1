#!/usr/bin/env python3
"""Bin a block policy's measured envelope into the table planning::PlanSave loads.

mjlab's ``measure_envelope`` rolls shots at a trained goalkeeper and writes one CSV row per
resolved shot (``envelope.csv``: dy, time_to_arrival, speed, on_target, saved, touched, conceded,
fell) plus a JSON sidecar carrying the SHA-256 of the ONNX it was measured from. This script counts
those shots into a grid over the three quantities the planner commands the policy with, signed
``dy`` x time to arrival x ball speed, and writes ``SaveEnvelope.yaml`` with the ONNX hash, so
PlanSave can refuse to plan with a policy the envelope was not measured on.

Only counts are written. PlanSave turns them into a conservative success rate (a Wilson lower
bound) itself, so the confidence it plans with stays in its own config.

It also writes ``SaveCapability.yaml`` next to the envelope (``--no-capability`` skips it): the save rate
smoothed from the same shots onto a fine grid, which PlanSave positions the goalie with. The smoothing is
save_positioning_prototype.py's, so it needs numpy, scipy and PyYAML (mjlab's environment has them).

**Shots the goalie fell during count as failed on-target shots.** ``measure_envelope`` (mjlab up to
at least 40db3bfab) reads ``on_target`` after the environment has already reset for the fall, so
every interrupted shot is written as off target and silently leaves the save rate. Pass
``--keep-fall-on-target`` once that is fixed upstream.

Usage:
    python3 tools/policy/make_save_envelope.py <run>/envelope.csv \\
        -o module/planning/PlanSave/data/config/SaveEnvelope.yaml --name "run 12 (zrhfjas8)"
"""

import argparse
import csv
import json
from bisect import bisect_right
from pathlib import Path

DY_EDGES = [round(-1.5 + 0.1 * i, 2) for i in range(31)]
"""Signed crossing offset (m, +left), out to the Block command's clip. The goalie is not symmetric in
practice, so neither is this."""
TIME_EDGES = [0.0, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 3.0]
"""Time to arrival at the kick (s). The last bin also takes everything beyond it."""
SPEED_EDGES = [0.0, 2.0, 3.5, 6.0]
"""Ball speed at the kick (m/s): the three bands in PLAN.md. The last bin takes everything beyond."""


def bin_of(value: float, edges: list[float], clamp: bool) -> int | None:
    """Index of the bin holding value, or None outside the edges when not clamping."""
    if not clamp and (value < edges[0] or value >= edges[-1]):
        return None
    return min(max(bisect_right(edges, value) - 1, 0), len(edges) - 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="envelope.csv from mjlab's measure_envelope")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, help="JSON sidecar (default: next to the CSV)")
    parser.add_argument("--name", default="", help="Human-readable name of the policy")
    parser.add_argument("--keep-fall-on-target", action="store_true", help="Trust the CSV's on_target on falls")
    parser.add_argument(
        "--capability", type=Path, help="Smoothed capability output (default: SaveCapability.yaml beside -o)"
    )
    parser.add_argument("--no-capability", action="store_true", help="Don't write the smoothed capability")
    args = parser.parse_args()

    sidecar_path = args.sidecar or args.csv.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    if "onnx_sha256" not in sidecar:
        raise SystemExit(f"{sidecar_path} has no onnx_sha256: the envelope cannot be tied to a policy")

    n_dy, n_t, n_v = len(DY_EDGES) - 1, len(TIME_EDGES) - 1, len(SPEED_EDGES) - 1
    trials = [[[0] * n_dy for _ in range(n_t)] for _ in range(n_v)]
    saves = [[[0] * n_dy for _ in range(n_t)] for _ in range(n_v)]
    falls = [[[0] * n_dy for _ in range(n_t)] for _ in range(n_v)]

    shots = on_target = saved = fell = outside = 0
    with args.csv.open() as f:
        for row in csv.DictReader(f):
            shots += 1
            is_fall = float(row["fell"]) > 0.5
            is_on_target = float(row["on_target"]) > 0.5 or (is_fall and not args.keep_fall_on_target)
            if not is_on_target:
                continue
            d = bin_of(float(row["dy"]), DY_EDGES, clamp=False)
            if d is None:
                outside += 1
                continue
            t = bin_of(float(row["time_to_arrival"]), TIME_EDGES, clamp=True)
            v = bin_of(float(row["speed"]), SPEED_EDGES, clamp=True)
            is_save = float(row["saved"]) > 0.5
            trials[v][t][d] += 1
            saves[v][t][d] += is_save
            falls[v][t][d] += is_fall
            on_target += 1
            saved += is_save
            fell += is_fall

    def table(name: str, cells: list[list[list[int]]]) -> list[str]:
        lines = [f"{name}:"]
        for v, plane in enumerate(cells):
            lines.append(f"  # speed {SPEED_EDGES[v]}-{SPEED_EDGES[v + 1]} m/s; rows are time bins, columns dy bins")
            lines.append("  -")
            lines += [f"    - [{', '.join(str(c) for c in row)}]" for row in plane]
        return lines

    level = sidecar.get("shot_level", {})
    lines = [
        "# Block policy capability envelope for planning::PlanSave.",
        "# GENERATED by tools/policy/make_save_envelope.py -- do not edit by hand, re-measure instead.",
        "#",
        f"# Policy: {args.name or sidecar.get('onnx', '?')}",
        f"# Checkpoint: {sidecar.get('checkpoint', '?')}",
        f"# Shot level measured: {level.get('name', '?')} (crossing {level.get('crossing')},"
        f" speed {level.get('speed')}, distance {level.get('distance')})",
        f"# {shots} shots, {on_target} on target inside the dy grid ({outside} outside it):"
        f" saved {saved / max(on_target, 1):.1%}, fell during {fell / max(on_target, 1):.1%}.",
        "# Falls are counted as failed on-target shots (see the tool's docstring).",
        "#",
        "# Axes are the Block command at the kick: signed dy (m, +left) where the ball crosses the",
        "# goalie's frontal plane, time to arrival (s) and ball speed (m/s). Cells are indexed",
        "# [speed][time][dy] and count on-target shots, saves and shots the goalie fell during.",
        "",
        f'onnx_sha256: "{sidecar["onnx_sha256"]}"',
        f"dy_edges: {DY_EDGES}",
        f"time_edges: {TIME_EDGES}",
        f"speed_edges: {SPEED_EDGES}",
        "",
        *table("trials", trials),
        "",
        *table("saves", saves),
        "",
        *table("falls", falls),
        "",
    ]
    args.output.write_text("\n".join(lines))

    print(f"{on_target} on-target shots binned ({outside} outside the dy grid) -> {args.output}")
    print(f"saved {saved / max(on_target, 1):.1%}, fell during {fell / max(on_target, 1):.1%}")
    empty = sum(1 for plane in trials for row in plane for c in row if c == 0)
    print(f"{empty} of {n_v * n_t * n_dy} cells are empty (PlanSave counts them as 0% success)")

    if not args.no_capability:
        write_capability(args, sidecar)


def write_capability(args: argparse.Namespace, sidecar: dict) -> None:
    """SaveCapability.yaml: the save rate smoothed onto a fine grid, which PlanSave positions the goalie with."""
    if args.keep_fall_on_target:
        raise SystemExit("The capability's smoothing always counts falls as failures: pass --no-capability as well")
    from save_positioning_prototype import Capability

    path = args.capability or args.output.with_name("SaveCapability.yaml")
    S = Capability.from_csv(args.csv)
    sm = S.smoothing
    lines = [
        "# Block policy capability, smoothed, for planning::PlanSave's goalie positioning.",
        "# GENERATED by tools/policy/make_save_envelope.py -- do not edit by hand, re-measure instead.",
        "#",
        f"# Policy: {args.name or sidecar.get('onnx', '?')}",
        f"# Checkpoint: {sidecar.get('checkpoint', '?')}",
        f"# {S.n_shots} on-target shots (falls count as failures), raw save rate {S.raw_rate:.1%}.",
        f"# Smoothing (save_positioning_prototype.py Capability.from_csv): Gaussian sigma {sm['sigma']} (dy m, time s,",
        f"# speed m/s), shrunk towards 0 with {sm['prior_failures']} pseudo-failures, then carried up the time axis",
        f"# (more time never hurts: {sm['monotone_time']}) and down the speed axis (a slower ball never hurts:",
        f"# {sm['monotone_speed']}).",
        "#",
        "# Each axis is start + step * i for i < count. The rate is indexed [dy][time][speed] and interpolated",
        "# trilinearly: dy off the grid is a miss, time and speed clamp to it, and time <= 0 is a miss.",
        "",
        f'onnx_sha256: "{sidecar["onnx_sha256"]}"',
        *S.axes_yaml(),
        "",
        *S.rate_yaml(),
        "",
    ]
    path.write_text("\n".join(lines))
    print(f"Smoothed capability ({len(S.DY)} x {len(S.T)} x {len(S.V)}) -> {path}")


if __name__ == "__main__":
    main()
