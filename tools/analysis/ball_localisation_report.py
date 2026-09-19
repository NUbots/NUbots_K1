#!/usr/bin/env python3
"""Summarise a tools::BallLocalisationBenchmark run: tables and plots of the ball UKF against NUSim truth.

Reads the run directory the benchmark writes (samples.csv, detections.csv, summary.csv) and writes
report.md plus PNG plots next to them:

- the per-shot metrics, overall and binned by ball speed and by start distance
- estimated vs true ball speed after the kick (how fast the filter follows a kick)
- position error against time since the kick
- velocity response time against ball speed
- raw detection error against range (detector error, separate from the filter)
- the effective-lag distribution
- consistency (NEES) of the published covariance against the true error

Needs numpy and matplotlib (e.g. any mjlab / booster_mjlab venv).

Usage:
    python3 tools/analysis/ball_localisation_report.py recordings/ball_localisation_benchmark/<run>
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# A detection further than this from the true ball is counted as a false positive / gross error
FALSE_POSITIVE_M = 0.5

# 95th percentile of the chi-squared distribution, by degrees of freedom
CHI2_95 = {2: 5.991, 4: 9.488}


def read_csv(path: Path) -> dict[str, np.ndarray]:
    """Read a CSV into named columns; numeric where possible, NaN for empty cells."""
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    columns: dict[str, np.ndarray] = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        try:
            columns[key] = np.array([float(v) if v not in ("", "nan") else math.nan for v in values])
        except ValueError:
            columns[key] = np.array(values)
    return columns


def fmt(x: float, unit: str = "", digits: int = 3) -> str:
    return "–" if not np.isfinite(x) else f"{x:.{digits}f}{unit}"


def metrics_table(summary: dict[str, np.ndarray], mask: np.ndarray) -> list[str]:
    """One markdown row of medians (and the response success rate) over the masked shots."""
    n = int(mask.sum())
    if n == 0:
        return ["0", "–", "–", "–", "–", "–", "–", "–"]
    response = summary["velocity_response_s"][mask]
    followed = np.isfinite(response)
    return [
        str(n),
        fmt(np.nanmedian(summary["pos_rmse"][mask]), " m"),
        fmt(np.nanmedian(summary["pos_max"][mask]), " m"),
        fmt(np.nanmedian(summary["vel_rmse"][mask]), " m/s"),
        f"{followed.sum()}/{n}" + (f" ({fmt(np.median(response[followed]), ' s', 2)})" if followed.any() else ""),
        fmt(np.nanmedian(summary["best_lag_s"][mask]), " s", 2),
        fmt(np.nanmedian(summary["detection_pos_rmse"][mask]), " m"),
        fmt(np.nanmedian(summary["detection_rate_hz"][mask]), " Hz", 1),
    ]


def nees_table(samples: dict[str, np.ndarray], mask: np.ndarray) -> list[str]:
    """Normalised estimation error squared for the position, velocity and full state over the masked samples.

    A consistent filter averages the number of degrees of freedom (2, 2 and 4). Larger means the filter claims
    more certainty than it has. The scale column is how much the standard deviations would have to grow to make
    it consistent, which is what a consumer should apply if the filter itself is not retuned.
    """
    names = ["cov_xx", "cov_xy", "cov_xvx", "cov_xvy", "cov_yy", "cov_yvx", "cov_yvy", "cov_vxvx", "cov_vxvy", "cov_vyvy"]
    if not all(n in samples for n in names):
        return []
    n = int(mask.sum())
    if n == 0:
        return []
    error = np.stack(
        [
            samples["est_x"][mask] - samples["gt_x"][mask],
            samples["est_y"][mask] - samples["gt_y"][mask],
            samples["est_vx"][mask] - samples["gt_vx"][mask],
            samples["est_vy"][mask] - samples["gt_vy"][mask],
        ],
        axis=1,
    )
    covariance = np.zeros((n, 4, 4))
    for name, (i, j) in zip(names, [(a, b) for a in range(4) for b in range(a, 4)]):
        covariance[:, i, j] = covariance[:, j, i] = samples[name][mask]

    rows = []
    for label, idx, dof in [("position", [0, 1], 2), ("velocity", [2, 3], 2), ("full state", [0, 1, 2, 3], 4)]:
        e = error[:, idx]
        P = covariance[np.ix_(np.arange(n), idx, idx)]
        finite = np.isfinite(e).all(axis=1) & np.isfinite(P).all(axis=(1, 2)) & (np.linalg.det(P) > 0)
        if not finite.any():
            continue
        nees = np.einsum("ni,nij,nj->n", e[finite], np.linalg.inv(P[finite]), e[finite])
        rows.append(
            f"| {label} | {dof} | {np.mean(nees):.1f} | {np.median(nees):.1f} | "
            f"{100 * np.mean(nees <= CHI2_95[dof]):.0f}% | {np.sqrt(np.mean(nees) / dof):.1f}x |"
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path, help="run directory written by the benchmark")
    args = parser.parse_args()

    summary = read_csv(args.run / "summary.csv")
    samples = read_csv(args.run / "samples.csv")
    detections = read_csv(args.run / "detections.csv")
    if not summary:
        raise SystemExit(f"{args.run / 'summary.csv'} has no shots")

    header = [
        "shots",
        "pos RMSE",
        "pos max",
        "vel RMSE",
        "velocity followed (median time)",
        "effective lag",
        "detection RMSE",
        "detection rate",
    ]
    lines = [
        f"# Ball localisation benchmark: {args.run.name}",
        "",
        "Medians over shots. Errors are in the robot frame over the roll after each kick. *Velocity followed* counts",
        "the shots where the estimated velocity came within 10% of the true speed and 20° of the true direction.",
        "",
        "| group | " + " | ".join(header) + " |",
        "|---" * (len(header) + 1) + "|",
    ]
    everything = np.ones_like(summary["shot"], dtype=bool)
    lines.append("| all | " + " | ".join(metrics_table(summary, everything)) + " |")
    speed = summary["speed"]
    for lo, hi in [(0.0, 2.0), (2.0, 3.0), (3.0, 99.0)]:
        label = f"speed {lo:g}–{hi:g} m/s" if hi < 99 else f"speed ≥ {lo:g} m/s"
        lines.append(f"| {label} | " + " | ".join(metrics_table(summary, (speed >= lo) & (speed < hi))) + " |")
    distance = np.hypot(summary["start_x"], summary["start_y"])
    for lo, hi in [(0.0, 3.0), (3.0, 4.0), (4.0, 99.0)]:
        label = f"start {lo:g}–{hi:g} m" if hi < 99 else f"start ≥ {lo:g} m"
        lines.append(f"| {label} | " + " | ".join(metrics_table(summary, (distance >= lo) & (distance < hi))) + " |")
    lines += [
        "",
        f"Median vision latency (image capture to `Balls`): {fmt(np.nanmedian(summary['vision_latency_median_s']), ' s', 3)}.",
    ]

    if samples:
        moving = samples["phase"] == "roll"
        for label, mask in [
            ("resting ball", samples["phase"] == "settle"),
            ("rolling, first 0.3 s after the kick", moving & (samples["t_kick"] < 0.3)),
            ("rolling, 0.3 s after the kick onwards", moving & (samples["t_kick"] >= 0.3)),
        ]:
            rows = nees_table(samples, mask)
            if not rows:
                continue
            lines += [
                "",
                f"### Covariance consistency: {label} ({int(mask.sum())} estimates)",
                "",
                "NEES averages the degrees of freedom when the covariance is honest. The scale column is the factor",
                "the standard deviations would need to grow by to make it so.",
                "",
                "| state | dof | mean NEES | median | within 95% gate | scale needed |",
                "|---|---|---|---|---|---|",
                *rows,
            ]
    if detections:
        valid = np.isfinite(detections["gt_x"])
        err = np.hypot(detections["det_x"] - detections["gt_x"], detections["det_y"] - detections["gt_y"])[valid]
        rng = np.hypot(detections["gt_x"], detections["gt_y"])[valid]
        false_pos = err > FALSE_POSITIVE_M
        lines += [
            "",
            f"Raw detections: {valid.sum()} with ground truth, {false_pos.sum()} ({100 * false_pos.mean():.1f}%) more than "
            f"{FALSE_POSITIVE_M} m from the ball (false positives or gross errors). Median error of the rest by true range:",
            "",
            "| range | detections | median error |",
            "|---|---|---|",
        ]
        for lo, hi in [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0), (4.0, 99.0)]:
            m = (rng >= lo) & (rng < hi) & ~false_pos
            label = f"{lo:g}–{hi:g} m" if hi < 99 else f"≥ {lo:g} m"
            lines.append(f"| {label} | {m.sum()} | {fmt(np.median(err[m]) if m.any() else math.nan, ' m')} |")
    lines.append("")

    # --- plots ---
    roll = samples["phase"] == "roll" if samples else np.array([], dtype=bool)
    figures = []

    if samples and roll.any():
        t = samples["t_kick"][roll]
        shot = samples["shot"][roll]
        est_speed = np.hypot(samples["est_vx"][roll], samples["est_vy"][roll])
        gt_speed = np.hypot(samples["gt_vx"][roll], samples["gt_vy"][roll])
        pos_err = np.hypot(samples["est_x"][roll] - samples["gt_x"][roll], samples["est_y"][roll] - samples["gt_y"][roll])

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for s in np.unique(shot)[:12]:
            m = shot == s
            (line,) = ax.plot(t[m], gt_speed[m], "-", lw=1)
            ax.plot(t[m], est_speed[m], ".", ms=3, color=line.get_color())
        ax.set(xlabel="time since kick (s)", ylabel="ball speed (m/s)", title="Estimated (dots) vs true (lines) speed")
        ax.grid(alpha=0.3)
        figures.append(("speed_tracking.png", fig))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(t, pos_err, ".", ms=2, alpha=0.4)
        bins = np.arange(0.0, np.nanmax(t) + 0.2, 0.2)
        idx = np.digitize(t, bins)
        med = [np.nanmedian(pos_err[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(bins))]
        ax.plot(bins[:-1] + 0.1, med, "-", lw=2, label="median")
        ax.set(xlabel="time since kick (s)", ylabel="position error (m)", title="Estimate position error after the kick")
        ax.legend()
        ax.grid(alpha=0.3)
        figures.append(("position_error.png", fig))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(summary["speed"], summary["velocity_response_s"], "o")
    missed = ~np.isfinite(summary["velocity_response_s"])
    if missed.any():
        ax.plot(summary["speed"][missed], np.zeros(missed.sum()), "x", color="red", label="never followed")
        ax.legend()
    ax.set(xlabel="kick speed (m/s)", ylabel="velocity response (s)", title="Time for the estimate to follow the kick")
    ax.grid(alpha=0.3)
    figures.append(("velocity_response.png", fig))

    if detections:
        valid = np.isfinite(detections["gt_x"])
        det_err = np.hypot(detections["det_x"] - detections["gt_x"], detections["det_y"] - detections["gt_y"])[valid]
        det_range = np.hypot(detections["gt_x"], detections["gt_y"])[valid]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(det_range, det_err, ".", ms=3, alpha=0.5)
        ax.set(xlabel="true range (m)", ylabel="detection error (m)", title="Raw detection error vs range")
        ax.grid(alpha=0.3)
        figures.append(("detection_error.png", fig))

    lags = summary["best_lag_s"][np.isfinite(summary["best_lag_s"])]
    if lags.size:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(lags, bins=20)
        ax.set(xlabel="effective lag (s)", ylabel="shots", title="Lag at which the estimate best matches truth")
        figures.append(("effective_lag.png", fig))

    for name, fig in figures:
        fig.tight_layout()
        fig.savefig(args.run / name, dpi=120)
        plt.close(fig)
        lines.append(f"![{name}]({name})")

    (args.run / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))
    print(f"\nwrote {args.run / 'report.md'} and {len(figures)} plots")


if __name__ == "__main__":
    main()
