#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "matplotlib"]
# ///
"""Prototype: where should the goalie stand, given where the ball is and what the block policy can do?

For a ball at b, every straight shot at the goal lies in the triangle (b, left post, right post). Stepping out
towards the ball narrows the triangle where the goalie meets it (less lateral ground to cover) but shortens the time
every shot takes to arrive (less time to react). The block policy's measured envelope S(dy, t, v) is exactly the
exchange rate between those two, so the positioning objective is the expected save probability over the triangle:

    P_save(g) = E_{aim y ~ p(y), speed v ~ p(v)} [ S(dy(g, b, y), t(g, b, y, v) - t_lat, v) ]

with the goalie at g facing the ball, dy the signed offset (+left) where the shot crosses the goalie's frontal plane,
t the rolling ball's time to get there (constant deceleration, as PlanSave and mjlab's ShotCommand predict it) and
t_lat the stack's delay between the kick and a live Block command (the envelope was measured with a perfect command
at the kick). Shots that stop short of the goal line are no threat and are left out.

S is smoothed from the raw shots of mjlab's measure_envelope (envelope.csv), not read from the planner's bins, so the
optimum doesn't jump between cells: counts are binned finely, Gaussian-smoothed, and shrunk towards 0 with a few
pseudo-failures, so thinly measured regions count as unlikely saves. Shots the goalie fell during count as failed
on-target shots, as in make_save_envelope.py.

The objective is, by default, the CVaR over aim points (mean of the worst fraction of them: a shooter who picks the
goalie's gaps), or the mean over all of them. The mean is nearly flat in depth and leaves the near post open on tight
angles, so the CVaR is the default. An optional penalty on distance from the goal line stands in for everything a
one-shot model ignores (dribbles, passes, rebounds).

The envelope is mjlab's, and the real goalie will be worse in ways nobody has measured yet. So each spot is also
scored on degraded copies of S, one per combination of an extra reaction delay and a reach scale (a real goalie that
reaches 70% as far sideways saves a shot at dy as mjlab's saves one at dy / 0.7). By default the spot chosen is the
one with the least worst-case regret over those copies: whichever the real robot turns out to be, it is never far
from that robot's own best spot. Degradations pull different ways (less time favours the line, less reach favours
stepping out), so this is not the same as planning for the most pessimistic copy.

The objective is nearly flat in depth for central balls (a metre or two of spots within a couple of points of the
best), so the depth the maximum lands on is noise. The spot chosen is therefore the shallowest (closest to the goal
line) within `near_best` of the best score: stay home unless stepping out is clearly worth it.

Frames: field frame with our goal line at x = 0, x out into the field, y to the left looking out from the goal.

Usage:
    uv run tools/policy/save_positioning_prototype.py <run>/envelope.csv -o <out dir>

The defaults (speeds 1.5-4 m/s, the CVaR objective, regret over 0/0.2 s extra delay x 100/70% reach, near best 0.02)
are the configuration that positioned the goalie most sensibly in the first comparisons (24 Sep 2026).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


@dataclass
class Config:
    goal_width: float = 2.5  # between the posts' inner edges (M-Field and KidSize)
    ball_radius: float = 0.0785
    penalty_area_length: float = 3.0
    penalty_area_width: float = 6.0
    rolling_deceleration: float = 0.5  # PlanSave's ball model, which the policy was trained on
    t_lat: float = 0.2  # kick -> live Block command; GoalieShotBenchmark measured a median 0.23 s through the stack
    speed_range: tuple[float, float] = (1.5, 4.0)  # shot speed prior (uniform), the envelope's measured range
    n_aim: int = 41
    n_speed: int = 11
    cvar_fraction: float = 0.2
    objective: str = "cvar"  # "cvar": the worst cvar_fraction of aim points; "mean": all of them
    line_penalty: float = 0.0  # per metre off the goal line
    min_ball_distance: float = 0.5  # the goalie can't stand on the ball
    # Sim-to-real degradations scored against: every combination of an extra reaction delay (s) and a reach scale.
    # Guesses until the real goalie has been measured.
    extra_lat: tuple[float, ...] = (0.0, 0.2)
    reach_scale: tuple[float, ...] = (1.0, 0.7)
    robust: str = "regret"  # "regret": least worst-case regret over the degradations; "mean": best mean over them
    near_best: float = 0.02  # take the shallowest spot within this of the best score (0: the best score itself)


# --------------------------------------------------------------------------------------------------------------------
# The block policy's capability, smoothed from the raw shots


class Capability:
    """S(dy, t, v): smoothed save rate of the block policy, shrunk to 0 where there is little data."""

    DY = np.arange(-2.0, 2.0 + 1e-9, 0.05)
    T = np.arange(0.0, 3.0 + 1e-9, 0.05)
    V = np.arange(1.5, 4.0 + 1e-9, 0.25)

    def __init__(
        self,
        csv: Path,
        sigma: tuple[float, float, float] = (0.08, 0.1, 0.3),
        prior_failures: float = 5.0,
        monotone_time: bool = True,
        monotone_speed: bool = True,
    ):
        d = np.genfromtxt(csv, delimiter=",", names=True)
        fell = d["fell"] > 0.5
        on_target = (d["on_target"] > 0.5) | fell  # falls are written off target upstream; count them as failures
        dy, t, v = d["dy"][on_target], d["time_to_arrival"][on_target], d["speed"][on_target]
        saved = (d["saved"][on_target] > 0.5).astype(float)

        # Bin centres are the grid points; smooth in physical units
        def edges(c):
            h = (c[1] - c[0]) / 2
            return np.append(c - h, c[-1] + h)

        e = [edges(self.DY), edges(self.T), edges(self.V)]
        sample = np.column_stack([dy, np.clip(t, 0, self.T[-1]), np.clip(v, self.V[0], self.V[-1])])
        trials, _ = np.histogramdd(sample, bins=e)
        saves, _ = np.histogramdd(sample, bins=e, weights=saved)
        steps = [self.DY[1] - self.DY[0], self.T[1] - self.T[0], self.V[1] - self.V[0]]
        s = [sg / st for sg, st in zip(sigma, steps)]
        # Undo the kernel's normalisation so the counts stay counts (the prior then means pseudo-shots)
        norm = (2 * np.pi) ** 1.5 * np.prod(s)
        trials_s = gaussian_filter(trials, s, mode="constant") * norm
        saves_s = gaussian_filter(saves, s, mode="constant") * norm
        self.rate = saves_s / (trials_s + prior_failures)
        if monotone_time:
            # mjlab shots start at most 4.5 m out, so fast balls are never measured arriving late, and the prior reads
            # that gap as failure. More time can't hurt as long as PlanSave holds the ready stance until the time to
            # arrival is back in the measured range, so carry the best rate so far up the time axis.
            self.rate = np.maximum.accumulate(self.rate, axis=1)
        if monotone_speed:
            # Likewise slow balls are never measured arriving early. A slower ball arriving at the same time and place
            # is no harder to stop, so carry the best rate so far down the speed axis.
            self.rate = np.maximum.accumulate(self.rate[:, :, ::-1], axis=2)[:, :, ::-1]
        self.support = trials_s
        self._interp = RegularGridInterpolator((self.DY, self.T, self.V), self.rate, bounds_error=False, fill_value=0.0)
        self.n_shots, self.raw_rate = len(saved), saved.mean()

    def __call__(self, dy, t, v):
        """Save rate; dy outside the grid is a miss, t and v clamp to the measured range, t <= 0 is a miss."""
        pts = np.stack([dy, np.clip(t, 0, self.T[-1]), np.clip(v, self.V[0], self.V[-1])], axis=-1)
        return np.where(t > 0, self._interp(pts), 0.0)


# --------------------------------------------------------------------------------------------------------------------
# The objective


def time_to_travel(s, v0, a):
    """Time for a ball kicked at v0 decelerating at a to roll s, nan if it stops first."""
    disc = v0**2 - 2 * a * s
    with np.errstate(invalid="ignore"):
        return np.where(disc >= 0, (v0 - np.sqrt(np.maximum(disc, 0))) / a, np.nan)


def save_probability(S: Capability, cfg: Config, ball: np.ndarray, gx: np.ndarray, gy: np.ndarray, facing=None):
    """P_save (mean over aim points, and CVaR) for goalie positions (gx, gy), same shape, and a ball at `ball`.

    The goalie faces the ball, or the unit vector `facing` if given (its frontal plane is perpendicular to it).

    Returns (mean, cvar, threat): threat is the fraction of (aim, speed) shots that reach the goal line at all.
    """
    half = cfg.goal_width / 2 - cfg.ball_radius
    aims = np.linspace(-half, half, cfg.n_aim)
    speeds = np.linspace(*cfg.speed_range, cfg.n_speed)

    # Shot directions and the distance to the goal line along each
    to_aim = np.stack([-ball[0] * np.ones_like(aims), aims - ball[1]], axis=-1)  # (A, 2)
    s_goal = np.linalg.norm(to_aim, axis=-1)
    u = to_aim / s_goal[:, None]

    # Goalie frame: facing the ball, left is +90 degrees
    g = np.stack([gx, gy], axis=-1)[..., None, :]  # (..., 1, 2)
    bg = ball - g
    dist = np.maximum(np.linalg.norm(bg, axis=-1, keepdims=True), 1e-6)
    f = bg / dist if facing is None else np.broadcast_to(np.asarray(facing, float), bg.shape)
    ahead = np.einsum("...k,...k->...", bg, f)  # (..., 1): ball's distance in front of the goalie's plane
    left = np.stack([-f[..., 1], f[..., 0]], axis=-1)

    # Where each shot crosses the goalie's frontal plane (p - g).f = 0: s = |b - g| / (-u.f)
    uf = np.einsum("ak,...ak->...a", u, np.broadcast_to(f, f.shape[:-2] + (len(aims), 2)))
    with np.errstate(divide="ignore", invalid="ignore"):
        s_plane = np.where((uf < -1e-6) & (ahead > 0), ahead / -uf, np.inf)  # (..., A)
    cross = ball + np.where(np.isfinite(s_plane), s_plane, 0.0)[..., None] * u  # (..., A, 2); unused where infinite
    dy = np.einsum("...ak,...ak->...a", cross - g, np.broadcast_to(left, cross.shape))

    # Per speed: shots that reach the goal line are threats; they meet the goalie's plane first unless it lies beyond
    # the goal line (then it's a goal)
    t_plane = time_to_travel(s_plane[..., None], speeds, cfg.rolling_deceleration)  # (..., A, V)
    t_goal = time_to_travel(s_goal[:, None], speeds, cfg.rolling_deceleration)  # (A, V)
    threat = ~np.isnan(t_goal)
    before_goal = np.isfinite(s_plane)[..., None] & (s_plane[..., None] <= s_goal[:, None])
    p = S(
        np.broadcast_to(dy[..., None], t_plane.shape),
        np.nan_to_num(t_plane, nan=0.0) - cfg.t_lat,
        np.broadcast_to(speeds, t_plane.shape),
    )
    p = np.where(before_goal, p, 0.0)

    # Average over speeds that are threats, per aim point
    n_threat = threat.sum(axis=-1)  # (A,)
    per_aim = np.where(threat, p, 0.0).sum(axis=-1) / np.maximum(n_threat, 1)  # (..., A)
    w = n_threat / max(n_threat.sum(), 1)  # aim points weighted by how many of their shots are threats
    mean = (per_aim * w).sum(axis=-1)

    k = max(1, int(round(cfg.cvar_fraction * len(aims))))
    cvar = np.sort(per_aim, axis=-1)[..., :k].mean(axis=-1)

    # Positions the goalie can't take
    bad = (dist[..., 0, 0] < cfg.min_ball_distance) | (gx >= ball[0])
    mean, cvar = np.where(bad, np.nan, mean), np.where(bad, np.nan, cvar)
    return mean, cvar, threat.mean()


def objective(mean, cvar, gx, cfg: Config, use_cvar: bool):
    return (cvar if use_cvar else mean) - cfg.line_penalty * gx


class Degraded:
    """S on a goalie that reacts extra_lat later and reaches reach_scale as far sideways as it did in mjlab."""

    def __init__(self, S: Capability, extra_lat: float, reach_scale: float):
        self.S, self.extra_lat, self.reach_scale = S, extra_lat, reach_scale

    def __call__(self, dy, t, v):
        return self.S(dy / self.reach_scale, t - self.extra_lat, v)


@dataclass
class Evaluation:
    """One ball position scored over a grid of goalie spots."""

    mean: np.ndarray  # nominal (mjlab) P_save, mean over aim points
    cvar: np.ndarray  # nominal P_save, CVaR over aim points
    score: np.ndarray  # robust score the spot is chosen on, higher is better
    regret: np.ndarray  # worst-case regret over the degradations
    threat: float  # fraction of shots that reach the goal line
    variant_best: np.ndarray  # each degradation's best objective over the grid, to score other spots against

    def regret_at(self, S, cfg, ball, point, use_cvar, facing=None):
        """Worst-case regret of a spot off the grid (e.g. today's), against the grid's best per degradation."""
        J = [
            objective(
                *save_probability(Degraded(S, l, k), cfg, ball, point[:1], point[1:], facing)[:2], point[:1], cfg, use_cvar
            )
            for l in cfg.extra_lat
            for k in cfg.reach_scale
        ]
        return float(np.max(self.variant_best - np.concatenate(J)))


def evaluate(S: Capability, cfg: Config, ball, GX, GY, use_cvar: bool) -> Evaluation:
    mean, cvar, threat = save_probability(S, cfg, ball, GX, GY)
    if threat == 0:
        nan = np.full_like(mean, np.nan)
        return Evaluation(mean, cvar, nan, nan, threat, np.array([]))
    J = np.stack(
        [
            objective(*save_probability(Degraded(S, l, k), cfg, ball, GX, GY)[:2], GX, cfg, use_cvar)
            for l in cfg.extra_lat
            for k in cfg.reach_scale
        ]
    )
    best = np.nanmax(J.reshape(len(J), -1), axis=1)
    regret = (best.reshape(-1, *[1] * GX.ndim) - J).max(axis=0)
    score = -regret if cfg.robust == "regret" else J.mean(axis=0)
    return Evaluation(mean, cvar, score, regret, threat, best)


def choose(score, GX, near_best: float) -> int:
    """Flat index of the shallowest spot within near_best of the best score (the better-scoring one on a tie)."""
    near = score >= np.nanmax(score) - near_best
    x = np.where(near, GX, np.inf)
    candidates = np.flatnonzero(x == x.min())
    return int(candidates[np.argmax(score.flat[candidates])])


TODAY_FACING = np.array([1.0, 0.0])
"""purpose::Goalie faces straight up the field, whatever the ball does."""


def current_goalie(ball, cfg: Config):
    """purpose::Goalie today (Goalie.cpp, Goalie.yaml): follows the ball's y, kept goal_post_clearance (0.35 m) inside
    each post, on the goal line in the middle and bowing forward in a parabola, strafe_curve_depth (0.15 m) at the
    widest."""
    y_max = cfg.goal_width / 2 - 0.35
    y = np.clip(ball[1], -y_max, y_max)
    return np.array([0.15 * (y / y_max) ** 2, y])


# --------------------------------------------------------------------------------------------------------------------
# Plots


def draw_field(ax, cfg: Config, xmax):
    hw = cfg.goal_width / 2
    ax.plot([0, 0], [-cfg.penalty_area_width / 2 - 1, cfg.penalty_area_width / 2 + 1], color="0.3", lw=1.5)
    ax.plot([0, -0.3, -0.3, 0], [hw, hw, -hw, -hw], color="0.3", lw=1.5)
    pa = cfg.penalty_area_length
    ax.plot([0, pa, pa, 0], [cfg.penalty_area_width / 2] * 2 + [-cfg.penalty_area_width / 2] * 2, color="0.6", lw=1)
    ax.plot([0, 1, 1, 0], [2, 2, -2, -2], color="0.6", lw=1)  # goal area (M-Field: 1 x 4)
    ax.set_xlim(-0.4, xmax)
    ax.set_aspect("equal")


def plot_ball_positions(S, cfg, balls, out: Path, use_cvar: bool):
    xs = np.arange(0.05, 5.0, 0.04)
    ys = np.arange(-3.0, 3.0 + 1e-9, 0.04)
    GX, GY = np.meshgrid(xs, ys, indexing="xy")

    cols = 3
    rows = int(np.ceil(len(balls) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), constrained_layout=True)
    label = f"CVaR{int(cfg.cvar_fraction * 100)}" if use_cvar else "mean"
    rows_out = []
    for ax, ball in zip(axes.flat, balls):
        ball = np.asarray(ball, float)
        ev = evaluate(S, cfg, ball, GX, GY, use_cvar)
        draw_field(ax, cfg, max(xs[-1], ball[0] + 0.3))
        if ev.threat == 0:
            ax.set_title(f"ball ({ball[0]:.1f}, {ball[1]:.1f}): no shot reaches the goal")
            ax.plot(*ball, "o", color="orange", ms=8, mec="k")
            rows_out.append((ball, None, None, None))
            continue
        i = choose(ev.score, GX, cfg.near_best)
        j = int(np.nanargmax(ev.score))
        cur = current_goalie(ball, cfg)
        cm, cc, _ = save_probability(S, cfg, ball, cur[:1], cur[1:], TODAY_FACING)
        cr = ev.regret_at(S, cfg, ball, cur, use_cvar, TODAY_FACING)

        field = ev.cvar if use_cvar else ev.mean
        im = ax.pcolormesh(xs, ys, field, vmin=0, vmax=1, cmap="viridis", shading="nearest")
        ax.contour(xs, ys, np.nan_to_num(field), levels=[0.2, 0.4, 0.6, 0.8], colors="w", linewidths=0.5, alpha=0.6)
        near = np.nan_to_num(ev.score, nan=-np.inf) >= np.nanmax(ev.score) - cfg.near_best
        ax.contour(xs, ys, near.astype(float), levels=[0.5], colors="red", linewidths=1.2)
        hw = cfg.goal_width / 2
        ax.fill([ball[0], 0, 0], [ball[1], hw, -hw], facecolor="none", edgecolor="orange", lw=1, ls="--")
        ax.plot(*ball, "o", color="orange", ms=8, mec="k")
        at = lambda k: (GX.flat[k], GY.flat[k])
        ax.plot(*at(j), "*", color="none", ms=14, mec="red", mew=1.2, label=f"best score: regret {ev.regret.flat[j]:.2f}")
        ax.plot(
            *at(i),
            "*",
            color="red",
            ms=16,
            mec="k",
            label=f"chosen: {ev.mean.flat[i]:.2f} / {ev.cvar.flat[i]:.2f}, regret {ev.regret.flat[i]:.2f}",
        )
        ax.plot(*cur, "s", color="w", ms=8, mec="k", label=f"today: {cm[0]:.2f} / {cc[0]:.2f}, regret {cr:.2f}")
        ax.set_title(f"ball ({ball[0]:.1f}, {ball[1]:.1f}), {ev.threat:.0%} of shots reach goal")
        ax.legend(loc="lower right", fontsize=7, framealpha=0.8)
        rows_out.append(
            (
                ball,
                (at(i), ev.mean.flat[i], ev.cvar.flat[i], ev.regret.flat[i]),
                (at(j), ev.mean.flat[j], ev.cvar.flat[j], ev.regret.flat[j]),
                (cur, cm[0], cc[0], cr),
            )
        )
    for ax in axes.flat[len(balls) :]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label=f"nominal P_save ({label})")
    fig.suptitle(
        f"Colour: nominal (mjlab) P_save, {label}. Legends: nominal mean / CVaR, worst-case regret.\n"
        f"Red outline: spots within {cfg.near_best} of the best {cfg.robust} score over extra delay {cfg.extra_lat} s"
        f" x reach {cfg.reach_scale}; filled star: the shallowest of them.\n"
        f"t_lat={cfg.t_lat}s, speeds {cfg.speed_range} m/s, line penalty {cfg.line_penalty}/m"
    )
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return rows_out


def plot_policy_map(S, cfg, out: Path, use_cvar: bool):
    """g*(ball) over a grid of ball positions: an arrow from each ball to where the goalie should stand."""
    xs = np.arange(0.05, 5.0, 0.1)
    ys = np.arange(-3.0, 3.0 + 1e-9, 0.1)
    GX, GY = np.meshgrid(xs, ys, indexing="xy")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    draw_field(ax, cfg, 7.5)
    for bx in np.arange(1.5, 7.6, 1.0):
        for by in np.arange(-4.0, 4.1, 1.0):
            ball = np.array([bx, by])
            ev = evaluate(S, cfg, ball, GX, GY, use_cvar)
            if ev.threat < 0.05:
                continue
            i = choose(ev.score, GX, cfg.near_best)
            best = np.array([GX.flat[i], GY.flat[i]])
            pv = (ev.cvar if use_cvar else ev.mean).flat[i]
            ax.annotate("", best, ball, arrowprops=dict(arrowstyle="->", color="0.5", lw=0.7))
            ax.plot(*ball, "o", color="orange", ms=4)
            sc = ax.scatter(*best, c=[pv], vmin=0, vmax=1, cmap="viridis", s=40, edgecolors="k", zorder=3)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="nominal P_save at the chosen spot")
    ax.set_title(
        f"Chosen goalie spot (dot) for each ball position (orange): shallowest within {cfg.near_best} of the best"
        f" {cfg.robust} score"
    )
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_capability(S: Capability, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    for ax, v in zip(axes, [2.0, 2.75, 3.5]):
        k = int(np.argmin(np.abs(S.V - v)))
        im = ax.pcolormesh(S.DY, S.T, S.rate[:, :, k].T, vmin=0, vmax=1, cmap="viridis", shading="nearest")
        ax.contour(S.DY, S.T, S.support[:, :, k].T, levels=[20, 100], colors="w", linewidths=0.6, linestyles=["--", "-"])
        ax.set_xlabel("dy (m, +left)")
        ax.set_ylabel("time to arrival (s)")
        ax.set_title(f"S at v = {S.V[k]:.2f} m/s (white: 20 / 100 shots)", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="save rate")
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="envelope.csv from mjlab's measure_envelope")
    parser.add_argument("-o", "--output", type=Path, default=Path("save_positioning"))
    parser.add_argument("--t-lat", type=float, default=Config.t_lat)
    parser.add_argument("--line-penalty", type=float, default=Config.line_penalty)
    parser.add_argument("--objective", choices=["cvar", "mean"], default=Config.objective, help="over aim points")
    parser.add_argument("--raw-time", action="store_true", help="don't assume more time never hurts (see Capability)")
    parser.add_argument("--raw-speed", action="store_true", help="don't assume slower never hurts (see Capability)")
    parser.add_argument("--speed-range", type=float, nargs=2, default=Config.speed_range, help="shot speed prior (m/s)")
    parser.add_argument("--no-map", action="store_true", help="skip the (slow) map over ball positions")
    parser.add_argument("--extra-lat", type=float, nargs="+", default=Config.extra_lat, help="extra delays (s) to score against")
    parser.add_argument("--reach-scale", type=float, nargs="+", default=Config.reach_scale, help="reach scales to score against")
    parser.add_argument("--robust", choices=["regret", "mean"], default=Config.robust)
    parser.add_argument("--near-best", type=float, default=Config.near_best, help="depth rule tolerance (0: plain best)")
    args = parser.parse_args()

    cfg = Config(
        t_lat=args.t_lat,
        line_penalty=args.line_penalty,
        speed_range=tuple(args.speed_range),
        extra_lat=tuple(args.extra_lat),
        reach_scale=tuple(args.reach_scale),
        robust=args.robust,
        near_best=args.near_best,
        objective=args.objective,
    )
    use_cvar = cfg.objective == "cvar"
    args.output.mkdir(parents=True, exist_ok=True)
    S = Capability(args.csv, monotone_time=not args.raw_time, monotone_speed=not args.raw_speed)
    print(f"{S.n_shots} on-target shots, raw save rate {S.raw_rate:.1%}")
    plot_capability(S, args.output / "capability.png")

    balls = [(2.0, 0.0), (3.0, 0.0), (4.5, 0.0), (3.0, 1.5), (3.0, -1.5), (2.0, 2.5), (5.0, -2.0), (6.5, 0.0), (1.0, 2.5)]
    rows = plot_ball_positions(S, cfg, balls, args.output / "positions.png", use_cvar)
    if not args.no_map:
        plot_policy_map(S, cfg, args.output / "policy_map.png", use_cvar)

    print(
        f"\nspeeds {cfg.speed_range} m/s, t_lat {cfg.t_lat} s, objective {cfg.objective},"
        f" line penalty {cfg.line_penalty}/m, {cfg.robust} over extra delay {cfg.extra_lat} s x reach {cfg.reach_scale},"
        f" near best {cfg.near_best}"
    )
    print("Each spot: position, nominal mean / CVaR, worst-case regret")
    print(f"{'ball':>12} | {'chosen':^29} | {'best score':^29} | {'today':^29}")

    def spot(r):
        (x, y), m, c, reg = r
        return f"({x:5.2f},{y:6.2f}) {m:4.2f} {c:4.2f} {reg:4.2f}"

    for ball, chosen, best, today in rows:
        if chosen is None:
            print(f"({ball[0]:4.1f},{ball[1]:5.1f}) | no shot reaches the goal")
            continue
        print(f"({ball[0]:4.1f},{ball[1]:5.1f}) | {spot(chosen)} | {spot(best)} | {spot(today)}")
    print(f"\nPlots in {args.output}/")


if __name__ == "__main__":
    main()
