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

Frames: field frame with our goal line at x = 0, x out into the field, y to the left looking out from the goal. This is
planning::PlanSave's goal frame {g}: x = field_length / 2 - x_f, y = -y_f in NUbots' field frame.

planning::PlanSave runs this (module/planning/PlanSave/src/positioning.hpp) on SaveCapability.yaml, which
make_save_envelope.py writes with Capability.from_csv. Spots stop at the penalty area line. --write-reference writes the
results its tests hold the port to; change the two together and regenerate them.

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
    # The spots considered: a grid from min_depth off the goal line out to the penalty area line (max_depth, the M-Field's
    # penalty_area_length), and across the penalty area (max_lateral, half its width). PlanSave uses a coarser step.
    grid_step: float = 0.04
    min_depth: float = 0.05
    max_depth: float = 3.0
    max_lateral: float = 3.0
    # Sim-to-real degradations scored against: every combination of an extra reaction delay (s) and a reach scale.
    # Guesses until the real goalie has been measured.
    extra_lat: tuple[float, ...] = (0.0, 0.2)
    reach_scale: tuple[float, ...] = (1.0, 0.7)
    robust: str = "regret"  # "regret": least worst-case regret over the degradations; "mean": best mean over them
    near_best: float = 0.02  # take the shallowest spot within this of the best score (0: the best score itself)


# --------------------------------------------------------------------------------------------------------------------
# The block policy's capability, smoothed from the raw shots


def axis(start: float, step: float, count: int) -> np.ndarray:
    """A uniform grid axis, built the way planning::PlanSave builds it from SaveCapability.yaml."""
    return start + step * np.arange(count)


class Capability:
    """S(dy, t, v): the block policy's save rate on a uniform grid, interpolated trilinearly.

    `from_csv` smooths it from mjlab's raw shots, shrunk to 0 where there is little data. `from_yaml` and `to_yaml` read
    and write the SaveCapability.yaml planning::PlanSave positions the goalie with, which uses the same interpolation.
    """

    def __init__(self, dy: np.ndarray, t: np.ndarray, v: np.ndarray, rate: np.ndarray, support=None):
        self.DY, self.T, self.V, self.rate = dy, t, v, rate
        self.support = support
        self._interp = RegularGridInterpolator((self.DY, self.T, self.V), self.rate, bounds_error=False, fill_value=0.0)
        self.n_shots, self.raw_rate = 0, float("nan")

    @classmethod
    def from_csv(
        cls,
        csv: Path,
        sigma: tuple[float, float, float] = (0.08, 0.1, 0.3),
        prior_failures: float = 5.0,
        monotone_time: bool = True,
        monotone_speed: bool = True,
    ) -> "Capability":
        DY, T, V = axis(-2.0, 0.05, 81), axis(0.0, 0.05, 61), axis(1.5, 0.25, 11)
        d = np.genfromtxt(csv, delimiter=",", names=True)
        fell = d["fell"] > 0.5
        on_target = (d["on_target"] > 0.5) | fell  # falls are written off target upstream; count them as failures
        dy, t, v = d["dy"][on_target], d["time_to_arrival"][on_target], d["speed"][on_target]
        saved = (d["saved"][on_target] > 0.5).astype(float)

        # Bin centres are the grid points; smooth in physical units
        def edges(c):
            h = (c[1] - c[0]) / 2
            return np.append(c - h, c[-1] + h)

        e = [edges(DY), edges(T), edges(V)]
        sample = np.column_stack([dy, np.clip(t, 0, T[-1]), np.clip(v, V[0], V[-1])])
        trials, _ = np.histogramdd(sample, bins=e)
        saves, _ = np.histogramdd(sample, bins=e, weights=saved)
        steps = [DY[1] - DY[0], T[1] - T[0], V[1] - V[0]]
        s = [sg / st for sg, st in zip(sigma, steps)]
        # Undo the kernel's normalisation so the counts stay counts (the prior then means pseudo-shots)
        norm = (2 * np.pi) ** 1.5 * np.prod(s)
        trials_s = gaussian_filter(trials, s, mode="constant") * norm
        saves_s = gaussian_filter(saves, s, mode="constant") * norm
        rate = saves_s / (trials_s + prior_failures)
        if monotone_time:
            # mjlab shots start at most 4.5 m out, so fast balls are never measured arriving late, and the prior reads
            # that gap as failure. More time can't hurt as long as PlanSave holds the ready stance until the time to
            # arrival is back in the measured range, so carry the best rate so far up the time axis.
            rate = np.maximum.accumulate(rate, axis=1)
        if monotone_speed:
            # Likewise slow balls are never measured arriving early. A slower ball arriving at the same time and place
            # is no harder to stop, so carry the best rate so far down the speed axis.
            rate = np.maximum.accumulate(rate[:, :, ::-1], axis=2)[:, :, ::-1]
        c = cls(DY, T, V, rate, trials_s)
        c.n_shots, c.raw_rate = len(saved), saved.mean()
        c.smoothing = dict(
            sigma=list(sigma), prior_failures=prior_failures, monotone_time=monotone_time, monotone_speed=monotone_speed
        )
        return c

    @classmethod
    def from_yaml(cls, path: Path) -> "Capability":
        import yaml

        d = yaml.safe_load(Path(path).read_text())
        ax = [axis(d[k]["start"], d[k]["step"], d[k]["count"]) for k in ("dy", "time", "speed")]
        return cls(*ax, np.asarray(d["rate"], float))

    def axes_yaml(self) -> list[str]:
        def one(name, a):
            return f"{name}: {{start: {a[0]:.6g}, step: {a[1] - a[0]:.6g}, count: {len(a)}}}"

        return [one("dy", self.DY), one("time", self.T), one("speed", self.V)]

    def rate_yaml(self, indent: str = "") -> list[str]:
        """The rate as nested [dy][time][speed] lists, one line per (dy, time)."""
        lines = [f"{indent}rate:"]
        for i in range(len(self.DY)):
            lines.append(f"{indent}  # dy = {self.DY[i]:.2f} m; rows are time, columns speed")
            lines.append(f"{indent}  -")
            lines += [f"{indent}    - [{', '.join(f'{r:.4f}' for r in row)}]" for row in self.rate[i]]
        return lines

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

    # CVaR over the aim points a shot reaches at all: one that can't is no threat, not a goal
    reached = n_threat > 0
    k = max(1, int(round(cfg.cvar_fraction * max(reached.sum(), 1))))
    cvar = np.sort(per_aim[..., reached], axis=-1)[..., :k].mean(axis=-1) if reached.any() else np.zeros_like(mean)

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
                *save_probability(Degraded(S, l, k), cfg, ball, point[:1], point[1:], facing)[:2],
                point[:1],
                cfg,
                use_cvar,
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


def spot_grid(cfg: Config, step: float | None = None):
    """The goalie spots considered, as (GX, GY) of shape (y, x): planning::PlanSave's grid, capped at the penalty area."""
    step = cfg.grid_step if step is None else step
    xs = cfg.min_depth + step * np.arange(int(np.floor((cfg.max_depth - cfg.min_depth) / step + 1e-9)) + 1)
    ys = -cfg.max_lateral + step * np.arange(int(np.floor(2 * cfg.max_lateral / step + 1e-9)) + 1)
    return np.meshgrid(xs, ys, indexing="xy")


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
    GX, GY = spot_grid(cfg)
    xs, ys = GX[0], GY[:, 0]

    cols = 3
    rows = int(np.ceil(len(balls) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4.6 * rows), constrained_layout=True)
    label = f"CVaR{int(cfg.cvar_fraction * 100)}" if use_cvar else "mean"
    rows_out = []
    for ax, ball in zip(axes.flat, balls):
        ball = np.asarray(ball, float)
        ev = evaluate(S, cfg, ball, GX, GY, use_cvar)
        draw_field(ax, cfg, max(cfg.max_depth + 0.5, ball[0] + 0.3))
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
        ax.plot(
            *at(j), "*", color="none", ms=14, mec="red", mew=1.2, label=f"best score: regret {ev.regret.flat[j]:.2f}"
        )
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
        f"t_lat={cfg.t_lat}s, speeds {cfg.speed_range} m/s, line penalty {cfg.line_penalty}/m,"
        f" spots out to {cfg.max_depth} m (penalty area line)"
    )
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return rows_out


def plot_policy_map(S, cfg, out: Path, use_cvar: bool):
    """g*(ball) over a grid of ball positions: an arrow from each ball to where the goalie should stand."""
    GX, GY = spot_grid(cfg, 0.1)
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
        if S.support is not None:
            ax.contour(
                S.DY, S.T, S.support[:, :, k].T, levels=[20, 100], colors="w", linewidths=0.6, linestyles=["--", "-"]
            )
        ax.set_xlabel("dy (m, +left)")
        ax.set_ylabel("time to arrival (s)")
        ax.set_title(f"S at v = {S.V[k]:.2f} m/s (white: 20 / 100 shots)", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="save rate")
    fig.savefig(out, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------------------------
# Reference results for planning::PlanSave's port (tests/TestPositioning.cpp)


def synthetic_capability() -> Capability:
    """A small made-up capability: a reach that grows with time, off centre, a little worse for faster balls. Rounded
    as the YAML writes it, so both sides of the comparison read the same numbers."""
    DY, T, V = axis(-2.0, 0.1, 41), axis(0.0, 0.1, 31), axis(1.5, 0.5, 6)
    dy, t, v = np.meshgrid(DY, T, V, indexing="ij")
    reach = 0.35 + 0.6 * np.maximum(t - 0.3, 0.0)
    rate = np.clip((reach - np.abs(dy - 0.05)) / 0.25 + 0.5, 0.0, 1.0) * (0.95 - 0.05 * (v - 1.5))
    return Capability(DY, T, V, np.round(rate, 4))


def write_reference(path: Path) -> None:
    """Positions chosen on the synthetic capability with PlanSave's default configuration and a few variations."""
    import dataclasses

    S = synthetic_capability()
    base = Config(grid_step=0.1)  # PlanSave.yaml's positioning defaults
    cases = [((2.0, 0.0), {}), ((3.0, 0.0), {}), ((4.5, 0.0), {}), ((3.0, 1.5), {}), ((3.0, -1.5), {})]
    cases += [((5.0, -2.0), {}), ((6.5, 0.0), {}), ((1.0, 2.5), {}), ((3.0, 1.5), {"objective": "mean"})]
    cases += [((3.0, 1.5), {"robust": "mean"}), ((3.0, 0.5), {"line_penalty": 0.05, "near_best": 0.0})]
    cases += [((4.5, 0.0), {"speed_range": (1.5, 2.0)})]  # no shot reaches the goal

    def config_yaml(c: Config) -> str:
        fields = [
            "goal_width", "ball_radius", "rolling_deceleration", "t_lat", "speed_range", "n_aim", "n_speed",
            "cvar_fraction", "objective", "line_penalty", "min_ball_distance", "extra_lat", "reach_scale", "robust",
            "near_best", "grid_step", "min_depth", "max_depth", "max_lateral",
        ]  # fmt: skip
        def val(x):
            if isinstance(x, str):
                return x
            if isinstance(x, (tuple, list)):
                return "[" + ", ".join(f"{float(e):.6g}" for e in x) + "]"
            return f"{x:.6g}"

        return "{" + ", ".join(f"{f}: {val(getattr(c, f))}" for f in fields) + "}"

    lines = [
        "# Reference results for planning::PlanSave's goalie positioning (tests/TestPositioning.cpp).",
        "# GENERATED by tools/policy/save_positioning_prototype.py --write-reference: do not edit by hand.",
        "",
        "capability:",
        *[f"  {line}" for line in S.axes_yaml()],
        *S.rate_yaml("  "),
        "",
        f"config: {config_yaml(base)}",
        "",
        "cases:",
    ]
    for ball, over in cases:
        cfg = dataclasses.replace(base, **over)
        GX, GY = spot_grid(cfg)
        ev = evaluate(S, cfg, np.asarray(ball, float), GX, GY, cfg.objective == "cvar")
        lines.append(f"  - ball: [{ball[0]}, {ball[1]}]")
        if over:
            lines.append(f"    overrides: {config_yaml(cfg)}")
        if ev.threat == 0:
            lines.append("    reaches: false")
            continue
        i = choose(ev.score, GX, cfg.near_best)
        lines += [
            "    reaches: true",
            f"    target: [{GX.flat[i]:.6f}, {GY.flat[i]:.6f}]",
            f"    mean: {ev.mean.flat[i]:.9f}",
            f"    cvar: {ev.cvar.flat[i]:.9f}",
            f"    regret: {ev.regret.flat[i]:.9f}",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(cases)} reference cases to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "csv", type=Path, nargs="?", help="envelope.csv from mjlab's measure_envelope, or a SaveCapability.yaml"
    )
    parser.add_argument("--write-reference", type=Path, help="write the C++ port's reference results here and stop")
    parser.add_argument("-o", "--output", type=Path, default=Path("save_positioning"))
    parser.add_argument("--t-lat", type=float, default=Config.t_lat)
    parser.add_argument("--line-penalty", type=float, default=Config.line_penalty)
    parser.add_argument("--objective", choices=["cvar", "mean"], default=Config.objective, help="over aim points")
    parser.add_argument("--raw-time", action="store_true", help="don't assume more time never hurts (see Capability)")
    parser.add_argument("--raw-speed", action="store_true", help="don't assume slower never hurts (see Capability)")
    parser.add_argument("--speed-range", type=float, nargs=2, default=Config.speed_range, help="shot speed prior (m/s)")
    parser.add_argument("--no-map", action="store_true", help="skip the (slow) map over ball positions")
    parser.add_argument(
        "--extra-lat", type=float, nargs="+", default=Config.extra_lat, help="extra delays (s) to score against"
    )
    parser.add_argument(
        "--reach-scale", type=float, nargs="+", default=Config.reach_scale, help="reach scales to score against"
    )
    parser.add_argument("--robust", choices=["regret", "mean"], default=Config.robust)
    parser.add_argument(
        "--near-best", type=float, default=Config.near_best, help="depth rule tolerance (0: plain best)"
    )
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
    if args.write_reference:
        write_reference(args.write_reference)
        return
    if args.csv is None:
        parser.error("an envelope.csv or SaveCapability.yaml is needed")
    if args.csv.suffix in (".yaml", ".yml"):
        S = Capability.from_yaml(args.csv)
        print(f"Capability from {args.csv}")
    else:
        S = Capability.from_csv(args.csv, monotone_time=not args.raw_time, monotone_speed=not args.raw_speed)
        print(f"{S.n_shots} on-target shots, raw save rate {S.raw_rate:.1%}")
    plot_capability(S, args.output / "capability.png")

    balls = [
        (2.0, 0.0),
        (3.0, 0.0),
        (4.5, 0.0),
        (3.0, 1.5),
        (3.0, -1.5),
        (2.0, 2.5),
        (5.0, -2.0),
        (6.5, 0.0),
        (1.0, 2.5),
    ]
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
