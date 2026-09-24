/*
 * MIT License
 *
 * Copyright (c) 2026 NUbots
 *
 * This file is part of the NUbots codebase.
 * See https://github.com/NUbots/NUbots for further info.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifndef MODULE_PLANNING_PLANSAVE_POSITIONING_HPP
#define MODULE_PLANNING_PLANSAVE_POSITIONING_HPP

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

/**
 * Where the goalie stands while no shot is on its way, from the block policy's capability. This is
 * tools/policy/save_positioning_prototype.py, which has the derivation, the plots and the reference results
 * tests/TestPositioning.cpp holds this to; change the two together.
 *
 * Everything here is in the goal frame {g}: our goal line is x = 0, x runs out into the field and y is to the left
 * looking out from the goal.
 */
namespace module::planning::save {

    /// A uniform grid axis: start + step * i for i < count
    struct Axis {
        double start      = 0.0;
        double step       = 1.0;
        std::size_t count = 0;

        [[nodiscard]] double at(const std::size_t i) const {
            return start + step * double(i);
        }
        [[nodiscard]] double back() const {
            return at(count - 1);
        }
        /// Lower grid index and fraction towards the next for a value inside the axis
        void locate(const double value, std::size_t& i, double& frac) const {
            const double u = (value - start) / step;
            i              = std::size_t(std::clamp(std::floor(u), 0.0, double(count - 2)));
            frac           = std::clamp((value - at(i)) / step, 0.0, 1.0);
        }
    };

    /**
     * S(dy, t, v): the block policy's save rate over the signed offset where a shot crosses the goalie's frontal
     * plane, its time to get there and its speed, smoothed from mjlab's shots by tools/policy/make_save_envelope.py
     * (SaveCapability.yaml).
     */
    struct Capability {
        /// SHA-256 (hex) of the ONNX the shots were measured from
        std::string onnx_sha256{};
        Axis dy{};
        Axis time{};
        Axis speed{};
        /// Save rate, [dy][time][speed] with speed fastest
        std::vector<double> rate{};

        [[nodiscard]] bool valid() const {
            return dy.count >= 2 && time.count >= 2 && speed.count >= 2 && dy.step > 0.0 && time.step > 0.0
                   && speed.step > 0.0 && rate.size() == dy.count * time.count * speed.count;
        }

        [[nodiscard]] double value(const std::size_t i, const std::size_t j, const std::size_t k) const {
            return rate[(i * time.count + j) * speed.count + k];
        }

        /// Trilinear save rate: dy off the grid is a miss, time and speed clamp to it, and t <= 0 is a miss
        [[nodiscard]] double operator()(const double d, const double t, const double v) const {
            if (!(t > 0.0) || d < dy.start || d > dy.back()) {
                return 0.0;
            }
            std::size_t i = 0, j = 0, k = 0;
            double fi = 0.0, fj = 0.0, fk = 0.0;
            dy.locate(d, i, fi);
            time.locate(std::min(t, time.back()), j, fj);
            speed.locate(std::clamp(v, speed.start, speed.back()), k, fk);
            double out = 0.0;
            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    for (int c = 0; c < 2; ++c) {
                        const double w = (a ? fi : 1.0 - fi) * (b ? fj : 1.0 - fj) * (c ? fk : 1.0 - fk);
                        if (w > 0.0) {
                            out += w * value(i + a, j + b, k + c);
                        }
                    }
                }
            }
            return out;
        }
    };

    struct PositioningConfig {
        /// Between the posts' inner edges (m), and the ball's radius: aim points stay a radius inside the posts
        double goal_width  = 2.5;
        double ball_radius = 0.0785;
        /// Rolling ball model the policy was trained on (m/s^2)
        double deceleration = 0.5;
        /// Kick to a live Block command (s)
        double t_lat = 0.2;
        /// Shot speed prior (m/s): n_speed speeds spread evenly over [min_speed, max_speed]
        double min_speed = 1.5;
        double max_speed = 4.0;
        int n_aim        = 41;
        int n_speed      = 11;
        /// Score a spot by the mean save rate over the worst cvar_fraction of aim points (a shooter who picks the
        /// goalie's gaps), or over all of them
        bool cvar            = true;
        double cvar_fraction = 0.2;
        /// Subtracted from a spot's score per metre off the goal line
        double line_penalty = 0.0;
        /// Closest the goalie stands to the ball (m)
        double min_ball_distance = 0.5;
        /// Guesses at how much worse the real goalie is: every combination of an extra delay (s) and a reach scale
        std::vector<double> extra_lat{0.0, 0.2};
        std::vector<double> reach_scale{1.0, 0.7};
        /// Choose on the least worst-case regret over those guesses, or else on the mean score over them
        bool regret = true;
        /// Take the spot closest to the goal line whose score is within this of the best
        double near_best = 0.02;
        /// The spots considered (m): a grid from min_depth off the goal line out to max_depth (the penalty area
        /// line), and across to max_lateral either side
        double grid_step   = 0.1;
        double min_depth   = 0.05;
        double max_depth   = 3.0;
        double max_lateral = 3.0;
    };

    /// Where one shot meets the goalie's frontal plane
    struct PlaneCrossing {
        /// Whether it meets the plane before our goal line (else it is a goal whatever the goalie does)
        bool before_goal = false;
        /// Distance along the shot to the plane (m)
        double distance = std::numeric_limits<double>::infinity();
        /// Signed offset where it crosses (m, +left of the goalie)
        double dy = 0.0;
    };

    /**
     * A shot from `ball` along `u` (unit), goal_distance from our goal line, against a goalie at `g` facing `f`
     * (unit): where it crosses the plane (p - g).f = 0.
     */
    inline PlaneCrossing cross_plane(const Eigen::Vector2d& ball,
                                     const Eigen::Vector2d& u,
                                     const double goal_distance,
                                     const Eigen::Vector2d& g,
                                     const Eigen::Vector2d& f) {
        PlaneCrossing c{};
        const double uf    = u.dot(f);
        const double ahead = (ball - g).dot(f);
        if (!(uf < -1e-6 && ahead > 0.0)) {
            return c;
        }
        c.distance    = ahead / -uf;
        c.before_goal = c.distance <= goal_distance;
        const Eigen::Vector2d left(-f.y(), f.x());
        c.dy = (ball + c.distance * u - g).dot(left);
        return c;
    }

    /// Time for a ball kicked at v, slowing at a, to roll s, or nothing if it stops first
    inline std::optional<double> time_to_travel(const double s, const double v, const double a) {
        const double disc = v * v - 2.0 * a * s;
        if (disc < 0.0) {
            return std::nullopt;
        }
        return (v - std::sqrt(disc)) / a;
    }

    /// Every straight shot at the goal from one ball position: the aim points, and which speeds reach the line
    struct Shots {
        Eigen::Vector2d ball = Eigen::Vector2d::Zero();
        std::vector<Eigen::Vector2d> direction{};
        std::vector<double> goal_distance{};
        std::vector<double> speeds{};
        /// [aim][speed]: whether a shot at that speed reaches the goal line at all
        std::vector<char> threat{};
        /// Threatening speeds per aim point, and in total
        std::vector<int> n_threat{};
        int total_threat = 0;
        /// Aim points reached by at least one speed, which the CVaR is over
        int n_reached = 0;

        Shots(const Eigen::Vector2d& b, const PositioningConfig& cfg) : ball(b) {
            const double half = cfg.goal_width / 2.0 - cfg.ball_radius;
            const double speed_step = cfg.n_speed > 1 ? (cfg.max_speed - cfg.min_speed) / double(cfg.n_speed - 1) : 0.0;
            for (int s = 0; s < cfg.n_speed; ++s) {
                speeds.push_back(cfg.min_speed + speed_step * s);
            }
            for (int a = 0; a < cfg.n_aim; ++a) {
                const double y = cfg.n_aim == 1 ? 0.0 : -half + 2.0 * half * a / double(cfg.n_aim - 1);
                const Eigen::Vector2d to_aim = Eigen::Vector2d(0.0, y) - ball;
                goal_distance.push_back(to_aim.norm());
                direction.push_back(to_aim / to_aim.norm());
                int n = 0;
                for (const double v : speeds) {
                    const bool reaches = v * v >= 2.0 * cfg.deceleration * goal_distance.back();
                    threat.push_back(char(reaches));
                    n += int(reaches);
                }
                n_threat.push_back(n);
                total_threat += n;
                n_reached += int(n > 0);
            }
        }
    };

    /// A goalie's save probability against every shot in Shots, over all aim points and over the worst of them
    struct SpotScore {
        double mean = 0.0;
        double cvar = 0.0;
    };

    /// A guess at how much worse the real goalie is than mjlab's: later by extra_lat, reaching reach_scale as far
    struct Degradation {
        double extra_lat   = 0.0;
        double reach_scale = 1.0;
    };

    /**
     * Save probability of a goalie at g facing f against every shot, once per degradation: the geometry is shared,
     * only the capability lookups differ.
     */
    inline void score_spot(const Capability& S,
                           const PositioningConfig& cfg,
                           const Shots& shots,
                           const Eigen::Vector2d& g,
                           const Eigen::Vector2d& f,
                           const std::vector<Degradation>& degradations,
                           std::vector<SpotScore>& out,
                           std::vector<double>& per_aim) {
        const std::size_t n_deg   = degradations.size();
        const std::size_t n_aim   = shots.direction.size();
        const std::size_t n_speed = shots.speeds.size();
        out.assign(n_deg, SpotScore{});
        per_aim.assign(n_deg * n_aim, 0.0);

        for (std::size_t a = 0; a < n_aim; ++a) {
            if (shots.n_threat[a] == 0) {
                continue;
            }
            const PlaneCrossing c = cross_plane(shots.ball, shots.direction[a], shots.goal_distance[a], g, f);
            if (c.before_goal) {
                for (std::size_t s = 0; s < n_speed; ++s) {
                    if (!shots.threat[a * n_speed + s]) {
                        continue;
                    }
                    const double v = shots.speeds[s];
                    const double t = time_to_travel(c.distance, v, cfg.deceleration).value_or(0.0) - cfg.t_lat;
                    for (std::size_t k = 0; k < n_deg; ++k) {
                        const Degradation& d = degradations[k];
                        per_aim[k * n_aim + a] += S(c.dy / d.reach_scale, t - d.extra_lat, v);
                    }
                }
            }
            for (std::size_t k = 0; k < n_deg; ++k) {
                out[k].mean += per_aim[k * n_aim + a];
                per_aim[k * n_aim + a] /= shots.n_threat[a];
            }
        }

        const std::size_t worst = std::max<std::size_t>(
            1,
            std::size_t(std::nearbyint(cfg.cvar_fraction * double(std::max(shots.n_reached, 1)))));
        std::vector<double> reached{};
        for (std::size_t k = 0; k < n_deg; ++k) {
            out[k].mean /= std::max(shots.total_threat, 1);
            reached.clear();
            for (std::size_t a = 0; a < n_aim; ++a) {
                if (shots.n_threat[a] > 0) {
                    reached.push_back(per_aim[k * n_aim + a]);
                }
            }
            if (reached.empty()) {
                continue;
            }
            const std::size_t n = std::min(worst, reached.size());
            std::partial_sort(reached.begin(), reached.begin() + std::ptrdiff_t(n), reached.end());
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                sum += reached[i];
            }
            out[k].cvar = sum / double(n);
        }
    }

    /// The spot chosen for one ball position
    struct Position {
        /// Where the goalie stands, in {g}
        Eigen::Vector2d rGg = Eigen::Vector2d::Zero();
        /// Its save probability on the mjlab capability, over all aim points and the worst of them
        double mean = 0.0;
        double cvar = 0.0;
        /// Its worst-case regret over the degradations: how far short of each one's best spot it falls, at most
        double regret = 0.0;
    };

    /// The grid of spots the goalie can stand on: x from min_depth to max_depth, y across +-max_lateral
    struct SpotGrid {
        Axis x{};
        Axis y{};

        explicit SpotGrid(const PositioningConfig& cfg) {
            const double h = cfg.grid_step;
            x = Axis{cfg.min_depth, h, std::size_t(std::floor((cfg.max_depth - cfg.min_depth) / h + 1e-9)) + 1};
            y = Axis{-cfg.max_lateral, h, std::size_t(std::floor(2.0 * cfg.max_lateral / h + 1e-9)) + 1};
        }
        /// Spots are ordered y-major (all x at the first y, then the next), as the prototype's meshgrid
        [[nodiscard]] std::size_t size() const {
            return x.count * y.count;
        }
        [[nodiscard]] Eigen::Vector2d at(const std::size_t i) const {
            return {x.at(i % x.count), y.at(i / x.count)};
        }
        /// The spot nearest a point, if it is on the grid
        [[nodiscard]] std::optional<std::size_t> nearest(const Eigen::Vector2d& p) const {
            const double i = std::round((p.x() - x.start) / x.step);
            const double j = std::round((p.y() - y.start) / y.step);
            if (i < 0 || j < 0 || i >= double(x.count) || j >= double(y.count)) {
                return std::nullopt;
            }
            return std::size_t(j) * x.count + std::size_t(i);
        }
    };

    /**
     * Where the goalie should stand against a ball at `ball` (in {g}), facing it: the spot closest to the goal line
     * among those within near_best of the best score, where a spot's score is its least worst-case regret over the
     * degradations (or its mean over them). Nothing if no shot from there reaches the goal, or no spot is allowed.
     *
     * `hold` is the spot the goalie is already going to. It is kept while it is still within near_best of the best,
     * so the target doesn't hop between spots that are as good as each other.
     */
    inline std::optional<Position> choose_position(const Capability& S,
                                                   const PositioningConfig& cfg,
                                                   const Eigen::Vector2d& ball,
                                                   const std::optional<Eigen::Vector2d>& hold = std::nullopt) {
        const Shots shots(ball, cfg);
        if (shots.total_threat == 0) {
            return std::nullopt;
        }

        std::vector<Degradation> degradations{};
        for (const double l : cfg.extra_lat) {
            for (const double k : cfg.reach_scale) {
                degradations.push_back({l, k});
            }
        }
        const std::size_t n_deg = degradations.size();
        if (n_deg == 0) {
            return std::nullopt;
        }

        // Every allowed spot's objective under every degradation
        const SpotGrid grid(cfg);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> J(n_deg * grid.size(), nan);
        std::vector<double> best(n_deg, -std::numeric_limits<double>::infinity());
        std::vector<SpotScore> scores{};
        std::vector<double> per_aim{};
        for (std::size_t i = 0; i < grid.size(); ++i) {
            const Eigen::Vector2d g = grid.at(i);
            const double distance   = (ball - g).norm();
            if (distance < cfg.min_ball_distance || g.x() >= ball.x()) {
                continue;
            }
            score_spot(S, cfg, shots, g, (ball - g) / distance, degradations, scores, per_aim);
            for (std::size_t k = 0; k < n_deg; ++k) {
                const double j       = (cfg.cvar ? scores[k].cvar : scores[k].mean) - cfg.line_penalty * g.x();
                J[k * grid.size() + i] = j;
                best[k]                = std::max(best[k], j);
            }
        }

        // Score each spot on its worst-case regret (higher is better), or its mean over the degradations
        std::vector<double> score(grid.size(), nan);
        std::vector<double> regret(grid.size(), nan);
        double top = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < grid.size(); ++i) {
            if (std::isnan(J[i])) {
                continue;
            }
            double r = -std::numeric_limits<double>::infinity();
            double m = 0.0;
            for (std::size_t k = 0; k < n_deg; ++k) {
                r = std::max(r, best[k] - J[k * grid.size() + i]);
                m += J[k * grid.size() + i];
            }
            regret[i] = r;
            score[i]  = cfg.regret ? -r : m / double(n_deg);
            top       = std::max(top, score[i]);
        }
        if (!std::isfinite(top)) {
            return std::nullopt;
        }

        // Keep going to the held spot while it is still near-best; else the shallowest near-best spot
        std::optional<std::size_t> chosen{};
        if (hold) {
            const auto h = grid.nearest(*hold);
            if (h && !std::isnan(score[*h]) && score[*h] >= top - cfg.near_best) {
                chosen = h;
            }
        }
        if (!chosen) {
            for (std::size_t i = 0; i < grid.size(); ++i) {
                if (std::isnan(score[i]) || score[i] < top - cfg.near_best) {
                    continue;
                }
                const double x_i = grid.at(i).x();
                if (!chosen) {
                    chosen = i;
                    continue;
                }
                const double x_c = grid.at(*chosen).x();
                if (x_i < x_c - 1e-9 || (x_i < x_c + 1e-9 && score[i] > score[*chosen])) {
                    chosen = i;
                }
            }
        }

        Position p{};
        p.rGg    = grid.at(*chosen);
        p.regret = regret[*chosen];
        score_spot(S, cfg, shots, p.rGg, (ball - p.rGg).normalized(), {Degradation{}}, scores, per_aim);
        p.mean = scores[0].mean;
        p.cvar = scores[0].cvar;
        return p;
    }

}  // namespace module::planning::save

#endif  // MODULE_PLANNING_PLANSAVE_POSITIONING_HPP
