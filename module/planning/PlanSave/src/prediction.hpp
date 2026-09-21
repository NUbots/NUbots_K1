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
#ifndef MODULE_PLANNING_PLANSAVE_PREDICTION_HPP
#define MODULE_PLANNING_PLANSAVE_PREDICTION_HPP

#include <Eigen/Core>
#include <algorithm>
#include <cmath>

namespace module::planning::save {

    /// Ball state on the field plane: position (x, y) then velocity (vx, vy), as the ball filter orders it
    using State      = Eigen::Matrix<double, 4, 1>;
    using Covariance = Eigen::Matrix<double, 4, 4>;

    /// Rolling ball model, shared with the mjlab goalkeeper task's shot command
    struct RollingModel {
        /// Constant deceleration (m/s^2) while the ball rolls
        double deceleration = 0.5;
        /// Slowest ball (m/s) counted as travelling at all
        double min_speed = 0.3;
    };

    /// Where and when a rolling ball crosses a line
    struct Crossing {
        /// Offset along the line (m) at which the ball crosses it
        double offset = 0.0;
        /// Time (s) until it does
        double time = 0.0;
        /// Whether it gets there at all: in front, closing, and not stopped short
        bool reaches = false;
    };

    /// Crossing with its uncertainty
    struct UncertainCrossing {
        Crossing crossing{};
        double sigma_offset = 0.0;
        double sigma_time   = 0.0;
    };

    /**
     * Where and when a ball rolling straight and slowing at a constant rate crosses the line x = 0 of the frame its
     * state is in, approaching from +x. This is ShotCommand._predict_crossing in the mjlab goalkeeper task, which is
     * what the block policy's command was computed with in training, so keep the two identical.
     */
    inline Crossing predict_crossing(const State& x, const RollingModel& model) {
        const Eigen::Vector2d p = x.head<2>();
        const Eigen::Vector2d v = x.tail<2>();
        const double speed      = v.norm();
        const Eigen::Vector2d direction = v / std::max(speed, 1e-6);

        // Distance along the path to the line, positive only while the ball is in front and closing
        const double closing     = -direction.x();
        const double distance    = p.x() / std::max(closing, 1e-6);
        const bool approaching   = closing > 1e-3 && p.x() > 0.0;
        const double a           = model.deceleration;
        const double discriminant = speed * speed - 2.0 * a * distance;

        Crossing c{};
        c.offset  = p.y() + direction.y() * distance;
        c.reaches = approaching && discriminant > 0.0 && speed > model.min_speed;
        // v t - a t^2 / 2 = distance, the first root; constant velocity without deceleration
        c.time = a > 1e-9 ? (speed - std::sqrt(std::max(discriminant, 0.0))) / a : distance / std::max(speed, 1e-6);
        return c;
    }

    /**
     * Crossing of the line x = 0 with its standard deviations, carrying the state covariance through a numerical
     * Jacobian. The crossing is smooth wherever the ball reaches the line; at the edge of reaching it, a perturbed
     * state that no longer reaches it falls back to a one-sided difference.
     */
    inline UncertainCrossing predict_crossing(const State& x, const Covariance& P, const RollingModel& model) {
        UncertainCrossing out{};
        out.crossing = predict_crossing(x, model);
        if (!out.crossing.reaches) {
            return out;
        }

        Eigen::Matrix<double, 2, 4> J = Eigen::Matrix<double, 2, 4>::Zero();
        for (int i = 0; i < 4; ++i) {
            const double h = 1e-4 * std::max(1.0, std::abs(x[i]));
            State plus     = x;
            State minus    = x;
            plus[i] += h;
            minus[i] -= h;
            const Crossing cp = predict_crossing(plus, model);
            const Crossing cm = predict_crossing(minus, model);
            const Crossing& hi = cp.reaches ? cp : out.crossing;
            const Crossing& lo = cm.reaches ? cm : out.crossing;
            const double span  = (cp.reaches ? h : 0.0) + (cm.reaches ? h : 0.0);
            if (span > 0.0) {
                J(0, i) = (hi.offset - lo.offset) / span;
                J(1, i) = (hi.time - lo.time) / span;
            }
        }
        const Eigen::Matrix2d S = J * P * J.transpose();
        out.sigma_offset        = std::sqrt(std::max(S(0, 0), 0.0));
        out.sigma_time          = std::sqrt(std::max(S(1, 1), 0.0));
        return out;
    }

    /// Standard normal cumulative distribution
    inline double normal_cdf(const double z) {
        return 0.5 * std::erfc(-z / std::sqrt(2.0));
    }

    /// Probability that a Gaussian with this mean and standard deviation lies in [lo, hi]
    inline double probability_between(const double mean, const double sigma, const double lo, const double hi) {
        if (hi <= lo) {
            return 0.0;
        }
        if (sigma < 1e-9) {
            return mean >= lo && mean < hi ? 1.0 : 0.0;
        }
        return std::max(0.0, normal_cdf((hi - mean) / sigma) - normal_cdf((lo - mean) / sigma));
    }

    /**
     * Moves a planar state and its covariance into another frame on the field plane: a rotation by yaw then a
     * translation for the position, the rotation alone for the velocity.
     */
    inline void transform(const double yaw,
                          const Eigen::Vector2d& translation,
                          State& x,
                          Covariance& P) {
        const Eigen::Matrix2d R = (Eigen::Matrix2d() << std::cos(yaw), -std::sin(yaw), std::sin(yaw), std::cos(yaw))
                                      .finished();
        Covariance T                = Covariance::Zero();
        T.topLeftCorner<2, 2>()     = R;
        T.bottomRightCorner<2, 2>() = R;
        x                           = T * x;
        x.head<2>() += translation;
        P = T * P * T.transpose();
    }

}  // namespace module::planning::save

#endif  // MODULE_PLANNING_PLANSAVE_PREDICTION_HPP
