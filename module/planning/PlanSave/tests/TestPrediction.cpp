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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "prediction.hpp"

using Catch::Approx;
using module::planning::save::Covariance;
using module::planning::save::predict_crossing;
using module::planning::save::probability_between;
using module::planning::save::RollingModel;
using module::planning::save::State;
using module::planning::save::transform;

namespace {
    State state(double x, double y, double vx, double vy) {
        State s;
        s << x, y, vx, vy;
        return s;
    }
}  // namespace

SCENARIO("A rolling ball's crossing of the goalie's line matches the training model", "[PlanSave]") {
    const RollingModel model{0.5, 0.3};

    GIVEN("a ball 3 m out rolling straight at the line at 3 m/s") {
        const auto c = predict_crossing(state(3.0, 0.2, -3.0, 0.0), model);
        THEN("it crosses in front of where it started, after the decelerating travel time") {
            REQUIRE(c.reaches);
            REQUIRE(c.offset == Approx(0.2));
            // 3 t - 0.25 t^2 = 3, first root
            REQUIRE(c.time == Approx((3.0 - std::sqrt(9.0 - 3.0)) / 0.5));
        }
    }

    GIVEN("a ball rolling across at an angle") {
        // Heading (-2, 1): over 2 m of x it moves 1 m of y, and travels sqrt(5) m
        const auto c = predict_crossing(state(2.0, -0.5, -2.0, 1.0), model);
        THEN("it crosses where the straight line meets x = 0") {
            REQUIRE(c.reaches);
            REQUIRE(c.offset == Approx(0.5));
            const double v = std::sqrt(5.0);
            const double d = std::sqrt(5.0);
            REQUIRE(c.time == Approx((v - std::sqrt(v * v - 2.0 * 0.5 * d)) / 0.5));
        }
    }

    GIVEN("balls that never get there") {
        THEN("a ball rolling away does not reach") {
            REQUIRE_FALSE(predict_crossing(state(3.0, 0.0, 2.0, 0.0), model).reaches);
        }
        THEN("a ball behind the line does not reach") {
            REQUIRE_FALSE(predict_crossing(state(-0.5, 0.0, -2.0, 0.0), model).reaches);
        }
        THEN("a slow ball stops short") {
            // Stops after v^2 / 2a = 1 m
            REQUIRE_FALSE(predict_crossing(state(1.5, 0.0, -1.0, 0.0), model).reaches);
            REQUIRE(predict_crossing(state(0.9, 0.0, -1.0, 0.0), model).reaches);
        }
        THEN("a ball slower than a shot is not one") {
            REQUIRE_FALSE(predict_crossing(state(0.05, 0.0, -0.2, 0.0), model).reaches);
        }
    }
}

SCENARIO("The crossing's uncertainty follows the ball covariance", "[PlanSave]") {
    const RollingModel model{0.5, 0.3};

    GIVEN("a ball rolling straight at the line with uncertain sideways position and velocity") {
        const double x0 = 3.0, v = 3.0, sy = 0.05, svy = 0.2;
        Covariance P = Covariance::Zero();
        P(1, 1)      = sy * sy;
        P(3, 3)      = svy * svy;
        const auto u = predict_crossing(state(x0, 0.0, -v, 0.0), P, model);
        THEN("sigma(dy) is the position spread plus the heading spread carried over the distance") {
            // dy = y + (vy / v) x0 to first order
            REQUIRE(u.sigma_offset == Approx(std::sqrt(sy * sy + (x0 / v) * (x0 / v) * svy * svy)).epsilon(1e-4));
        }
        THEN("sideways uncertainty barely moves the arrival time") {
            REQUIRE(u.sigma_time == Approx(0.0).margin(1e-3));
        }
    }

    GIVEN("uncertain speed along the path") {
        Covariance P = Covariance::Zero();
        P(2, 2)      = 0.3 * 0.3;
        const auto u = predict_crossing(state(3.0, 0.0, -3.0, 0.0), P, model);
        THEN("sigma(t) is |dt/dv| sigma(v)") {
            // t(v) = (v - sqrt(v^2 - 2 a d)) / a, so dt/dv = (1 - v / sqrt(v^2 - 2 a d)) / a
            const double root = std::sqrt(9.0 - 3.0);
            REQUIRE(u.sigma_time == Approx(std::abs((1.0 - 3.0 / root) / 0.5) * 0.3).epsilon(1e-4));
            REQUIRE(u.sigma_offset == Approx(0.0).margin(1e-6));
        }
    }
}

SCENARIO("Probabilities and frames", "[PlanSave]") {
    THEN("a Gaussian centred in an interval one sigma wide each way holds 68%") {
        REQUIRE(probability_between(0.0, 1.0, -1.0, 1.0) == Approx(0.6827).epsilon(1e-3));
    }
    THEN("a point estimate is in or out") {
        REQUIRE(probability_between(0.5, 0.0, 0.0, 1.0) == 1.0);
        REQUIRE(probability_between(1.5, 0.0, 0.0, 1.0) == 0.0);
    }

    GIVEN("a ball rolling at our goal, which is at +x on a 9 m field") {
        // The goal frame {g} is the field frame turned half way round and moved to the goal line
        State x      = state(3.0, 0.5, 2.0, -0.2);
        Covariance P = Covariance::Identity();
        P(0, 0)      = 0.04;
        P(1, 1)      = 0.01;
        transform(M_PI, Eigen::Vector2d(4.5, 0.0), x, P);
        THEN("it is in front of the goal line and closing on it, mirrored sideways") {
            REQUIRE(x[0] == Approx(1.5));
            REQUIRE(x[1] == Approx(-0.5));
            REQUIRE(x[2] == Approx(-2.0));
            REQUIRE(x[3] == Approx(0.2));
            REQUIRE(P(0, 0) == Approx(0.04));
            REQUIRE(P(1, 1) == Approx(0.01));
            REQUIRE(P(0, 1) == Approx(0.0).margin(1e-12));
        }
    }
}
