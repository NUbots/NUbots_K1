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
#include <functional>
#include <string>
#include <yaml-cpp/yaml.h>

#include "positioning.hpp"

using Catch::Approx;
using module::planning::save::Axis;
using module::planning::save::Capability;
using module::planning::save::choose_position;
using module::planning::save::cross_plane;
using module::planning::save::PositioningConfig;
using module::planning::save::time_to_travel;

namespace {
    /// A capability filled from a function of the grid values
    Capability make_capability(const Axis& dy,
                               const Axis& time,
                               const Axis& speed,
                               const std::function<double(double, double, double)>& rate) {
        Capability c{};
        c.dy    = dy;
        c.time  = time;
        c.speed = speed;
        for (std::size_t i = 0; i < dy.count; ++i) {
            for (std::size_t j = 0; j < time.count; ++j) {
                for (std::size_t k = 0; k < speed.count; ++k) {
                    c.rate.push_back(rate(dy.at(i), time.at(j), speed.at(k)));
                }
            }
        }
        return c;
    }

    /// Saves a shot the less likely the further from the goalie it crosses, out to `reach`, however long it has
    Capability reach_only(const double reach) {
        const auto rate = [&](double dy, double, double) { return std::max(0.0, 1.0 - std::abs(dy) / reach); };
        return make_capability(Axis{-2.0, 0.05, 81}, Axis{0.0, 0.1, 31}, Axis{1.5, 0.5, 6}, rate);
    }

    /// Nominal only: no sim-to-real guesses
    PositioningConfig nominal() {
        PositioningConfig cfg{};
        cfg.extra_lat   = {0.0};
        cfg.reach_scale = {1.0};
        return cfg;
    }

    Axis read_axis(const YAML::Node& node) {
        return Axis{node["start"].as<double>(), node["step"].as<double>(), node["count"].as<std::size_t>()};
    }

    PositioningConfig read_config(const YAML::Node& node) {
        PositioningConfig cfg{};
        cfg.goal_width        = node["goal_width"].as<double>();
        cfg.ball_radius       = node["ball_radius"].as<double>();
        cfg.deceleration      = node["rolling_deceleration"].as<double>();
        cfg.t_lat             = node["t_lat"].as<double>();
        const auto speeds     = node["speed_range"].as<std::vector<double>>();
        cfg.min_speed         = speeds.at(0);
        cfg.max_speed         = speeds.at(1);
        cfg.n_aim             = node["n_aim"].as<int>();
        cfg.n_speed           = node["n_speed"].as<int>();
        cfg.cvar              = node["objective"].as<std::string>() == "cvar";
        cfg.cvar_fraction     = node["cvar_fraction"].as<double>();
        cfg.line_penalty      = node["line_penalty"].as<double>();
        cfg.min_ball_distance = node["min_ball_distance"].as<double>();
        cfg.extra_lat         = node["extra_lat"].as<std::vector<double>>();
        cfg.reach_scale       = node["reach_scale"].as<std::vector<double>>();
        cfg.regret            = node["robust"].as<std::string>() == "regret";
        cfg.near_best         = node["near_best"].as<double>();
        cfg.grid_step         = node["grid_step"].as<double>();
        cfg.min_depth         = node["min_depth"].as<double>();
        cfg.max_depth         = node["max_depth"].as<double>();
        cfg.max_lateral       = node["max_lateral"].as<double>();
        return cfg;
    }
}  // namespace

SCENARIO("The capability interpolates its grid trilinearly", "[PlanSave][positioning]") {
    // Linear in every axis, so trilinear interpolation is exact inside the grid
    const auto linear = [](double dy, double t, double v) { return (dy + 1.0) + 10.0 * t + 100.0 * (v - 1.0); };
    const Capability S = make_capability(Axis{-1.0, 1.0, 3}, Axis{0.0, 1.0, 3}, Axis{1.0, 1.0, 2}, linear);
    REQUIRE(S.valid());

    THEN("it reproduces a linear rate inside the grid") {
        REQUIRE(S(0.3, 1.4, 1.25) == Approx(linear(0.3, 1.4, 1.25)));
        REQUIRE(S(-1.0, 0.5, 2.0) == Approx(linear(-1.0, 0.5, 2.0)));
    }
    THEN("dy off the grid is a miss") {
        REQUIRE(S(1.01, 1.0, 1.5) == 0.0);
        REQUIRE(S(-1.01, 1.0, 1.5) == 0.0);
    }
    THEN("no time left is a miss") {
        REQUIRE(S(0.0, 0.0, 1.5) == 0.0);
        REQUIRE(S(0.0, -0.3, 1.5) == 0.0);
    }
    THEN("time and speed beyond the grid clamp to its edges") {
        REQUIRE(S(0.0, 7.0, 1.5) == Approx(linear(0.0, 2.0, 1.5)));
        REQUIRE(S(0.0, 1.0, 0.2) == Approx(linear(0.0, 1.0, 1.0)));
        REQUIRE(S(0.0, 1.0, 9.0) == Approx(linear(0.0, 1.0, 2.0)));
    }
}

SCENARIO("A shot meets the goalie's plane where similar triangles say", "[PlanSave][positioning]") {
    const Eigen::Vector2d ball(3.0, 0.0);
    const Eigen::Vector2d aim(0.0, 0.6);
    const Eigen::Vector2d u = (aim - ball).normalized();
    const double to_goal    = (aim - ball).norm();

    GIVEN("a goalie 1 m out facing the ball") {
        const auto c = cross_plane(ball, u, to_goal, Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(1.0, 0.0));
        THEN("the shot crosses two thirds of the way along, 0.4 m to the goalie's left") {
            REQUIRE(c.before_goal);
            REQUIRE(c.dy == Approx(0.4));
            REQUIRE(c.distance == Approx(std::sqrt(4.0 + 0.16)));
        }
    }
    GIVEN("a goalie facing away from the ball") {
        const auto c = cross_plane(ball, u, to_goal, Eigen::Vector2d(1.0, 0.0), Eigen::Vector2d(-1.0, 0.0));
        THEN("the shot never meets its plane") {
            REQUIRE_FALSE(c.before_goal);
        }
    }
    GIVEN("a goalie behind the goal line") {
        const auto c = cross_plane(ball, u, to_goal, Eigen::Vector2d(-0.5, 0.0), Eigen::Vector2d(1.0, 0.0));
        THEN("the shot is in the goal before it gets there") {
            REQUIRE_FALSE(c.before_goal);
        }
    }
}

SCENARIO("A rolling ball takes the decelerating time to travel", "[PlanSave][positioning]") {
    REQUIRE(time_to_travel(2.0, 2.0, 0.5).value() == Approx((2.0 - std::sqrt(2.0)) / 0.5));
    REQUIRE_FALSE(time_to_travel(5.0, 2.0, 0.5).has_value());
}

SCENARIO("The goalie's spot", "[PlanSave][positioning]") {
    GIVEN("a goalie that saves whatever crosses near it, however late") {
        const Capability S = reach_only(1.5);
        PositioningConfig cfg = nominal();
        cfg.near_best         = 0.0;
        const Eigen::Vector2d ball(6.0, 0.0);

        THEN("against a far ball it goes out as far as the penalty area line and no further") {
            // Further out narrows the triangle, and time doesn't matter: the best spot is the furthest allowed
            const auto p = choose_position(S, cfg, ball);
            REQUIRE(p.has_value());
            REQUIRE(p->rGg.x() <= cfg.max_depth);
            REQUIRE(p->rGg.x() > cfg.max_depth - cfg.grid_step);
            REQUIRE(p->rGg.y() == Approx(0.0).margin(1e-9));

            cfg.max_depth = 2.0;
            const auto q  = choose_position(S, cfg, ball);
            REQUIRE(q->rGg.x() <= 2.0);
            REQUIRE(q->rGg.x() > 2.0 - cfg.grid_step);
        }
        THEN("it keeps the spot it is going to while that is still near-best") {
            cfg.near_best                 = 1.0;  // every spot is near-best
            const Eigen::Vector2d current = Eigen::Vector2d(1.05, 0.3);  // on the grid
            REQUIRE(choose_position(S, cfg, ball, current)->rGg.isApprox(current));

            cfg.near_best = 0.0;  // only the best
            REQUIRE_FALSE(choose_position(S, cfg, ball, current)->rGg.isApprox(current));
        }
    }

    GIVEN("a ball no shot from which reaches the goal") {
        PositioningConfig cfg = nominal();
        cfg.min_speed         = 1.5;
        cfg.max_speed         = 2.0;  // rolls at most 4 m
        THEN("there is nothing to position for") {
            REQUIRE_FALSE(choose_position(reach_only(0.3), cfg, Eigen::Vector2d(4.5, 0.0)).has_value());
        }
    }

    GIVEN("a goalie that saves everything that reaches it") {
        const Capability S = make_capability(Axis{-2.0, 0.1, 41},
                                             Axis{0.0, 0.1, 31},
                                             Axis{1.5, 0.5, 6},
                                             [](double, double, double) { return 1.0; });
        PositioningConfig cfg = nominal();
        cfg.t_lat             = 0.0;
        THEN("it saves every shot, and has no regret") {
            const auto p = choose_position(S, cfg, Eigen::Vector2d(3.0, 1.0));
            REQUIRE(p->mean == Approx(1.0));
            REQUIRE(p->cvar == Approx(1.0));
            REQUIRE(p->regret == Approx(0.0).margin(1e-12));
            // Every spot is as good, so it stays as close to the goal line as it can
            REQUIRE(p->rGg.x() == Approx(cfg.min_depth));
        }
    }
}

SCENARIO("The deployed capability is usable and belongs to the deployed envelope's policy", "[PlanSave][positioning]") {
    // The module's config files, as the build copies them
    const YAML::Node cap      = YAML::LoadFile("config/SaveCapability.yaml");
    const YAML::Node envelope = YAML::LoadFile("config/SaveEnvelope.yaml");

    Capability S{};
    S.onnx_sha256 = cap["onnx_sha256"].as<std::string>();
    S.dy          = read_axis(cap["dy"]);
    S.time        = read_axis(cap["time"]);
    S.speed       = read_axis(cap["speed"]);
    for (const auto& plane : cap["rate"].as<std::vector<std::vector<std::vector<double>>>>()) {
        for (const auto& row : plane) {
            S.rate.insert(S.rate.end(), row.begin(), row.end());
        }
    }
    REQUIRE(S.valid());
    REQUIRE(S.onnx_sha256 == envelope["onnx_sha256"].as<std::string>());

    THEN("it positions the goalie inside the penalty area against a ball in front of goal") {
        const PositioningConfig cfg{};
        const auto p = choose_position(S, cfg, Eigen::Vector2d(4.5, 0.5));
        REQUIRE(p.has_value());
        REQUIRE(p->rGg.x() >= cfg.min_depth);
        REQUIRE(p->rGg.x() <= cfg.max_depth);
        REQUIRE(std::abs(p->rGg.y()) < cfg.goal_width / 2.0);
        REQUIRE(p->cvar > 0.0);
    }
}

SCENARIO("The goalie's spot matches tools/policy/save_positioning_prototype.py", "[PlanSave][positioning]") {
    // Regenerate with: python3 tools/policy/save_positioning_prototype.py --write-reference
    //                  module/planning/PlanSave/tests/data/PlanSave/positioning_reference.yaml
    const YAML::Node ref = YAML::LoadFile("tests/PlanSave/positioning_reference.yaml");

    const YAML::Node& cap = ref["capability"];
    Capability S{};
    S.dy    = read_axis(cap["dy"]);
    S.time  = read_axis(cap["time"]);
    S.speed = read_axis(cap["speed"]);
    for (const auto& plane : cap["rate"].as<std::vector<std::vector<std::vector<double>>>>()) {
        for (const auto& row : plane) {
            S.rate.insert(S.rate.end(), row.begin(), row.end());
        }
    }
    REQUIRE(S.valid());

    const PositioningConfig base = read_config(ref["config"]);
    for (const auto& c : ref["cases"]) {
        const auto ball             = c["ball"].as<std::vector<double>>();
        const PositioningConfig cfg = c["overrides"] ? read_config(c["overrides"]) : base;
        INFO("ball (" << ball[0] << ", " << ball[1] << ")" << (c["overrides"] ? " with overrides" : ""));

        const auto p = choose_position(S, cfg, Eigen::Vector2d(ball[0], ball[1]));
        REQUIRE(p.has_value() == c["reaches"].as<bool>());
        if (!p) {
            continue;
        }
        const auto target = c["target"].as<std::vector<double>>();
        CHECK(p->rGg.x() == Approx(target[0]).margin(1e-6));
        CHECK(p->rGg.y() == Approx(target[1]).margin(1e-6));
        CHECK(p->mean == Approx(c["mean"].as<double>()).margin(1e-6));
        CHECK(p->cvar == Approx(c["cvar"].as<double>()).margin(1e-6));
        CHECK(p->regret == Approx(c["regret"].as<double>()).margin(1e-6));
    }
}
