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

#include "handoff.hpp"

using Catch::Approx;
using module::planning::save::ankle_height_difference;
using module::planning::save::ankle_in_trunk;
using module::planning::save::HandoffConfig;
using module::planning::save::HandoffGate;
using module::planning::save::LegAngles;

SCENARIO("A K1's ankles are where its leg chain puts them", "[PlanSave][handoff]") {
    const LegAngles zero{};
    // The ready stance: hips and knees bent as in K1BlockPolicy's default pose
    const LegAngles stance{-0.2, 0.0, 0.0, 0.4};
    const Eigen::Matrix3d upright = Eigen::Matrix3d::Identity();

    THEN("with straight legs each ankle is the URDF's offsets below its hip") {
        const Eigen::Vector3d left = ankle_in_trunk(zero, true);
        REQUIRE(left.x() == Approx(0.012 - 0.014 + 0.00019706));
        REQUIRE(left.y() == Approx(0.096 + 0.0002));
        REQUIRE(left.z() == Approx(-0.077 - 0.026 - 0.0485 - 0.117 - 0.24519));
        REQUIRE(ankle_in_trunk(zero, false).y() == Approx(-left.y()));
    }
    THEN("standing on both feet the ankles are level") {
        REQUIRE(ankle_height_difference(stance, stance, upright) == Approx(0.0).margin(1e-12));
    }
    THEN("a leg drawn up has its ankle higher") {
        const LegAngles swing{-0.6, 0.0, 0.0, 1.0};
        REQUIRE(ankle_height_difference(swing, stance, upright) > 0.03);
        REQUIRE(ankle_height_difference(stance, swing, upright) < -0.03);
    }
    THEN("a trunk rolled to its right lifts its left side") {
        const double roll         = 0.1;
        const Eigen::Matrix3d Rwt = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
        const double spread       = ankle_in_trunk(stance, true).y() - ankle_in_trunk(stance, false).y();
        REQUIRE(ankle_height_difference(stance, stance, Rwt) == Approx(spread * std::sin(roll)));
    }
}

SCENARIO("A walking goalie is handed to the block policy with both feet down", "[PlanSave][handoff]") {
    const HandoffConfig cfg{};  // 0.5 cm, 0.35 s
    const double dt = 0.02;
    HandoffGate gate{};

    THEN("nothing is handed over while the planner doesn't want the block policy") {
        REQUIRE_FALSE(gate.step(false, true, 0.0, dt, cfg));
    }
    THEN("a goalie that isn't walking is handed over at once") {
        REQUIRE(gate.step(true, false, 0.05, dt, cfg));
        REQUIRE_FALSE(gate.timed_out);
    }
    THEN("a walking goalie with a foot up waits, and is handed over when the foot is down") {
        REQUIRE_FALSE(gate.step(true, true, 0.04, dt, cfg));
        REQUIRE_FALSE(gate.step(true, true, 0.02, dt, cfg));
        REQUIRE(gate.waiting());
        REQUIRE(gate.step(true, true, 0.002, dt, cfg));
        REQUIRE(gate.last_wait == Approx(2 * dt));
        REQUIRE_FALSE(gate.timed_out);

        AND_THEN("it stays with the block policy while the planner wants it, whatever the feet do") {
            REQUIRE(gate.step(true, true, 0.05, dt, cfg));
            REQUIRE_FALSE(gate.step(false, true, 0.05, dt, cfg));
            REQUIRE_FALSE(gate.step(true, true, 0.05, dt, cfg));
        }
    }
    THEN("a walking goalie whose feet never come down is handed over after max_wait") {
        int ticks = 0;
        while (!gate.step(true, true, 0.05, dt, cfg)) {
            ++ticks;
            REQUIRE(ticks < 100);
        }
        REQUIRE(ticks * dt >= cfg.max_wait - 1e-9);
        REQUIRE(ticks * dt < cfg.max_wait + dt + 1e-9);
        REQUIRE(gate.timed_out);
    }
    THEN("without a way to see the feet it waits out max_wait") {
        REQUIRE_FALSE(gate.step(true, true, std::nullopt, dt, cfg));
    }
    THEN("disabled, it hands over at once") {
        HandoffConfig off = cfg;
        off.enabled       = false;
        REQUIRE(gate.step(true, true, 0.05, dt, off));
    }
}
