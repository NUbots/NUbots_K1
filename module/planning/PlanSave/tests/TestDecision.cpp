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

#include <catch2/catch_test_macros.hpp>

#include "decision.hpp"

using module::planning::save::DecisionConfig;
using module::planning::save::Decider;
using module::planning::save::Mode;
using module::planning::save::Situation;

namespace {
    constexpr double DT = 0.02;

    Situation ball_at(double distance, double speed = 0.0) {
        Situation s{};
        s.ball_valid = true;
        s.distance   = distance;
        s.ahead      = distance;
        s.speed      = speed;
        return s;
    }

    Situation shot(double p_on_target) {
        Situation s   = ball_at(3.5, 3.0);
        s.active      = true;
        s.p_on_target = p_on_target;
        return s;
    }
}  // namespace

SCENARIO("The goalie guards a near ball and blocks shots on target", "[PlanSave]") {
    const DecisionConfig cfg{};
    Decider decider{};

    THEN("a far ball leaves it to the walk") {
        REQUIRE(decider.step(ball_at(6.0), DT, cfg) == Mode::IDLE);
    }
    THEN("no ball leaves it to the walk") {
        REQUIRE(decider.step(Situation{}, DT, cfg) == Mode::IDLE);
    }
    THEN("a near ball holds the ready stance, with hysteresis") {
        REQUIRE(decider.step(ball_at(2.5), DT, cfg) == Mode::GUARD);
        REQUIRE(decider.step(ball_at(3.2), DT, cfg) == Mode::GUARD);
        REQUIRE(decider.step(ball_at(3.6), DT, cfg) == Mode::IDLE);
        REQUIRE(decider.step(ball_at(3.2), DT, cfg) == Mode::IDLE);
    }
    THEN("a shot going wide is not blocked") {
        REQUIRE(decider.step(shot(0.2), DT, cfg) == Mode::IDLE);
    }
    THEN("a shot on target is blocked from anywhere") {
        REQUIRE(decider.step(shot(0.9), DT, cfg) == Mode::BLOCK);
    }
}

SCENARIO("A block sticks with its shot until the shot is over", "[PlanSave]") {
    const DecisionConfig cfg{};
    Decider decider{};
    REQUIRE(decider.step(shot(0.9), DT, cfg) == Mode::BLOCK);

    THEN("it holds while the prediction wanders off target") {
        REQUIRE(decider.step(shot(0.1), DT, cfg) == Mode::BLOCK);
    }
    THEN("it holds through a short loss of the ball") {
        REQUIRE(decider.step(Situation{}, DT, cfg) == Mode::BLOCK);
    }
    THEN("it releases release_delay after the ball stops, to guard the ball it stopped") {
        Mode mode = Mode::BLOCK;
        int ticks = 0;
        while (mode == Mode::BLOCK && ticks < 100) {
            mode = decider.step(ball_at(0.4, 0.0), DT, cfg);
            ++ticks;
        }
        REQUIRE(mode == Mode::GUARD);
        REQUIRE(ticks == 25);  // 0.5 s at 50 Hz
    }
    THEN("the delay restarts if the ball moves again") {
        for (int i = 0; i < 20; ++i) {
            decider.step(ball_at(0.4, 0.0), DT, cfg);
        }
        REQUIRE(decider.step(shot(0.9), DT, cfg) == Mode::BLOCK);
        for (int i = 0; i < 20; ++i) {
            REQUIRE(decider.step(ball_at(0.4, 0.0), DT, cfg) == Mode::BLOCK);
        }
    }
    THEN("a ball that got past is over, and does not start another block") {
        Situation behind = shot(0.9);
        behind.ahead     = -0.5;
        Mode mode        = Mode::BLOCK;
        for (int i = 0; i < 30; ++i) {
            mode = decider.step(behind, DT, cfg);
        }
        REQUIRE(mode != Mode::BLOCK);
    }
}
