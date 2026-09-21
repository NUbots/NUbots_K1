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
#ifndef MODULE_PLANNING_PLANSAVE_DECISION_HPP
#define MODULE_PLANNING_PLANSAVE_DECISION_HPP

namespace module::planning::save {

    /// What the goalie is doing about the ball
    enum class Mode {
        /// No threat: the walk positions the goalie
        IDLE,
        /// Ball near: hold the block policy's ready stance, where its envelope was measured from
        GUARD,
        /// Shot on its way: run the block policy with a live command
        BLOCK,
    };

    struct DecisionConfig {
        double min_p_on_target  = 0.5;
        double guard_distance   = 3.0;
        double guard_hysteresis = 0.5;
        double guard_min_ahead  = 0.3;
        double release_delay    = 0.5;
        double min_shot_speed   = 0.3;
    };

    /// One tick's view of the ball, from PlanSave's prediction
    struct Situation {
        /// Whether a fresh ball estimate from our own vision exists; nothing below means anything without one
        bool ball_valid = false;
        /// Ball distance from the goalie (m) and how far in front of its line (m, negative behind)
        double distance = 0.0;
        double ahead    = 0.0;
        double speed    = 0.0;
        /// The block policy's training rule for an active command: the ball reaches the line in time
        bool active = false;
        /// Probability the shot goes between the posts
        double p_on_target = 0.0;
    };

    /**
     * Picks the goalie's mode tick to tick. A block starts on a shot that is both on its way (the training rule) and
     * likely to go in, and is stuck with until that shot is over: the ball no longer on its way (stopped, past the
     * goalie or going away) or lost, for release_delay seconds. There is no other skill to switch to mid-shot yet; when there is (a dive),
     * switching is where "unless the prediction jumps" belongs.
     */
    class Decider {
    public:
        Mode step(const Situation& s, const double dt, const DecisionConfig& cfg) {
            if (mode == Mode::BLOCK) {
                const bool over = !s.ball_valid || !s.active;
                release_timer   = over ? release_timer + dt : 0.0;
                if (release_timer < cfg.release_delay) {
                    return mode;
                }
                release_timer = 0.0;
                mode          = Mode::IDLE;
            }

            if (!s.ball_valid) {
                mode = Mode::IDLE;
            }
            else if (s.active && s.p_on_target >= cfg.min_p_on_target && s.ahead > 0.0) {
                mode = Mode::BLOCK;
            }
            // Only a ball out in front: the ready stance faces the field, and vision's false positives on the
            // goalie's own feet and hands would otherwise hold it there for good (and keep its head on them)
            else if (s.ahead > cfg.guard_min_ahead
                     && (s.distance < cfg.guard_distance
                         || (mode == Mode::GUARD && s.distance < cfg.guard_distance + cfg.guard_hysteresis))) {
                mode = Mode::GUARD;
            }
            else {
                mode = Mode::IDLE;
            }
            return mode;
        }

        [[nodiscard]] Mode current() const {
            return mode;
        }

        void reset() {
            mode          = Mode::IDLE;
            release_timer = 0.0;
        }

    private:
        Mode mode            = Mode::IDLE;
        double release_timer = 0.0;
    };

}  // namespace module::planning::save

#endif  // MODULE_PLANNING_PLANSAVE_DECISION_HPP
