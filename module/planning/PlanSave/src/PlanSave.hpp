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
#ifndef MODULE_PLANNING_PLANSAVE_HPP
#define MODULE_PLANNING_PLANSAVE_HPP

#include <Eigen/Core>
#include <atomic>
#include <memory>
#include <mutex>
#include <nuclear>
#include <optional>
#include <string>

#include "decision.hpp"
#include "envelope.hpp"
#include "handoff.hpp"
#include "positioning.hpp"
#include "prediction.hpp"

#include "extension/Behaviour.hpp"

#include "message/localisation/Ball.hpp"

namespace module::planning {

    class PlanSave : public ::extension::behaviour::BehaviourReactor {
    private:
        /// @brief Stores configuration values
        struct Config {
            /// @brief Ball model the block policy's command was trained with
            save::RollingModel ball{};
            /// @brief Longest time to arrival (s) the policy is told about a shot
            double max_time = 0.0;
            /// @brief Oldest ball estimate (s) planned with
            double ball_timeout = 0.0;
            /// @brief Window after a kick (s) over which the velocity covariance is widened, and the factor in sigma
            double kick_window = 0.0;
            double kick_factor = 1.0;
            /// @brief Mode switching
            save::DecisionConfig decision{};
            /// @brief Expected success below which a block is logged as a gap in what the goalie can do
            double block_threshold = 0.0;
            /// @brief Wilson lower bound z and minimum shots per envelope cell
            double confidence_z = 0.0;
            int min_trials      = 0;
            /// @brief Whether PlanSave positions the goalie while no shot is on its way, instead of the walk below it
            bool positioning = false;
            /// @brief Oldest target (s) walked to; older, PlanSave leaves positioning to the walk below it
            double target_timeout = 0.0;
            /// @brief When a walking goalie is handed to the block policy
            save::HandoffConfig handoff{};
        } cfg;

        /// @brief Guards the positioning configuration and target, shared with the positioning reaction
        std::mutex positioning_mutex{};
        /// @brief How the goalie's spot is chosen. The field's goal, ball and penalty area fill in the rest.
        save::PositioningConfig positioning_cfg{};
        /// @brief The spot the goalie is walking to while no shot is on its way, in the goal frame {g}
        struct Target {
            save::Position position{};
            /// When it was chosen, and how long choosing it took (s)
            NUClear::clock::time_point time{};
            double compute_time = 0.0;
        };
        std::optional<Target> target{};

        /// @brief Whether the Save task is running, and the mode of the last tick, for the positioning reaction
        std::atomic<bool> save_running{false};
        std::atomic<bool> blocking{false};

        /// @brief Guards the envelope and the policy check, which configuration updates change under the planner
        std::mutex envelope_mutex{};
        /// @brief The block policy's capability envelope, and the policy file it must have been measured from
        std::optional<save::Envelope> envelope{};
        std::optional<std::string> policy_path{};
        /// @brief Whether the envelope matches the deployed policy; without that, PlanSave only ever lets the walk run
        bool envelope_ok = false;
        /// @brief The block policy's smoothed capability, which the goalie is positioned with, and whether it matches
        /// the deployed policy. Without that, positioning is left to the walk.
        std::shared_ptr<const save::Capability> capability{};
        bool capability_ok = false;
        /// @brief Checks the deployed policy's SHA-256 against the envelope and capability once they are known. Call
        /// with envelope_mutex held.
        void check_policy();

        /// @brief Mode switching, stuck to one shot at a time
        save::Decider decider{};
        /// @brief Holds a walking goalie's hand-off to the block policy until both its feet are down
        save::HandoffGate handoff_gate{};

        /// @brief The last ball estimate seen, to notice a new one, and when the current shot started rolling
        std::shared_ptr<const message::localisation::Ball> last_ball{};
        bool rolling = false;
        NUClear::clock::time_point kick_time{};
        NUClear::clock::time_point last_tick{};
        /// Throttles the warning that NUSim's forecast is missing while planning on it
        NUClear::clock::time_point last_forecast_warning{};

    public:
        /// @brief Called by the powerplant to build and setup the PlanSave reactor.
        explicit PlanSave(std::unique_ptr<NUClear::Environment> environment);
    };

}  // namespace module::planning

#endif  // MODULE_PLANNING_PLANSAVE_HPP
