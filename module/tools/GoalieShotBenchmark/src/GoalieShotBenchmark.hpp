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
#ifndef MODULE_TOOLS_GOALIESHOTBENCHMARK_HPP
#define MODULE_TOOLS_GOALIESHOTBENCHMARK_HPP

#include <Eigen/Core>
#include <array>
#include <chrono>
#include <deque>
#include <fstream>
#include <nuclear>
#include <random>
#include <string>
#include <vector>

#include "extension/Behaviour.hpp"

namespace module::tools {

    /**
     * Rolls shots at the goalie in NUSim and scores them against ground truth, with planning::PlanSave guarding the
     * goal through the full stack (YOLO -> ball UKF -> PlanSave -> skill::K1BlockPolicy). See README.md.
     */
    class GoalieShotBenchmark : public ::extension::behaviour::BehaviourReactor {
    private:
        using Clock = std::chrono::system_clock;

        struct Config {
            double start_delay = 0.0;
            /// Publish Field from NUSim's ground truth instead of running field localisation
            bool ground_truth_field = false;
            /// Which ball the stack runs on: estimate, true_state or true_crossing (message::booster::NUSimBallSource)
            std::string ground_truth_ball{};
            unsigned int seed  = 0;
            int shots          = 0;
            /// Goalie's home spot in the simulator world {s}, facing +x; the goal line is line_offset behind it
            Eigen::Vector2d home = Eigen::Vector2d::Zero();
            double line_offset   = 0.0;
            /// Shot distribution around home, the mjlab goalkeeper task's "full" level
            std::array<double, 2> distance{};
            std::array<double, 2> lateral{};
            std::array<double, 2> crossing{};
            std::array<double, 2> speed{};
            double rolling_deceleration = 0.0;
            double settle_time          = 0.0;
            double max_roll_time        = 0.0;
            /// Torso heights (m): below fall_height the goalie has fallen, above upright_height it is up again
            double fall_height    = 0.0;
            double upright_height = 0.0;
            double recover_time   = 0.0;
            /// How close to home (m, rad) the goalie must be before a shot, and how long to wait for it (s)
            double home_tolerance = 0.0;
            double yaw_tolerance  = 0.0;
            double home_timeout   = 0.0;
            std::string output_dir{};
            bool shutdown_when_done = false;
            int save_priority       = 0;
            int walk_priority       = 0;
            int fall_recovery_priority = 0;
            int look_at_ball_priority  = 0;
            int look_around_priority   = 0;
        } cfg;

        enum class Phase { WAITING, SETTLING, ROLLING, RECOVERING, DONE };
        Phase phase = Phase::WAITING;
        Clock::time_point phase_since{};
        Clock::time_point start_after{};
        bool localisation_reset = false;

        struct Shot {
            int id = 0;
            Eigen::Vector2d start = Eigen::Vector2d::Zero();
            double crossing_y     = 0.0;
            double speed          = 0.0;
            Clock::time_point kicked{};
            bool on_target = false;
            /// Where the ball's path crosses the goalie's frontal plane, in its frame, and when, just after the kick
            double true_dy = NAN;
            double true_t  = NAN;
            /// Goalie's offset from home at the kick
            Eigen::Vector2d robot_offset = Eigen::Vector2d::Constant(NAN);
            double robot_yaw             = NAN;
            /// NUSim blew up and reset itself during the shot, which says nothing about the goalie
            bool sim_reset = false;
            /// PlanSave at the kick and at its first BLOCK tick of the shot
            int mode_at_kick      = -1;
            double block_reaction = NAN;
            double plan_dy        = NAN;
            double plan_sigma_dy  = NAN;
            double plan_t         = NAN;
            double plan_p_on_target = NAN;
            double plan_expected  = NAN;
            double closest        = INFINITY;
            bool fell             = false;
            std::string outcome{};
            bool conceded = false;
            double stopped_for = 0.0;
        } shot{};
        int shots_done = 0;

        std::mt19937 rng{};
        std::ofstream shots_csv{};
        std::vector<Shot> results{};

        /// Latest ground truth: ball centre and velocity, torso position, height and yaw, in {s}
        bool have_truth = false;
        Eigen::Vector2d ball  = Eigen::Vector2d::Zero();
        Eigen::Vector2d ball_v = Eigen::Vector2d::Zero();
        Eigen::Vector2d torso = Eigen::Vector2d::Zero();
        Clock::time_point last_waiting_log{};
        /// The last few seconds of ground truth and ball commands, dumped when NUSim resets itself
        std::deque<std::string> recent{};
        void remember(std::string line);
        /// Since when the goalie has been ready at home, while recovering
        Clock::time_point home_since{};
        double torso_z        = 0.0;
        double torso_yaw      = 0.0;
        int plan_mode         = -1;
        /// PlanSave's latest ball estimate in the robot frame, and whether it planned with one
        Eigen::Vector2d plan_ball = Eigen::Vector2d::Constant(NAN);
        bool plan_ball_valid      = false;
        /// Half the goal mouth (m), from FieldDescription
        double half_goal = 0.0;

        void place_next_shot();
        void kick();
        void track(double dt);
        void finish_shot(const std::string& outcome);
        /// Emits Save while a shot is set up or rolling, and withdraws it otherwise so the walk takes the goalie home
        void emit_save();
        void finish_run();

    public:
        explicit GoalieShotBenchmark(std::unique_ptr<NUClear::Environment> environment);
    };

}  // namespace module::tools

#endif  // MODULE_TOOLS_GOALIESHOTBENCHMARK_HPP
