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
#ifndef MODULE_TOOLS_BALLLOCALISATIONBENCHMARK_HPP
#define MODULE_TOOLS_BALLLOCALISATIONBENCHMARK_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <array>
#include <chrono>
#include <deque>
#include <fstream>
#include <nuclear>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "extension/Behaviour.hpp"

namespace module::tools {

    /// Validates localisation::BallLocalisation against NUSim ground truth.
    ///
    /// The robot stands still and looks at the ball while NUSim (through input::NUSimGroundTruth)
    /// rolls the ball at it from a seeded random schedule: each shot places the ball, lets it rest so
    /// the estimate converges, then kicks it. Every UKF estimate and every raw vision detection is
    /// compared with the ground truth in the robot frame {r} (so odometry drift cannot pollute the
    /// comparison), and each shot is summarised: position/velocity error, how fast the velocity
    /// estimate follows the kick, the effective latency, raw detection error and vision latency.
    class BallLocalisationBenchmark : public ::extension::behaviour::BehaviourReactor {
    public:
        explicit BallLocalisationBenchmark(std::unique_ptr<NUClear::Environment> environment);

    private:
        using SysTime = std::chrono::system_clock::time_point;

        struct Config {
            double start_delay = 20.0;
            unsigned seed      = 1;
            int shots          = 30;
            double settle_time = 2.0;
            double roll_time   = 3.0;
            std::array<double, 2> start_distance{2.0, 5.0};
            std::array<double, 2> start_lateral{-1.5, 1.5};
            std::array<double, 2> target_lateral{-1.2, 1.2};
            std::array<double, 2> speed{1.0, 4.0};
            std::array<double, 3> lag_search{0.0, 0.4, 0.01};
            std::string output_dir;
            bool shutdown_when_done   = true;
            int stand_still_priority  = 1;
            int look_at_ball_priority = 2;
            int look_around_priority  = 1;
        } cfg;

        /// One ground-truth sample, already reduced to the planar quantities compared here
        struct BallTruth {
            SysTime t;
            Eigen::Vector3d rBSs;
            Eigen::Vector3d vBs;
        };
        struct RobotTruth {
            SysTime t;
            Eigen::Vector2d position;  ///< torso ground projection in {s}
            double yaw;                ///< torso yaw in {s}
        };

        /// Ground truth of the ball in the robot frame {r} at time t (xy position and velocity),
        /// interpolated from the buffers; empty if t is outside them
        std::optional<std::pair<Eigen::Vector2d, Eigen::Vector2d>> truth_in_robot(SysTime t) const;

        /// A UKF estimate (robot frame) kept for the per-shot effective-latency search
        struct Estimate {
            SysTime t;
            Eigen::Vector2d r;
            Eigen::Vector2d v;
        };

        enum class Phase { WAITING, SETTLING, ROLLING, DONE };

        struct Shot {
            int id = -1;
            Eigen::Vector2d start{};
            double target_y = 0.0;
            double speed    = 0.0;
            SysTime placed{};
            SysTime kicked{};
            std::vector<Estimate> estimates{};
            std::vector<double> detection_errors{};
            std::vector<double> vision_latencies{};
            std::optional<double> velocity_response{};  ///< s from the kick to the estimate following it
        };

        void place_next_shot();
        void kick();
        void finish_shot();
        void finish_run();
        void open_outputs();

        /// NUClear time (possibly rate-adjusted) to wall-clock time, which the ground truth uses
        static SysTime to_sys(NUClear::clock::time_point t);

        std::deque<BallTruth> ball_truth{};
        std::deque<RobotTruth> robot_truth{};

        std::mt19937 rng{};
        Phase phase = Phase::WAITING;
        SysTime phase_since{};
        SysTime start_after{};
        Shot shot{};
        int shots_done = 0;

        std::ofstream samples_csv{};
        std::ofstream detections_csv{};
        std::ofstream summary_csv{};
        /// Per-shot summaries, kept for the end-of-run table: pos rmse, vel rmse, response, lag
        std::vector<std::array<double, 4>> run_summary{};
    };

}  // namespace module::tools

#endif  // MODULE_TOOLS_BALLLOCALISATIONBENCHMARK_HPP
