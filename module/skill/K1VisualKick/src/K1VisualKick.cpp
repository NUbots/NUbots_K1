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
#include "K1VisualKick.hpp"

#include <cmath>

#include "extension/Behaviour.hpp"
#include "extension/Configuration.hpp"

#include "message/booster/BoosterMode.hpp"
#include "message/booster/BoosterVisualKick.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Robot.hpp"
#include "message/skill/Kick.hpp"
#include "message/vision/Goal.hpp"

namespace module::skill {

    using extension::Configuration;
    using message::booster::BoosterMode;
    using message::booster::K1Mode;
    using message::input::Sensors;
    using message::localisation::Robots;
    using message::skill::Kick;
    using message::vision::Goals;
    using VisualKick = message::booster::BoosterVisualKick;
    using KickVer    = message::booster::VisualKickVer;

    K1VisualKick::K1VisualKick(std::unique_ptr<NUClear::Environment> environment)
        : BehaviourReactor(std::move(environment)) {

        on<Configuration>("K1VisualKick.yaml").then([this](const Configuration& config) {
            this->log_level = config["log_level"].as<NUClear::LogLevel>();
            cfg.balanced_version = config["profiles"]["balanced"]["version"].as<int>() == 1 ? KickVer::V1 : KickVer::V2;
            cfg.powerful_version = config["profiles"]["powerful"]["version"].as<int>() == 1 ? KickVer::V1 : KickVer::V2;
            cfg.balanced_duration = std::chrono::duration_cast<NUClear::clock::duration>(
                std::chrono::duration<double>(config["profiles"]["balanced"]["duration"].as<double>()));
            cfg.powerful_duration = std::chrono::duration_cast<NUClear::clock::duration>(
                std::chrono::duration<double>(config["profiles"]["powerful"]["duration"].as<double>()));
            cfg.goal_angle_threshold      = config["autonomous"]["goal_angle_threshold"].as<double>();
            cfg.goal_distance_threshold   = config["autonomous"]["goal_distance_threshold"].as<double>();
            cfg.obstacle_angle_threshold  = config["autonomous"]["obstacle_angle_threshold"].as<double>();
            cfg.obstacle_distance_threshold = config["autonomous"]["obstacle_distance_threshold"].as<double>();
        });

        on<Trigger<Goals>, With<Sensors>>().then([this](const Goals& goals, const Sensors& sensors) {
            (void)sensors;
            bool found_clear_goal = false;

            for (const auto& goal : goals.goals) {
                const bool centered = std::abs(goal.screen_angular.x()) < cfg.goal_angle_threshold;
                const bool low_elevation = std::abs(goal.screen_angular.y()) < cfg.goal_angle_threshold * 0.75;
                const bool close_enough = goal.post.distance < cfg.goal_distance_threshold;
                if (centered && low_elevation && close_enough) {
                    found_clear_goal = true;
                    break;
                }
            }

            has_clear_goal_context = found_clear_goal;
        });

        on<Trigger<Robots>, With<Sensors>>().then([this](const Robots& robots, const Sensors& sensors) {
            bool found_obstacle = false;

            for (const auto& robot : robots.robots) {
                if (robot.teammate) {
                    continue;
                }

                const Eigen::Vector3d rORr = sensors.Hrw * robot.rRWw;
                const bool in_front = rORr.x() > 0.0;
                const bool laterally_aligned = std::abs(rORr.y()) < cfg.obstacle_angle_threshold;
                const bool close_enough = rORr.x() < cfg.obstacle_distance_threshold;

                if (in_front && laterally_aligned && close_enough) {
                    found_obstacle = true;
                    break;
                }
            }

            has_obstacle_context = found_obstacle;
        });

        on<Provide<Kick>, Every<10, Per<std::chrono::seconds>>>().then(
            [this](const RunReason& run_reason) {
                // The SDK kick is autonomous, so the Kick task's leg/target/direction are unused
                if (run_reason == RunReason::NEW_TASK) {
                    if (has_obstacle_context) {
                        active_profile = KickProfile::BALANCED;
                    }
                    else if (has_clear_goal_context) {
                        active_profile = KickProfile::POWERFUL;
                    }
                    else {
                        active_profile = KickProfile::BALANCED;
                    }

                    log<INFO>("Starting visual kick with "
                              + std::string(active_profile == KickProfile::POWERFUL ? "powerful" : "balanced")
                              + " profile");

                    // Kicking requires movement, so request soccer mode from the Booster hardware.
                    auto mode  = std::make_unique<BoosterMode>();
                    mode->mode = K1Mode::SOCCER;
                    emit(std::move(mode));
                    auto vk     = std::make_unique<VisualKick>();
                    vk->start   = true;
                    vk->version = active_profile == KickProfile::POWERFUL ? cfg.powerful_version : cfg.balanced_version;
                    emit(std::move(vk));
                    kick_start_time = NUClear::clock::now();
                    last_profile_duration = active_profile == KickProfile::POWERFUL ? cfg.powerful_duration : cfg.balanced_duration;
                }

                // No completion feedback from the SDK, so finish after the selected profile duration
                if (NUClear::clock::now() - kick_start_time > last_profile_duration) {
                    emit<Task>(std::make_unique<Done>());
                }
                else {
                    emit<Task>(std::make_unique<Continue>());
                }
            });

        // Stop the SDK behaviour when the Kick task ends
        on<Stop<Kick>>().then([this] {
            log<INFO>("Stopping visual kick");
            auto vk     = std::make_unique<VisualKick>();
            vk->start   = false;
            vk->version = active_profile == KickProfile::POWERFUL ? cfg.powerful_version : cfg.balanced_version;
            emit(std::move(vk));
        });
    }

}  // namespace module::skill
