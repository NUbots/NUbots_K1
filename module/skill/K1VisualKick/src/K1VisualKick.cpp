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

#include <boost/interprocess/sync/scoped_lock.hpp>
#include <cmath>

#include "extension/Behaviour.hpp"
#include "extension/Configuration.hpp"

#include "message/booster/BoosterMode.hpp"
#include "message/booster/BoosterVisualKick.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Ball.hpp"
#include "message/localisation/Field.hpp"
#include "message/localisation/Robot.hpp"
#include "message/skill/Kick.hpp"
#include "message/support/FieldDescription.hpp"
#include "message/vision/Goal.hpp"

#include "utility/math/euler.hpp"

namespace module::skill {

    using extension::Configuration;
    using message::booster::BoosterMode;
    using message::booster::K1Mode;
    using message::input::Sensors;
    using message::localisation::Ball;
    using message::localisation::Field;
    using message::localisation::Robots;
    using message::skill::Kick;
    using message::support::FieldDescription;
    using message::vision::Goals;
    using utility::math::euler::mat_to_rpy_intrinsic;
    using VisualKick = message::booster::BoosterVisualKick;
    using KickVer    = message::booster::VisualKickVer;

    void K1VisualKick::connect_kick_shm() {
        if (cfg.kick_decision_segment.empty() || kick_shared_memory != nullptr) {
            return;
        }

        try {
            kick_shared_memory          = std::make_unique<KickSharedMemory>(cfg.kick_decision_segment);
            kick_shm_unavailable_logged = false;
            log<INFO>("K1VisualKick mapped kick decision segment", cfg.kick_decision_segment);
        }
        catch (const bip::interprocess_exception& ex) {
            if (!kick_shm_unavailable_logged) {
                log<WARN>("K1VisualKick kick decision segment unavailable", cfg.kick_decision_segment, ex.what());
                kick_shm_unavailable_logged = true;
            }
        }
    }

    void K1VisualKick::publish_kick_decision(const Kick& kick,
                                             const Sensors& sensors,
                                             const Field& field,
                                             const FieldDescription& field_description) {
        connect_kick_shm();
        if (kick_shared_memory == nullptr) {
            return;
        }

        try {
            // Opposition goal midpoint (the one we kick towards), field frame -> world frame -> robot frame.
            // Field space has +x pointing towards our own goal, so the opposition goal sits at -x;
            // goalpost_opp_l/r already account for this, no sign flip needed here.
            const Eigen::Vector3d goal_field(
                0.5 * (field_description.goalpost_opp_l.x() + field_description.goalpost_opp_r.x()),
                0.5 * (field_description.goalpost_opp_l.y() + field_description.goalpost_opp_r.y()),
                0.0);
            const Eigen::Vector3d goal_robot = sensors.Hrw * (field.Hfw.inverse() * goal_field);

            // Robot heading relative to field frame
            const Eigen::Isometry3d Hfr       = field.Hfw * sensors.Hrw.inverse();
            const double robot_theta_to_field = mat_to_rpy_intrinsic(Hfr.linear())(2);

            bip::scoped_lock<bip::interprocess_mutex> lock(kick_shared_memory->header->mutex, bip::try_to_lock);
            if (!lock.owns()) {
                return;
            }

            auto* header                 = kick_shared_memory->header;
            header->x                    = kick.target.x();
            header->y                    = kick.target.y();
            header->dir                  = std::atan2(kick.direction.y(), kick.direction.x());
            header->power                = kick.direction.norm();
            header->goal_x               = goal_robot.x();
            header->goal_y               = goal_robot.y();
            header->robot_theta_to_field = robot_theta_to_field;
            header->sequence++;
            header->has_new_data.notify_all();
        }
        catch (const bip::interprocess_exception& ex) {
            kick_shared_memory.reset();
            if (!kick_shm_unavailable_logged) {
                log<WARN>("K1VisualKick lost access to kick decision segment", cfg.kick_decision_segment, ex.what());
                kick_shm_unavailable_logged = true;
            }
        }
    }

    K1VisualKick::K1VisualKick(std::unique_ptr<NUClear::Environment> environment)
        : BehaviourReactor(std::move(environment)) {

        on<Configuration>("K1VisualKick.yaml").then([this](const Configuration& config) {
            this->log_level = config["log_level"].as<NUClear::LogLevel>();
            cfg.kick_decision_segment = config["kick_decision_segment"].as<std::string>();
            cfg.max_kick_distance     = config["max_kick_distance"].as<double>();
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

            kick_shared_memory.reset();
            kick_shm_unavailable_logged = false;
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

        on<Provide<Kick>,
           With<Sensors>,
           With<Field>,
           With<FieldDescription>,
           With<Ball>,
           Every<10, Per<std::chrono::seconds>>>()
            .then([this](const Kick& kick,
                         const RunReason& run_reason,
                         const Sensors& sensors,
                         const std::shared_ptr<const Field>& field,
                         const std::shared_ptr<const FieldDescription>& field_description,
                         const std::shared_ptr<const Ball>& ball) {
                // The SDK kick is autonomous, so the Kick task's leg/target/direction are unused for the
                // kick itself; the ball position gates whether we kick, and the decision is published to
                // NUbridge for telemetry.
                const double ball_distance = (sensors.Hrw * ball->rBWw).norm();
                if (ball_distance > cfg.max_kick_distance) {
                    log<INFO>("Ball out of kicking range, stopping visual kick", ball_distance);
                    emit<Task>(std::make_unique<Done>());
                    return;
                }

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
                    last_profile_duration =
                        active_profile == KickProfile::POWERFUL ? cfg.powerful_duration : cfg.balanced_duration;

                    // Report the kick decision to NUbridge for off-robot consumers (telemetry only, so it is fine
                    // for the field estimate to be missing).
                    if (field != nullptr && field_description != nullptr) {
                        publish_kick_decision(kick, sensors, *field, *field_description);
                    }
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
