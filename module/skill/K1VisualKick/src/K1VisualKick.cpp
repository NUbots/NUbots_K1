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

#include "extension/Behaviour.hpp"
#include "extension/Configuration.hpp"

#include "message/booster/BoosterVisualKick.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Ball.hpp"
#include "message/localisation/Field.hpp"
#include "message/skill/Kick.hpp"
#include "message/support/FieldDescription.hpp"

#include "utility/math/euler.hpp"

namespace module::skill {

    using extension::Configuration;
    using message::input::Sensors;
    using message::localisation::Ball;
    using message::localisation::Field;
    using message::skill::Kick;
    using message::support::FieldDescription;
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
            this->log_level   = config["log_level"].as<NUClear::LogLevel>();
            cfg.version       = config["version"].as<int>() == 1 ? KickVer::V1 : KickVer::V2;
            cfg.kick_duration = std::chrono::duration_cast<NUClear::clock::duration>(
                std::chrono::duration<double>(config["kick_duration"].as<double>()));
            cfg.kick_decision_segment = config["kick_decision_segment"].as<std::string>();
            cfg.max_kick_distance     = config["max_kick_distance"].as<double>();

            kick_shared_memory.reset();
            kick_shm_unavailable_logged = false;
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
                log<INFO>("Beginning visual kick");
                const double ball_distance = (sensors.Hrw * ball->rBWw).norm();
                if (ball_distance > cfg.max_kick_distance) {
                    log<INFO>("Ball out of kicking range, stopping visual kick", ball_distance);
                    emit<Task>(std::make_unique<Done>());
                    return;
                }

                if (run_reason == RunReason::NEW_TASK) {
                    log<INFO>("Starting visual kick");
                    auto vk     = std::make_unique<VisualKick>();
                    vk->start   = true;
                    vk->version = cfg.version;
                    emit(std::move(vk));
                    kick_start_time = NUClear::clock::now();

                    // Report the kick decision to NUbridge for off-robot consumers (telemetry only, so it is fine
                    // for the field estimate to be missing).
                    if (field != nullptr && field_description != nullptr) {
                        publish_kick_decision(kick, sensors, *field, *field_description);
                    }
                }

                // No completion feedback from the SDK, so finish after kick_duration
                if (NUClear::clock::now() - kick_start_time > cfg.kick_duration) {
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
            vk->version = cfg.version;
            emit(std::move(vk));
        });
    }

}  // namespace module::skill
