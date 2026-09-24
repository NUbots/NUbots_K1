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
#include "PlanSave.hpp"

#include <chrono>
#include <cmath>
#include <vector>

#include "sha256.hpp"

#include "extension/Configuration.hpp"

#include "message/behaviour/state/WalkState.hpp"
#include "message/booster/NUSimGroundTruth.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Field.hpp"
#include "message/planning/Save.hpp"
#include "message/platform/RawSensors.hpp"
#include "message/skill/Block.hpp"
#include "message/strategy/WalkToFieldPosition.hpp"
#include "message/support/FieldDescription.hpp"

#include "utility/math/euler.hpp"

namespace module::planning {

    using extension::Configuration;

    using SaveTask  = message::planning::Save;
    using BlockTask = message::skill::Block;
    using message::behaviour::state::WalkState;
    using message::booster::NUSimBallCrossings;
    using message::booster::NUSimBallSource;
    using message::input::Sensors;
    using message::localisation::Ball;
    using message::localisation::Field;
    using message::planning::SavePlan;
    using message::platform::RawSensors;
    using message::strategy::WalkToFieldPosition;
    using message::support::FieldDescription;

    using utility::math::euler::pos_rpy_to_transform;
    using utility::math::euler::rpy_intrinsic_to_mat;

    using save::Mode;

    namespace {
        double seconds(const NUClear::clock::duration& d) {
            return std::chrono::duration_cast<std::chrono::duration<double>>(d).count();
        }

        /// Yaw of a transform whose rotation is about z, as the planar robot and field frames are
        double yaw_of(const Eigen::Isometry3d& H) {
            return std::atan2(H.linear()(1, 0), H.linear()(0, 0));
        }

        /// Flattens a [speed][time][dy] table from the envelope file, checking it has the grid's shape
        std::vector<int> flatten(const std::vector<std::vector<std::vector<int>>>& table, const save::Envelope& e) {
            std::vector<int> flat{};
            if (table.size() != e.n_speed()) {
                return flat;
            }
            for (const auto& plane : table) {
                if (plane.size() != e.n_time()) {
                    return {};
                }
                for (const auto& row : plane) {
                    if (row.size() != e.n_dy()) {
                        return {};
                    }
                    flat.insert(flat.end(), row.begin(), row.end());
                }
            }
            return flat;
        }

        /// A crossing known exactly, from NUSim's ground truth: no uncertainty, and the time counted down to it
        save::UncertainCrossing known_crossing(const bool reaches,
                                               const double offset,
                                               const NUClear::clock::time_point& at,
                                               const NUClear::clock::time_point& now) {
            save::UncertainCrossing c{};
            c.crossing.reaches = reaches;
            c.crossing.offset  = offset;
            c.crossing.time    = std::max(0.0, seconds(at - now));
            return c;
        }

        /// A point on the field plane in the goal frame {g}: {f} turned half way round about our goal's centre, so x
        /// runs out of the goal and y is to the left looking out
        Eigen::Vector2d field_to_goal(const Eigen::Vector2d& rPFf, const FieldDescription& fd) {
            return {fd.dimensions.field_length / 2.0 - rPFf.x(), -rPFf.y()};
        }

        /// Standing at rGg in {g} and facing the point rBGg, as a pose in the field frame
        Eigen::Isometry3d goal_pose_to_field(const Eigen::Vector2d& rGg,
                                             const Eigen::Vector2d& rBGg,
                                             const FieldDescription& fd) {
            const Eigen::Vector2d to_ball = rBGg - rGg;
            // Out of the goal (+x in {g}) if the ball is on top of the spot
            const double yaw_g = to_ball.norm() > 1e-3 ? std::atan2(to_ball.y(), to_ball.x()) : 0.0;
            return pos_rpy_to_transform(Eigen::Vector3d(fd.dimensions.field_length / 2.0 - rGg.x(), -rGg.y(), 0.0),
                                        Eigen::Vector3d(0.0, 0.0, yaw_g + M_PI));
        }

        /// How far the left ankle is above the right (m), from the servos and the IMU
        double ankle_height_difference(const RawSensors& raw) {
            const auto& s = raw.servo;
            const save::LegAngles left{s.l_hip_pitch.present_position,
                                       s.l_hip_roll.present_position,
                                       s.l_hip_yaw.present_position,
                                       s.l_knee.present_position};
            const save::LegAngles right{s.r_hip_pitch.present_position,
                                        s.r_hip_roll.present_position,
                                        s.r_hip_yaw.present_position,
                                        s.r_knee.present_position};
            const Eigen::Matrix3d Rwt =
                rpy_intrinsic_to_mat(Eigen::Vector3d(raw.imu_rpy.x(), raw.imu_rpy.y(), raw.imu_rpy.z()));
            return save::ankle_height_difference(left, right, Rwt);
        }

        SavePlan::State to_message(const Mode mode) {
            switch (mode) {
                case Mode::GUARD: return SavePlan::State::GUARD;
                case Mode::BLOCK: return SavePlan::State::BLOCK;
                default: return SavePlan::State::IDLE;
            }
        }
    }  // namespace

    PlanSave::PlanSave(std::unique_ptr<NUClear::Environment> environment)
        : BehaviourReactor(std::move(environment)) {

        on<Configuration>("PlanSave.yaml").then([this](const Configuration& config) {
            this->log_level = config["log_level"].as<NUClear::LogLevel>();

            cfg.ball.deceleration = config["ball"]["rolling_deceleration"].as<double>();
            cfg.ball.min_speed    = config["ball"]["min_shot_speed"].as<double>();
            cfg.max_time          = config["ball"]["max_time"].as<double>();
            cfg.ball_timeout      = config["ball"]["timeout"].as<double>();
            cfg.kick_window       = config["ball"]["kick_widening"]["window"].as<double>();
            cfg.kick_factor       = config["ball"]["kick_widening"]["factor"].as<double>();

            cfg.decision.min_p_on_target  = config["decision"]["min_p_on_target"].as<double>();
            cfg.decision.guard_distance   = config["decision"]["guard_distance"].as<double>();
            cfg.decision.guard_hysteresis = config["decision"]["guard_hysteresis"].as<double>();
            cfg.decision.guard_min_ahead  = config["decision"]["guard_min_ahead"].as<double>();
            cfg.decision.release_delay    = config["decision"]["release_delay"].as<double>();
            cfg.decision.min_shot_speed   = cfg.ball.min_speed;
            cfg.block_threshold           = config["decision"]["block_threshold"].as<double>();

            cfg.confidence_z = config["envelope"]["confidence_z"].as<double>();
            cfg.min_trials   = config["envelope"]["min_trials"].as<int>();

            cfg.handoff.enabled               = config["handoff"]["enabled"].as<bool>();
            cfg.handoff.foot_height_tolerance = config["handoff"]["foot_height_tolerance"].as<double>();
            cfg.handoff.max_wait              = config["handoff"]["max_wait"].as<double>();

            const auto& pos = config["positioning"];
            save::PositioningConfig p{};
            p.deceleration      = cfg.ball.deceleration;
            p.t_lat             = pos["t_lat"].as<double>();
            const auto speeds   = pos["speed_range"].as<std::vector<double>>();
            p.min_speed         = speeds.at(0);
            p.max_speed         = speeds.at(1);
            p.n_aim             = pos["n_aim"].as<int>();
            p.n_speed           = pos["n_speed"].as<int>();
            p.cvar              = pos["objective"].as<std::string>() != "mean";
            p.cvar_fraction     = pos["cvar_fraction"].as<double>();
            p.line_penalty      = pos["line_penalty"].as<double>();
            p.min_ball_distance = pos["min_ball_distance"].as<double>();
            p.extra_lat         = pos["extra_lat"].as<std::vector<double>>();
            p.reach_scale       = pos["reach_scale"].as<std::vector<double>>();
            p.regret            = pos["robust"].as<std::string>() != "mean";
            p.near_best         = pos["near_best"].as<double>();
            p.grid_step         = pos["grid_step"].as<double>();
            p.min_depth         = pos["min_depth"].as<double>();

            const std::lock_guard<std::mutex> lock(positioning_mutex);
            cfg.positioning    = pos["enabled"].as<bool>();
            cfg.target_timeout = pos["target_timeout"].as<double>();
            positioning_cfg    = p;
            target.reset();
        });

        on<Configuration>("SaveCapability.yaml").then([this](const Configuration& config) {
            auto c         = std::make_shared<save::Capability>();
            c->onnx_sha256 = config["onnx_sha256"].as<std::string>();
            const auto axis = [&](const char* name) {
                return save::Axis{config[name]["start"].as<double>(),
                                  config[name]["step"].as<double>(),
                                  config[name]["count"].as<std::size_t>()};
            };
            c->dy    = axis("dy");
            c->time  = axis("time");
            c->speed = axis("speed");
            for (const auto& plane : config["rate"].as<std::vector<std::vector<std::vector<double>>>>()) {
                for (const auto& row : plane) {
                    c->rate.insert(c->rate.end(), row.begin(), row.end());
                }
            }
            const std::lock_guard<std::mutex> lock(envelope_mutex);
            if (!c->valid()) {
                log<ERROR>("SaveCapability.yaml does not describe a grid: positioning is left to the walk");
                capability.reset();
            }
            else {
                capability = std::move(c);
            }
            check_policy();
        });

        on<Configuration>("SaveEnvelope.yaml").then([this](const Configuration& config) {
            save::Envelope e{};
            e.onnx_sha256 = config["onnx_sha256"].as<std::string>();
            e.dy_edges    = config["dy_edges"].as<std::vector<double>>();
            e.time_edges  = config["time_edges"].as<std::vector<double>>();
            e.speed_edges = config["speed_edges"].as<std::vector<double>>();
            using Table   = std::vector<std::vector<std::vector<int>>>;
            if (e.dy_edges.size() >= 2 && e.time_edges.size() >= 2 && e.speed_edges.size() >= 2) {
                e.trials = flatten(config["trials"].as<Table>(), e);
                e.saves  = flatten(config["saves"].as<Table>(), e);
                e.falls  = flatten(config["falls"].as<Table>(), e);
            }
            const std::lock_guard<std::mutex> lock(envelope_mutex);
            if (!e.valid()) {
                log<ERROR>("SaveEnvelope.yaml does not describe a grid: planning positioning only");
                envelope.reset();
            }
            else {
                envelope = std::move(e);
            }
            check_policy();
        });

        // The block policy's own config says which file it runs, so the check follows it when it changes
        on<Configuration>("K1BlockPolicy.yaml").then([this](const Configuration& config) {
            const std::lock_guard<std::mutex> lock(envelope_mutex);
            policy_path = config["model_path"].as<std::string>();
            check_policy();
        });

        on<Start<SaveTask>>().then([this] {
            decider.reset();
            handoff_gate.reset();
            last_tick    = NUClear::clock::now();
            save_running = true;
        });

        on<Stop<SaveTask>>().then([this] {
            decider.reset();
            handoff_gate.reset();
            save_running = false;
            blocking     = false;
            const std::lock_guard<std::mutex> lock(positioning_mutex);
            target.reset();
        });

        // Choosing the goalie's spot scores a few thousand spots against a few hundred shots, too slow for every tick,
        // so it runs on its own while Save does. Not during a block, which has the goalie's attention and the CPU.
        on<Every<5, Per<std::chrono::seconds>>,
           Optional<With<Ball>>,
           With<Field>,
           With<FieldDescription>,
           Single>()
            .then("Position goalie",
                  [this](const std::shared_ptr<const Ball>& ball, const Field& field, const FieldDescription& fd) {
                      if (!save_running || blocking) {
                          return;
                      }
                      std::shared_ptr<const save::Capability> S{};
                      {
                          const std::lock_guard<std::mutex> lock(envelope_mutex);
                          if (!capability_ok) {
                              return;
                          }
                          S = capability;
                      }
                      save::PositioningConfig pcfg{};
                      std::optional<Eigen::Vector2d> hold{};
                      {
                          const std::lock_guard<std::mutex> lock(positioning_mutex);
                          if (!cfg.positioning) {
                              return;
                          }
                          pcfg = positioning_cfg;
                          if (target) {
                              hold = target->position.rGg;
                          }
                      }

                      // Positioning needs only where the ball is, so a teammate's ball does as well as our own
                      const auto now = NUClear::clock::now();
                      if (ball == nullptr || seconds(now - ball->time_of_measurement) > cfg.ball_timeout) {
                          const std::lock_guard<std::mutex> lock(positioning_mutex);
                          target.reset();
                          return;
                      }

                      // The field says where the goal is, and how far out the goalie may go: the penalty area
                      pcfg.goal_width  = fd.dimensions.goal_width;
                      pcfg.ball_radius = fd.ball_radius;
                      pcfg.max_depth   = fd.dimensions.penalty_area_length;
                      pcfg.max_lateral = fd.dimensions.penalty_area_width / 2.0;

                      const Eigen::Vector2d rBGg = field_to_goal((field.Hfw * ball->rBWw).head<2>(), fd);
                      const auto start           = std::chrono::steady_clock::now();
                      const auto position        = save::choose_position(*S, pcfg, rBGg, hold);
                      const double took =
                          std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

                      const std::lock_guard<std::mutex> lock(positioning_mutex);
                      if (position) {
                          target = Target{*position, now, took};
                      }
                      else {
                          target.reset();
                      }
                  });

        // 50 Hz, the block policy's rate. Between ball estimates (30 Hz) the command is recomputed from the held
        // estimate and the robot's latest pose, which is also how the training command behaves.
        on<Provide<SaveTask>,
           Optional<With<Ball>>,
           With<Sensors>,
           With<Field>,
           With<FieldDescription>,
           Optional<With<NUSimBallSource>>,
           Optional<With<NUSimBallCrossings>>,
           Optional<With<RawSensors>>,
           Optional<With<WalkState>>,
           Every<50, Per<std::chrono::seconds>>,
           Single>()
            .then([this](const std::shared_ptr<const Ball>& ball,
                         const Sensors& sensors,
                         const Field& field,
                         const FieldDescription& fd,
                         const std::shared_ptr<const NUSimBallSource>& source,
                         const std::shared_ptr<const NUSimBallCrossings>& crossings,
                         const std::shared_ptr<const RawSensors>& raw,
                         const std::shared_ptr<const WalkState>& walk) {
                const auto now = NUClear::clock::now();
                const double dt = std::clamp(seconds(now - last_tick), 0.0, 0.1);
                last_tick       = now;

                auto plan = std::make_unique<SavePlan>();

                const std::lock_guard<std::mutex> lock(envelope_mutex);
                if (!envelope_ok) {
                    // No envelope for the policy being run: never block, leave the goalie to positioning
                    plan->state = SavePlan::State::DISABLED;
                    emit(plan);
                    return;
                }

                // In NUSim a harness can put the crossings on ground truth (NUSimBallSource TRUE_CROSSING): then
                // they come from NUSim rolling the ball ahead, and without a fresh forecast there is no shot to plan
                const bool true_crossing   = source != nullptr && source->source == NUSimBallSource::Source::TRUE_CROSSING;
                const bool crossings_fresh = crossings != nullptr
                                             && seconds(now - crossings->timestamp) < cfg.ball_timeout;
                plan->true_crossing = true_crossing;
                if (true_crossing && !crossings_fresh && now - last_forecast_warning > std::chrono::seconds(5)) {
                    last_forecast_warning = now;
                    log<WARN>("On NUSim's true crossings, but no fresh forecast from NUSim (rt/nusim/gt/ball_crossing/*)");
                }

                // Only our own estimate carries a velocity; teammates' balls come with confidence 0 and none
                const bool ball_valid = ball != nullptr && ball->confidence > 0.0
                                        && seconds(now - ball->time_of_measurement) < cfg.ball_timeout
                                        && (!true_crossing || crossings_fresh);

                save::Situation situation{};
                situation.ball_valid = ball_valid;
                plan->ball_valid     = ball_valid;

                save::UncertainCrossing at_robot{};
                // Ball in the goal frame {g}, for the shot log: x < 0 is behind our goal line
                Eigen::Vector2d rBGg = Eigen::Vector2d::Constant(NAN);
                if (ball_valid) {
                    save::State x{};
                    x << ball->rBWw.x(), ball->rBWw.y(), ball->vBw.x(), ball->vBw.y();
                    save::Covariance P = ball->covariance;
                    const double speed = x.tail<2>().norm();

                    // Notice each shot starting, from each new estimate, to widen the velocity covariance while the
                    // filter is still catching up with the kick
                    if (ball != last_ball) {
                        last_ball = ball;
                        if (speed > cfg.ball.min_speed && !rolling) {
                            kick_time = now;
                        }
                        rolling = speed > cfg.ball.min_speed;
                    }
                    plan->time_since_kick = rolling ? seconds(now - kick_time) : 0.0;
                    plan->widened         = rolling && plan->time_since_kick < cfg.kick_window;
                    if (plan->widened) {
                        P.bottomRightCorner<2, 2>() *= cfg.kick_factor * cfg.kick_factor;
                    }

                    // The goalie's frontal plane: x = 0 in the robot frame {r}, as in training
                    save::State xr      = x;
                    save::Covariance Pr = P;
                    save::transform(yaw_of(sensors.Hrw), sensors.Hrw.translation().head<2>(), xr, Pr);
                    at_robot = save::predict_crossing(xr, Pr, cfg.ball);

                    // Our goal line, in a frame {g} at the goal's centre with x out into the field: our goal is at
                    // +x in the field frame, so {g} is {f} turned half way round and moved there
                    save::State xg      = x;
                    save::Covariance Pg = P;
                    save::transform(yaw_of(field.Hfw), field.Hfw.translation().head<2>(), xg, Pg);
                    save::transform(M_PI, Eigen::Vector2d(fd.dimensions.field_length / 2.0, 0.0), xg, Pg);
                    save::UncertainCrossing at_goal = save::predict_crossing(xg, Pg, cfg.ball);
                    rBGg                            = xg.head<2>();
                    const double half_goal          = fd.dimensions.goal_width / 2.0;

                    if (true_crossing) {
                        // The goalie's line: NUSim's crossing is already in {r}. The training rule still needs the
                        // ball moving faster than min_shot_speed now.
                        const auto& robot = crossings->robot;
                        at_robot          = known_crossing(robot.crosses && speed > cfg.ball.min_speed,
                                                  robot.rBRr.y(),
                                                  robot.time,
                                                  now);
                        // The goal line: NUSim reports the first one the ball leaves the field over, at either end,
                        // in {r}; it is ours if it lands on our half of {g}
                        const auto& goal     = crossings->goal_line;
                        const Eigen::Isometry3d Hwr = Eigen::Isometry3d(sensors.Hrw).inverse();
                        save::State xc{};
                        xc << (Hwr * goal.rBRr).head<2>(), (Hwr.linear() * goal.vBr).head<2>();
                        save::Covariance Pc = save::Covariance::Zero();
                        save::transform(yaw_of(field.Hfw), field.Hfw.translation().head<2>(), xc, Pc);
                        save::transform(M_PI, Eigen::Vector2d(fd.dimensions.field_length / 2.0, 0.0), xc, Pc);
                        at_goal = known_crossing(goal.crosses && xc.x() < fd.dimensions.field_length / 2.0,
                                                 xc.y(),
                                                 goal.time,
                                                 now);
                    }

                    plan->rBRr          = xr.head<2>();
                    plan->vBr           = xr.tail<2>();
                    plan->reaches_robot = at_robot.crossing.reaches;
                    plan->dy            = at_robot.crossing.offset;
                    plan->sigma_dy      = at_robot.sigma_offset;
                    plan->time          = at_robot.crossing.time;
                    plan->sigma_time    = at_robot.sigma_time;
                    plan->reaches_goal  = at_goal.crossing.reaches;
                    plan->goal_y        = at_goal.crossing.offset;
                    plan->sigma_goal_y  = at_goal.sigma_offset;
                    plan->p_on_target   = at_goal.crossing.reaches ? save::probability_between(at_goal.crossing.offset,
                                                                                               at_goal.sigma_offset,
                                                                                               -half_goal,
                                                                                               half_goal)
                                                                   : 0.0;

                    plan->block_fall_rate = -1.0;
                    if (at_robot.crossing.reaches) {
                        plan->expected_block_success = envelope->expected_success(at_robot.crossing.offset,
                                                                                  at_robot.sigma_offset,
                                                                                  at_robot.crossing.time,
                                                                                  speed,
                                                                                  cfg.confidence_z,
                                                                                  cfg.min_trials);
                        plan->block_fall_rate =
                            envelope->fall_rate(at_robot.crossing.offset, at_robot.crossing.time, speed, cfg.min_trials)
                                .value_or(-1.0);
                    }

                    situation.distance    = xr.head<2>().norm();
                    situation.ahead       = xr.x();
                    situation.speed       = speed;
                    situation.active      = at_robot.crossing.reaches && at_robot.crossing.time < cfg.max_time;
                    situation.p_on_target = plan->p_on_target;
                }
                else {
                    last_ball.reset();
                    rolling = false;
                }

                const Mode previous = decider.current();
                const Mode mode     = decider.step(situation, dt, cfg.decision);
                plan->state         = to_message(mode);
                blocking            = mode == Mode::BLOCK;

                // A walking goalie goes to the block policy (GUARD or BLOCK) only with both feet down; until then it
                // carries on as in IDLE. The policy was trained from a standing start and topples taking over
                // mid-stride.
                const bool walking = walk != nullptr && walk->state == WalkState::State::WALKING;
                const std::optional<double> feet =
                    raw != nullptr ? std::optional<double>(ankle_height_difference(*raw)) : std::nullopt;
                const bool was_waiting = handoff_gate.waiting();
                const bool handed      = handoff_gate.step(mode != Mode::IDLE, walking, feet, dt, cfg.handoff);
                const Mode acting      = handed ? mode : Mode::IDLE;
                plan->handoff_waiting  = handoff_gate.waiting();
                plan->handoff_wait     = handed ? handoff_gate.last_wait : handoff_gate.waited_so_far();
                plan->ankle_height_difference = feet.value_or(NAN);
                if (handed && was_waiting) {
                    log<INFO>("Handed the walking goalie to the block policy after",
                              handoff_gate.last_wait,
                              "s:",
                              handoff_gate.timed_out ? "gave up waiting for both feet down" : "both feet down");
                }

                // The latest spot chosen for the goalie, if it is fresh
                std::optional<Target> spot{};
                {
                    const std::lock_guard<std::mutex> lock(positioning_mutex);
                    if (cfg.positioning && target && seconds(now - target->time) < cfg.target_timeout) {
                        spot = target;
                    }
                }
                if (spot) {
                    plan->has_target          = true;
                    plan->rTGg                = spot->position.rGg;
                    plan->target_mean         = spot->position.mean;
                    plan->target_cvar         = spot->position.cvar;
                    plan->target_regret       = spot->position.regret;
                    plan->target_compute_time = spot->compute_time;
                }

                if (mode == Mode::BLOCK && previous != Mode::BLOCK) {
                    log<INFO>("Shot: dy",
                              plan->dy,
                              "+/-",
                              plan->sigma_dy,
                              "m, t",
                              plan->time,
                              "+/-",
                              plan->sigma_time,
                              "s, v",
                              situation.speed,
                              "m/s, P(on target)",
                              plan->p_on_target,
                              ", E[save]",
                              plan->expected_block_success,
                              ", falls",
                              plan->block_fall_rate);
                    if (plan->expected_block_success < cfg.block_threshold) {
                        log<WARN>("Blocking a shot the policy is not expected to save: E[save]",
                                  plan->expected_block_success,
                                  "<",
                                  cfg.block_threshold);
                    }
                }
                else if (previous == Mode::BLOCK && mode != Mode::BLOCK) {
                    // Negative x inside the posts is a goal; the log is for matching predictions against outcomes
                    log<INFO>("Shot over, ball at (x, y) =", rBGg.x(), rBGg.y(), "from our goal, facing the field");
                }

                switch (acting) {
                    case Mode::GUARD: emit<Task>(std::make_unique<BlockTask>()); break;
                    case Mode::BLOCK: {
                        // The training rule: a command only while the ball is on its way within max_time, zeros
                        // otherwise. K1BlockPolicy clips each field to the training ranges.
                        auto block = std::make_unique<BlockTask>();
                        if (situation.active) {
                            block->active          = true;
                            block->dy              = at_robot.crossing.offset;
                            block->time_to_arrival = at_robot.crossing.time;
                            block->ball_speed      = situation.speed;
                        }
                        emit<Task>(block);
                        break;
                    }
                    default: {
                        // Walk to the chosen spot facing the ball. Without one, emitting nothing hands the goalie back
                        // to the positioning walk below Save.
                        if (spot && ball != nullptr) {
                            const Eigen::Vector2d rBGg = field_to_goal((field.Hfw * ball->rBWw).head<2>(), fd);
                            emit<Task>(std::make_unique<WalkToFieldPosition>(
                                goal_pose_to_field(spot->position.rGg, rBGg, fd),
                                true));
                            plan->positioning = true;
                        }
                        break;
                    }
                }

                emit(plan);
            });
    }

    void PlanSave::check_policy() {
        if (!policy_path) {
            envelope_ok   = false;
            capability_ok = false;
            return;
        }
        const auto hash = save::sha256_file(*policy_path);

        capability_ok = capability && hash && *hash == capability->onnx_sha256;
        if (capability && hash && !capability_ok) {
            log<ERROR>("The block policy",
                       *policy_path,
                       "is not the one SaveCapability.yaml was measured from (sha256",
                       *hash,
                       "vs",
                       capability->onnx_sha256,
                       "): positioning is left to the walk. Regenerate it with tools/policy/make_save_envelope.py.");
        }

        if (!envelope) {
            envelope_ok = false;
            return;
        }
        envelope_ok = hash && *hash == envelope->onnx_sha256;
        if (!hash) {
            log<ERROR>("Cannot read the block policy", *policy_path, "to check it: planning positioning only");
        }
        else if (!envelope_ok) {
            log<ERROR>("The block policy",
                       *policy_path,
                       "is not the one SaveEnvelope.yaml was measured from (sha256",
                       *hash,
                       "vs",
                       envelope->onnx_sha256,
                       "): planning positioning only. Re-measure the envelope for this policy.");
        }
        else {
            log<INFO>("Block envelope matches", *policy_path);
        }
    }

}  // namespace module::planning
