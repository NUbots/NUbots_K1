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

#include <cmath>
#include <vector>

#include "sha256.hpp"

#include "extension/Configuration.hpp"

#include "message/booster/NUSimGroundTruth.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Field.hpp"
#include "message/planning/Save.hpp"
#include "message/skill/Block.hpp"
#include "message/support/FieldDescription.hpp"

namespace module::planning {

    using extension::Configuration;

    using SaveTask  = message::planning::Save;
    using BlockTask = message::skill::Block;
    using message::booster::NUSimBallCrossings;
    using message::booster::NUSimBallSource;
    using message::input::Sensors;
    using message::localisation::Ball;
    using message::localisation::Field;
    using message::planning::SavePlan;
    using message::support::FieldDescription;

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
            last_tick = NUClear::clock::now();
        });

        on<Stop<SaveTask>>().then([this] { decider.reset(); });

        // 50 Hz, the block policy's rate. Between ball estimates (30 Hz) the command is recomputed from the held
        // estimate and the robot's latest pose, which is also how the training command behaves.
        on<Provide<SaveTask>,
           Optional<With<Ball>>,
           With<Sensors>,
           With<Field>,
           With<FieldDescription>,
           Optional<With<NUSimBallSource>>,
           Optional<With<NUSimBallCrossings>>,
           Every<50, Per<std::chrono::seconds>>,
           Single>()
            .then([this](const std::shared_ptr<const Ball>& ball,
                         const Sensors& sensors,
                         const Field& field,
                         const FieldDescription& fd,
                         const std::shared_ptr<const NUSimBallSource>& source,
                         const std::shared_ptr<const NUSimBallCrossings>& crossings) {
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

                switch (mode) {
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
                    // Emitting nothing hands the goalie back to the positioning walk
                    default: break;
                }

                emit(plan);
            });
    }

    void PlanSave::check_policy() {
        if (!envelope || !policy_path) {
            envelope_ok = false;
            return;
        }
        const auto hash = save::sha256_file(*policy_path);
        envelope_ok     = hash && *hash == envelope->onnx_sha256;
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
