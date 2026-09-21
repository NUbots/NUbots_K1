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
#include "GoalieShotBenchmark.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>

#include "extension/Configuration.hpp"

#include "message/booster/NUSimGroundTruth.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Field.hpp"
#include "message/planning/LookAround.hpp"
#include "message/planning/Save.hpp"
#include "message/strategy/FallRecovery.hpp"
#include "message/strategy/LookAtFeature.hpp"
#include "message/strategy/WalkToFieldPosition.hpp"
#include "message/support/FieldDescription.hpp"

#include "utility/math/euler.hpp"

namespace module::tools {

    using extension::Configuration;

    using message::booster::NUSimBallCommand;
    using message::booster::NUSimBallGroundTruth;
    using message::booster::NUSimRobotGroundTruth;
    using message::input::Sensors;
    using message::localisation::Field;
    using message::localisation::PenaltyReset;
    using message::planning::LookAround;
    using message::planning::Save;
    using message::planning::SavePlan;
    using message::strategy::FallRecovery;
    using message::strategy::LookAtBall;
    using message::strategy::WalkToFieldPosition;
    using message::support::FieldDescription;

    using utility::math::euler::pos_rpy_to_transform;

    namespace {
        double seconds(const std::chrono::system_clock::duration d) {
            return std::chrono::duration<double>(d).count();
        }

        std::array<double, 2> pair_of(const Configuration& config, const char* key) {
            const auto v = config["shot"][key].as<std::vector<double>>();
            if (v.size() != 2) {
                throw std::runtime_error(std::string("GoalieShotBenchmark.yaml: shot.") + key + " needs [min, max]");
            }
            return {v[0], v[1]};
        }

        const char* mode_name(const int mode) {
            switch (mode) {
                case SavePlan::State::IDLE: return "IDLE";
                case SavePlan::State::GUARD: return "GUARD";
                case SavePlan::State::BLOCK: return "BLOCK";
                case SavePlan::State::DISABLED: return "DISABLED";
                default: return "NONE";
            }
        }

        /// The goalie's home pose in the field frame: on our goal's centre line, line_offset out, facing the field
        Eigen::Isometry3d home_in_field(const FieldDescription& fd, const double line_offset) {
            return pos_rpy_to_transform(Eigen::Vector3d(fd.dimensions.field_length / 2.0 - line_offset, 0.0, 0.0),
                                        Eigen::Vector3d(0.0, 0.0, M_PI));
        }

        double median(std::vector<double> v) {
            v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return !std::isfinite(x); }), v.end());
            if (v.empty()) {
                return NAN;
            }
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        }
    }  // namespace

    GoalieShotBenchmark::GoalieShotBenchmark(std::unique_ptr<NUClear::Environment> environment)
        : BehaviourReactor(std::move(environment)) {

        on<Configuration>("GoalieShotBenchmark.yaml").then([this](const Configuration& config) {
            this->log_level          = config["log_level"].as<NUClear::LogLevel>();
            cfg.start_delay          = config["start_delay"].as<double>();
            cfg.ground_truth_field   = config["ground_truth_field"].as<bool>();
            cfg.seed                 = config["seed"].as<unsigned int>();
            cfg.shots                = config["shots"].as<int>();
            const auto home          = config["home"].as<std::vector<double>>();
            cfg.home                 = Eigen::Vector2d(home.at(0), home.at(1));
            cfg.line_offset          = config["line_offset"].as<double>();
            cfg.distance             = pair_of(config, "distance");
            cfg.lateral              = pair_of(config, "lateral");
            cfg.crossing             = pair_of(config, "crossing");
            cfg.speed                = pair_of(config, "speed");
            cfg.rolling_deceleration = config["rolling_deceleration"].as<double>();
            cfg.settle_time          = config["settle_time"].as<double>();
            cfg.max_roll_time        = config["max_roll_time"].as<double>();
            cfg.fall_height          = config["fall_height"].as<double>();
            cfg.upright_height       = config["upright_height"].as<double>();
            cfg.recover_time         = config["recover_time"].as<double>();
            cfg.home_tolerance       = config["home_tolerance"].as<double>();
            cfg.yaw_tolerance        = config["yaw_tolerance"].as<double>();
            cfg.home_timeout         = config["home_timeout"].as<double>();
            cfg.output_dir           = config["output_dir"].as<std::string>();
            cfg.shutdown_when_done   = config["shutdown_when_done"].as<bool>();
            cfg.save_priority          = config["tasks"]["save_priority"].as<int>();
            cfg.walk_priority          = config["tasks"]["walk_priority"].as<int>();
            cfg.fall_recovery_priority = config["tasks"]["fall_recovery_priority"].as<int>();
            cfg.look_at_ball_priority  = config["tasks"]["look_at_ball_priority"].as<int>();
            cfg.look_around_priority   = config["tasks"]["look_around_priority"].as<int>();
        });

        on<Startup>().then([this] {
            rng.seed(cfg.seed);
            start_after = Clock::now()
                          + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(cfg.start_delay));
            log<INFO>("Goalie shot benchmark:", cfg.shots, "shots starting in", cfg.start_delay, "s");
        });

        // What the Goalie does while it defends: PlanSave above a walk back to the home spot, and the head on the ball
        on<Every<10, Per<std::chrono::seconds>>, Optional<With<FieldDescription>>>().then(
            [this](const std::shared_ptr<const FieldDescription>& fd) {
                if (cfg.fall_recovery_priority > 0) {
                    emit<Task>(std::make_unique<FallRecovery>(), cfg.fall_recovery_priority);
                }
                if (cfg.look_around_priority > 0) {
                    emit<Task>(std::make_unique<LookAround>(), cfg.look_around_priority);
                }
                if (cfg.look_at_ball_priority > 0) {
                    emit<Task>(std::make_unique<LookAtBall>(), cfg.look_at_ball_priority);
                }
                if (!localisation_reset || fd == nullptr) {
                    return;
                }
                if (cfg.walk_priority > 0) {
                    emit<Task>(std::make_unique<WalkToFieldPosition>(home_in_field(*fd, cfg.line_offset), true),
                               cfg.walk_priority);
                }
                if (cfg.save_priority > 0) {
                    emit<Task>(std::make_unique<Save>(), cfg.save_priority);
                }
            });

        on<Trigger<NUSimBallGroundTruth>, Sync<GoalieShotBenchmark>>().then([this](const NUSimBallGroundTruth& gt) {
            ball   = gt.rBSs.head<2>();
            ball_v = gt.vBs.head<2>();
            std::ostringstream line{};
            line << "ball p " << gt.rBSs.transpose() << " v " << gt.vBs.transpose() << " w " << gt.omegaBs.transpose();
            remember(line.str());
        });

        on<Trigger<NUSimRobotGroundTruth>, Optional<With<Sensors>>, Sync<GoalieShotBenchmark>>().then(
            [this](const NUSimRobotGroundTruth& gt, const std::shared_ptr<const Sensors>& sensors) {
                // NUSim resets itself when its physics blows up, which teleports the robot: 0.15 m between two
                // ground-truth samples (50 Hz) is 7.5 m/s, which no torso does
                const Eigen::Vector2d now_at = gt.Hst.translation().head<2>();
                if (have_truth && (now_at - torso).norm() > 0.15) {
                    log<WARN>("NUSim reset itself (the robot jumped", (now_at - torso).norm(), "m). Before it:");
                    for (const auto& line : recent) {
                        log<WARN>("  ", line);
                    }
                    if (phase == Phase::ROLLING || phase == Phase::SETTLING) {
                        shot.sim_reset = true;
                    }
                }
                torso      = gt.Hst.translation().head<2>();
                torso_z    = gt.Hst.translation().z();
                {
                    std::ostringstream line{};
                    line << "torso p " << gt.Hst.translation().transpose() << " v " << gt.vTs.transpose() << " mode "
                         << mode_name(plan_mode);
                    remember(line.str());
                }
                torso_yaw  = std::atan2(gt.Hst.linear()(1, 0), gt.Hst.linear()(0, 0));
                have_truth = true;

                // Ground-truth localisation: the field frame is NUSim's world turned half way round (our goal is
                // at -x in NUSim and at +x in {f}), and odometry's world is tied to NUSim's through the torso
                if (cfg.ground_truth_field && sensors != nullptr) {
                    const Eigen::Isometry3d Hfs(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ()));
                    const Eigen::Isometry3d Hfw = Hfs * Eigen::Isometry3d(gt.Hst) * Eigen::Isometry3d(sensors->Htw);
                    // Keep it planar, as field localisation publishes it
                    const double yaw = std::atan2(Hfw.linear()(1, 0), Hfw.linear()(0, 0));
                    auto field       = std::make_unique<Field>();
                    field->Hfw =
                        pos_rpy_to_transform(Eigen::Vector3d(Hfw.translation().x(), Hfw.translation().y(), 0.0),
                                             Eigen::Vector3d(0.0, 0.0, yaw));
                    field->localised = true;
                    emit(field);
                }
            });

        on<Trigger<SavePlan>, Sync<GoalieShotBenchmark>>().then([this](const SavePlan& plan) {
            plan_mode       = plan.state;
            plan_ball       = plan.rBRr;
            plan_ball_valid = plan.ball_valid;
            if (phase != Phase::ROLLING || !std::isnan(shot.block_reaction) || plan.state != SavePlan::State::BLOCK) {
                return;
            }
            shot.block_reaction   = seconds(Clock::now() - shot.kicked);
            shot.plan_dy          = plan.dy;
            shot.plan_sigma_dy    = plan.sigma_dy;
            shot.plan_t           = plan.time;
            shot.plan_p_on_target = plan.p_on_target;
            shot.plan_expected    = plan.expected_block_success;
        });

        // The shot schedule
        on<Every<50, Per<std::chrono::seconds>>,
           Optional<With<Sensors>>,
           Optional<With<FieldDescription>>,
           Sync<GoalieShotBenchmark>>()
            .then([this](const std::shared_ptr<const Sensors>& sensors,
                         const std::shared_ptr<const FieldDescription>& fd) {
                const auto now = Clock::now();
                switch (phase) {
                    case Phase::WAITING: {
                        if (now < start_after) {
                            return;
                        }
                        if (!have_truth || sensors == nullptr || fd == nullptr) {
                            log<WARN>("Waiting for NUSim ground truth, Sensors and FieldDescription");
                            start_after = now + std::chrono::seconds(2);
                            return;
                        }
                        // NUSim starts the goalie at home (the "goalie" keyframe), so field localisation is told
                        // where it is rather than left to pick an end of a symmetric field. Its state is Hfw.
                        if (!cfg.ground_truth_field) {
                            const Eigen::Isometry3d Hfw = home_in_field(*fd, cfg.line_offset) * sensors->Hrw;
                            emit(std::make_unique<PenaltyReset>(
                                Eigen::Vector3d(Hfw.translation().x(),
                                                Hfw.translation().y(),
                                                std::atan2(Hfw.linear()(1, 0), Hfw.linear()(0, 0)))));
                        }
                        localisation_reset = true;
                        half_goal          = fd->dimensions.goal_width / 2.0;
                        if ((torso - cfg.home).norm() > 0.3) {
                            log<WARN>("The goalie is",
                                      (torso - cfg.home).norm(),
                                      "m from home: launch NUSim with --keyframe goalie");
                        }

                        const std::time_t t = std::time(nullptr);
                        std::ostringstream stamp{};
                        stamp << std::put_time(std::localtime(&t), "%Y%m%d-%H%M%S");
                        const auto dir = std::filesystem::path(cfg.output_dir) / stamp.str();
                        std::error_code ec{};
                        std::filesystem::create_directories(dir, ec);
                        if (!ec) {
                            shots_csv.open(dir / "shots.csv");
                            shots_csv << "shot,start_x,start_y,crossing_y,speed,on_target,mode_at_kick,true_dy,true_t,"
                                         "robot_dx,robot_dy,robot_yaw,block_reaction_s,plan_dy,plan_sigma_dy,plan_t,"
                                         "plan_p_on_target,plan_expected,closest,fell,outcome,saved,conceded\n";
                            log<INFO>("Writing goalie shot results to", dir.string());
                        }
                        // Give the walk and localisation a moment with the reset before the first shot
                        phase       = Phase::RECOVERING;
                        phase_since = now;
                        home_since  = now;
                        return;
                    }
                    case Phase::RECOVERING: {
                        // Next shot once the goalie is upright, back home and facing the field for recover_time,
                        // or after home_timeout whatever state it is in
                        const bool home = torso_z > cfg.upright_height
                                          && (torso - cfg.home).norm() < cfg.home_tolerance
                                          && std::abs(torso_yaw) < cfg.yaw_tolerance;
                        if (!home) {
                            home_since = now;
                            if (seconds(now - last_waiting_log) > 10.0) {
                                log<INFO>("Waiting for the goalie to get home: at",
                                          (torso - cfg.home).norm(),
                                          "m, yaw",
                                          torso_yaw,
                                          "height",
                                          torso_z,
                                          "| PlanSave",
                                          mode_name(plan_mode),
                                          "ball",
                                          plan_ball_valid ? "valid" : "invalid",
                                          "at",
                                          plan_ball.x(),
                                          plan_ball.y(),
                                          "| true ball",
                                          ball.x(),
                                          ball.y());
                                last_waiting_log = now;
                            }
                            if (seconds(now - phase_since) > cfg.home_timeout) {
                                log<WARN>("The goalie did not get home in", cfg.home_timeout, "s: shooting anyway");
                                place_next_shot();
                            }
                            return;
                        }
                        if (seconds(now - home_since) >= cfg.recover_time) {
                            place_next_shot();
                        }
                        return;
                    }
                    case Phase::SETTLING:
                        if (seconds(now - phase_since) >= cfg.settle_time) {
                            kick();
                        }
                        return;
                    case Phase::ROLLING: track(0.02); return;
                    case Phase::DONE: return;
                }
            });
    }

    void GoalieShotBenchmark::remember(std::string line) {
        std::ostringstream stamp{};
        stamp << std::fixed << std::setprecision(3)
              << std::chrono::duration<double>(Clock::now().time_since_epoch()).count() << ' ';
        recent.push_back(stamp.str() + line);
        // Both truths arrive at about 50 Hz: keep about 2 s
        while (recent.size() > 200) {
            recent.pop_front();
        }
    }

    void GoalieShotBenchmark::place_next_shot() {
        std::uniform_real_distribution<double> distance(cfg.distance[0], cfg.distance[1]);
        std::uniform_real_distribution<double> lateral(cfg.lateral[0], cfg.lateral[1]);
        std::uniform_real_distribution<double> crossing(cfg.crossing[0], cfg.crossing[1]);
        std::uniform_real_distribution<double> speed(cfg.speed[0], cfg.speed[1]);

        shot            = Shot{};
        shot.id         = shots_done;
        shot.start      = cfg.home + Eigen::Vector2d(distance(rng), lateral(rng));
        shot.crossing_y = cfg.home.y() + crossing(rng);
        shot.speed      = speed(rng);

        auto cmd      = std::make_unique<NUSimBallCommand>();
        // Eigen members are not zeroed by the message's default constructor: a command that leaves them
        // unset hands NUSim whatever was in memory, which blew up its physics
        cmd->velocity         = Eigen::Vector3d::Zero();
        cmd->angular_velocity = Eigen::Vector3d::Zero();
        cmd->frame            = NUSimBallCommand::Frame::WORLD;
        cmd->position         = Eigen::Vector3d(shot.start.x(), shot.start.y(), -1.0);
        remember("command place " + std::to_string(shot.start.x()) + " " + std::to_string(shot.start.y()));
        emit(cmd);

        phase       = Phase::SETTLING;
        phase_since = Clock::now();
    }

    void GoalieShotBenchmark::kick() {
        // Kick from wherever the ball rests, at the crossing point on the goalie's home line, as mjlab aims its shots
        const Eigen::Vector2d from   = ball;
        const Eigen::Vector2d target = Eigen::Vector2d(cfg.home.x(), shot.crossing_y);
        const Eigen::Vector2d dir    = (target - from).normalized();
        const double a               = cfg.rolling_deceleration;
        // A shot that stops short of the goalie teaches nothing, as in mjlab
        shot.speed = std::max(shot.speed, 1.1 * std::sqrt(2.0 * a * (target - from).norm()));

        // Whether it was ever going in: crosses the goal line between the posts before it stops
        const double line_x = cfg.home.x() - cfg.line_offset;
        const double to_line = (from.x() - line_x) / std::max(-dir.x(), 1e-6);
        const double goal_y  = from.y() + dir.y() * to_line;
        shot.on_target       = dir.x() < 0.0 && shot.speed * shot.speed > 2.0 * a * to_line
                         && std::abs(goal_y - cfg.home.y()) < half_goal;

        // The same, from the goalie where it actually stands: the axes the envelope is binned on
        const Eigen::Rotation2Dd Rrs(-torso_yaw);
        const Eigen::Vector2d p = Rrs * (from - torso);
        const Eigen::Vector2d d = Rrs * dir;
        if (d.x() < -1e-3 && p.x() > 0.0) {
            const double to_robot = p.x() / -d.x();
            const double disc     = shot.speed * shot.speed - 2.0 * a * to_robot;
            shot.true_dy          = p.y() + d.y() * to_robot;
            shot.true_t           = disc > 0.0 ? (shot.speed - std::sqrt(disc)) / a : NAN;
        }
        shot.robot_offset = torso - cfg.home;
        shot.robot_yaw    = torso_yaw;
        shot.mode_at_kick = plan_mode;

        auto cmd      = std::make_unique<NUSimBallCommand>();
        // Eigen members are not zeroed by the message's default constructor: a command that leaves them
        // unset hands NUSim whatever was in memory, which blew up its physics
        cmd->velocity         = Eigen::Vector3d::Zero();
        cmd->angular_velocity = Eigen::Vector3d::Zero();
        cmd->frame            = NUSimBallCommand::Frame::WORLD;
        cmd->position         = Eigen::Vector3d(from.x(), from.y(), -1.0);
        cmd->velocity         = Eigen::Vector3d(shot.speed * dir.x(), shot.speed * dir.y(), 0.0);
        cmd->rolling          = true;
        remember("command kick from " + std::to_string(from.x()) + " " + std::to_string(from.y()) + " v "
                 + std::to_string(shot.speed));
        emit(cmd);

        phase       = Phase::ROLLING;
        phase_since = Clock::now();
        shot.kicked = phase_since;
    }

    void GoalieShotBenchmark::track(const double dt) {
        shot.closest = std::min(shot.closest, (ball - torso).norm());
        shot.fell |= torso_z < cfg.fall_height;

        // Past the goal line is a goal between the posts and a miss outside them. The ball's centre crossing is
        // counted, not the whole ball, as mjlab counts it.
        const double line_x = cfg.home.x() - cfg.line_offset;
        if (ball.x() < line_x) {
            const bool in_goal = std::abs(ball.y() - cfg.home.y()) < half_goal;
            shot.conceded      = in_goal;
            finish_shot(in_goal ? "goal" : "wide");
            return;
        }
        shot.stopped_for = ball_v.norm() < 0.05 ? shot.stopped_for + dt : 0.0;
        if (shot.stopped_for > 0.3) {
            finish_shot("stopped");
        }
        else if (seconds(Clock::now() - phase_since) > cfg.max_roll_time) {
            finish_shot("timeout");
        }
    }

    void GoalieShotBenchmark::finish_shot(const std::string& outcome) {
        // Park the ball out of play at once, so it does not roll about the goalie while it recovers
        auto park      = std::make_unique<NUSimBallCommand>();
        // Eigen members are not zeroed by the message's default constructor: a command that leaves them
        // unset hands NUSim whatever was in memory, which blew up its physics
        park->velocity         = Eigen::Vector3d::Zero();
        park->angular_velocity = Eigen::Vector3d::Zero();
        park->frame            = NUSimBallCommand::Frame::WORLD;
        park->position         = Eigen::Vector3d(cfg.home.x() + 6.0, cfg.home.y() + 4.0, -1.0);
        remember("command park");
        emit(park);

        shot.outcome     = shot.sim_reset ? "sim_reset" : outcome;
        const bool saved = shot.on_target && !shot.conceded;
        if (!shot.sim_reset) {
            results.push_back(shot);
        }
        ++shots_done;

        log<INFO>("Shot",
                  shots_done,
                  "/",
                  cfg.shots,
                  shot.outcome,
                  shot.on_target ? (saved ? "SAVED" : "CONCEDED") : "(off target)",
                  "| true dy",
                  shot.true_dy,
                  "t",
                  shot.true_t,
                  "v",
                  shot.speed,
                  "| kick in",
                  mode_name(shot.mode_at_kick),
                  ", BLOCK after",
                  shot.block_reaction,
                  "s, plan dy",
                  shot.plan_dy,
                  "+/-",
                  shot.plan_sigma_dy,
                  "E[save]",
                  shot.plan_expected,
                  shot.fell ? "| FELL" : "");

        if (shots_csv.is_open()) {
            shots_csv << shot.id << ',' << shot.start.x() << ',' << shot.start.y() << ',' << shot.crossing_y << ','
                      << shot.speed << ',' << shot.on_target << ',' << mode_name(shot.mode_at_kick) << ','
                      << shot.true_dy << ',' << shot.true_t << ',' << shot.robot_offset.x() << ','
                      << shot.robot_offset.y() << ',' << shot.robot_yaw << ',' << shot.block_reaction << ','
                      << shot.plan_dy << ',' << shot.plan_sigma_dy << ',' << shot.plan_t << ','
                      << shot.plan_p_on_target << ','
                      << shot.plan_expected << ',' << shot.closest << ',' << shot.fell << ',' << shot.outcome << ','
                      << saved << ',' << shot.conceded << '\n';
            shots_csv.flush();
        }

        if (shots_done >= cfg.shots) {
            finish_run();
            return;
        }
        phase       = Phase::RECOVERING;
        phase_since = Clock::now();
        home_since  = phase_since;
    }

    void GoalieShotBenchmark::finish_run() {
        phase         = Phase::DONE;
        int on_target = 0, saved = 0, blocked = 0, fell = 0, false_blocks = 0;
        double expected = 0.0;
        std::vector<double> reaction{};
        for (const auto& s : results) {
            fell += s.fell;
            const bool block = std::isfinite(s.block_reaction);
            if (!s.on_target) {
                false_blocks += block;
                continue;
            }
            ++on_target;
            saved += !s.conceded;
            blocked += block;
            expected += block ? s.plan_expected : 0.0;
            reaction.push_back(s.block_reaction);
        }
        log<INFO>("Goalie shot benchmark done:",
                  saved,
                  "/",
                  on_target,
                  "on-target shots saved; PlanSave blocked",
                  blocked,
                  "of them (median",
                  median(reaction),
                  "s after the kick) and",
                  false_blocks,
                  "off-target shots; envelope expected",
                  expected,
                  "saves; fell during",
                  fell,
                  "of",
                  results.size(),
                  "shots");
        shots_csv.close();
        if (cfg.shutdown_when_done) {
            powerplant.shutdown();
        }
    }

}  // namespace module::tools
