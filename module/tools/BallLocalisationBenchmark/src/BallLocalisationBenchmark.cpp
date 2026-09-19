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
#include "BallLocalisationBenchmark.hpp"

#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>

#include "extension/Configuration.hpp"

#include "message/booster/NUSimGroundTruth.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Ball.hpp"
#include "message/planning/LookAround.hpp"
#include "message/strategy/LookAtFeature.hpp"
#include "message/strategy/StandStill.hpp"
#include "message/support/FieldDescription.hpp"
#include "message/vision/Ball.hpp"

namespace module::tools {

    using extension::Configuration;

    using message::booster::NUSimBallCommand;
    using message::booster::NUSimBallGroundTruth;
    using message::booster::NUSimRobotGroundTruth;
    using message::input::Sensors;
    using message::planning::LookAround;
    using message::strategy::LookAtBall;
    using message::strategy::StandStill;
    using message::support::FieldDescription;
    using LocalisationBall = message::localisation::Ball;
    using VisionBalls      = message::vision::Balls;

    namespace {

        /// How long ground truth is kept: a shot plus the latency search, with margin
        constexpr double TRUTH_HISTORY_S = 15.0;
        /// How far past the newest ground-truth sample it may be extrapolated (it arrives at 50 Hz)
        constexpr double MAX_EXTRAPOLATION_S = 0.06;
        /// Ball speed below which the velocity-response test is not meaningful
        constexpr double MIN_RESPONSE_SPEED = 0.3;

        double seconds(std::chrono::system_clock::duration d) {
            return std::chrono::duration<double>(d).count();
        }

        std::array<double, 2> pair_of(const Configuration& config, const char* key) {
            const auto v = config[key].as<std::vector<double>>();
            if (v.size() != 2) {
                throw std::runtime_error(std::string("BallLocalisationBenchmark.yaml: ") + key + " needs [min, max]");
            }
            return {v[0], v[1]};
        }

        double wrap_angle(double a) {
            return std::atan2(std::sin(a), std::cos(a));
        }

        Eigen::Matrix2d rot(double yaw) {
            return Eigen::Rotation2Dd(yaw).toRotationMatrix();
        }

        double rms(const std::vector<double>& v) {
            if (v.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0) / double(v.size()));
        }

        double median(std::vector<double> v) {
            if (v.empty()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        }

        /// Interpolate between the two samples of a time-sorted buffer bracketing t
        template <typename Buffer, typename F>
        auto bracket(const Buffer& buffer, std::chrono::system_clock::time_point t, F&& lerp)
            -> std::optional<decltype(lerp(buffer.front(), buffer.front(), 0.0))> {
            if (buffer.empty() || t < buffer.front().t) {
                return std::nullopt;
            }
            if (t >= buffer.back().t) {
                return lerp(buffer.back(), buffer.back(), 0.0);
            }
            const auto hi  = std::upper_bound(buffer.begin(), buffer.end(), t, [](const auto& time, const auto& s) {
                return time < s.t;
            });
            const auto lo  = std::prev(hi);
            const double a = seconds(t - lo->t) / std::max(1e-9, seconds(hi->t - lo->t));
            return lerp(*lo, *hi, a);
        }

    }  // namespace

    BallLocalisationBenchmark::SysTime BallLocalisationBenchmark::to_sys(NUClear::clock::time_point t) {
        return std::chrono::system_clock::now()
               + std::chrono::duration_cast<std::chrono::system_clock::duration>(t - NUClear::clock::now());
    }

    std::optional<std::pair<Eigen::Vector2d, Eigen::Vector2d>> BallLocalisationBenchmark::truth_in_robot(
        SysTime t) const {
        if (ball_truth.empty() || robot_truth.empty()) {
            return std::nullopt;
        }
        // Truth arrives at 50 Hz, so a query can land just after the newest sample: extrapolate the ball
        // a little with its own velocity (the robot stands still, so its pose is simply held)
        const double ahead = seconds(t - ball_truth.back().t);
        if (ahead > MAX_EXTRAPOLATION_S) {
            return std::nullopt;
        }

        const auto ball =
            bracket(ball_truth, std::min(t, ball_truth.back().t), [](const BallTruth& a, const BallTruth& b, double s) {
                return std::make_pair(Eigen::Vector3d(a.rBSs + s * (b.rBSs - a.rBSs)),
                                      Eigen::Vector3d(a.vBs + s * (b.vBs - a.vBs)));
            });
        const auto robot =
            bracket(robot_truth,
                    std::min(t, robot_truth.back().t),
                    [](const RobotTruth& a, const RobotTruth& b, double s) {
                        return std::make_pair(Eigen::Vector2d(a.position + s * (b.position - a.position)),
                                              a.yaw + s * wrap_angle(b.yaw - a.yaw));
                    });
        if (!ball || !robot) {
            return std::nullopt;
        }

        Eigen::Vector3d rBSs = ball->first;
        if (ahead > 0.0) {
            rBSs += ball->second * ahead;
        }
        const Eigen::Matrix2d Rrs = rot(robot->second).transpose();
        return std::make_pair(Eigen::Vector2d(Rrs * (rBSs.head<2>() - robot->first)),
                              Eigen::Vector2d(Rrs * ball->second.head<2>()));
    }

    BallLocalisationBenchmark::BallLocalisationBenchmark(std::unique_ptr<NUClear::Environment> environment)
        : BehaviourReactor(std::move(environment)) {

        on<Configuration>("BallLocalisationBenchmark.yaml").then([this](const Configuration& config) {
            log_level          = config["log_level"].as<NUClear::LogLevel>();
            cfg.start_delay    = config["start_delay"].as<double>();
            cfg.seed           = config["seed"].as<unsigned>();
            cfg.shots          = config["shots"].as<int>();
            cfg.settle_time    = config["settle_time"].as<double>();
            cfg.roll_time      = config["roll_time"].as<double>();
            cfg.start_distance = pair_of(config, "start_distance");
            cfg.start_lateral  = pair_of(config, "start_lateral");
            cfg.target_lateral = pair_of(config, "target_lateral");
            cfg.speed          = pair_of(config, "speed");
            const auto lag     = config["lag_search"].as<std::vector<double>>();
            if (lag.size() != 3 || lag[2] <= 0.0) {
                throw std::runtime_error("BallLocalisationBenchmark.yaml: lag_search needs [min, max, step > 0]");
            }
            cfg.lag_search            = {lag[0], lag[1], lag[2]};
            cfg.output_dir            = config["output_dir"].as<std::string>();
            cfg.shutdown_when_done    = config["shutdown_when_done"].as<bool>();
            cfg.stand_still_priority  = config["tasks"]["stand_still_priority"].as<int>();
            cfg.look_at_ball_priority = config["tasks"]["look_at_ball_priority"].as<int>();
            cfg.look_around_priority  = config["tasks"]["look_around_priority"].as<int>();
        });

        on<Startup>().then([this] {
            rng.seed(cfg.seed);
            start_after = std::chrono::system_clock::now()
                          + std::chrono::duration_cast<std::chrono::system_clock::duration>(
                              std::chrono::duration<double>(cfg.start_delay));
            log<INFO>("Ball localisation benchmark:", cfg.shots, "shots starting in", cfg.start_delay, "s");
        });

        // Keep the robot standing and looking at the ball for the whole run
        on<Every<10, Per<std::chrono::seconds>>>().then([this] {
            if (cfg.stand_still_priority > 0) {
                emit<Task>(std::make_unique<StandStill>(), cfg.stand_still_priority);
            }
            if (cfg.look_around_priority > 0) {
                emit<Task>(std::make_unique<LookAround>(), cfg.look_around_priority);
            }
            if (cfg.look_at_ball_priority > 0) {
                emit<Task>(std::make_unique<LookAtBall>(), cfg.look_at_ball_priority);
            }
        });

        on<Trigger<NUSimBallGroundTruth>, Sync<BallLocalisationBenchmark>>().then(
            [this](const NUSimBallGroundTruth& gt) {
                ball_truth.push_back({gt.timestamp, gt.rBSs, gt.vBs});
                while (!ball_truth.empty() && seconds(ball_truth.back().t - ball_truth.front().t) > TRUTH_HISTORY_S) {
                    ball_truth.pop_front();
                }
            });

        on<Trigger<NUSimRobotGroundTruth>, Sync<BallLocalisationBenchmark>>().then(
            [this](const NUSimRobotGroundTruth& gt) {
                const Eigen::Matrix3d R = gt.Hst.linear();
                robot_truth.push_back({gt.timestamp, gt.Hst.translation().head<2>(), std::atan2(R(1, 0), R(0, 0))});
                while (!robot_truth.empty()
                       && seconds(robot_truth.back().t - robot_truth.front().t) > TRUTH_HISTORY_S) {
                    robot_truth.pop_front();
                }
            });

        // The shot schedule
        on<Every<50, Per<std::chrono::seconds>>, Optional<With<FieldDescription>>, Sync<BallLocalisationBenchmark>>()
            .then([this](const std::shared_ptr<const FieldDescription>& fd) {
                const auto now = std::chrono::system_clock::now();
                switch (phase) {
                    case Phase::WAITING:
                        if (now < start_after) {
                            return;
                        }
                        if (ball_truth.empty() || robot_truth.empty()) {
                            log<WARN>(
                                "No NUSim ground truth yet (is input::NUSimGroundTruth in the role and NUSim "
                                "running?)");
                            start_after = now + std::chrono::seconds(2);
                            return;
                        }
                        open_outputs();
                        place_next_shot();
                        return;
                    case Phase::SETTLING:
                        if (seconds(now - phase_since) >= cfg.settle_time) {
                            // The detector turns apparent size into range with the configured radius, so
                            // a sim ball of a different size biases every detection
                            if (fd != nullptr && shots_done == 0) {
                                const double sim_radius = ball_truth.back().rBSs.z();
                                if (std::abs(sim_radius - fd->ball_radius) > 0.005) {
                                    log<WARN>("NUSim ball radius",
                                              sim_radius,
                                              "differs from FieldDescription",
                                              fd->ball_radius,
                                              "- detections will be range-biased");
                                }
                            }
                            kick();
                        }
                        return;
                    case Phase::ROLLING:
                        if (seconds(now - phase_since) >= cfg.roll_time) {
                            finish_shot();
                            if (shots_done >= cfg.shots) {
                                finish_run();
                            }
                            else {
                                place_next_shot();
                            }
                        }
                        return;
                    case Phase::DONE: return;
                }
            });

        // Every UKF estimate against the ground truth at the moment it is available to consumers
        on<Trigger<LocalisationBall>, With<Sensors>, Sync<BallLocalisationBenchmark>>().then(
            [this](const LocalisationBall& ball, const Sensors& sensors) {
                if (phase != Phase::SETTLING && phase != Phase::ROLLING) {
                    return;
                }
                const auto now          = std::chrono::system_clock::now();
                const auto gt           = truth_in_robot(now);
                const Eigen::Vector2d r = (sensors.Hrw * ball.rBWw).head<2>();
                const Eigen::Vector2d v = (sensors.Hrw.linear() * ball.vBw).head<2>();
                const bool rolling      = phase == Phase::ROLLING;

                if (rolling) {
                    shot.estimates.push_back({now, r, v});
                    if (gt && !shot.velocity_response && gt->second.norm() > MIN_RESPONSE_SPEED) {
                        const double cos_angle = v.dot(gt->second) / std::max(1e-9, v.norm() * gt->second.norm());
                        if (v.norm() >= 0.9 * gt->second.norm() && cos_angle > std::cos(M_PI / 9.0)) {
                            shot.velocity_response = seconds(now - shot.kicked);
                        }
                    }
                }
                if (samples_csv) {
                    samples_csv << shot.id << ',' << (rolling ? "roll" : "settle") << ',' << seconds(now - shot.placed)
                                << ',' << (rolling ? seconds(now - shot.kicked) : std::nan("")) << ',' << r.x() << ','
                                << r.y() << ',' << v.x() << ',' << v.y() << ',';
                    if (gt) {
                        samples_csv << gt->first.x() << ',' << gt->first.y() << ',' << gt->second.x() << ','
                                    << gt->second.y();
                    }
                    else {
                        samples_csv << ",,,";
                    }
                    samples_csv << ',' << ball.confidence << '\n';
                }
            });

        // Every raw vision detection against the ground truth at its image time: separates detector
        // error from filter error, and measures the vision pipeline's latency
        on<Trigger<VisionBalls>, With<Sensors>, Sync<BallLocalisationBenchmark>>().then(
            [this](const VisionBalls& balls, const Sensors& sensors) {
                if (phase != Phase::SETTLING && phase != Phase::ROLLING) {
                    return;
                }
                const auto now              = std::chrono::system_clock::now();
                const auto image_time       = to_sys(balls.timestamp);
                const double latency        = seconds(now - image_time);
                const auto gt               = truth_in_robot(image_time);
                const Eigen::Isometry3d Hwc = Eigen::Isometry3d(balls.Hcw).inverse();
                const bool rolling          = phase == Phase::ROLLING;
                shot.vision_latencies.push_back(latency);

                // Several candidates can come back; score the one nearest the truth (the association
                // the filter should make) and log them all
                double best = std::numeric_limits<double>::infinity();
                for (const auto& b : balls.balls) {
                    if (b.is_invalid || b.measurements.empty()) {
                        continue;
                    }
                    const Eigen::Vector2d r = (sensors.Hrw * (Hwc * b.measurements[0].rBCc)).head<2>();
                    if (gt) {
                        best = std::min(best, (r - gt->first).norm());
                    }
                    if (detections_csv) {
                        detections_csv << shot.id << ',' << (rolling ? "roll" : "settle") << ','
                                       << seconds(now - shot.placed) << ','
                                       << (rolling ? seconds(image_time - shot.kicked) : std::nan("")) << ',' << latency
                                       << ',' << r.x() << ',' << r.y() << ',';
                        if (gt) {
                            detections_csv << gt->first.x() << ',' << gt->first.y() << '\n';
                        }
                        else {
                            detections_csv << ",\n";
                        }
                    }
                }
                if (rolling && std::isfinite(best)) {
                    shot.detection_errors.push_back(best);
                }
            });
    }

    void BallLocalisationBenchmark::open_outputs() {
        const std::time_t now = std::time(nullptr);
        std::ostringstream stamp{};
        stamp << std::put_time(std::localtime(&now), "%Y%m%d-%H%M%S");
        const std::filesystem::path dir = std::filesystem::path(cfg.output_dir) / stamp.str();
        std::error_code ec{};
        std::filesystem::create_directories(dir, ec);
        if (ec) {
            log<ERROR>("Cannot create", dir.string(), ec.message(), "- continuing without CSV output");
            return;
        }
        samples_csv.open(dir / "samples.csv");
        detections_csv.open(dir / "detections.csv");
        summary_csv.open(dir / "summary.csv");
        samples_csv << "shot,phase,t_shot,t_kick,est_x,est_y,est_vx,est_vy,gt_x,gt_y,gt_vx,gt_vy,confidence\n";
        detections_csv << "shot,phase,t_shot,t_kick,latency,det_x,det_y,gt_x,gt_y\n";
        summary_csv << "shot,start_x,start_y,target_y,speed,n_estimates,n_detections,estimate_rate_hz,"
                       "detection_rate_hz,pos_rmse,pos_max,vel_rmse,speed_rmse,velocity_response_s,best_lag_s,"
                       "pos_rmse_at_best_lag,detection_pos_rmse,vision_latency_median_s\n";
        log<INFO>("Writing ball localisation benchmark results to", dir.string());
    }

    void BallLocalisationBenchmark::place_next_shot() {
        std::uniform_real_distribution<double> distance(cfg.start_distance[0], cfg.start_distance[1]);
        std::uniform_real_distribution<double> lateral(cfg.start_lateral[0], cfg.start_lateral[1]);
        std::uniform_real_distribution<double> target(cfg.target_lateral[0], cfg.target_lateral[1]);
        std::uniform_real_distribution<double> speed(cfg.speed[0], cfg.speed[1]);

        shot          = Shot{};
        shot.id       = shots_done;
        shot.start    = Eigen::Vector2d(distance(rng), lateral(rng));
        shot.target_y = target(rng);
        shot.speed    = speed(rng);

        // Rest the ball in front of the robot; the estimate gets settle_time to converge on it
        auto cmd      = std::make_unique<NUSimBallCommand>();
        cmd->frame    = NUSimBallCommand::Frame::ROBOT;
        cmd->position = Eigen::Vector3d(shot.start.x(), shot.start.y(), -1.0);
        emit(cmd);

        phase       = Phase::SETTLING;
        phase_since = std::chrono::system_clock::now();
        shot.placed = phase_since;
        log<INFO>("Shot",
                  shot.id + 1,
                  "/",
                  cfg.shots,
                  ": ball at (",
                  shot.start.x(),
                  ",",
                  shot.start.y(),
                  "), speed",
                  shot.speed,
                  "m/s towards y =",
                  shot.target_y);
    }

    void BallLocalisationBenchmark::kick() {
        // Kick from wherever the ball actually rests, so the only thing that changes is its velocity
        const auto now             = std::chrono::system_clock::now();
        const auto gt              = truth_in_robot(now);
        const Eigen::Vector2d from = gt ? gt->first : shot.start;
        const Eigen::Vector2d dir  = (Eigen::Vector2d(0.0, shot.target_y) - from).normalized();

        auto cmd      = std::make_unique<NUSimBallCommand>();
        cmd->frame    = NUSimBallCommand::Frame::ROBOT;
        cmd->position = Eigen::Vector3d(from.x(), from.y(), -1.0);
        cmd->velocity = Eigen::Vector3d(shot.speed * dir.x(), shot.speed * dir.y(), 0.0);
        cmd->rolling  = true;
        emit(cmd);

        phase       = Phase::ROLLING;
        phase_since = now;
        shot.kicked = now;
    }

    void BallLocalisationBenchmark::finish_shot() {
        std::vector<double> pos_err{}, vel_err{}, speed_err{};
        for (const auto& e : shot.estimates) {
            if (const auto gt = truth_in_robot(e.t)) {
                pos_err.push_back((e.r - gt->first).norm());
                vel_err.push_back((e.v - gt->second).norm());
                speed_err.push_back(e.v.norm() - gt->second.norm());
            }
        }

        // Effective latency: the delay at which the estimate best matches the ground truth
        double best_lag = std::numeric_limits<double>::quiet_NaN();
        double best_rms = std::numeric_limits<double>::infinity();
        for (double lag = cfg.lag_search[0]; lag <= cfg.lag_search[1] + 1e-9; lag += cfg.lag_search[2]) {
            std::vector<double> err{};
            for (const auto& e : shot.estimates) {
                const auto shifted = e.t
                                     - std::chrono::duration_cast<std::chrono::system_clock::duration>(
                                         std::chrono::duration<double>(lag));
                if (const auto gt = truth_in_robot(shifted)) {
                    err.push_back((e.r - gt->first).norm());
                }
            }
            const double r = rms(err);
            if (err.size() >= 5 && r < best_rms) {
                best_rms = r;
                best_lag = lag;
            }
        }

        const double n_est    = double(shot.estimates.size());
        const double n_det    = double(shot.detection_errors.size());
        const double pos_rmse = rms(pos_err);
        const double pos_max  = pos_err.empty() ? std::nan("") : *std::max_element(pos_err.begin(), pos_err.end());
        const double vel_rmse = rms(vel_err);
        const double response = shot.velocity_response.value_or(std::numeric_limits<double>::quiet_NaN());

        if (summary_csv) {
            summary_csv << shot.id << ',' << shot.start.x() << ',' << shot.start.y() << ',' << shot.target_y << ','
                        << shot.speed << ',' << n_est << ',' << n_det << ',' << n_est / cfg.roll_time << ','
                        << n_det / cfg.roll_time << ',' << pos_rmse << ',' << pos_max << ',' << vel_rmse << ','
                        << rms(speed_err) << ',' << response << ',' << best_lag << ','
                        << (std::isfinite(best_rms) ? best_rms : std::nan("")) << ',' << rms(shot.detection_errors)
                        << ',' << median(shot.vision_latencies) << '\n';
            summary_csv.flush();
            samples_csv.flush();
            detections_csv.flush();
        }
        run_summary.push_back({pos_rmse, vel_rmse, response, best_lag});
        log<INFO>("Shot",
                  shot.id + 1,
                  "done: pos rmse",
                  pos_rmse,
                  "m, vel rmse",
                  vel_rmse,
                  "m/s, velocity response",
                  response,
                  "s, effective lag",
                  best_lag,
                  "s,",
                  n_est,
                  "estimates,",
                  n_det,
                  "detections");
        ++shots_done;
    }

    void BallLocalisationBenchmark::finish_run() {
        phase = Phase::DONE;
        std::vector<double> pos{}, vel{}, response{}, lag{};
        int responded = 0;
        for (const auto& s : run_summary) {
            pos.push_back(s[0]);
            vel.push_back(s[1]);
            if (std::isfinite(s[2])) {
                response.push_back(s[2]);
                ++responded;
            }
            if (std::isfinite(s[3])) {
                lag.push_back(s[3]);
            }
        }
        log<INFO>("Ball localisation benchmark done over",
                  run_summary.size(),
                  "shots: median pos rmse",
                  median(pos),
                  "m, median vel rmse",
                  median(vel),
                  "m/s, velocity followed the kick in",
                  responded,
                  "/",
                  run_summary.size(),
                  "shots (median",
                  median(response),
                  "s), median effective lag",
                  median(lag),
                  "s");
        samples_csv.close();
        detections_csv.close();
        summary_csv.close();
        if (cfg.shutdown_when_done) {
            powerplant.shutdown();
        }
    }

}  // namespace module::tools
