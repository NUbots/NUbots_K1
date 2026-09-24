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
#include "LocalisationPlayback.hpp"

#include <Eigen/Geometry>
#include <cmath>
#include <filesystem>
#include <iomanip>

#include "extension/Configuration.hpp"

#include "message/input/Sensors.hpp"
#include "message/localisation/Field.hpp"

namespace module::tools {

    using extension::Configuration;

    using message::nbs::player::LoadRequest;
    using message::nbs::player::PlaybackFinished;
    using message::nbs::player::PlaybackState;
    using message::nbs::player::PlayRequest;
    using message::nbs::player::SetModeRequest;
    using message::nbs::player::PlaybackMode::FAST;
    using message::nbs::player::PlaybackMode::REALTIME;
    using message::nbs::player::PlaybackMode::SEQUENTIAL;

    using message::input::Sensors;
    using message::localisation::Field;

    using NUClear::message::CommandLineArguments;

    namespace fs = std::filesystem;

    LocalisationPlayback::LocalisationPlayback(std::unique_ptr<NUClear::Environment> environment)
        : Reactor(std::move(environment)), config{} {

        on<Configuration>("LocalisationPlayback.yaml").then([this](const Configuration& cfg) {
            this->log_level = cfg["log_level"].as<NUClear::LogLevel>();

            auto playback_mode = cfg["playback_mode"].as<std::string>();
            if (playback_mode == "FAST") {
                config.mode = FAST;
            }
            else if (playback_mode == "SEQUENTIAL") {
                config.mode = SEQUENTIAL;
            }
            else if (playback_mode == "REALTIME") {
                config.mode = REALTIME;
            }
            else {
                log<ERROR>("Playback mode is invalid, stopping playback");
                powerplant.shutdown();
            }

            config.output_directory = cfg["output_directory"].as<std::string>();

            // Update which types we will be playing
            config.messages.clear();
            for (const auto& setting : cfg["messages"]) {
                auto name    = setting.first.as<std::string>();
                bool enabled = setting.second.as<bool>();
                if (enabled) {
                    config.messages.push_back(name);
                }
            }
        });

        on<Startup, With<CommandLineArguments>>().then([this](const CommandLineArguments& args) {
            if (args.size() < 2) {
                log<ERROR>("Pass the nbs files to play back on the command line");
                powerplant.shutdown();
                return;
            }

            // Name the output after the recording and the binary, so the NLopt and SRIF roles write side by side
            fs::create_directories(config.output_directory);
            csv_path = (fs::path(config.output_directory)
                        / (fs::path(args[1]).stem().string() + "_" + fs::path(args[0]).filename().string() + ".csv"))
                           .string();
            csv.open(csv_path);
            if (!csv) {
                log<ERROR>("Could not open", csv_path, "for writing");
                powerplant.shutdown();
                return;
            }
            csv << "t_ns,x,y,yaw,localised,uncertainty,cost\n" << std::setprecision(9);
            log<INFO>("Writing field pose estimates to", csv_path);

            // Set playback mode
            auto set_mode_request  = std::make_unique<SetModeRequest>();
            set_mode_request->mode = config.mode;
            emit<Scope::INLINE>(set_mode_request);

            // Load the files
            auto load_request      = std::make_unique<LoadRequest>();
            load_request->files    = std::vector<std::string>(std::next(args.begin()), args.end());
            load_request->messages = config.messages;
            emit<Scope::INLINE>(std::move(load_request));

            // Start playback
            emit<Scope::INLINE>(std::make_unique<PlayRequest>());
        });

        // The torso in the field, Hft = Hfw * Htw^-1, against the latest odometry. The Player time travels
        // NUClear's clock along the recording, so now() is the recording's time.
        on<Trigger<Field>, With<Sensors>>().then([this](const Field& field, const Sensors& sensors) {
            if (!csv.is_open()) {
                return;
            }
            const Eigen::Isometry3d Hft = Eigen::Isometry3d(field.Hfw) * Eigen::Isometry3d(sensors.Htw).inverse();
            // Integer nanoseconds: seconds since the epoch at 9 significant figures would lose the fraction
            const int64_t t =
                std::chrono::duration_cast<std::chrono::nanoseconds>(NUClear::clock::now().time_since_epoch()).count();
            const double yaw = std::atan2(Hft.linear()(1, 0), Hft.linear()(0, 0));
            csv << t << "," << Hft.translation().x() << "," << Hft.translation().y() << "," << yaw << ","
                << int(field.localised) << "," << field.uncertainty << "," << field.cost << "\n";
            ++count;
        });

        on<Trigger<PlaybackState>>().then([this](const PlaybackState& playback_state) {
            progress_bar.update(playback_state.current_message, playback_state.total_messages, "", "NBS Playback");
        });

        on<Trigger<PlaybackFinished>>().then([this] {
            progress_bar.close();
            csv.close();
            log<INFO>("Finished playback: wrote", count, "field pose estimates to", csv_path);
            powerplant.shutdown();
        });
    }

}  // namespace module::tools
