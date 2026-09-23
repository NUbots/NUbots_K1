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
#include "PlanSavePlayback.hpp"

#include <stdexcept>

#include "extension/Configuration.hpp"

#include "message/planning/Save.hpp"

namespace module::tools {

    using extension::Configuration;

    using message::nbs::player::LoadRequest;
    using message::nbs::player::PlaybackFinished;
    using message::nbs::player::PlaybackMode;
    using message::nbs::player::PlayRequest;
    using message::nbs::player::SetModeRequest;
    using message::planning::Save;
    using NUClear::message::CommandLineArguments;

    PlanSavePlayback::PlanSavePlayback(std::unique_ptr<NUClear::Environment> environment)
        : BehaviourReactor(std::move(environment)) {

        on<Configuration>("PlanSavePlayback.yaml").then([this](const Configuration& config) {
            this->log_level = config["log_level"].as<NUClear::LogLevel>();

            const auto mode = config["playback_mode"].as<std::string>();
            if (mode == "SEQUENTIAL") {
                cfg.mode = PlaybackMode::SEQUENTIAL;
            }
            else if (mode == "REALTIME") {
                cfg.mode = PlaybackMode::REALTIME;
            }
            else if (mode == "FAST") {
                cfg.mode = PlaybackMode::FAST;
            }
            else {
                throw std::runtime_error("PlanSavePlayback.yaml: playback_mode must be SEQUENTIAL, REALTIME or FAST, not "
                                         + mode);
            }

            cfg.files = config["files"].as<std::vector<std::string>>();

            cfg.messages.clear();
            for (const auto& setting : config["messages"]) {
                if (setting.second.as<bool>()) {
                    cfg.messages.push_back(setting.first.as<std::string>());
                }
            }
        });

        on<Startup, With<CommandLineArguments>>().then([this](const CommandLineArguments& args) {
            // Recordings on the command line, or else from the config, which survives `./b configure` rewriting the
            // VS Code launch configurations
            auto files = std::vector<std::string>(std::next(args.begin()), args.end());
            if (files.empty()) {
                files = cfg.files;
            }
            if (files.empty()) {
                log<ERROR>("Give the recording(s) to play: ./b run data/plansaveplayback recordings/<file>.nbs, or "
                           "files in PlanSavePlayback.yaml");
                powerplant.shutdown();
                return;
            }

            // What the harness did live: guard the goal. With nothing else asking for the servos, PlanSave's Block
            // commands have no provider and go nowhere, which is what a replay wants.
            emit<Task>(std::make_unique<Save>());

            auto set_mode  = std::make_unique<SetModeRequest>();
            set_mode->mode = cfg.mode;
            emit<Scope::INLINE>(set_mode);

            auto load      = std::make_unique<LoadRequest>();
            load->files    = files;
            load->messages = cfg.messages;
            emit<Scope::INLINE>(load);

            emit<Scope::INLINE>(std::make_unique<PlayRequest>());
        });

        on<Trigger<PlaybackFinished>>().then([this] {
            log<INFO>("Finished playback");
            powerplant.shutdown();
        });
    }

}  // namespace module::tools
