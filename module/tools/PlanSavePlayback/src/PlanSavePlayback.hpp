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
#ifndef MODULE_TOOLS_PLANSAVEPLAYBACK_HPP
#define MODULE_TOOLS_PLANSAVEPLAYBACK_HPP

#include <nuclear>
#include <string>
#include <vector>

#include "extension/Behaviour.hpp"

#include "message/nbs/player/Player.hpp"

namespace module::tools {

    /**
     * Replays a goalie recording into planning::PlanSave, one message at a time, so PlanSave can be stepped through
     * under a debugger: nbs::Player's SEQUENTIAL mode holds time still while a breakpoint is hit. It also asks for the
     * Save task, which the harness that made the recording asked for live. See README.md.
     */
    class PlanSavePlayback : public ::extension::behaviour::BehaviourReactor {
    private:
        struct Config {
            message::nbs::player::PlaybackMode mode{};
            std::vector<std::string> messages{};
            /// Recordings to play when none are given on the command line
            std::vector<std::string> files{};
        } cfg;

    public:
        /// @brief Called by the powerplant to build and setup the PlanSavePlayback reactor.
        explicit PlanSavePlayback(std::unique_ptr<NUClear::Environment> environment);
    };

}  // namespace module::tools

#endif  // MODULE_TOOLS_PLANSAVEPLAYBACK_HPP
