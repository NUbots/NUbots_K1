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
#ifndef MODULE_SKILL_K1VISUALKICK_HPP
#define MODULE_SKILL_K1VISUALKICK_HPP

#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <memory>
#include <nuclear>
#include <string>

#include "extension/Behaviour.hpp"

#include "message/booster/BoosterVisualKick.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Ball.hpp"
#include "message/localisation/Field.hpp"
#include "message/skill/Kick.hpp"
#include "message/support/FieldDescription.hpp"

namespace module::skill {

    namespace bip = boost::interprocess;

    // Mirrors NUbridge::SharedKickHeader (NUbridge/include/NUbridge/KickBallWriter.hpp) -
    // layout must stay in sync between the two repos.
    struct SharedKickHeader {
        bip::interprocess_mutex mutex;
        bip::interprocess_condition has_new_data;
        uint64_t sequence{0};
        double x{0.0};
        double y{0.0};
        double dir{0.0};
        double goal_x{0.0};
        double goal_y{0.0};
        double robot_theta_to_field{0.0};
        double power{0.0};
    };

    struct KickSharedMemory {
        explicit KickSharedMemory(const std::string& segment)
            : shm(bip::open_only, segment.c_str(), bip::read_write), region(shm, bip::read_write) {
            header = reinterpret_cast<SharedKickHeader*>(region.get_address());
        }

        bip::shared_memory_object shm;
        bip::mapped_region region;
        SharedKickHeader* header = nullptr;
    };

    class K1VisualKick : public ::extension::behaviour::BehaviourReactor {
    private:
        /// @brief Stores configuration values
        struct Config {
            /// @brief Which version of Booster's visual kick to use
            message::booster::VisualKickVer version = message::booster::VisualKickVer::V2;
            /// @brief How long to let the kick run before reporting the task as done
            NUClear::clock::duration kick_duration{};
            /// @brief Name of the shared memory segment NUbridge publishes kick decisions from
            std::string kick_decision_segment{};
            /// @brief Maximum distance (metres) the ball can be from the robot for a kick to proceed
            double max_kick_distance = 0.0;
        } cfg;

        /// @brief The time the current visual kick was started
        NUClear::clock::time_point kick_start_time{};

        /// @brief Shared memory segment that publishes kick decisions to NUbridge
        std::unique_ptr<KickSharedMemory> kick_shared_memory;
        bool kick_shm_unavailable_logged = false;

        /// @brief Lazily (re)connects to the kick decision shared memory segment
        void connect_kick_shm();

        /// @brief Writes the given Kick task's decision data into shared memory, if the segment is available
        void publish_kick_decision(const message::skill::Kick& kick,
                                    const message::input::Sensors& sensors,
                                    const message::localisation::Field& field,
                                    const message::support::FieldDescription& field_description);

    public:
        /// @brief Called by the powerplant to build and setup the K1VisualKick reactor.
        explicit K1VisualKick(std::unique_ptr<NUClear::Environment> environment);
    };

}  // namespace module::skill

#endif  // MODULE_SKILL_K1VISUALKICK_HPP
