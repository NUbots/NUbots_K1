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
#ifndef MODULE_INPUT_NUSIMGROUNDTRUTH_HPP
#define MODULE_INPUT_NUSIMGROUNDTRUTH_HPP

#include <Eigen/Geometry>
#include <booster/idl/nav_msgs/Odometry.h>
#include <booster/robot/channel/channel_factory.hpp>
#include <mutex>
#include <nuclear>
#include <string>

#include "message/booster/NUSimGroundTruth.hpp"

namespace module::input {

    /// Bridges NUSim's test surface into NUClear: ground truth for the ball and the robot torso in
    /// (rt/nusim/gt/ball, rt/nusim/gt/robot -> message::booster::NUSimBallGroundTruth /
    /// NUSimRobotGroundTruth), where the ball will cross the robot and the goal line
    /// (rt/nusim/gt/ball_crossing/* -> NUSimBallCrossings), and ball commands out
    /// (message::booster::NUSimBallCommand -> rt/nusim/ball_command). Only useful against NUSim; on a real
    /// robot the topics never appear.
    class NUSimGroundTruth : public NUClear::Reactor {
    public:
        explicit NUSimGroundTruth(std::unique_ptr<NUClear::Environment> environment);

    private:
        struct Config {
            std::string ball_topic;
            std::string robot_topic;
            std::string robot_crossing_topic;
            std::string goal_crossing_topic;
            std::string command_topic;
        } cfg;

        void ball_handler(const void* msg);
        void robot_handler(const void* msg);
        /// One handler per crossing topic; each emits both crossings, the other one as last received
        void crossing_handler(const void* msg, bool goal_line);

        /// The latest torso ground truth, to put the ball in the robot frame {r}, and the latest crossings. The DDS
        /// callbacks run on their own threads.
        std::mutex truth_mutex;
        bool have_Hst         = false;
        Eigen::Isometry3d Hst = Eigen::Isometry3d::Identity();
        message::booster::NUSimBallCrossings crossings{};

        booster::robot::ChannelPtr<nav_msgs::msg::Odometry> ball_channel;
        booster::robot::ChannelPtr<nav_msgs::msg::Odometry> robot_channel;
        booster::robot::ChannelPtr<nav_msgs::msg::Odometry> robot_crossing_channel;
        booster::robot::ChannelPtr<nav_msgs::msg::Odometry> goal_crossing_channel;
        booster::robot::ChannelPtr<nav_msgs::msg::Odometry> command_channel;
    };

}  // namespace module::input

#endif  // MODULE_INPUT_NUSIMGROUNDTRUTH_HPP
