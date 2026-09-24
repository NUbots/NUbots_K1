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
#ifndef MODULE_PLANNING_PLANSAVE_HANDOFF_HPP
#define MODULE_PLANNING_PLANSAVE_HANDOFF_HPP

#include <Eigen/Geometry>
#include <cmath>
#include <optional>

namespace module::planning::save {

    /// The leg joints that place a K1's ankle, in radians as the servos report them
    struct LegAngles {
        double hip_pitch = 0.0;
        double hip_roll  = 0.0;
        double hip_yaw   = 0.0;
        double knee      = 0.0;
    };

    /**
     * Where a K1's ankle (the foot link's origin, on the ankle joints) is in the trunk frame, from the leg chain in
     * shared/utility/platform/models/k1/robot.urdf. The ankle joints turn the foot about that point, so they don't
     * move it.
     */
    inline Eigen::Vector3d ankle_in_trunk(const LegAngles& q, const bool left) {
        const double side   = left ? 1.0 : -1.0;
        Eigen::Isometry3d H = Eigen::Isometry3d::Identity();
        H.translate(Eigen::Vector3d(0.0, side * 0.096, -0.077));
        H.rotate(Eigen::AngleAxisd(q.hip_pitch, Eigen::Vector3d::UnitY()));
        H.translate(Eigen::Vector3d(0.0, 0.0, -0.026));
        H.rotate(Eigen::AngleAxisd(q.hip_roll, Eigen::Vector3d::UnitX()));
        H.translate(Eigen::Vector3d(0.012, 0.0, -0.0485));
        H.rotate(Eigen::AngleAxisd(q.hip_yaw, Eigen::Vector3d::UnitZ()));
        H.translate(Eigen::Vector3d(-0.014, 0.0, -0.117));
        H.rotate(Eigen::AngleAxisd(q.knee, Eigen::Vector3d::UnitY()));
        H.translate(Eigen::Vector3d(0.00019706, side * 0.0002, -0.24519));
        return H.translation();
    }

    /**
     * How far the left ankle is above the right (m), with the trunk turned upright by its orientation in the world
     * Rwt. The K1 has no foot contact sensors, so both feet are taken to be down when this is near zero: on flat ground
     * a planted foot's ankle is as high as the other's, and a swinging one is centimetres higher.
     */
    inline double ankle_height_difference(const LegAngles& left, const LegAngles& right, const Eigen::Matrix3d& Rwt) {
        return (Rwt * (ankle_in_trunk(left, true) - ankle_in_trunk(right, false))).z();
    }

    struct HandoffConfig {
        /// Off: hand a walking goalie to the block policy at once
        bool enabled = true;
        /// Ankles within this height of each other (m) count as both feet down
        double foot_height_tolerance = 0.005;
        /// Longest wait (s) for both feet down before handing off anyway: half a step of the walk
        double max_wait = 0.35;
    };

    /**
     * When to hand a walking goalie to the block policy. The policy was trained from a standing start, and taking
     * over mid-stride, with one foot in the air and the body moving, topples it; so a hand-off waits for both feet to
     * be down, for at most max_wait. A goalie that isn't walking is handed over at once, and one already handed over
     * stays with the block policy until the planner no longer wants it.
     */
    class HandoffGate {
    public:
        /// Whether the block policy should have the goalie this tick
        bool step(const bool wanted,
                  const bool walking,
                  const std::optional<double>& height_difference,
                  const double dt,
                  const HandoffConfig& cfg) {
            if (!wanted) {
                reset();
                return false;
            }
            if (handed) {
                return true;
            }
            feet_down = height_difference.has_value() && std::abs(*height_difference) < cfg.foot_height_tolerance;
            if (!cfg.enabled || !walking || feet_down || waited >= cfg.max_wait) {
                handed      = true;
                last_wait   = waited;
                timed_out   = cfg.enabled && walking && !feet_down;
                was_walking = walking;
                return true;
            }
            waited += dt;
            return false;
        }

        void reset() {
            handed = false;
            waited = 0.0;
        }

        /// Whether a hand-off is being held back for the feet
        [[nodiscard]] bool waiting() const {
            return !handed && waited > 0.0;
        }
        [[nodiscard]] double waited_so_far() const {
            return waited;
        }

        /// The last hand-off: how long it waited (s), whether the goalie was walking, and whether it gave up
        /// waiting for the feet
        double last_wait = 0.0;
        bool was_walking = false;
        bool timed_out   = false;

    private:
        bool handed    = false;
        bool feet_down = false;
        double waited  = 0.0;
    };

}  // namespace module::planning::save

#endif  // MODULE_PLANNING_PLANSAVE_HANDOFF_HPP
