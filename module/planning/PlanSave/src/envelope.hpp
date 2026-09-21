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
#ifndef MODULE_PLANNING_PLANSAVE_ENVELOPE_HPP
#define MODULE_PLANNING_PLANSAVE_ENVELOPE_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "prediction.hpp"

namespace module::planning::save {

    /**
     * Wilson score lower bound on a success rate: the rate the skill beats with the confidence z stands for. A
     * cell nobody measured has no evidence of success, so it scores 0.
     */
    inline double wilson_lower_bound(const int successes, const int trials, const double z) {
        if (trials <= 0) {
            return 0.0;
        }
        const double n      = trials;
        const double p      = std::clamp(successes / n, 0.0, 1.0);
        const double z2     = z * z;
        const double centre = p + z2 / (2.0 * n);
        const double spread = z * std::sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n));
        return std::max(0.0, (centre - spread) / (1.0 + z2 / n));
    }

    /**
     * A block policy's capability envelope: counts of on-target shots, saves and falls over the command it was
     * driven with (signed dy x time to arrival x ball speed), measured in mjlab and binned by
     * tools/policy/make_save_envelope.py. Cells are stored [speed][time][dy], flattened.
     */
    struct Envelope {
        /// SHA-256 (hex) of the ONNX the envelope was measured from
        std::string onnx_sha256{};
        std::vector<double> dy_edges{};
        std::vector<double> time_edges{};
        std::vector<double> speed_edges{};
        std::vector<int> trials{};
        std::vector<int> saves{};
        std::vector<int> falls{};

        [[nodiscard]] std::size_t n_dy() const {
            return dy_edges.size() - 1;
        }
        [[nodiscard]] std::size_t n_time() const {
            return time_edges.size() - 1;
        }
        [[nodiscard]] std::size_t n_speed() const {
            return speed_edges.size() - 1;
        }

        /// Whether the axes are increasing and the cell counts fill the grid
        [[nodiscard]] bool valid() const {
            const auto increasing = [](const std::vector<double>& e) {
                return e.size() >= 2 && std::is_sorted(e.begin(), e.end())
                       && std::adjacent_find(e.begin(), e.end()) == e.end();
            };
            if (!increasing(dy_edges) || !increasing(time_edges) || !increasing(speed_edges)) {
                return false;
            }
            const std::size_t cells = n_dy() * n_time() * n_speed();
            return trials.size() == cells && saves.size() == cells && falls.size() == cells;
        }

        [[nodiscard]] std::size_t index(const std::size_t speed, const std::size_t time, const std::size_t dy) const {
            return (speed * n_time() + time) * n_dy() + dy;
        }

        /// Bin of a value on an axis whose outer bins take everything beyond them
        static std::size_t clamped_bin(const std::vector<double>& edges, const double value) {
            const auto it = std::upper_bound(edges.begin(), edges.end(), value);
            const auto i  = static_cast<std::ptrdiff_t>(it - edges.begin()) - 1;
            return static_cast<std::size_t>(std::clamp<std::ptrdiff_t>(i, 0, std::ptrdiff_t(edges.size()) - 2));
        }

        /// Bin of a dy inside the grid, or nothing outside it
        [[nodiscard]] std::optional<std::size_t> dy_bin(const double dy) const {
            if (dy < dy_edges.front() || dy >= dy_edges.back()) {
                return std::nullopt;
            }
            return clamped_bin(dy_edges, dy);
        }

        /// Conservative success rate of one cell, 0 when it has fewer than min_trials shots
        [[nodiscard]] double success(const std::size_t cell, const double z, const int min_trials) const {
            return trials[cell] < min_trials ? 0.0 : wilson_lower_bound(saves[cell], trials[cell], z);
        }

        /**
         * Expected success of a block, averaging the cells' conservative success over a Gaussian on dy at the
         * predicted time and speed. Probability mass outside the dy grid is a shot the policy was never measured
         * reaching, so it counts as a miss.
         */
        [[nodiscard]] double expected_success(const double dy,
                                              const double sigma_dy,
                                              const double time,
                                              const double speed,
                                              const double z,
                                              const int min_trials) const {
            const std::size_t s = clamped_bin(speed_edges, speed);
            const std::size_t t = clamped_bin(time_edges, time);
            double expected     = 0.0;
            for (std::size_t d = 0; d < n_dy(); ++d) {
                const double p = probability_between(dy, sigma_dy, dy_edges[d], dy_edges[d + 1]);
                if (p > 0.0) {
                    expected += p * success(index(s, t, d), z, min_trials);
                }
            }
            return expected;
        }

        /// Measured fall rate at the predicted command, nothing if that cell has too few shots to say
        [[nodiscard]] std::optional<double> fall_rate(const double dy,
                                                      const double time,
                                                      const double speed,
                                                      const int min_trials) const {
            const auto d = dy_bin(dy);
            if (!d) {
                return std::nullopt;
            }
            const std::size_t cell = index(clamped_bin(speed_edges, speed), clamped_bin(time_edges, time), *d);
            if (trials[cell] < min_trials) {
                return std::nullopt;
            }
            return double(falls[cell]) / trials[cell];
        }
    };

}  // namespace module::planning::save

#endif  // MODULE_PLANNING_PLANSAVE_ENVELOPE_HPP
