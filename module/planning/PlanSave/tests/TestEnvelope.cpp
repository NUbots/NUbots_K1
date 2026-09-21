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

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "envelope.hpp"
#include "sha256.hpp"

using Catch::Approx;
using module::planning::save::Envelope;
using module::planning::save::Sha256;
using module::planning::save::wilson_lower_bound;

namespace {
    /// Two dy bins ([-1, 0) and [0, 1)), one time bin, one speed bin
    Envelope two_bins(int trials_left, int saves_left, int trials_right, int saves_right) {
        Envelope e{};
        e.dy_edges    = {-1.0, 0.0, 1.0};
        e.time_edges  = {0.0, 3.0};
        e.speed_edges = {0.0, 6.0};
        e.trials      = {trials_left, trials_right};
        e.saves       = {saves_left, saves_right};
        e.falls       = {0, trials_right / 10};
        return e;
    }
}  // namespace

SCENARIO("Wilson lower bounds are conservative success rates", "[PlanSave]") {
    THEN("no trials is no evidence of success") {
        REQUIRE(wilson_lower_bound(0, 0, 1.64) == 0.0);
    }
    THEN("it matches the closed form for 8 of 10 at z = 1.96") {
        REQUIRE(wilson_lower_bound(8, 10, 1.96) == Approx(0.4902).epsilon(1e-3));
    }
    THEN("more trials at the same rate tighten it towards the rate") {
        REQUIRE(wilson_lower_bound(80, 100, 1.64) > wilson_lower_bound(8, 10, 1.64));
        REQUIRE(wilson_lower_bound(800, 1000, 1.64) < 0.8);
    }
}

SCENARIO("Expected block success averages the envelope over the uncertainty in dy", "[PlanSave]") {
    const Envelope e = two_bins(1000, 900, 1000, 500);
    REQUIRE(e.valid());
    const double left  = wilson_lower_bound(900, 1000, 1.64);
    const double right = wilson_lower_bound(500, 1000, 1.64);

    THEN("a certain dy reads its own cell") {
        REQUIRE(e.expected_success(-0.5, 0.0, 1.0, 3.0, 1.64, 10) == Approx(left));
        REQUIRE(e.expected_success(0.5, 0.0, 1.0, 3.0, 1.64, 10) == Approx(right));
    }
    THEN("dy on the boundary splits evenly") {
        REQUIRE(e.expected_success(0.0, 0.1, 1.0, 3.0, 1.64, 10) == Approx((left + right) / 2.0).epsilon(1e-6));
    }
    THEN("probability outside the grid counts as a miss") {
        // Centred on the grid's edge: half the mass is off it
        REQUIRE(e.expected_success(1.0, 0.05, 1.0, 3.0, 1.64, 10) == Approx(right / 2.0).epsilon(1e-3));
        REQUIRE(e.expected_success(5.0, 0.1, 1.0, 3.0, 1.64, 10) == 0.0);
    }
    THEN("time and speed beyond the grid use its outer bins") {
        REQUIRE(e.expected_success(-0.5, 0.0, 10.0, 20.0, 1.64, 10) == Approx(left));
    }
    THEN("a thinly measured cell counts for nothing") {
        const Envelope thin = two_bins(5, 5, 1000, 500);
        REQUIRE(thin.expected_success(-0.5, 0.0, 1.0, 3.0, 1.64, 10) == 0.0);
    }
    THEN("the fall rate is read from the cell, when there is one") {
        REQUIRE(e.fall_rate(0.5, 1.0, 3.0, 10).value() == Approx(0.1));
        REQUIRE_FALSE(e.fall_rate(2.0, 1.0, 3.0, 10).has_value());
    }
}

SCENARIO("A malformed envelope is refused", "[PlanSave]") {
    Envelope e = two_bins(10, 5, 10, 5);
    e.trials.pop_back();
    REQUIRE_FALSE(e.valid());

    Envelope unsorted = two_bins(10, 5, 10, 5);
    unsorted.dy_edges = {1.0, 0.0, -1.0};
    REQUIRE_FALSE(unsorted.valid());
}

SCENARIO("SHA-256 ties an envelope to its policy", "[PlanSave]") {
    THEN("it matches the FIPS 180-4 test vectors") {
        Sha256 abc{};
        abc.update("abc");
        REQUIRE(abc.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

        Sha256 empty{};
        REQUIRE(empty.hex() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

        // 56 bytes: the padding spills into a second block
        Sha256 two_blocks{};
        two_blocks.update("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq");
        REQUIRE(two_blocks.hex() == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    }
}
