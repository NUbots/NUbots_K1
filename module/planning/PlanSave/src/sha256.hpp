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
#ifndef MODULE_PLANNING_PLANSAVE_SHA256_HPP
#define MODULE_PLANNING_PLANSAVE_SHA256_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>

namespace module::planning::save {

    /// SHA-256 (FIPS 180-4), to tie an envelope to the exact policy file it was measured from. The build image has
    /// no OpenSSL headers, and this only ever hashes one file at startup.
    class Sha256 {
    public:
        void update(const unsigned char* data, std::size_t size) {
            length += std::uint64_t(size) * 8;
            while (size > 0) {
                const std::size_t take = std::min(size, block.size() - filled);
                std::memcpy(block.data() + filled, data, take);
                filled += take;
                data += take;
                size -= take;
                if (filled == block.size()) {
                    compress();
                    filled = 0;
                }
            }
        }

        void update(const std::string_view data) {
            update(reinterpret_cast<const unsigned char*>(data.data()), data.size());
        }

        /// Finishes the hash and returns it as lowercase hex
        std::string hex() {
            const std::uint64_t bits = length;
            const unsigned char pad  = 0x80;
            update(&pad, 1);
            const unsigned char zero = 0;
            while (filled != 56) {
                update(&zero, 1);
            }
            for (int i = 7; i >= 0; --i) {
                block[filled++] = static_cast<unsigned char>(bits >> (8 * i));
            }
            compress();

            static constexpr char digits[] = "0123456789abcdef";
            std::string out{};
            for (const std::uint32_t word : h) {
                for (int i = 28; i >= 0; i -= 4) {
                    out.push_back(digits[(word >> i) & 0xF]);
                }
            }
            return out;
        }

    private:
        static constexpr std::array<std::uint32_t, 64> k = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

        static std::uint32_t rotr(const std::uint32_t x, const int n) {
            return (x >> n) | (x << (32 - n));
        }

        void compress() {
            std::array<std::uint32_t, 64> w{};
            for (int i = 0; i < 16; ++i) {
                w[i] = std::uint32_t(block[4 * i]) << 24 | std::uint32_t(block[4 * i + 1]) << 16
                       | std::uint32_t(block[4 * i + 2]) << 8 | std::uint32_t(block[4 * i + 3]);
            }
            for (int i = 16; i < 64; ++i) {
                const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                w[i]                   = w[i - 16] + s0 + w[i - 7] + s1;
            }
            std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
            for (int i = 0; i < 64; ++i) {
                const std::uint32_t t1 =
                    hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
                const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
                hh                     = g;
                g                      = f;
                f                      = e;
                e                      = d + t1;
                d                      = c;
                c                      = b;
                b                      = a;
                a                      = t1 + t2;
            }
            h[0] += a;
            h[1] += b;
            h[2] += c;
            h[3] += d;
            h[4] += e;
            h[5] += f;
            h[6] += g;
            h[7] += hh;
        }

        std::array<std::uint32_t, 8> h = {0x6a09e667,
                                          0xbb67ae85,
                                          0x3c6ef372,
                                          0xa54ff53a,
                                          0x510e527f,
                                          0x9b05688c,
                                          0x1f83d9ab,
                                          0x5be0cd19};
        std::array<unsigned char, 64> block{};
        std::size_t filled   = 0;
        std::uint64_t length = 0;
    };

    /// SHA-256 of a file as lowercase hex, nothing if it cannot be read
    inline std::optional<std::string> sha256_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            return std::nullopt;
        }
        Sha256 hash{};
        std::array<char, 1 << 16> buffer{};
        while (file.read(buffer.data(), buffer.size()) || file.gcount() > 0) {
            hash.update(reinterpret_cast<const unsigned char*>(buffer.data()), std::size_t(file.gcount()));
        }
        return hash.hex();
    }

}  // namespace module::planning::save

#endif  // MODULE_PLANNING_PLANSAVE_SHA256_HPP
