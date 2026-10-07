// SPDX-License-Identifier: GPL-3.0-or-later
// Self-contained SHA-1 (NID generation) and SHA-256 (dump validation).
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace bb {

class Sha256 {
public:
    using Digest = std::array<uint8_t, 32>;
    Sha256();
    void update(std::span<const uint8_t> data);
    Digest finish();

private:
    void block(const uint8_t* p);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t buf_len_ = 0;
    uint64_t total_ = 0;
};

class Sha1 {
public:
    using Digest = std::array<uint8_t, 20>;
    Sha1();
    void update(std::span<const uint8_t> data);
    Digest finish();

private:
    void block(const uint8_t* p);
    uint32_t h_[5];
    uint8_t buf_[64];
    size_t buf_len_ = 0;
    uint64_t total_ = 0;
};

std::string to_hex(std::span<const uint8_t> bytes);

// Streams the file; nullopt if it cannot be read.
std::optional<std::string> sha256_file_hex(const std::filesystem::path& path);

} // namespace bb
