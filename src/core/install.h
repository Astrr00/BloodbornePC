// SPDX-License-Identifier: GPL-3.0-or-later
// Validate and install a dump (shared by bbinstall and the launcher).
#pragma once
#include <atomic>
#include <optional>

#include "core/dump.h"

namespace bb::dump {

struct Source {
    fs::path dump;
    std::optional<fs::path> manifest, dlc;
};

// Structure checks + DLC + manifest (if given). A manifest that cannot be read is an error; a missing one a warning.
Report validate(const Source& src, const Progress& progress);

// Bytes of the application files that install() copies.
uintmax_t install_bytes(const Source& src);
// Free space on the volume that will hold `dest` (checked at its closest existing ancestor).
std::optional<uintmax_t> free_bytes(fs::path dest);

inline constexpr const char* kIncompleteMarker = ".bbinstall-incomplete";

using Log = std::function<void(Severity, const std::string&)>;
// Validates, then copies to dest/game (and dest/dlc) and verifies the copy. `cancel` is polled between files.
// `validated` (optional) skips the second validation when the caller already ran validate(). Returns false on any error (logged).
bool install(const Source& src, const fs::path& dest, const Progress& progress, const Log& log, const Report* validated = nullptr,
             const std::atomic<bool>* cancel = nullptr);

} // namespace bb::dump
