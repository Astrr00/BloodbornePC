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
    std::optional<fs::path> update;  // 01.00 base + update 01.09 folder, merged virtually (validate) or on disk (install)
};

// What merging `update` onto `dump` yields (application files only).
struct MergeStats {
    size_t base_only = 0, from_update = 0, overridden = 0;  // files taken from the base, from the update (incl. overridden ones), base files replaced
    uintmax_t bytes = 0;                                     // size of the merged file set
};
MergeStats merge_stats(const fs::path& dump, const fs::path& update);

// Structure checks + DLC + update + manifest (if given; with an update the manifest is checked against the virtual merge, nothing is copied).
// A manifest that cannot be read is an error; a missing one a warning.
Report validate(const Source& src, const Progress& progress);

// Bytes of the application files that install() copies (the merged set with an update).
uintmax_t install_bytes(const Source& src);
// Free space on the volume that will hold `dest` (checked at its closest existing ancestor).
std::optional<uintmax_t> free_bytes(fs::path dest);

inline constexpr const char* kIncompleteMarker = ".bbinstall-incomplete";

using Log = std::function<void(Severity, const std::string&)>;
// Validates, then copies to dest/game (base, then the update overlaid) and dest/dlc and verifies the copy. `cancel` is polled between files.
// `validated` (optional) skips the second validation when the caller already ran validate(). Returns false on any error (logged).
bool install(const Source& src, const fs::path& dest, const Progress& progress, const Log& log, const Report* validated = nullptr,
             const std::atomic<bool>* cancel = nullptr);

} // namespace bb::dump
