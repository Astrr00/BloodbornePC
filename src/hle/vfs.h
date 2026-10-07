// SPDX-License-Identifier: GPL-3.0-or-later
// Virtual file system: guest paths (/app0/..., /savedata0/..., /temp0/...) map to host directories.
#pragma once
#include <filesystem>
#include <optional>
#include <string>

namespace bb::hle {

// Mounts a guest prefix ("/app0") on a host directory. Later mounts of the same prefix replace earlier ones.
void vfs_mount(const std::string& guest_prefix, const std::filesystem::path& host_dir);

void vfs_unmount(const std::string& guest_prefix);

// Host path for a guest path, or nullopt if no mount matches or the path escapes its mount ("..").
std::optional<std::filesystem::path> vfs_resolve(const std::string& guest_path);

} // namespace bb::hle
