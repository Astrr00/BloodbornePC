// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/install.h"

#include <cstdio>
#include <fstream>

namespace bb::dump {
namespace {

bool cancelled(const std::atomic<bool>* c) { return c && c->load(); }

uintmax_t tree_bytes(const fs::path& p) {
    uintmax_t n = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec)) n += it->file_size(ec);
    return n;
}

// File-by-file copy so the caller gets progress and can cancel.
bool copy_tree(const fs::path& from, const fs::path& to, const Progress& progress, const Log& log, const std::atomic<bool>* cancel) {
    std::error_code ec;
    std::vector<fs::path> files;
    for (fs::recursive_directory_iterator it(from, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec)) files.push_back(it->path());
    if (ec) {
        log(Severity::Error, "cannot enumerate " + from.string() + ": " + ec.message());
        return false;
    }
    size_t done = 0;
    for (const fs::path& f : files) {
        if (cancelled(cancel)) {
            log(Severity::Error, "Install cancelled.");
            return false;
        }
        const fs::path rel = fs::relative(f, from, ec), dst = to / rel;
        fs::create_directories(dst.parent_path(), ec);
        if (!fs::copy_file(f, dst, fs::copy_options::overwrite_existing, ec)) {
            log(Severity::Error, "copy " + f.string() + " -> " + dst.string() + " failed: " + ec.message());
            return false;
        }
        if (progress) progress(++done, files.size(), rel.generic_string());
    }
    return true;
}

// Manifest paths are relative to the dump root; the install normalizes the layout (copy_layout below), so map each entry
// to where its file was copied. Entries of files that are not installed (license/np data in Sc0/) are dropped.
std::vector<ManifestEntry> map_to_install(const std::vector<ManifestEntry>& manifest, const fs::path& root, const fs::path& to) {
    const Layout l = resolve_layout(root);
    const bool split = l.sce_sys != l.app / "sce_sys";
    std::vector<ManifestEntry> out;
    for (const ManifestEntry& e : manifest) {
        const fs::path src = (root / fs::path(e.path)).lexically_normal();
        const fs::path in_app = src.lexically_relative(l.app.lexically_normal()), in_sys = src.lexically_relative(l.sce_sys.lexically_normal());
        if (!in_app.empty() && *in_app.begin() != "..")
            out.push_back({e.sha256, (to / in_app).generic_string()});
        else if (split && !in_sys.empty() && *in_sys.begin() != ".." &&
                 (in_sys == "param.sfo" || in_sys == "icon0.png" || *in_sys.begin() == "trophy"))
            out.push_back({e.sha256, (to / "sce_sys" / in_sys).generic_string()});
    }
    return out;
}

// Normalize to the console layout: app files at to/, system files at to/sce_sys/. From a PKG-extractor Sc0/ only what the
// port uses is taken (license/np binding data is not needed and not staged).
bool copy_layout(const fs::path& src, const fs::path& to, const Progress& progress, const Log& log, const std::atomic<bool>* cancel) {
    Layout l = resolve_layout(src);
    std::error_code ec;
    if (fs::is_directory(l.app, ec) && !copy_tree(l.app, to, progress, log, cancel)) return false;  // entitlement-only DLC: no app files
    if (l.sce_sys == l.app / "sce_sys") return true;
    fs::create_directories(to / "sce_sys", ec);
    for (const char* f : {"param.sfo", "icon0.png"})
        if (fs::exists(l.sce_sys / f, ec) && !fs::copy_file(l.sce_sys / f, to / "sce_sys" / f, fs::copy_options::overwrite_existing, ec)) {
            log(Severity::Error, "Copying " + (l.sce_sys / f).string() + " failed: " + ec.message());
            return false;
        }
    return !fs::is_directory(l.sce_sys / "trophy", ec) || copy_tree(l.sce_sys / "trophy", to / "sce_sys" / "trophy", progress, log, cancel);
}

} // namespace

Report validate(const Source& s, const Progress& progress) {
    Report r;
    check_structure(s.dump, r);
    if (s.dlc) check_dlc(*s.dlc, r.title_id, r);
    else r.add(Severity::Warning, "No DLC folder given: the add-on entitlement (The Old Hunters, SPEXPANSIONDLC03) will not be "
                                  "reported, so the game will not offer the expansion.");
    if (s.manifest) {
        std::string err;
        auto m = load_manifest(*s.manifest, err);
        if (!err.empty()) r.add(Severity::Error, err);
        else if (r.ok()) check_manifest(s.dump, m, r, progress);
    } else {
        r.add(Severity::Warning, "No hash manifest given: file contents not verified.");
    }
    return r;
}

uintmax_t install_bytes(const Source& s) {
    return tree_bytes(resolve_layout(s.dump).app) + (s.dlc ? tree_bytes(resolve_layout(*s.dlc).app) : 0);
}

std::optional<uintmax_t> free_bytes(fs::path dest) {
    std::error_code ec;
    dest = fs::absolute(dest, ec);
    while (!fs::exists(dest, ec) && dest.has_parent_path() && dest.parent_path() != dest) dest = dest.parent_path();
    const fs::space_info s = fs::space(dest, ec);
    if (ec) return std::nullopt;
    return s.available;
}

bool install(const Source& src, const fs::path& dest, const Progress& progress, const Log& log, const Report* validated,
             const std::atomic<bool>* cancel) {
    const Report r = validated ? *validated : validate(src, progress);
    if (!validated)
        for (auto& f : r.findings) log(f.severity, f.message);
    if (!r.ok()) return false;
    const uintmax_t need = install_bytes(src);
    if (const auto avail = free_bytes(dest); avail && *avail < need) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "Not enough free space for %s: need %.1f GiB, %.1f GiB available.", dest.string().c_str(),
                      double(need) / (1u << 30), double(*avail) / (1u << 30));
        log(Severity::Error, buf);
        return false;
    }
    std::error_code ec;
    fs::create_directories(dest, ec);
    // Present until the copy is complete and verified: a later run (or the user) can tell a half-finished install.
    const fs::path marker = dest / kIncompleteMarker;
    if (fs::exists(marker, ec)) log(Severity::Info, "A previous install into " + dest.string() + " did not finish; copying again.");
    if (!std::ofstream(marker)) {
        log(Severity::Error, "Cannot write to " + dest.string() + " (permissions?).");
        return false;
    }
    log(Severity::Info, "Installing to " + dest.string() + " ...");
    if (!copy_layout(src.dump, dest / "game", progress, log, cancel)) return false;
    if (src.dlc && !copy_layout(*src.dlc, dest / "dlc", progress, log, cancel)) return false;
    if (src.manifest) {  // the source was verified above; verify the installed copy too (disk/copy errors)
        std::string err;
        Report v;
        check_manifest(dest, map_to_install(load_manifest(*src.manifest, err), src.dump, "game"), v, progress);
        for (auto& f : v.findings) log(f.severity == Severity::Error ? Severity::Error : Severity::Info, f.message);
        if (!v.ok()) {
            log(Severity::Error, "Installed copy differs from the dump (disk or copy error) - run the install again.");
            return false;
        }
    }
    fs::create_directories(dest / "mods", ec);  // override folder, mirrors game/ layout
    fs::remove(marker, ec);
    log(Severity::Info, "Done.");
    return true;
}

} // namespace bb::dump
