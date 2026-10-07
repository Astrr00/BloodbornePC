// SPDX-License-Identifier: GPL-3.0-or-later
// bbinstall: validate a user-supplied Bloodborne dump and install it into the user data directory.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "core/dump.h"

namespace fs = std::filesystem;
using namespace bb::dump;

namespace {

constexpr const char* kUsage = R"(usage:
  bbinstall validate <dump-dir> [--manifest <file.sha256>] [--dlc <dlc-dir>]
  bbinstall hash     <dir> -o <file.sha256>
  bbinstall install  <dump-dir> [--manifest <file.sha256>] [--dlc <dlc-dir>] [--dest <dir>]

<dump-dir> is the decrypted, extracted application folder (eboot.bin, sce_sys/, dvdroot_ps4/)
with update 1.09 merged in. Manifests use `sha256sum` format; see manifests/README.md.
Exit codes: 0 ok, 1 validation failed, 2 usage error.
)";

void print_progress(size_t done, size_t total, const std::string& path) {
    std::fprintf(stderr, "\r[%zu/%zu] %-60.60s", done, total, path.c_str());
    if (done == total) std::fputc('\n', stderr);
}

bool print_report(const Report& r) {
    for (auto& f : r.findings) {
        const char* tag = f.severity == Severity::Error ? "ERROR" : f.severity == Severity::Warning ? "WARN " : "info ";
        std::printf("%s %s\n", tag, f.message.c_str());
    }
    std::printf(r.ok() ? "Dump OK.\n" : "Dump INVALID - see errors above.\n");
    return r.ok();
}

struct Args {
    std::string command;
    fs::path input;
    std::optional<fs::path> manifest, dlc, dest, out;
};

std::optional<Args> parse_args(int argc, char** argv) {
    if (argc < 3) return std::nullopt;
    Args a{argv[1], fs::path(argv[2])};
    for (int i = 3; i < argc; ++i) {
        std::string k = argv[i];
        if (i + 1 >= argc) return std::nullopt;
        fs::path v = argv[++i];
        if (k == "--manifest") a.manifest = v;
        else if (k == "--dlc") a.dlc = v;
        else if (k == "--dest") a.dest = v;
        else if (k == "-o") a.out = v;
        else return std::nullopt;
    }
    return a;
}

Report validate(const Args& a) {
    Report r;
    check_structure(a.input, r);
    if (a.dlc) check_dlc(*a.dlc, r.title_id, r);
    else r.add(Severity::Warning, "No DLC folder given (--dlc): the add-on entitlement (The Old Hunters, SPEXPANSIONDLC03) will not be "
                                  "reported, so the game will not offer the expansion.");
    if (a.manifest) {
        std::string err;
        auto m = load_manifest(*a.manifest, err);
        if (!err.empty()) r.add(Severity::Error, err);
        else if (r.ok()) check_manifest(a.input, m, r, print_progress);
    } else {
        r.add(Severity::Warning, "No hash manifest given (--manifest): file contents not verified.");
    }
    return r;
}

bool copy_tree(const fs::path& from, const fs::path& to) {
    std::error_code ec;
    fs::create_directories(to, ec);
    fs::copy(from, to, fs::copy_options::recursive | fs::copy_options::overwrite_existing, ec);
    if (ec) std::fprintf(stderr, "copy %s -> %s failed: %s\n", from.string().c_str(), to.string().c_str(),
                         ec.message().c_str());
    return !ec;
}

uintmax_t tree_bytes(const fs::path& p) {
    uintmax_t n = 0;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(p, ec), end; !ec && it != end; it.increment(ec))
        if (it->is_regular_file(ec)) n += it->file_size(ec);
    return n;
}

// Free space on the volume that will hold `dest` (checked at its closest existing ancestor).
std::optional<uintmax_t> free_bytes(fs::path dest) {
    std::error_code ec;
    dest = fs::absolute(dest, ec);
    while (!fs::exists(dest, ec) && dest.has_parent_path() && dest.parent_path() != dest) dest = dest.parent_path();
    const fs::space_info s = fs::space(dest, ec);
    if (ec) return std::nullopt;
    return s.available;
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

} // namespace

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (!args) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    if (args->command == "validate") return print_report(validate(*args)) ? 0 : 1;

    if (args->command == "hash") {
        if (!args->out || !fs::is_directory(args->input)) {
            std::fputs(kUsage, stderr);
            return 2;
        }
        std::string error;
        auto entries = hash_tree(args->input, print_progress, error);
        if (!error.empty()) {
            std::fprintf(stderr, "error: %s\n", error.c_str());
            return 1;
        }
        if (!write_manifest(*args->out, entries)) {
            std::fprintf(stderr, "cannot write %s\n", args->out->string().c_str());
            return 1;
        }
        std::printf("wrote %zu entries to %s\n", entries.size(), args->out->string().c_str());
        return 0;
    }

    if (args->command == "install") {
        if (!print_report(validate(*args))) return 1;
        fs::path dest = args->dest.value_or(default_install_dir());
        const uintmax_t need = tree_bytes(resolve_layout(args->input).app) + (args->dlc ? tree_bytes(resolve_layout(*args->dlc).app) : 0);
        if (const auto avail = free_bytes(dest); avail && *avail < need) {
            std::printf("ERROR Not enough free space for %s: need %.1f GiB, %.1f GiB available.\n", dest.string().c_str(),
                        double(need) / (1u << 30), double(*avail) / (1u << 30));
            return 1;
        }
        std::error_code ec;
        fs::create_directories(dest, ec);
        // Present until the copy is complete and verified: a later run (or the user) can tell a half-finished install.
        const fs::path marker = dest / ".bbinstall-incomplete";
        if (fs::exists(marker, ec)) std::printf("info  A previous install into %s did not finish; copying again.\n", dest.string().c_str());
        if (!std::ofstream(marker)) {
            std::printf("ERROR Cannot write to %s (permissions?).\n", dest.string().c_str());
            return 1;
        }
        std::printf("Installing to %s ...\n", dest.string().c_str());
        // Normalize to the console layout: app files at game/, system files at game/sce_sys/. From a PKG-extractor
        // Sc0/ only what the port uses is taken (license/np binding data is not needed and not staged).
        auto copy_layout = [](const fs::path& src, const fs::path& to) {
            Layout l = resolve_layout(src);
            std::error_code ec;
            if (fs::is_directory(l.app, ec) && !copy_tree(l.app, to)) return false;  // entitlement-only DLC: no app files
            if (l.sce_sys == l.app / "sce_sys") return true;
            fs::create_directories(to / "sce_sys", ec);
            for (const char* f : {"param.sfo", "icon0.png"})
                if (fs::exists(l.sce_sys / f, ec) &&
                    !fs::copy_file(l.sce_sys / f, to / "sce_sys" / f, fs::copy_options::overwrite_existing, ec)) {
                    std::printf("ERROR Copying %s failed: %s\n", (l.sce_sys / f).string().c_str(), ec.message().c_str());
                    return false;
                }
            return !fs::is_directory(l.sce_sys / "trophy", ec) || copy_tree(l.sce_sys / "trophy", to / "sce_sys" / "trophy");
        };
        if (!copy_layout(args->input, dest / "game")) return 1;
        if (args->dlc && !copy_layout(*args->dlc, dest / "dlc")) return 1;
        if (args->manifest) {  // the source was verified above; verify the installed copy too (disk/copy errors)
            std::string err;
            Report r;
            check_manifest(dest, map_to_install(load_manifest(*args->manifest, err), args->input, "game"), r, print_progress);
            for (auto& f : r.findings)
                std::printf("%s %s\n", f.severity == Severity::Error ? "ERROR" : "info ", f.message.c_str());
            if (!r.ok()) {
                std::printf("Installed copy differs from the dump (disk or copy error) - run the install again.\n");
                return 1;
            }
        }
        fs::create_directories(dest / "mods", ec);  // override folder, mirrors game/ layout
        fs::remove(marker, ec);
        std::printf("Done.\n");
        return 0;
    }

    std::fputs(kUsage, stderr);
    return 2;
}
