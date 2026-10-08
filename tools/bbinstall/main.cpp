// SPDX-License-Identifier: GPL-3.0-or-later
// bbinstall: validate a user-supplied Bloodborne dump and install it into the user data directory.
#include <cstdio>
#include <optional>
#include <string>

#include "core/install.h"

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
    Args a{argv[1], fs::path(argv[2]), {}, {}, {}, {}};
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

Source source(const Args& a) { return {a.input, a.manifest, a.dlc}; }

} // namespace

int main(int argc, char** argv) {
    auto args = parse_args(argc, argv);
    if (!args) {
        std::fputs(kUsage, stderr);
        return 2;
    }
    if (args->command == "validate") return print_report(validate(source(*args), print_progress)) ? 0 : 1;

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
        const fs::path dest = args->dest.value_or(default_install_dir());
        const Report r = validate(source(*args), print_progress);
        if (!print_report(r)) return 1;
        const auto log = [](Severity s, const std::string& m) {
            std::printf("%s %s\n", s == Severity::Error ? "ERROR" : s == Severity::Warning ? "WARN " : "info ", m.c_str());
        };
        if (!install(source(*args), dest, print_progress, log, &r)) return 1;
        return 0;
    }

    std::fputs(kUsage, stderr);
    return 2;
}
