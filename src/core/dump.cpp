// SPDX-License-Identifier: GPL-3.0-or-later
#include "core/dump.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <thread>

#include "core/hash.h"
#include "core/orbis_elf.h"
#include "core/sfo.h"
#include "core/util.h"

namespace bb::dump {
namespace {

constexpr uint8_t kPkgMagic[4] = {0x7F, 'C', 'N', 'T'};

// Hashes files in parallel; result[i] is nullopt if file i could not be read.
std::vector<std::optional<std::string>> hash_files(const std::vector<fs::path>& files,
                                                   const std::vector<std::string>& names, const Progress& progress) {
    std::vector<std::optional<std::string>> out(files.size());
    std::atomic<size_t> next{0}, done{0};
    std::mutex progress_mutex;
    auto worker = [&] {
        for (size_t i; (i = next++) < files.size();) {
            out[i] = sha256_file_hex(files[i]);
            size_t n = ++done;
            if (progress) {
                std::lock_guard lock(progress_mutex);
                progress(n, files.size(), names[i]);
            }
        }
    };
    // ponytail: one file per task; a single huge archive bounds wall time, chunked hashing can't help SHA-256 anyway.
    unsigned n = std::max(1u, std::min<unsigned>(std::thread::hardware_concurrency(), unsigned(files.size())));
    std::vector<std::jthread> threads;
    for (unsigned t = 0; t < n; ++t) threads.emplace_back(worker);
    threads.clear();  // join before `out` is returned (moved without NRVO)
    return out;
}

bool is_pkg_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    char m[4] = {};
    return f.read(m, 4) && std::memcmp(m, kPkgMagic, 4) == 0;
}

} // namespace

bool Report::ok() const {
    return std::none_of(findings.begin(), findings.end(), [](auto& f) { return f.severity == Severity::Error; });
}

std::vector<ManifestEntry> load_manifest(const fs::path& file, std::string& error) {
    std::vector<ManifestEntry> entries;
    std::ifstream f(file);
    if (!f) {
        error = "cannot open manifest " + file.string();
        return {};
    }
    std::string line;
    for (size_t no = 1; std::getline(f, line); ++no) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        if (line.size() < 67 || line[64] != ' ' || (line[65] != ' ' && line[65] != '*')) {
            error = file.string() + ":" + std::to_string(no) + ": expected '<sha256>  <path>'";
            return {};
        }
        std::string hex = line.substr(0, 64);
        std::transform(hex.begin(), hex.end(), hex.begin(), [](unsigned char c) { return char(std::tolower(c)); });
        if (hex.find_first_not_of("0123456789abcdef") != std::string::npos) {
            error = file.string() + ":" + std::to_string(no) + ": invalid hex digest";
            return {};
        }
        entries.push_back({hex, line.substr(66)});
    }
    return entries;
}

bool write_manifest(const fs::path& file, const std::vector<ManifestEntry>& entries) {
    std::ofstream f(file, std::ios::binary);
    for (auto& e : entries) f << e.sha256 << "  " << e.path << '\n';
    return bool(f);
}

std::vector<ManifestEntry> hash_tree(const fs::path& root, const Progress& progress, std::string& error) {
    std::vector<std::string> names;
    std::error_code ec;
    for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code type_ec;
        if (it->is_regular_file(type_ec)) names.push_back(fs::relative(it->path(), root).generic_string());
    }
    if (ec) {
        error = "cannot enumerate " + root.string() + ": " + ec.message();
        return {};
    }
    std::sort(names.begin(), names.end());
    std::vector<fs::path> files;
    for (auto& n : names) files.push_back(root / fs::path(n));
    auto hashes = hash_files(files, names, progress);
    std::vector<ManifestEntry> out;
    for (size_t i = 0; i < names.size(); ++i) {
        if (!hashes[i]) {
            error = "cannot read " + names[i];
            return {};
        }
        out.push_back({*hashes[i], names[i]});
    }
    return out;
}

Layout resolve_layout(const fs::path& root) {
    std::error_code ec;
    // Image0/ may be absent: an entitlement-only DLC package (the GOTY data already carries the content) has just Sc0/
    if (fs::is_regular_file(root / "Sc0" / "param.sfo", ec))
        return {root / "Image0", root / "Sc0"};
    return {root, root / "sce_sys"};
}

void check_structure(const fs::path& game, Report& r, bool base_for_update) {
    if (fs::is_regular_file(game) && is_pkg_file(game)) {
        r.add(Severity::Error,
              "Input is a PKG file. PKG contents are PFS-encrypted and this project ships no keys and performs no "
              "decryption. Extract the PKG of your own copy with an external tool, then pass the extracted folder.");
        return;
    }
    if (!fs::is_directory(game)) {
        r.add(Severity::Error, "Dump folder not found: " + game.string());
        return;
    }
    const Layout L = resolve_layout(game);
    if (L.app != game) r.add(Severity::Info, "PKG-extractor layout detected (Image0/ + Sc0/).");
    if (!fs::is_directory(L.app)) {
        r.add(Severity::Error, "Image0/ missing next to Sc0/ - the application files were not extracted.");
        return;
    }

    auto sfo = load_sfo(L.sce_sys / "param.sfo");
    if (!sfo) {
        r.add(Severity::Error, "sce_sys/param.sfo missing or unreadable - not a PS4 application folder.");
    } else {
        r.title_id = sfo->str("TITLE_ID").value_or("");
        r.app_ver = sfo->str("APP_VER").value_or("");
        std::string category = sfo->str("CATEGORY").value_or("");
        bool known = std::any_of(std::begin(kKnownTitleIds), std::end(kKnownTitleIds),
                                 [&](const char* id) { return r.title_id == id; });
        if (!known)
            r.add(Severity::Error, "TITLE_ID '" + r.title_id + "' is not a known Bloodborne release.");
        // A merged base+update folder carries the update's param.sfo (CATEGORY gp). Whether the base data is
        // present is decided by the data-dir check and the hash manifest, not by the category.
        if (category != "gd" && category != "gp")
            r.add(Severity::Error, "Unexpected CATEGORY '" + category + "' (expected 'gd' or 'gp' - game or merged "
                                   "update).");
        else if (category == "gp")
            r.add(Severity::Info, "param.sfo is from the update (CATEGORY gp); base game files must be merged in.");
        r.add(Severity::Info, "param.sfo: " + r.title_id + " v" + r.app_ver + " (" + category + ")");
        if (base_for_update && r.app_ver == "01.00")
            r.add(Severity::Info, "base is version 01.00: the update will be merged onto it.");
        else if (base_for_update && r.app_ver == kRequiredAppVer)
            r.add(Severity::Warning, std::string("this folder is already version ") + kRequiredAppVer + ": the update is not needed (and would only overwrite files).");
        else if (base_for_update)
            r.add(Severity::Error, "Base version is " + (r.app_ver.empty() ? "unknown" : r.app_ver) + ", the update " + kRequiredAppVer + " is merged onto 01.00.");
        else if (r.app_ver != kRequiredAppVer)
            r.add(Severity::Error, "Game version is " + (r.app_ver.empty() ? "unknown" : r.app_ver) +
                                       ", required " + kRequiredAppVer + ". Apply update 1.09 to the dump.");
    }

    fs::path eboot = L.app / "eboot.bin";
    if (auto data = read_file(eboot); !data) {
        r.add(Severity::Error, "eboot.bin missing or unreadable.");
    } else {
        auto img = elf::parse(*data);
        if (auto* e = std::get_if<elf::Error>(&img))
            r.add(Severity::Error, "eboot.bin: " + e->message);
        else
            r.add(Severity::Info, std::string("eboot.bin: ") + (std::get<elf::Image>(img).from_self ? "fake-signed SELF" : "ELF") +
                                      ", " + elf::elf_type_name(std::get<elf::Image>(img).elf_type));
    }

    // Confirmed on a real 1.09 EU dump: game data is loose files under dvdroot_ps4/ (chr/, map/, param/, ...).
    if (!fs::is_directory(L.app / "dvdroot_ps4"))
        r.add(Severity::Error, "dvdroot_ps4/ missing - game data incomplete.");
    if (!fs::is_directory(L.app / "sce_module"))
        r.add(Severity::Info, "sce_module/ absent (fine: system modules are replaced by HLE).");
}

void check_dlc(const fs::path& dlc, const std::string& game_title_id, Report& r) {
    auto sfo = load_sfo(resolve_layout(dlc).sce_sys / "param.sfo");
    if (!sfo) {
        r.add(Severity::Error, "DLC: sce_sys/param.sfo missing in " + dlc.string());
        return;
    }
    if (sfo->str("CATEGORY").value_or("") != "ac")
        r.add(Severity::Error, "DLC: CATEGORY is not 'ac' (additional content).");
    std::string tid = sfo->str("TITLE_ID").value_or("");
    // Unverified whether the DLC shares the game's TITLE_ID, so a mismatch only warns.
    if (!game_title_id.empty() && tid != game_title_id)
        r.add(Severity::Warning, "DLC: TITLE_ID " + tid + " differs from game " + game_title_id + " (region mismatch?).");
    r.add(Severity::Info, "DLC: " + sfo->str("CONTENT_ID").value_or(tid));
}

void check_manifest(const fs::path& root, const std::vector<ManifestEntry>& manifest, Report& r,
                    const Progress& progress) {
    check_manifest(manifest, [&](const ManifestEntry& e) { return root / fs::path(e.path); }, r, progress);
}

void check_update(const fs::path& update, const std::string& game_title_id, Report& r) {
    std::error_code ec;
    if (fs::is_regular_file(update, ec) && is_pkg_file(update)) {
        r.add(Severity::Error, "Update is a PKG file. Extract the update package with an external tool first (this project performs no decryption), then pass the extracted folder.");
        return;
    }
    if (!fs::is_directory(update, ec)) {
        r.add(Severity::Error, "Update folder not found: " + update.string());
        return;
    }
    auto sfo = load_sfo(resolve_layout(update).sce_sys / "param.sfo");
    if (!sfo) {
        r.add(Severity::Error, "Update folder has no param.sfo (expected Sc0/param.sfo or sce_sys/param.sfo) or is empty: " + update.string());
        return;
    }
    const std::string tid = sfo->str("TITLE_ID").value_or(""), ver = sfo->str("APP_VER").value_or(""), cat = sfo->str("CATEGORY").value_or("");
    if (!game_title_id.empty() && tid != game_title_id) {
        const bool known = std::any_of(std::begin(kKnownTitleIds), std::end(kKnownTitleIds), [&](const char* id) { return tid == id; });
        r.add(Severity::Error, "Update folder is not for this game (title " + tid + " vs " + game_title_id + ")." +
                                   (known ? " It belongs to another Bloodborne release; the update must match the base release." : ""));
    }
    if (ver != kRequiredAppVer)
        r.add(Severity::Error, "Update is version " + (ver.empty() ? "unknown" : ver) + ", expected " + kRequiredAppVer + ".");
    if (cat != "gp")
        r.add(Severity::Warning, "Update CATEGORY is '" + cat + "' (an update package has 'gp').");
    if (!fs::is_directory(resolve_layout(update).app, ec))
        r.add(Severity::Error, "Update folder has no application files (Image0/ or dvdroot_ps4/).");
    r.add(Severity::Info, "update: " + tid + " v" + ver + " (" + cat + ")");
}

void check_manifest(const std::vector<ManifestEntry>& manifest, const Resolver& resolve, Report& r, const Progress& progress) {
    std::vector<fs::path> files;
    std::vector<std::string> names;
    std::vector<const ManifestEntry*> present;
    for (auto& e : manifest) {
        fs::path p = resolve(e);
        std::error_code ec;
        if (!fs::is_regular_file(p, ec)) {
            r.add(Severity::Error, "missing: " + e.path);
            continue;
        }
        files.push_back(p);
        names.push_back(e.path);
        present.push_back(&e);
    }
    auto hashes = hash_files(files, names, progress);
    size_t bad = 0;
    for (size_t i = 0; i < present.size(); ++i) {
        if (!hashes[i])
            r.add(Severity::Error, "unreadable: " + present[i]->path), ++bad;
        else if (*hashes[i] != present[i]->sha256)
            r.add(Severity::Error, "hash mismatch (wrong version/region or corrupt): " + present[i]->path), ++bad;
    }
    r.add(Severity::Info, "manifest: " + std::to_string(present.size() - bad) + "/" + std::to_string(manifest.size()) +
                              " files verified");
}

fs::path default_install_dir() {
#ifdef _WIN32
    if (const char* appdata = std::getenv("APPDATA")) return fs::path(appdata) / "BloodbornePC";
#else
    if (const char* xdg = std::getenv("XDG_DATA_HOME"); xdg && *xdg) return fs::path(xdg) / "BloodbornePC";
    if (const char* home = std::getenv("HOME")) return fs::path(home) / ".local/share/BloodbornePC";
#endif
    return fs::current_path() / "BloodbornePC";
}

} // namespace bb::dump
