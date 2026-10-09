// SPDX-License-Identifier: GPL-3.0-or-later
// PKG import without decryption: the launcher only reads the plain PKG header and runs an external extractor the user configured.
// Nothing here knows keys or the PFS format; extracted folders (Image0/ + Sc0/) go through the normal validate/install flow.
#pragma once
#include <atomic>
#include <string>
#include <string_view>
#include <vector>

#include "core.h"
#include "core/dump.h"

namespace bbl {

// ---- extractor command template -------------------------------------------------------------------------------------------------------
// "python \"C:\\my tools\\x.py\" {pkg} {out}" -> argv list. Split on spaces, "..." groups (no escapes), no shell: the result is passed to
// CreateProcess/execv as separate arguments, so $, &, |, %, ; etc. have no meaning. Empty `err` on success.
std::vector<std::string> split_command(std::string_view cmd, std::string& err);
// Every argument containing {pkg} / {out} gets them replaced (the argument stays one argument, whatever the paths contain).
std::vector<std::string> substitute_args(const std::vector<std::string>& tmpl, const std::string& pkg, const std::string& out);
// The template must name a program and use both placeholders; returns "" or the reason.
std::string check_template(const std::vector<std::string>& tmpl);
// `"path"` (quoted when it has spaces) for building a template from a browsed file; ".py" gets "python " in front.
std::string template_for_program(const std::string& path);

// ---- plain PKG header (offset 0x00 magic 7F 'C' 'N' 'T', 0x40 content id, 36 ASCII bytes) -------------------------------------------------
struct PkgInfo {
    std::string content_id;  // e.g. EP9000-CUSA03173_00-BLOODBORNE0000EU
    std::string title_id;    // CUSA03173 (chars 7..15 of the content id) if the id has the usual shape, else empty
    uint64_t size = 0;
};
bool read_pkg_header(const fs::path& file, PkgInfo& out, std::string& err);
// Cross-checks the chosen packages; only what the header exposes (title id; the same file twice). `dlc` / `update` may be null.
// An unknown game title id (not in bb::dump::kKnownTitleIds) is an error here, before tens of GiB are extracted.
std::vector<bb::dump::Finding> check_pkg_set(const PkgInfo& game, const fs::path& game_path, const PkgInfo* dlc, const fs::path* dlc_path,
                                             const PkgInfo* update, const fs::path* update_path);

// ---- Add dialog state rules (kept free of UI so they can be tested) ----------------------------------------------------------------------
enum class AddPhase { Checking, Checked, Installing, Installed, Failed };
// A new check (Check button, a picked folder, a finished extraction) may replace the dialog's state only when no job runs and no finished
// install is still waiting to be registered as a game; otherwise it has to wait (Defer).
enum class CheckAction { Start, Defer };
CheckAction check_request(AddPhase phase, bool install_registered);

// ---- work directory owned by the launcher: <data_dir>/pkg_work/<run id> -----------------------------------------------------------------
enum class PkgKind { Game, Dlc, Update };
inline constexpr const char* kWorkMarker = ".bbl-pkgwork";
std::string new_run_id();
// Creates a FRESH directory (refuses an existing one) and a marker file in it; empty path + `err` on failure.
fs::path make_work_dir(const fs::path& data_dir, const std::string& run_id, std::string& err);
// Removes `dir` only if it is a direct child of <data_dir>/pkg_work that carries the marker this launcher wrote. The extractor logs live
// OUTSIDE the work folder (<run>-<kind>.log next to it) so they survive a failed run; they are removed too unless `keep_logs`.
bool remove_work_dir(const fs::path& dir, const fs::path& data_dir, std::string& err, bool keep_logs = false);
const char* kind_name(PkgKind kind);
fs::path work_log(const fs::path& work_dir, PkgKind kind);
// Every direct child of <data_dir>/pkg_work that carries the marker (also those left by a closed launcher), sorted by name.
std::vector<fs::path> list_work_dirs(const fs::path& data_dir);
// "" if data_dir's volume has room for the estimate (unpacked size ~ PKG size, needed once for the extraction and once more for the install).
std::string check_extract_space(uintmax_t pkg_bytes, const fs::path& data_dir);

// ---- running the extractor ---------------------------------------------------------------------------------------------------------------
struct ExtractResult {
    bool ok = false;
    int exit_code = 0;
    std::string error;  // understandable text when !ok
};
// Runs `tmpl` with {pkg}/{out} for one package (the program is looked up next to the launcher and on PATH, never in the current directory; the
// 64-bit PE gate used for bbgame does not apply to it); output (stdout+stderr) goes to `log_file`; `out` must not exist yet. `cancel` kills the
// child and its children. Afterwards `out` must hold Sc0/param.sfo (and Image0/ for a game or update).
ExtractResult run_extractor(const std::vector<std::string>& tmpl, const fs::path& pkg, const fs::path& out, PkgKind kind, const fs::path& log_file,
                            const std::atomic<bool>* cancel);

} // namespace bbl
