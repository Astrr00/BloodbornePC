// SPDX-License-Identifier: GPL-3.0-or-later
// All UI text in one table (English). A translation = a second table with the same ids, selected in T().
#pragma once

#define BBL_STRINGS(X)                                                                                                                       \
    X(window_title, "BloodbornePC Launcher")                                                                                                 \
    X(tab_games, "Games")                                                                                                                    \
    X(tab_settings, "Settings")                                                                                                              \
    X(tab_cheats, "Cheats")                                                                                                                  \
    X(play, "Play")                                                                                                                          \
    X(running, "Running...")                                                                                                                 \
    X(stop, "Stop game")                                                                                                                     \
    X(add_game, "Add game...")                                                                                                               \
    X(remove_game, "Remove from list")                                                                                                       \
    X(remove_hint, "Only removes the entry; no game files are deleted.")                                                                     \
    X(no_games, "No game added yet. Click 'Add game...' and choose the folder of your own decrypted dump.")                                   \
    X(no_game_selected, "Add or select a game on the Games tab first.")                                                                      \
    X(dump_path, "Dump")                                                                                                                     \
    X(build_found, "Matching build: ")                                                                                                       \
    X(build_none, "No build for this dump yet.")                                                                                             \
    X(build_none_help, "bbgame is recompiled from one specific eboot.bin, so every dump needs its own build (README, \"Game\"):")            \
    X(build_none_steps, "cmake -S . -B build_game -G Ninja -DCMAKE_BUILD_TYPE=Release -DBB_EBOOT=%s -DBB_GAME_OPT=O2\n"                      \
                        "cmake --build build_game --target bbgen\n"                                                                          \
                        "cmake --build build_game --target bbgame -j 2")                                                                     \
    X(build_none_after, "Then put bbgame next to bblauncher, or add its folder below.")                                                      \
    X(build_dirs, "Build folders (searched for bbgame*)")                                                                                    \
    X(add_build_dir, "Add build folder...")                                                                                                  \
    X(rescan, "Rescan")                                                                                                                      \
    X(scanning, "Scanning builds...")                                                                                                        \
    X(hashing, "Reading eboot.bin...")                                                                                                       \
    X(no_eboot, "eboot.bin cannot be read.")                                                                                                 \
    X(add_title, "Add a game")                                                                                                               \
    X(add_path, "Dump folder")                                                                                                               \
    X(browse, "Browse...")                                                                                                                   \
    X(check, "Check")                                                                                                                        \
    X(checking, "Checking the dump...")                                                                                                      \
    X(pkg_section, "From PKG files")                                                                                                          \
    X(pkg_note, "The launcher does not decrypt PKG files and contains no keys: it runs an external extractor that you trust and configure once.")  \
    X(pkg_game, "Game PKG (.pkg)")                                                                                                            \
    X(pkg_dlc, "DLC PKG (optional)")                                                                                                          \
    X(pkg_update, "Update PKG (optional)")                                                                                                    \
    X(pkg_extractor, "PKG extractor (external tool, configured once)")                                                                        \
    X(pkg_extractor_hint, "The launcher does not decrypt PKGs itself. Enter the command of an extractor you trust, e.g. python C:\\path\\tool.py {pkg} {out}; it must write Image0/ and Sc0/ below {out}.") \
    X(pkg_extract, "Extract and check")                                                                                                       \
    X(pkg_space_note, "Extraction needs the unpacked size once and Install the same again (a game with its update: about 30 GiB + 30 GiB).")  \
    X(pkg_need_game, "Choose the game PKG first.")                                                                                            \
    X(pkg_done, "Extracted. The folders below were filled in; check them and press Install.")                                                \
    X(pkg_work_note, "Extracted PKG files (a temporary copy; installed games do not need it): ")                                              \
    X(pkg_delete, "Delete extracted files")                                                                                                   \
    X(pkg_delete_confirm, "Delete the extracted files of this PKG import? Your PKG files and installed games are not touched.")           \
    X(yes, "Yes")                                                                                                                              \
    X(no, "No")                                                                                                                                \
    X(folders_section, "Or use folders that are already extracted")                                                                           \
    X(use_in_place_pkg, "The extracted files are temporary: use Install.")                                                                    \
    X(dlc_folder, "DLC folder (optional)")                                                                                                   \
    X(dlc_hint, "Add-on (The Old Hunters): the folder with its param.sfo.")                                                                  \
    X(update_folder, "Update 1.09 folder (optional)")                                                                                        \
    X(update_hint, "Base 01.00 + the extracted 1.09 update package (folder with its Sc0 and Image0): merged by Install; nothing is decrypted.")  \
    X(add_hint, "Choose the folder that holds eboot.bin and sce_sys, or the one with Image0 and Sc0 (decrypted dump of your own copy).")     \
    X(version_recognised, "Recognised: ")                                                                                                    \
    X(manifest_used, "Hash manifest: ")                                                                                                      \
    X(manifest_none, "no manifest for this version (contents not verified)")                                                                 \
    X(need_space, "Needs %.1f GiB; %.1f GiB free at %s")                                                                                     \
    X(install, "Install (copy)")                                                                                                             \
    X(install_hint, "Copies the dump into the launcher's data folder and verifies the copy.")                                                \
    X(use_in_place, "Use in place")                                                                                                          \
    X(use_in_place_unverified, "Use in place (unverified)")                                                                                  \
    X(use_in_place_hint, "Keeps the folder where it is; nothing is copied.")                                                                 \
    X(use_in_place_update, "An update has to be merged: use Install.")                                                                      \
    X(installing, "Installing...")                                                                                                           \
    X(cancel, "Cancel")                                                                                                                      \
    X(close, "Close")                                                                                                                        \
    X(install_done, "Installed. The game was added to the list.")                                                                            \
    X(parent_hint, "Hint: this folder looks like Image0; choose its parent folder (the one that also holds Sc0).")                           \
    X(last_run, "Last run")                                                                                                                  \
    X(exit_code, "Exit code")                                                                                                                \
    X(log_tail, "Last lines of the game log:")                                                                                               \
    X(log_file, "Log file: ")                                                                                                                \
    X(reset_defaults, "Reset to defaults")                                                                                                   \
    X(settings_for, "Settings for: ")                                                                                                        \
    X(fps, "Frame rate")                                                                                                                     \
    X(fps_hint, "Presentation rate (BB_FPS). Above 30 the game renders interpolated in-between images as far as the GPU allows.")            \
    X(fps_native, "Native (one image per game frame)")                                                                                       \
    X(fps_refresh, "Monitor refresh rate")                                                                                                   \
    X(fps_unlimited, "Unlimited")                                                                                                            \
    X(window_mode, "Window mode")                                                                                                            \
    X(window_mode_hint, "Fullscreen starts borderless on the current display (BB_FULLSCREEN); F11 toggles in the game.")                     \
    X(windowed, "Windowed")                                                                                                                  \
    X(fullscreen, "Fullscreen")                                                                                                              \
    X(resolution, "Window resolution")                                                                                                       \
    X(resolution_hint, "Window size (BB_WINDOW); the picture scales to it, FSR 1 upscales when it is larger than 1920x1080.")                \
    X(custom, "Custom")                                                                                                                      \
    X(res_scale, "Internal resolution")                                                                                                      \
    X(res_scale_hint, "Edge-length multiplier for screen-sized targets (BB_RES_SCALE); 2x = 3840x2160, costs about 28 ms GPU per frame.")    \
    X(upscaler, "Upscaler")                                                                                                                  \
    X(upscaler_hint, "AMD FSR 1 (EASU + RCAS) or plain bilinear stretching when the window is larger than the game image (BB_UPSCALE).")     \
    X(upscaler_fsr, "FSR 1")                                                                                                                 \
    X(upscaler_linear, "Bilinear")                                                                                                           \
    X(sharpness, "FSR sharpness")                                                                                                            \
    X(sharpness_hint, "Stops below full sharpness (BB_FSR_SHARP): 0 = sharpest, default 0.2, higher = softer.")                              \
    X(interp, "Frame interpolation")                                                                                                         \
    X(interp_hint, "On: in-between images are interpolated; off: images are repeated (BB_INTERP=0). Only matters above 30 fps.")            \
    X(save_dir, "Save folder")                                                                                                               \
    X(save_dir_hint, "Where the game writes its saves (BB_SAVE_DIR). Empty = a folder inside this game's launcher profile.")                 \
    X(stub_imports, "Skip missing imports")                                                                                                  \
    X(stub_imports_hint, "Continue when the game calls a system function that is not implemented yet (BB_STUB_IMPORTS); please report them.") \
    X(cheats_note, "Cheats only work on builds whose memory layout is known (currently the EU 1.00 build); on others nothing is written.")   \
    X(cheats_before_run, "Whether the build supports a cheat is shown here after the first run.")                                            \
    X(cheat_unsupported, "not supported by this build")                                                                                      \
    X(cheat_active, "active in the last run")                                                                                                \
    X(cheat_god, "God mode")                                                                                                                 \
    X(cheat_god_desc, "Health is refilled every frame.")                                                                                     \
    X(cheat_nohit, "No hit")                                                                                                                 \
    X(cheat_nohit_desc, "No damage, no stagger and no fall death.")                                                                          \
    X(cheat_stamina, "Infinite stamina")                                                                                                     \
    X(cheat_stamina_desc, "Actions use no stamina.")                                                                                         \
    X(status_ready, "Ready.")                                                                                                                \
    X(status_no_build, "Cannot play: no matching bbgame build.")                                                                             \
    X(status_error, "Could not start the game: ")

namespace bbl {

enum class S {
#define X(id, text) id,
    BBL_STRINGS(X)
#undef X
};

inline const char* T(S s) {
    static const char* const en[] = {
#define X(id, text) text,
        BBL_STRINGS(X)
#undef X
    };
    return en[int(s)];
}

} // namespace bbl
