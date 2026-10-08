# BloodbornePC

Native, open-source PC port of Bloodborne (PS4, v1.09 + The Old Hunters) for Windows and Linux. The project statically
recompiles `eboot.bin` to C++ and replaces the PS4 system libraries with its own implementations. It is not an emulator.

**This repository contains no game content, no original code, no keys, and decrypts nothing.**
You need a decrypted folder dump of your own copy, made by yourself.

Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Detailed, dated evidence for everything below: [docs/ROADMAP.md](docs/ROADMAP.md)
(written in German; it lists what has been shown to work and what has not).

## Progress

**Overall: roughly 60 % (estimate, equal weight per row). Not finished, not yet playable to the end.** The evidence behind every row is in
[docs/ROADMAP.md](docs/ROADMAP.md).

| Area | Progress | Basis |
|---|---|---|
| Installer (hash check, copy, error messages) | 85 % | works; a build in a fresh directory has not been run yet |
| Windows 11 (Windows 10 untested) | 85 % | runs natively, saves, audio and movies work; a few rendering oddities and untranslated shaders remain |
| Linux / Steam Deck | 10 % | libraries and tests build in CI; the game has not been run with a window on Linux |
| Base game, playthrough | 70 % | 7 of 10 required bosses defeated (list below); the ending is untested |
| The Old Hunters (DLC) | 0 % | untested |
| Frame rate (30 Hz simulation, up to 165 Hz output) | 85 % | interpolated output measured at 165 Hz; not every rate checked in every area |
| Resolution up to 4K, post effects | 90 % | internal 4K and FSR 1 upscaling work |

Tested with the EU 1.00 `eboot.bin` on one machine (Windows 11, RTX 3070 Ti, 165 Hz monitor); the 1.09 build only reaches the clinic so far.
Story progress was reached by scripted play with test aids that are not part of normal builds, and some stretches were bridged by
teleporting, so "reached" means the game logic, loading, rendering and cutscenes worked there, not that a player can do it unaided.

Required bosses counted (shortest route to an ending; optional bosses such as the Witch of Hemwick are not counted): defeated – Scourge Beast,
Cleric Beast, Father Gascoigne, Blood-starved Beast, Vicar Amelia, Shadows of Yharnam (counted because the way to Byrgenwerth leads through
its arena), Rom; not reached – Mergo's Wet Nurse, Micolash, the final boss. The list is the maintainers' own count.

## Build

Requirements: CMake ≥ 3.24, Ninja, a C++20 compiler (Clang recommended, MSVC 2022 tested), Python 3 for the tests.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

CI (GitHub Actions: Windows `cl` and `clang-cl`, Ubuntu `clang`) runs exactly this: it builds and tests the recompiler, tools, HLE/GPU
libraries and unit tests. It does not build the game (`bbgame`), which needs your own dump.

### Game (Windows, needs your own dump)

1. Check the dump (a hash manifest for EU 1.09 is included; `install` additionally copies the checked dump to
   `%APPDATA%\BloodbornePC` and verifies the copy):
   ```
   build\bbinstall validate <dump folder> --manifest manifests\CUSA03173-01.09.sha256 [--dlc <DLC folder>]
   ```
2. Recompile and build (Visual Studio 2022 Build Tools, "x64 Native Tools" command prompt, Ninja):
   ```
   cmake -S . -B build_game -G Ninja -DCMAKE_BUILD_TYPE=Release -DBB_EBOOT=<path\to\eboot.bin> -DBB_GAME_OPT=O2
   cmake --build build_game --target bbgen
   cmake --build build_game --target bbgame -j 2
   ```
   `bbgen` generates the shard list; the next build re-configures with it (this two-step order follows `CMakeLists.txt` and has not
   yet been run in a fresh build directory). The first `bbgame` build compiles ≈ 90 generated shards: ≈ 3.5 CPU hours (measured:
   73 min with `-j 4` on 8 cores / 16 threads; expect roughly twice that with `-j 2`), up to 5.6 GB RAM per compiler process. The
   generated sources stay local (never commit them). An NID name list is optional (`-DBB_NAMES=<file>`, e.g. from ps4libdoc; none is
   bundled); it only adds import names to logs and stub reports.
3. Run from the repository folder, with the same `eboot.bin` as for the build:
   ```
   set BB_FPS=refresh
   build_game\bbgame.exe <path\to\eboot.bin>
   ```
   Controller via SDL3; keyboard: arrows = D-pad, Enter/Space = Cross, Backspace/Esc = Circle, WASD/IJKL = sticks, F11 = fullscreen.
   Saves are written to `savedata\` (`BB_SAVE_DIR`). If the game aborts with `unimplemented import`, `BB_STUB_IMPORTS=1` skips such
   imports (please report them).

- **Windows:** MSVC 2022 (reference). **Linux:** GCC/Clang with AVX; additionally the SDL3 build dependencies
  ([list](https://github.com/libsdl-org/SDL/blob/main/docs/README-linux.md#build-dependencies), e.g. `libx11-dev`, `libwayland-dev`) and a Vulkan driver.
  Shown at an earlier revision (not re-run since the recent Windows renderer work): `bbgame` built with g++ 15 (WSL2 Ubuntu,
  `-DSDL_UNIX_CONSOLE_BUILD=ON` because the X11/Wayland headers are missing there) and the guest ran headless (`BB_HEADLESS=1 BB_NO_GPU=1`)
  for 100 s through startup (logos, sound banks, > 2000 flips). Window/Vulkan on Linux is **not** tested.
- Runtime options (environment variables): `BB_FPS=native|30|60|120|144|165|unlimited|refresh` (presentation rate; `native` = one image
  per game frame, `30` = newest image at 30 Hz). Above 30 (`refresh` = every monitor refresh, FIFO) the hybrid interpolates: per game frame
  the backend renders up to `ceil(rate/30)` in-between images with interpolated camera and objects, as many as the GPU budget allows;
  presents without their own image show the nearest rendered image reprojected by depth to their time, with the HUD undistorted.
  `BB_INTERP=0` repeats images instead; `BB_INTERP_N=<n>` fixed count, `BB_INTERP_LERP=0` no interpolation, `BB_INTERP_OBJ=0` camera only,
  `BB_INTERP_RESTORE=0` no save/restore of temporal state, `BB_INTERP_LOG=1` one line per game frame, `BB_INTERP_WARP=0` no reprojection,
  `BB_INTERP_WARP_CELL=<px>` grid cell (default 2); the presenter uses a second, higher-priority queue if the queue family has one,
  `BB_PRESENT_QUEUE=0` disables it. `BB_WINDOW=3840x2160`, F11 = fullscreen, `BB_SAVE_DIR=<folder>` (saves), `BB_STUB_IMPORTS=1`
  (missing imports as stubs).
- Cheats: `BB_CHEATS=god,nohit,stamina` (comma list): `god` refills HP every frame, `nohit` takes no damage, stagger or fall death, `stamina` uses no
  stamina. Applies to the EU 1.00 build only: the player's memory layout is checked at run time, and on any other build the game logs
  `cheats: <name> unavailable on this build` and writes nothing.
- Upscaling: if the window is larger than the game image (1920×1080), the presenter upscales with AMD FSR 1 (EASU + RCAS);
  `BB_UPSCALE=linear` = bilinear, `BB_FSR_SHARP=<stops>` (0 = sharpest, default 0.2). `BB_SHOT_OUT=1`: screenshots (`BB_SHOT_AT`) show the
  presented image at window size instead of the game image. Diagnostics for internal scaling: `BB_SCALE_LOG=<s>` logs one frame from
  second s (targets/images per draw/dispatch, summary).
- Internal resolution: `BB_RES_SCALE=2` (or 3) renders screen-sized targets at twice (three times) the edge length, e.g. 3840×2160 instead
  of 1920×1080, including post-processing and HUD; shadow maps, cube and small effect targets stay native. Needs 148 bytes of push
  constants (otherwise factor 1). Use with `BB_WINDOW=3840x2160` or fullscreen. On an RTX 3070 Ti this costs ≈ 28.5 ms GPU per frame
  (30 fps, no room for interpolated frames); 1080p internal with FSR to 4K output costs ≈ 0.45 ms.
- Scripted play (testing): `BB_TELEMETRY=<file>` writes ~4 lines/s `T t= f= st=play|menu|load cam=x,y,z yaw= pitch= pos=x,y,z hp=cur/max
  cmd=<id> stk=<0..1> [sprint]` (`pos` = player feet, world coordinates; `menu` = a camera but not near the player, e.g. the title; `stk` = stick
  magnitude nav applies) plus `E t= <id> start|done|stuck|fail ...` per command. Commands, as lines appended to the `BB_PAD_LIVE` file or
  listed in a `BB_ROUTE=<file>` (run in order, `#` comments; a failed command does not stop the queue by default; after `failfast 1` it
  drops the rest, `E ... dropped N queued`; `failfast 0` switches back):
  - `goto <x> <z> [tol] [t=<s>]` (a trailing `sprint` of older routes is ignored): closed-loop run relative to the camera. Consecutive
    gotos form one route that is run through without stopping (look-ahead steering along the route, corners cut by ≤ ~1 m, `done d= pass`
    per waypoint); sprint is automatic on straight stretches, not into sharp turns or the last 6 m. A goto with an explicit `tol` is an
    exact stop (prompts), as is the last goto before another command; default tol 0.8 m. `t=` = time limit (`fail timeout`). < 0.3 m
    progress in 1 s = `stuck`: side-step / back off and blend back into the route (the same side again while that gains > 1 m, else the
    other); a pass-through waypoint within 3 m after two `stuck`s is skipped (`done ... skip`); `fail stuck` after 6 without gain.
  - `ladder <x> <z> [<fx> <fz>] [down]`: exact goto to the ladder's foot (top for `down`), turn toward `fx fz`, cross, hold the stick up
    (down) until the player stops moving along the ladder or steps off; `done dy=` / `fail no-ladder`.
  - `face <x> <z>` (camera; presses R3 once if a lock-on blocks the turn, else `fail no-turn`), `wait_load`, `wait <s>`, `press <button>`.
  - `abort` or `clear` (live): drops the running and queued commands and the pad script's pending presses and held buttons / stick values.
  - `BB_ROUTE_REC=<file>[,<metres>]` records the walk as a reusable route: a point every 3 m, loops and backtracking (fights, searching)
    cut, simplified; the file is rewritten after every point.

## Launcher

`bblauncher` (built by default, SDL3 + Dear ImGui; mouse, keyboard and gamepad: D-pad/stick, A = select, B = back, L1/R1 = tabs) is the
front end for the game: **Games** (add a dump folder and optionally the DLC folder: validated with the same checks as `bbinstall`, then installed or used in
place), **Settings** (frame rate, window mode and size, internal resolution, upscaler, sharpness, interpolation, save folder, missing imports)
and **Cheats** (`god`, `nohit`, `stamina`; they only work on builds whose memory layout is known, currently EU 1.00). **Play** starts the matching `bbgame`
and keeps its output in `game.log`; on a failed run the last lines are shown.

- `bbgame` is recompiled from one specific `eboot.bin`, so a dump needs the build made from it. The launcher looks for `bbgame*` next to itself
  (and in folders added on the Games tab) and matches the SHA-256 of the dump's `eboot.bin` with the one the build recorded: the file
  `<exe>.eboot.sha256` that the CMake build writes next to `bbgame`, a line `bbgame.exe=<sha256>` in a `games.ini` in that folder, or the output of
  `bbgame --eboot-hash`. Without a match it says so and shows the build steps above.
- Settings are written per game to `bbconfig.ini` (`KEY=VALUE` with the variable names of the runtime options below, `#` comments) in
  `profiles\<game>\` and handed to `bbgame` via `BB_CONFIG`. `bbgame` itself reads `bbconfig.ini` next to its executable, else
  `%APPDATA%\BloodbornePC\bbconfig.ini` (Linux: `$XDG_CONFIG_HOME/BloodbornePC`); real environment variables always win. `BB_FULLSCREEN=1` starts fullscreen.
- The launcher's own state is `launcher.ini` in `%APPDATA%\BloodbornePC` (Linux: `$XDG_CONFIG_HOME/BloodbornePC`); installs go to
  `games\<TITLE_ID>-<version>\` there. Command line: `--dry-run [--game <id>]` prints command, environment and `bbconfig.ini` without starting,
  `--validate <dump> [--manifest <file>]` checks a dump, `--run <seconds>` starts the game without a window of its own, waits and stops it, `--config-dir <dir>` and
  `--builds <dir>` use other folders, `--shot <png> [--tab games|settings|cheats] [--size WxH]` renders a frame.
- `python tools/package.py --build <build dir> --game <path\to\bbgame.exe>` assembles `dist/BloodbornePC` (launcher, `bbinstall`, builds, manifests,
  README, LICENSE; never game data). On Linux the launcher needs the SDL3 build dependencies (X11/Wayland headers) to open a window.

## Tools

```
bbinstall validate <dump> [--dlc <dlc>] [--manifest <file.sha256>]   # check a dump
bbinstall hash     <dump> -o <file.sha256>                             # create a manifest
bbinstall install  <dump> [--dlc <dlc>] [--manifest ...] [--dest <dir>]
bbelf <eboot.bin> [--names symbols.txt] [--csv imports.csv] [--relocs] [--functions functions.csv] [--jumptables jt.csv]
bbrecomp <eboot.bin> --stats | --coverage | --emit <out.cpp> | --emit-leaves <out.cpp> <max> | --emit-dir <dir> <n>  [--bias <hex>] [--names <txt>]
bbdiff <eboot.bin> [inputs]   # local only: cmake -DBB_EBOOT=<eboot.bin>; compares recompiled with native execution
```

License: GPL-3.0-or-later. `src/gpu/fsr1_spv.inc` contains shaders translated from AMD FidelityFX FSR 1 (MIT, notice in the file),
generated with `tools/gen_fsr1_spv.py`. The launcher uses Dear ImGui (MIT, © Omar Cornut) and SDL3 (zlib), both fetched at configure time.
