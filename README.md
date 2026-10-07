# BloodbornePC

Native, open-source PC port of Bloodborne (PS4, v1.09 + The Old Hunters) for Windows and Linux. The project statically
recompiles `eboot.bin` to C++ and replaces the PS4 system libraries with its own implementations. It is not an emulator.

**This repository contains no game content, no original code, no keys, and decrypts nothing.**
You need a decrypted folder dump of your own copy, made by yourself.

Status: see [docs/ROADMAP.md](docs/ROADMAP.md) (an honest list of what has been shown to work and what has not; written in German).
On Windows the opening part is playable: Iosefka's Clinic, the first fight, death, Hunter's Dream, headstone travel, save/continue,
audio, up to 165 Hz with interpolated frames, internal rendering up to 4K. This was tested with the EU 1.00 eboot; the 1.09 build has
so far only been shown to reach the clinic. Beyond that (Central Yharnam onwards) everything is untested – the game is not yet playable
to the end. Linux: the libraries, tools and tests build and pass in CI; the game itself is not tested on Linux.
Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

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
- Upscaling: if the window is larger than the game image (1920×1080), the presenter upscales with AMD FSR 1 (EASU + RCAS);
  `BB_UPSCALE=linear` = bilinear, `BB_FSR_SHARP=<stops>` (0 = sharpest, default 0.2). `BB_SHOT_OUT=1`: screenshots (`BB_SHOT_AT`) show the
  presented image at window size instead of the game image. Diagnostics for internal scaling: `BB_SCALE_LOG=<s>` logs one frame from
  second s (targets/images per draw/dispatch, summary).
- Internal resolution: `BB_RES_SCALE=2` (or 3) renders screen-sized targets at twice (three times) the edge length, e.g. 3840×2160 instead
  of 1920×1080, including post-processing and HUD; shadow maps, cube and small effect targets stay native. Needs 148 bytes of push
  constants (otherwise factor 1). Use with `BB_WINDOW=3840x2160` or fullscreen. On an RTX 3070 Ti this costs ≈ 28.5 ms GPU per frame
  (30 fps, no room for interpolated frames); 1080p internal with FSR to 4K output costs ≈ 0.45 ms.
- Scripted play (testing): `BB_TELEMETRY=<file>` writes ~4 lines/s `T t= f= st=play|menu|load cam=x,y,z yaw= pitch= pos=x,y,z hp=cur/max`
  (`pos` = player feet, world coordinates; `menu` = a camera but not near the player, e.g. the title) plus `E t= <id> start|done|stuck|fail ...` per command. Commands, as lines appended to the
  `BB_PAD_LIVE` file or listed in a `BB_ROUTE=<file>` (run in order, `#` comments): `goto <x> <z> [tol] [sprint]` (closed-loop run
  relative to the camera; < 0.3 m progress in 1 s = `stuck`: back off, side-step, retry; `fail stuck` after 6), `face <x> <z>`,
  `wait_load`, `wait <s>`, `press <button>`. `BB_ROUTE_REC=<file>[,<metres>]` appends `goto x z` every 3 m walked (a reusable route).

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
generated with `tools/gen_fsr1_spv.py`.
