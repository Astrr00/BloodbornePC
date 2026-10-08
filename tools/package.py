#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Assemble dist/BloodbornePC: launcher, installer, bbgame builds (+ their eboot hash sidecars), manifests, README, LICENSE.

  python tools/package.py --build build [--game build_game/bbgame.exe ...] [--out dist]

Never copies game data: only the files named here. Each --game build keeps its `<exe>.eboot.sha256` sidecar, which is how the
launcher matches a dump to a build; a second build is renamed bbgame_<n>.exe by passing `--game path/to/bbgame.exe=bbgame_eu109.exe`.
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXE = ".exe" if sys.platform == "win32" else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="build dir with bblauncher and bbinstall")
    ap.add_argument("--game", action="append", default=[], help="bbgame executable, optionally =<new name>")
    ap.add_argument("--out", default=str(ROOT / "dist"))
    a = ap.parse_args()
    dest = Path(a.out) / "BloodbornePC"
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    build = Path(a.build)
    for name in ("bblauncher", "bbinstall"):
        src = build / (name + EXE)
        if not src.is_file():
            sys.exit(f"missing {src} (build the default target first)")
        shutil.copy2(src, dest / src.name)
    for dll in build.glob("SDL3*.dll"):  # only if SDL was built shared
        shutil.copy2(dll, dest)
    for g in a.game:
        path, _, new = g.partition("=")
        src = Path(path)
        if not src.is_file():
            sys.exit(f"missing {src}")
        name = new or src.name
        shutil.copy2(src, dest / name)
        side = Path(str(src) + ".eboot.sha256")
        if side.is_file():
            shutil.copy2(side, dest / (name + ".eboot.sha256"))
        else:
            print(f"warning: no {side.name}; the launcher falls back to `{name} --eboot-hash`")
    (dest / "manifests").mkdir()
    for f in (ROOT / "manifests").iterdir():
        if f.suffix in (".sha256", ".md"):
            shutil.copy2(f, dest / "manifests" / f.name)
    for f in ("README.md", "LICENSE"):
        shutil.copy2(ROOT / f, dest / f)
    print("packaged:", ", ".join(sorted(p.name for p in dest.iterdir())), "->", dest)


main()
