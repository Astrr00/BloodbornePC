# SPDX-License-Identifier: GPL-3.0-or-later
"""End-to-end checks of bbelf/bbinstall against synthetic fixtures. Usage: run_tests.py <bin-dir> <work-dir>"""
import os
import shutil
import subprocess
import sys

import make_fixtures

BIN, WORK = sys.argv[1], sys.argv[2]
EXE = ".exe" if os.name == "nt" else ""
failures = []


def run(tool, *args):
    p = subprocess.run([os.path.join(BIN, tool + EXE), *args], capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def check(name, cond, out=""):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        failures.append(name)
        print(out)


shutil.rmtree(WORK, ignore_errors=True)
make_fixtures.main(WORK)
w = lambda *p: os.path.join(WORK, *p)

# bbelf: plain ELF with name resolution and CSV export
code, out = run("bbelf", w("eboot.elf"), "--names", w("names.txt"), "--csv", w("imports.csv"))
check("elf parses", code == 0, out)
check("elf type", "ET_SCE_DYNEXEC" in out, out)
check("needed module", "[1] libkernel v1.1" in out, out)
check("imports resolved", "imports: 2 symbols, 2 named; exports: 1" in out, out)
check("nid name", "9BcDykPmo1I __error" in out and "1jfXLRVzisc sceKernelUsleep" in out, out)
check("sdk version", "sdk       0x04508101" in out, out)
check("reloc summary", "R_X86_64_JUMP_SLOT   1" in out and "R_X86_64_RELATIVE    1" in out, out)
csv = open(w("imports.csv")).read()
check("csv rows", "0x402010,R_X86_64_JUMP_SLOT,libkernel,libkernel,1jfXLRVzisc,sceKernelUsleep" in csv, csv)

code, out2 = run("bbelf", w("eboot.self"), "--names", w("names.txt"))
check("fself unwraps identically", code == 0 and "(SELF" in out2 and "2 named" in out2, out2)
code, out = run("bbelf", w("eboot.self"), "--functions", w("functions.csv"))
fcsv = open(w("functions.csv")).read() if os.path.exists(w("functions.csv")) else ""
check("eh_frame functions found via fSELF", code == 0
      and f"functions (.eh_frame): {3 + len(make_fixtures.RECOMP_TESTS)}" in out
      and "0x400000,8,eh_frame\n0x400008,6,call\n0x400010,8,eh_frame\n0x400018,40,eh_frame\n" in fcsv, out + fcsv)
code, out = run("bbelf", w("eboot.elf"), "--jumptables", w("jt.csv"))
jcsv = open(w("jt.csv")).read() if os.path.exists(w("jt.csv")) else ""
check("switch tables resolved (cmp bound and unbounded)", code == 0
      and "2 switch tables (6 case targets, 0 outside" in out and "1 jmp [rip]" in out and "classified 3 of 3" in out
      and "0x40002b,0x400018,0x400034,3\n" in jcsv and "0x400085,0x400070,0x4000a0,3\n" in jcsv, out + jcsv)
code, out = run("bbelf", w("eboot.encrypted.self"))
check("encrypted self rejected", code == 1 and "does not decrypt" in out, out)

# bbinstall
code, out = run("bbinstall", "validate", w("dump"), "--dlc", w("dlc"))
check("valid dump", code == 0 and "Dump OK" in out and "CUSA00207 v01.09" in out, out)
code, out = run("bbinstall", "validate", w("merged"))
check("merged gp dump accepted", code == 0 and "CATEGORY gp" in out, out)
code, out = run("bbinstall", "install", w("extracted"), "--dest", w("installed_ext"))
check("extractor layout validates and normalizes", code == 0 and "Image0/ + Sc0/" in out
      and os.path.isfile(w("installed_ext", "game", "eboot.bin"))
      and os.path.isfile(w("installed_ext", "game", "sce_sys", "param.sfo")), out)
code, out = run("bbinstall", "validate", w("old_version"))
check("old version rejected", code == 1 and "required 01.09" in out and "eboot.bin missing" in out, out)
code, out = run("bbinstall", "validate", w("fake.pkg"))
check("pkg rejected with explanation", code == 1 and "PKG file" in out, out)

code, out = run("bbinstall", "hash", w("dump"), "-o", w("dump.sha256"))
check("hash manifest", code == 0 and "wrote 3 entries" in out, out)
code, out = run("bbinstall", "validate", w("dump"), "--dlc", w("dlc"), "--manifest", w("dump.sha256"))
check("manifest verifies", code == 0 and "3/3 files verified" in out, out)
with open(w("dump", "dvdroot_ps4", "data.bin"), "ab") as f:
    f.write(b"tamper")
code, out = run("bbinstall", "validate", w("dump"), "--manifest", w("dump.sha256"))
check("tamper detected", code == 1 and "hash mismatch" in out and "dvdroot_ps4/data.bin" in out, out)

run("bbinstall", "hash", w("dump"), "-o", w("dump.sha256"))
code, out = run("bbinstall", "install", w("dump"), "--dlc", w("dlc"), "--manifest", w("dump.sha256"),
                "--dest", w("installed"))
check("install copies", code == 0 and os.path.isfile(w("installed", "game", "eboot.bin"))
      and os.path.isfile(w("installed", "dlc", "sce_sys", "param.sfo")), out)
code, out = run("bbinstall", "install", w("dump"), "--dlc", w("dlc_sc0"), "--dest", w("installed_sc0"))
check("Sc0-only DLC accepted and staged", code == 0 and "TESTDLC000000000" in out
      and os.path.isfile(w("installed_sc0", "dlc", "sce_sys", "param.sfo")), out)
code, out = run("bbinstall", "validate", w("no_image0"))
check("game without Image0 rejected", code == 1 and "Image0/ missing" in out, out)

# update merge (base 01.00 + update 01.09 -> manifest of the merged tree; nothing is copied for validate)
import hashlib
sha = lambda b: hashlib.sha256(b).hexdigest()
elf = open(w("eboot.elf"), "rb").read()
sfo109 = open(w("upd", "Sc0", "param.sfo"), "rb").read()
merged_entries = [(sha(elf), "Image0/eboot.bin"), (sha(b"base a"), "Image0/dvdroot_ps4/a.bin"), (sha(b"update b"), "Image0/dvdroot_ps4/b.bin"),
                  (sha(b"update d"), "Image0/dvdroot_ps4/d.bin"), (sha(sfo109), "Sc0/param.sfo")]
def manifest(name, entries):
    with open(w(name), "w") as f:
        f.writelines(f"{h}  {p}\n" for h, p in entries)
manifest("merged.sha256", merged_entries)
manifest("merged_missing.sha256", merged_entries + [(sha(b"x"), "Image0/dvdroot_ps4/zzz.bin")])
code, out = run("bbinstall", "validate", w("upd_base"), "--update", w("upd"), "--manifest", w("merged.sha256"))
check("update merge validates against the merged manifest", code == 0 and "5/5 files verified" in out and "base is version 01.00" in out
      and "Merged set: 2 files from the base, 2 from the update (1 of them replace base files)" in out, out)
code, out = run("bbinstall", "validate", w("upd_base"), "--manifest", w("merged.sha256"))
check("01.00 base without the update is rejected", code == 1 and "required 01.09" in out, out)
code, out = run("bbinstall", "validate", w("upd_base"), "--update", w("upd"), "--manifest", w("merged_missing.sha256"))
check("missing merged file reported", code == 1 and "missing: Image0/dvdroot_ps4/zzz.bin" in out, out)
code, out = run("bbinstall", "validate", w("upd_base"), "--update", w("upd_wrong_title"))
check("update for another title rejected", code == 1 and "not for this game (title CUSA99999 vs CUSA00207)" in out, out)
code, out = run("bbinstall", "validate", w("upd_base"), "--update", w("upd_other_release"))
check("update of another known Bloodborne release gets a hint", code == 1 and "belongs to another Bloodborne release" in out, out)
code, out = run("bbinstall", "validate", w("upd_base"), "--update", w("upd_old"))
check("update with another version rejected", code == 1 and "Update is version 01.05, expected 01.09" in out, out)
code, out = run("bbinstall", "validate", w("upd_base"), "--update", w("upd_empty"))
check("update folder without param.sfo rejected", code == 1 and "no param.sfo" in out, out)
code, out = run("bbinstall", "validate", w("upd_base"), "--update", w("does_not_exist"))
check("missing update folder rejected", code == 1 and "Update folder not found" in out, out)
code, out = run("bbinstall", "validate", w("extracted"), "--update", w("upd"))
check("base already 01.09: warning only", code == 0 and "already version 01.09" in out, out)
code, out = run("bbinstall", "install", w("upd_base"), "--update", w("upd"), "--manifest", w("merged.sha256"), "--dest", w("installed_upd"))
g = lambda *p: os.path.join(w("installed_upd", "game"), *p)
rd = lambda *p: open(g(*p), "rb").read()
check("update install: override wins, base-only kept, update-only added, sfo is 01.09",
      code == 0 and os.path.isfile(g("eboot.bin")) and rd("dvdroot_ps4", "a.bin") == b"base a" and rd("dvdroot_ps4", "b.bin") == b"update b"
      and rd("dvdroot_ps4", "d.bin") == b"update d" and rd("sce_sys", "param.sfo") == sfo109
      and not os.path.exists(w("installed_upd", ".bbinstall-incomplete")) and "Merging update" in out, out)
# console-layout base + PKG-layout update: validate and install must agree (the update's Sc0/extra.dat replaces the base's sce_sys/extra.dat)
manifest("merged_mixed.sha256", [(sha(elf), "eboot.bin"), (sha(b"update a"), "dvdroot_ps4/a.bin"), (sha(sfo109), "sce_sys/param.sfo"), (sha(b"update extra"), "sce_sys/extra.dat")])
code, out = run("bbinstall", "validate", w("upd_cbase"), "--update", w("upd_pkglayout"), "--manifest", w("merged_mixed.sha256"))
check("mixed layouts: validate against the merged manifest", code == 0 and "4/4 files verified" in out, out)
code, out = run("bbinstall", "install", w("upd_cbase"), "--update", w("upd_pkglayout"), "--manifest", w("merged_mixed.sha256"), "--dest", w("installed_mixed"))
check("mixed layouts: install agrees with validate (sce_sys replaced by the update's Sc0)", code == 0
      and open(w("installed_mixed", "game", "sce_sys", "extra.dat"), "rb").read() == b"update extra"
      and open(w("installed_mixed", "game", "dvdroot_ps4", "a.bin"), "rb").read() == b"update a"
      and not os.path.exists(w("installed_mixed", ".bbinstall-incomplete")), out)
code, out = run("bbinstall", "install", w("upd_base"), "--update", w("upd_wrong_title"), "--dest", w("installed_bad"))
check("bad update installs nothing", code == 1 and not os.path.exists(w("installed_bad", "game")), out)

sys.exit(1 if failures else 0)
