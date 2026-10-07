# SPDX-License-Identifier: GPL-3.0-or-later
"""Builds a symbol-name list for `bbelf --names` from shadPS4's NID table (GPL-2.0-or-later, fetched, not vendored).

usage: python tools/nidnames.py build/ps4_names.txt
"""
import re
import sys
import urllib.request

URL = "https://raw.githubusercontent.com/shadps4-emu/shadPS4/main/src/core/aerolib/aerolib.inl"

text = urllib.request.urlopen(URL).read().decode("utf-8")
names = [n.strip() for n in re.findall(r'STUB\("[^"]+",\s*([^)]+)\)', text)]
# Entries with spaces are sanitized local symbols whose NIDs cannot be recomputed from the name.
names = [n for n in names if " " not in n]
with open(sys.argv[1], "w", encoding="utf-8") as f:
    f.write("\n".join(names) + "\n")
print(f"wrote {len(names)} names to {sys.argv[1]}")
