# Hash-Manifeste

Format: `sha256sum`-kompatibel, ein Eintrag pro Datei, Pfade relativ zum Dump-Ordner mit `/`:

```
<64 hex>  eboot.bin
<64 hex>  sce_sys/param.sfo
```

Dateiname: `<TITLE_ID>-<APP_VER>[-dlc].sha256`, z. B. `CUSA00207-01.09.sha256`.

Die Manifeste enthalten nur Hashes, keine Spielinhalte. Ein Referenzmanifest wird nur eingecheckt, wenn zwei unabhängige
Quellen es identisch bestätigen: zwei eigene Dumps derselben Version, oder ein eigener Dump plus der Inhalt der
Original-PKGs, deren PFS-Image-Digest (PKG-Header) und Entry-Digests stimmen. Bei Fake-PKGs (DRM-Typ 0xF) belegen diese
Digests nur die innere Konsistenz des Pakets, nicht die Identität mit der Retail-Disc.

```
bbinstall hash <dump-ordner> -o CUSA00207-01.09.sha256
```

Ohne `--manifest` prüft `bbinstall` nur die Struktur und warnt.

## Vorhandene Manifeste

- `CUSA03173-01.09.sha256`: Bloodborne EU (`EP9000-CUSA03173_00-BLOODBORNE0000EU`), Basis 01.00 + Update 01.09
  zusammengeführt, PKG-Extractor-Layout (Pfade `Image0/…`, `Sc0/…`), 28 844 Einträge. Gegengeprüft gegen den Inhalt
  des Basis-PKGs (Kategorie `gd`, APP_VER 01.00, 28 825 Dateien) und des offiziellen Update-PKGs (Kategorie `gp`,
  APP_VER 01.09, 436 Dateien; Update überschreibt gleiche Pfade): Header-, Body-, Entry- und PFS-Image-Digests beider
  PKGs stimmen, alle 28 844 Zeilen identisch.
