# Hash manifests

Format: `sha256sum`-compatible, one entry per file, paths relative to the dump folder with `/`:

```
<64 hex>  eboot.bin
<64 hex>  sce_sys/param.sfo
```

File name: `<TITLE_ID>-<APP_VER>[-dlc].sha256`, e.g. `CUSA00207-01.09.sha256`.

The manifests contain hashes only, no game content. A reference manifest is committed only if two independent sources
confirm it identically: two own dumps of the same version, or one own dump plus the contents of the original PKGs whose
PFS image digest (PKG header) and entry digests match. For fake PKGs (DRM type 0xF) these digests only prove the package's
internal consistency, not identity with the retail disc.

```
bbinstall hash <dump-folder> -o CUSA00207-01.09.sha256
```

Without `--manifest`, `bbinstall` only checks the structure and warns.

## Existing manifests

- `CUSA03173-01.09.sha256`: Bloodborne EU (`EP9000-CUSA03173_00-BLOODBORNE0000EU`), base 01.00 + update 01.09
  merged, PKG extractor layout (paths `Image0/…`, `Sc0/…`), 28,844 entries. Cross-checked against the contents of the
  base PKG (category `gd`, APP_VER 01.00, 28,825 files) and the official update PKG (category `gp`,
  APP_VER 01.09, 436 files; the update overwrites identical paths): header, body, entry and PFS image digests of both
  PKGs match, all 28,844 lines identical.
