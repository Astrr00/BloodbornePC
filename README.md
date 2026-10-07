# BloodbornePC

Nativer, quelloffener PC-Port von Bloodborne (PS4, v1.09 + The Old Hunters) für Windows und Linux. Das Projekt
rekompiliert den `eboot.bin` statisch und ersetzt die PS4-Systembibliotheken durch eigene Implementierungen.
Es ist kein Emulator.

**Dieses Repository enthält keine Spielinhalte, keinen Originalcode, keine Schlüssel und entschlüsselt nichts.**
Du brauchst einen selbst erstellten, entschlüsselten Ordner-Dump deiner eigenen Kopie mit Update 1.09.

Stand: siehe [docs/ROADMAP.md](docs/ROADMAP.md) (ehrliche Liste: was belegt läuft und was nicht). Unter Windows spielbar ist bisher der Anfang:
Iosefkas Klinik, erster Kampf, Tod, Hunter's Dream, Grabstein-Reise, Speichern/Fortsetzen, Ton, bis 165 Hz mit Zwischenbildern und intern bis 4K.
Alles danach ist ungeprüft – das Spiel ist noch nicht durchspielbar. Linux: baut, Fenster/Vulkan ungetestet. Architektur: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Build

Voraussetzungen: CMake ≥ 3.24, Ninja, ein C++20-Compiler (Clang empfohlen, MSVC 2022 getestet), Python 3 für die Tests.

```
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

### Spiel (Windows, braucht deinen eigenen Dump)

1. Dump prüfen (Hash-Manifest für EU 1.09 liegt bei; `install` kopiert den geprüften Dump zusätzlich nach `%APPDATA%\BloodbornePC`
   und prüft die Kopie):
   ```
   build\bbinstall validate <Dump-Ordner> --manifest manifests\CUSA03173-01.09.sha256 [--dlc <DLC-Ordner>]
   ```
2. Rekompilieren und bauen (Visual Studio 2022 Build Tools, „x64 Native Tools“-Eingabeaufforderung, Ninja):
   ```
   cmake -S . -B build_game -G Ninja -DCMAKE_BUILD_TYPE=Release -DBB_EBOOT=<Pfad\zu\eboot.bin> -DBB_GAME_OPT=O2
   cmake --build build_game --target bbgame -j 2
   ```
   Der erste Build übersetzt ≈ 90 generierte Shards: ≈ 3,5 CPU-Stunden (gemessen 73 min mit `-j 4` auf 8 Kernen), je Compiler-Prozess
   bis 5,6 GB RAM. Die erzeugten Quellen bleiben lokal (nie einchecken).
3. Starten, vom Repo-Ordner aus, mit derselben `eboot.bin` wie beim Bauen:
   ```
   set BB_FPS=refresh
   build_game\bbgame.exe <Pfad\zu\eboot.bin>
   ```
   Controller über SDL3; Tastatur: Pfeile = Steuerkreuz, Enter/Leertaste = Kreuz, Backspace/Esc = Kreis, WASD/IJKL = Sticks, F11 = Vollbild.
   Spielstände liegen in `savedata\` (`BB_SAVE_DIR`). Bricht das Spiel mit `unimplemented import` ab, überspringt `BB_STUB_IMPORTS=1` solche
   Importe (bitte melden).

- **Windows:** MSVC 2022 (Referenz). **Linux:** GCC/Clang mit AVX; zusätzlich die SDL3-Build-Abhängigkeiten
  ([Liste](https://github.com/libsdl-org/SDL/blob/main/docs/README-linux.md#build-dependencies), u. a. `libx11-dev`, `libwayland-dev`) und einen Vulkan-Treiber.
  Belegt: `bbgame` baut unter g++ 15 (WSL2 Ubuntu, `-DSDL_UNIX_CONSOLE_BUILD=ON` weil dort die X11/Wayland-Header fehlen) und der Gast läuft headless
  (`BB_HEADLESS=1 BB_NO_GPU=1`) 100 s durch den Start (Logos, Sound-Bänke, > 2000 Flips). Fenster/Vulkan unter Linux ist **nicht** getestet.
- Laufzeitoptionen (Umgebungsvariablen): `BB_FPS=native|30|60|120|144|165|unlimited|refresh` (Anzeige-Bildrate, siehe ROADMAP Punkt 21;
  `native` = ein Bild je Spielbild, `30` = neuestes Bild im 30-Hz-Takt). Über 30 (`refresh` = jede Bildwiederholung des Monitors mit FIFO)
  interpoliert der Hybrid: das Backend rendert je Spielbild bis zu `ceil(Rate/30)` Zwischenbilder mit interpolierter Kamera und Objekten,
  so viele wie das GPU-Budget erlaubt; Presents ohne eigenes Zwischenbild zeigen das nächstliegende gerenderte Bild per Tiefe auf ihren
  Zeitpunkt reprojiziert, HUD unverzerrt. `BB_INTERP=0` wiederholt stattdessen Bilder; `BB_INTERP_N=<n>` feste Anzahl,
  `BB_INTERP_LERP=0` ohne Interpolation, `BB_INTERP_OBJ=0` nur Kamera, `BB_INTERP_RESTORE=0` ohne Sichern/Zurückspielen des temporalen
  Zustands, `BB_INTERP_LOG=1` eine Zeile je Spielbild, `BB_INTERP_WARP=0` ohne Reprojektion, `BB_INTERP_WARP_CELL=<px>` Gitterzelle
  (Standard 2); der Presenter nutzt eine zweite Queue mit höherer Priorität, falls die Queue-Familie eine hat, `BB_PRESENT_QUEUE=0` ohne.
  `BB_WINDOW=3840x2160`, F11 = Vollbild, `BB_SAVE_DIR=<Ordner>` (Spielstände), `BB_STUB_IMPORTS=1` (fehlende Importe als Stubs).
- Hochskalierung: ist das Fenster größer als das Spielbild (1920×1080), skaliert der Presenter mit AMD FSR 1 (EASU + RCAS) hoch;
  `BB_UPSCALE=linear` = bilinear wie früher, `BB_FSR_SHARP=<Stufen>` (0 = schärfste, Standard 0.2). `BB_SHOT_OUT=1`: Screenshots
  (`BB_SHOT_AT`) zeigen das ausgegebene Bild in Fenstergröße statt des Spielbilds. Diagnose für spätere interne Skalierung:
  `BB_SCALE_LOG=<s>` protokolliert einen Frame ab Sekunde s (Ziele/Bilder je Draw/Dispatch, Zusammenfassung).
- Interne Auflösung: `BB_RES_SCALE=2` (oder 3) rendert bildschirmgroße Ziele intern mit doppelter (dreifacher) Kantenlänge, z. B.
  3840×2160 statt 1920×1080, inklusive Nachbearbeitung und HUD; Schattenkarten, Würfel- und kleine Effektziele bleiben nativ.
  Braucht 148 Byte Push-Konstanten (sonst Faktor 1). Passend dazu `BB_WINDOW=3840x2160` bzw. Vollbild.

## Werkzeuge

```
bbinstall validate <dump> [--dlc <dlc>] [--manifest <file.sha256>]   # Dump prüfen
bbinstall hash     <dump> -o <file.sha256>                             # Manifest erzeugen
bbinstall install  <dump> [--dlc <dlc>] [--manifest ...] [--dest <dir>]
bbelf <eboot.bin> [--names symbols.txt] [--csv imports.csv] [--relocs] [--functions functions.csv] [--jumptables jt.csv]
bbrecomp <eboot.bin> --stats | --coverage | --emit <out.cpp> | --emit-leaves <out.cpp> <max> | --emit-dir <dir> <n>  [--bias <hex>] [--names <txt>]
bbdiff <eboot.bin> [inputs]   # nur lokal: cmake -DBB_EBOOT=<eboot.bin>; vergleicht rekompilierte mit nativer Ausführung
```

Lizenz: GPL-3.0-or-later. `src/gpu/fsr1_spv.inc` enthält aus AMD FidelityFX FSR 1 (MIT, Hinweis in der Datei) übersetzte Shader
(erzeugt mit `tools/gen_fsr1_spv.py`).
