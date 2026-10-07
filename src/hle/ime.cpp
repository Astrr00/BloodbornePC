// SPDX-License-Identifier: GPL-3.0-or-later
// libSceImeDialog. The game opens the on-screen keyboard for the character name; there is no host text UI yet, so the
// dialog "finishes" a few polls after Init with a default text (BB_IME_TEXT, else "Hunter") written into the
// guest's input buffer as UTF-16 and end status OK.  ponytail: no keyboard UI; SDL3 text input overlay when needed.
// Layout (SDK, as documented by shadPS4 src/core/libraries/ime/ime_dialog.h): OrbisImeDialogParam { s32 userId @0, type @4,
// u64 supportedLanguages @8, enterLabel @16, inputMethod @20, filter* @24, u32 option @32, maxTextLength @36,
// char16_t* inputTextBuffer @40 ... }; OrbisImeDialogResult { s32 endstatus; s32 reserved[12]; }.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "hle/abi.h"

namespace bb::hle {
namespace {

enum Status { kNone = 0, kRunning = 1, kFinished = 2 };
std::atomic<int> g_status{kNone}, g_polls{0};

void ime_init(Context& c) {  // (const OrbisImeDialogParam*, const void* extended)
    const uint64_t param = arg(c, 0);
    if (!param) return ret(c, uint64_t(int64_t(int32_t(0x80bc0002))));  // SCE_IME_DIALOG_ERROR_INVALID_ADDRESS
    const uint32_t max_len = rt::ld<uint32_t>(c, param + 36);
    const uint64_t buf = rt::ld<uint64_t>(c, param + 40);
    if (buf && max_len) {
        const char* env = std::getenv("BB_IME_TEXT");
        const std::string text = env && *env ? env : "Hunter";
        std::fprintf(stderr, "ime: Init type=%u max=%u text='%s'\n", rt::ld<uint32_t>(c, param + 4), max_len, text.c_str());
        auto* out = ptr<uint16_t>(c, buf);
        uint32_t n = 0;
        for (; n < text.size() && n < max_len; ++n) out[n] = uint8_t(text[n]);  // ponytail: ASCII only
        out[n] = 0;
    }
    g_polls = 0;
    g_status = kRunning;
    ret(c, 0);
}

void ime_status(Context& c) {
    if (g_status == kRunning && ++g_polls > 10) g_status = kFinished;
    static int logged = 0;
    if (logged < 20) { ++logged; std::fprintf(stderr, "ime: GetStatus -> %d\n", g_status.load()); }
    ret(c, g_status.load());
}

void ime_result(Context& c) {  // (OrbisImeDialogResult*)
    std::fprintf(stderr, "ime: GetResult\n");
    if (const uint64_t res = arg(c, 0)) rt::st<int32_t>(c, res, 0);  // SCE_IME_DIALOG_END_STATUS_OK; the reserved tail is left alone (the game's struct may be shorter)
    ret(c, 0);
}

void ime_term(Context& c) {
    std::fprintf(stderr, "ime: Term/Abort\n");
    g_status = kNone;
    ret(c, 0);
}

} // namespace

void register_ime() {
    reg("sceImeDialogInit", ime_init);
    reg("sceImeDialogGetStatus", ime_status);
    reg("sceImeDialogGetResult", ime_result);
    reg("sceImeDialogTerm", ime_term);
    reg("sceImeDialogAbort", ime_term);
    reg("sceImeDialogForceClose", ime_term);
}

} // namespace bb::hle
