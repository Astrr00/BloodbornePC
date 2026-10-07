// SPDX-License-Identifier: GPL-3.0-or-later
// Small system services: Rtc (UTC only), Sysmodule, SaveData/Trophy initialisation (real storage comes with the save
// system milestone).
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "core/sfo.h"
#include "hle/abi.h"
#include "hle/vfs.h"

namespace bb::hle {
namespace {

constexpr uint64_t kTicksPerSecond = 1000000;
// Days between 0001-01-01 and 1970-01-01 (proleptic Gregorian).
constexpr int64_t kEpochDays = 719162;

int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {  // Howard Hinnant's algorithm, relative to 1970-01-01
    y -= m <= 2;
    const int64_t era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = unsigned(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + int64_t(doe) - 719468;
}
void civil_from_days(int64_t z, int64_t& y, unsigned& m, unsigned& d) {
    z += 719468;
    const int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const unsigned doe = unsigned(z - era * 146097);
    const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    y = int64_t(yoe) + era * 400;
    const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const unsigned mp = (5 * doy + 2) / 153;
    d = doy - (153 * mp + 2) / 5 + 1;
    m = mp < 10 ? mp + 3 : mp - 9;
    y += m <= 2;
}

// SceRtcDateTime: u16 year, month, day, hour, minute, second; u32 microsecond.
void write_datetime(Context& c, uint64_t at, uint64_t tick) {
    const int64_t secs = int64_t(tick / kTicksPerSecond), days = secs / 86400 - kEpochDays, rem = secs % 86400;
    int64_t y;
    unsigned m, d;
    civil_from_days(days, y, m, d);
    rt::st<uint16_t>(c, at, uint16_t(y));
    rt::st<uint16_t>(c, at + 2, uint16_t(m));
    rt::st<uint16_t>(c, at + 4, uint16_t(d));
    rt::st<uint16_t>(c, at + 6, uint16_t(rem / 3600));
    rt::st<uint16_t>(c, at + 8, uint16_t(rem % 3600 / 60));
    rt::st<uint16_t>(c, at + 10, uint16_t(rem % 60));
    rt::st<uint32_t>(c, at + 12, uint32_t(tick % kTicksPerSecond));
}
uint64_t read_datetime(Context& c, uint64_t at) {
    const int64_t days = days_from_civil(rt::ld<uint16_t>(c, at), rt::ld<uint16_t>(c, at + 2), rt::ld<uint16_t>(c, at + 4)) + kEpochDays;
    const uint64_t secs = uint64_t(days) * 86400 + rt::ld<uint16_t>(c, at + 6) * 3600u + rt::ld<uint16_t>(c, at + 8) * 60u + rt::ld<uint16_t>(c, at + 10);
    return secs * kTicksPerSecond + rt::ld<uint32_t>(c, at + 12);
}
uint64_t now_tick() {
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    return (uint64_t(kEpochDays) * 86400) * kTicksPerSecond + uint64_t(us);
}

void ok(Context& c) { ret(c, 0); }

// Installed add-on content: BB_DLC_DIR (';'-separated folders) or <app0>/../dlc (bbinstall layout). Each folder holds the DLC's
// param.sfo (sce_sys/ or a PKG extractor's Sc0/); the entitlement label is CONTENT_ID[20..36), e.g. SPEXPANSIONDLC03 (The Old
// Hunters - its data ships with the game, the package only grants the entitlement).
const std::vector<std::string>& addcont_labels() {
    static const std::vector<std::string> labels = [] {
        std::vector<std::filesystem::path> dirs;
        if (const char* e = std::getenv("BB_DLC_DIR")) {
            for (std::string s = e; !s.empty();) {
                const size_t k = s.find(';');
                dirs.emplace_back(s.substr(0, k));
                s = k == std::string::npos ? "" : s.substr(k + 1);
            }
        } else if (auto app0 = vfs_resolve("/app0")) {
            dirs.push_back(app0->parent_path() / "dlc");
        }
        std::vector<std::string> out;
        for (const auto& d : dirs)
            for (const char* sub : {"sce_sys", "Sc0"})
                if (auto sfo = load_sfo(d / sub / "param.sfo")) {
                    const std::string id = sfo->str("CONTENT_ID").value_or("");
                    if (id.size() >= 36 && sfo->str("CATEGORY").value_or("") == "ac") out.push_back(id.substr(20, 16));
                    break;
                }
        for (const auto& l : out) std::fprintf(stderr, "appcontent: add-on content %s installed\n", l.c_str());
        return out;
    }();
    return labels;
}

} // namespace

void register_services() {
    // ponytail: local time == UTC; add the host time zone offset if the game shows wall-clock time.
    reg("sceRtcGetCurrentClockLocalTime", [](Context& c) { write_datetime(c, arg(c, 0), now_tick()); ret(c, 0); });
    reg("sceRtcGetCurrentNetworkTick", [](Context& c) { rt::st<uint64_t>(c, arg(c, 0), now_tick()); ret(c, 0); });
    reg("sceRtcGetTick", [](Context& c) { rt::st<uint64_t>(c, arg(c, 1), read_datetime(c, arg(c, 0))); ret(c, 0); });
    reg("sceRtcSetTick", [](Context& c) { write_datetime(c, arg(c, 0), rt::ld<uint64_t>(c, arg(c, 1))); ret(c, 0); });
    reg("sceRtcGetTickResolution", [](Context& c) { ret(c, kTicksPerSecond); });
    reg("sceRtcSetTime_t", [](Context& c) {  // (SceRtcDateTime*, time_t)
        write_datetime(c, arg(c, 0), (uint64_t(kEpochDays) * 86400 + arg(c, 1)) * kTicksPerSecond);
        ret(c, 0);
    });
    reg("sceRtcConvertLocalTimeToUtc", [](Context& c) { rt::st<uint64_t>(c, arg(c, 1), rt::ld<uint64_t>(c, arg(c, 0))); ret(c, 0); });
    reg("sceRtcConvertUtcToLocalTime", [](Context& c) { rt::st<uint64_t>(c, arg(c, 1), rt::ld<uint64_t>(c, arg(c, 0))); ret(c, 0); });
    reg("sceRtcGetDayOfWeek", [](Context& c) {  // (year, month, day) -> 0 = Sunday
        const int64_t d = days_from_civil(int64_t(arg(c, 0)), unsigned(arg(c, 1)), unsigned(arg(c, 2)));
        ret(c, uint64_t((d % 7 + 11) % 7));  // 1970-01-01 was a Thursday
    });

    reg("sceSystemServiceHideSplashScreen", [](Context& c) { if (std::getenv("BB_SAVE_LOG")) std::fputs("service: HideSplashScreen\n", stderr); ret(c, 0); });
    // SceAppContentAddcontInfo: entitlement label char[17] + 3 padding, u32 status (4 = installed); 24 bytes each
    reg("sceAppContentGetAddcontInfoList", [](Context& c) {  // (serviceLabel, list*, maxCount, hitNumber*)
        const auto& labels = addcont_labels();
        const uint64_t list = arg(c, 1), max = uint32_t(arg(c, 2));
        uint32_t n = 0;
        for (const auto& l : labels) {
            if (list && n < max) {
                uint8_t e[24] = {};
                std::memcpy(e, l.data(), std::min<size_t>(l.size(), 16));
                e[20] = 4;
                std::memcpy(ptr<void>(c, list + 24ull * n), e, 24);
            }
            ++n;
        }
        if (std::getenv("BB_SAVE_LOG")) std::fprintf(stderr, "service: AddcontInfoList(label=%llu, list=%llx, max=%llu) -> %u\n", (unsigned long long)arg(c, 0), (unsigned long long)list, (unsigned long long)max, n);
        if (arg(c, 3)) rt::st<uint32_t>(c, arg(c, 3), list ? std::min<uint32_t>(n, uint32_t(max)) : n);
        ret(c, 0);
    });
    // SceSystemServiceStatus: s32 eventNum, then bool flags (system UI overlaid, background execution, cpu mode 7, ...): all off.
    reg("sceSystemServiceGetStatus", [](Context& c) { if (arg(c, 0)) std::memset(ptr<void>(c, arg(c, 0)), 0, 12); ret(c, 0); });
    reg("sceSystemServiceReceiveEvent", [](Context& c) { ret(c, uint64_t(int64_t(int32_t(0x80A10003)))); });  // SCE_SYSTEM_SERVICE_ERROR_NO_EVENT
    reg("sceSystemServiceLaunchWebBrowser", ok);
    reg("sceSysmoduleUnloadModule", ok);
    reg("sceSysmoduleIsLoaded", ok);
    reg("sceNpTrophyCreateContext", [](Context& c) { rt::st<int32_t>(c, arg(c, 0), 1); ret(c, 0); });  // (ctx*, userId, label, opts)
    reg("sceNpTrophyCreateHandle", [](Context& c) { rt::st<int32_t>(c, arg(c, 0), 1); ret(c, 0); });
    reg("sceNpTrophyRegisterContext", ok);
    // (ctx, handle, trophyId, platinumId*): no trophy store; report "no platinum unlocked" (SCE_NP_TROPHY_INVALID_TROPHY_ID)
    reg("sceNpTrophyUnlockTrophy", [](Context& c) { if (arg(c, 3)) rt::st<int32_t>(c, arg(c, 3), -1); ret(c, 0); });
}

} // namespace bb::hle
