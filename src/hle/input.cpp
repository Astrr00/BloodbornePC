// SPDX-License-Identifier: GPL-3.0-or-later
// UserService, Pad, PlayGo: a signed-in local user with one connected pad that reports a neutral state until the SDL3
// backend replaces scePadReadState.
#include <chrono>
#include <cstring>

#include "hle/abi.h"

namespace bb::hle {
namespace {

constexpr int32_t kUserId = 1;
PadProvider g_pad_provider = nullptr;

void ok(Context& c) { ret(c, 0); }

void pad_read_state(Context& c) {  // (handle, ScePadData*): neutral sticks, no buttons, connected
    uint8_t* d = ptr<uint8_t>(c, arg(c, 1));
    std::memset(d, 0, 0x80);
    d[4] = d[5] = d[6] = d[7] = 0x80;       // sticks centred
    if (g_pad_provider) {
        const PadInput in = g_pad_provider();
        std::memcpy(d, &in.buttons, 4);
        d[4] = in.lx; d[5] = in.ly; d[6] = in.rx; d[7] = in.ry; d[8] = in.l2; d[9] = in.r2;
    }
    const uint64_t ts = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
    std::memcpy(d + 80, &ts, 8);            // timestamp (µs)
    const float w = 1.0f;
    std::memcpy(d + 12 + 12, &w, 4);        // orientation quaternion (x,y,z,w) = identity
    const float g = 1.0f;
    std::memcpy(d + 28 + 4, &g, 4);         // acceleration: 1 g on y
    d[76] = 1;                              // connected
    d[77] = 1;                              // connectedCount
    ret(c, 0);
}
void pad_info(Context& c) {  // ScePadControllerInformation
    uint8_t* d = ptr<uint8_t>(c, arg(c, 1));
    std::memset(d, 0, 0x20);
    const float density = 44.86f;
    std::memcpy(d, &density, 4);
    const uint16_t rx = 1920, ry = 950;
    std::memcpy(d + 4, &rx, 2);
    std::memcpy(d + 6, &ry, 2);
    d[8] = d[9] = 13;  // dead zones
    d[11] = 1;         // connectedCount
    d[12] = 1;         // connected
    ret(c, 0);
}

} // namespace

void set_pad_provider(PadProvider p) { g_pad_provider = p; }

void register_input() {
    reg("sceUserServiceGetInitialUser", [](Context& c) { rt::st<int32_t>(c, arg(c, 0), kUserId); ret(c, 0); });
    reg("sceUserServiceGetLoginUserIdList", [](Context& c) {  // int userId[4]
        for (int i = 0; i < 4; ++i) rt::st<int32_t>(c, arg(c, 0) + 4 * uint64_t(i), i == 0 ? kUserId : -1);
        ret(c, 0);
    });
    reg("sceUserServiceGetUserName", [](Context& c) {  // (userId, char* name, size)
        const char* n = "Hunter";
        if (arg(c, 2) > 0) std::strncpy(ptr<char>(c, arg(c, 1)), n, arg(c, 2));
        ret(c, 0);
    });
    reg("sceUserServiceGetEvent", [](Context& c) { ret(c, 0x80960007); });  // SCE_USER_SERVICE_ERROR_NO_EVENT
    reg("sceUserServiceTerminate", ok);

    reg("scePadInit", ok);
    reg("scePadOpen", [](Context& c) { ret(c, 1); });
    reg("scePadClose", ok);
    reg("scePadReadState", pad_read_state);
    reg("scePadGetControllerInformation", pad_info);
    for (const char* n : {"scePadSetVibration", "scePadResetOrientation", "scePadSetAngularVelocityDeadbandState",
                          "scePadSetTiltCorrectionState"})
        reg(n, ok);

    reg("scePlayGoInitialize", ok);
    reg("scePlayGoOpen", [](Context& c) { rt::st<int32_t>(c, arg(c, 0), 1); ret(c, 0); });
    reg("scePlayGoSetInstallSpeed", ok);
    reg("scePlayGoGetLocus", [](Context& c) {  // (handle, chunkIds*, n, locus*): everything is local and fast
        for (uint64_t i = 0; i < arg(c, 2); ++i) rt::st<uint8_t>(c, arg(c, 3) + i, 3);
        ret(c, 0);
    });
    reg("scePlayGoGetChunkId", [](Context& c) { if (arg(c, 3)) rt::st<uint32_t>(c, arg(c, 3), 0); ret(c, 0); });
}

} // namespace bb::hle
