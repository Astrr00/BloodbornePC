// SPDX-License-Identifier: GPL-3.0-or-later
// Offline network stack: the console is "not connected". Sockets cannot be created, NetCtl reports disconnected,
// byte-order helpers work. Np*/Http answers live in np.cpp.
#include <atomic>
#include <cstring>

#include "hle/abi.h"
#include "hle/guest_heap.h"

namespace bb::hle {
namespace {

constexpr uint16_t swap16(uint16_t v) { return uint16_t((v << 8) | (v >> 8)); }
constexpr uint32_t swap32(uint32_t v) { return (uint32_t(swap16(uint16_t(v))) << 16) | swap16(uint16_t(v >> 16)); }

constexpr int kENetDown = 50;  // FreeBSD errno

int* errno_slot() { return guest_errno(); }

int64_t fail_net(Context&) {
    *errno_slot() = kENetDown;
    return -1;
}

std::atomic<uint64_t> g_ctl_cb{0}, g_ctl_arg{0};
std::atomic<bool> g_ctl_pending{false};

} // namespace

void register_net() {
    reg("sceNetHtons", wrap<+[](uint16_t v) -> uint16_t { return swap16(v); }>);
    reg("sceNetNtohs", wrap<+[](uint16_t v) -> uint16_t { return swap16(v); }>);
    reg("sceNetHtonl", wrap<+[](uint32_t v) -> uint32_t { return swap32(v); }>);
    reg("sceNetNtohl", wrap<+[](uint32_t v) -> uint32_t { return swap32(v); }>);
    reg("sceNetErrnoLoc", wrap<+[]() { return errno_slot(); }>);
    reg("sceNetInit", [](Context& c) { ret(c, 0); });
    reg("sceNetTerm", [](Context& c) { ret(c, 0); });
    reg("sceNetPoolCreate", [](Context& c) { ret(c, 1); });
    reg("sceNetPoolDestroy", [](Context& c) { ret(c, 0); });
    for (const char* n : {"sceNetSocket", "sceNetSetsockopt", "sceNetGetsockopt", "sceNetBind", "sceNetConnect",
                          "sceNetListen", "sceNetAccept", "sceNetSend", "sceNetSendto", "sceNetRecv", "sceNetRecvfrom",
                          "sceNetShutdown", "sceNetSocketClose", "sceNetSocketAbort", "sceNetGetsockname",
                          "sceNetEpollCreate", "sceNetEpollControl", "sceNetEpollWait", "sceNetEpollAbort",
                          "sceNetEpollDestroy", "sceNetResolverCreate", "sceNetResolverStartNtoa",
                          "sceNetResolverStartAton", "sceNetResolverDestroy", "sceNetInetNtop", "sceNetInetPton"})
        reg(n, [](Context& c) { ret(c, uint64_t(fail_net(c))); });
    reg("sceNetCtlGetState", [](Context& c) { rt::st<int32_t>(c, arg(c, 0), 0); ret(c, 0); });  // disconnected
    reg("sceNetCtlRegisterCallback", [](Context& c) {  // (callback(eventType, arg), arg, int* cid)
        g_ctl_cb = arg(c, 0);
        g_ctl_arg = arg(c, 1);
        g_ctl_pending = g_ctl_cb != 0;
        rt::st<int32_t>(c, arg(c, 2), 0);
        ret(c, 0);
    });
    reg("sceNetCtlUnregisterCallback", [](Context& c) { ret(c, 0); });
    reg("sceNetCtlCheckCallback", [](Context& c) {  // reports the (never changing) disconnected state once after registration
        if (g_ctl_pending.exchange(false)) call_guest(c, g_ctl_cb, 1, g_ctl_arg);  // SCE_NET_CTL_EVENT_TYPE_DISCONNECTED
        ret(c, 0);
    });
    for (const char* n : {"sceNetCtlGetInfo", "sceNetCtlGetNatInfo"})
        reg(n, [](Context& c) { ret(c, 0x80412101); });  // SCE_NET_CTL_ERROR_NOT_CONNECTED
}

} // namespace bb::hle
