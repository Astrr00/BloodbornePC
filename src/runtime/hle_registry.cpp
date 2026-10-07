// SPDX-License-Identifier: GPL-3.0-or-later
// HLE registry: import NID -> host implementation. Implementations follow the guest SysV ABI through Context
// (args in rdi, rsi, rdx, rcx, r8, r9; result in rax). Libraries register their functions at startup (M1.3);
// imports without an entry are reported by call_indirect with library, NID and name.
#include <string>
#include <unordered_map>

#include "runtime/guest.h"

namespace bb::rt {
namespace {
std::unordered_map<std::string, GuestFn>& table() {
    static std::unordered_map<std::string, GuestFn> t;
    return t;
}
} // namespace

void register_hle(const char* nid, GuestFn fn) { table()[nid] = fn; }

GuestFn find_hle(const char* nid) {
    auto it = table().find(nid);
    return it != table().end() ? it->second : nullptr;
}

} // namespace bb::rt
