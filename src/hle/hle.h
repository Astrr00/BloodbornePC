// SPDX-License-Identifier: GPL-3.0-or-later
// HLE helpers: guest SysV ABI argument access and registration by symbol name (NID computed at startup).
#pragma once
#include <cstdint>
#include <functional>
#include <string>

#include "core/orbis_elf.h"
#include "runtime/guest.h"

namespace bb::hle {

using rt::Context;

// i-th integer argument (rdi, rsi, rdx, rcx, r8, r9, then the stack above the pushed return address).
inline uint64_t arg(Context& c, int i) {
    static constexpr int kReg[6] = {7, 6, 2, 1, 8, 9};
    return i < 6 ? c.r[kReg[i]] : rt::ld<uint64_t>(c, c.r[4] + 8 + 8 * uint64_t(i - 6));
}
// Narrow guest parameters (int/short/char/bool) arrive with undefined upper register bits: read them through these.
inline uint32_t arg32(Context& c, int i) { return uint32_t(arg(c, i)); }
inline int32_t argi(Context& c, int i) { return int32_t(arg(c, i)); }
inline uint16_t arg16(Context& c, int i) { return uint16_t(arg(c, i)); }
inline uint8_t arg8(Context& c, int i) { return uint8_t(arg(c, i)); }
inline void ret(Context& c, uint64_t v) { c.r[0] = v; }
// Guest address as host pointer (identity mapping: Context::base is 0 in the game, a buffer offset in tests).
template <class T = void>
inline T* ptr(Context& c, uint64_t a) { return reinterpret_cast<T*>(rt::host(c, a)); }

void reg(const char* symbol, rt::GuestFn fn);  // registers under generate_nid(symbol)

// Orbis error codes returned by libkernel.
inline constexpr uint64_t kErrBusy = 0x80020010, kErrInval = 0x80020016, kErrNoMem = 0x8002000C,
                          kErrTimedOut = 0x8002003C, kErrSrch = 0x80020003;

// Process-wide state set once before the guest starts.
struct Process {
    const elf::Image* image = nullptr;
    uint64_t canary = 0;
};
Process& process();

// Maps `len` bytes of zeroed read/write memory at `hint` (exactly, if non-null) or anywhere; nullptr on failure.
// Identity mapping: the host address is the guest address.
void* host_map(void* hint, uint64_t len);

// Guest mappings created through sceKernelMap*Memory: queries (sceKernelQueryMemoryProtection, VirtualQuery) report the
// protection the guest asked for, e.g. GPU-readable (0x10) blocks that the engine verifies before using them.
struct Mapping {
    uint64_t start, end;
    int prot;
};
void note_mapping(uint64_t start, uint64_t len, int prot);
bool find_mapping(uint64_t addr, Mapping& out);

// Host thread with a large stack: generated code recurses on the host stack, once per guest call frame.
class HostThread {
public:
    ~HostThread();
    bool start(uint64_t stack_bytes, std::function<void()> fn);
    void join();
private:
    void* handle_ = nullptr;
};

// Library slices (one file each); register_all calls them all.
void register_libc();
void register_libc_wrapped();
void register_kernel();
void register_thread();
void register_system();
void register_net();
void register_np();
void register_audio();
void register_vfs();
void register_gnm();
// Called from sceVideoOutSubmitFlip / sceGnmSubmitAndFlipCommandBuffers with the display buffer index (any thread).
using FlipHook = void (*)(uint32_t buffer_index, uint64_t guest_addr, uint32_t width, uint32_t height);  // addr of the flipped display buffer (0 if unregistered)
void set_flip_hook(FlipHook hook);
void install_gpu_mem_hook();
struct PadInput { uint32_t buttons; uint8_t lx, ly, rx, ry, l2, r2; bool connected; };
using PadProvider = PadInput (*)();
void set_pad_provider(PadProvider p);  // host input source for scePadReadState (neutral pad if unset)  // gpu::hooks().mem_valid from the mapping table
// Host audio device: one stream per AudioOut port (SDL3 in the game; unset = ports are only paced, samples dropped).
struct AudioSink {
    int (*open)(uint32_t rate, uint32_t channels, bool is_float);  // stream id or -1
    void (*write)(int id, const void* data, uint32_t bytes);
    void (*close)(int id);
};
void set_audio_sink(const AudioSink& sink);
void print_gnm_stats();  // PM4 opcode histogram of submitted command buffers (BB_PM4_STATS=1)
void register_input();
void register_services();
void register_savedata();
void register_ajm();
void register_ime();
void set_save_root(const std::string& dir);  // host directory holding all save data (<root>/<user>/<title>/<dir>)
void register_all();

// Prints guest registers and a guest backtrace for fatal host exceptions (access violation etc.).
void install_crash_handler();
uint64_t current_os_thread_id();
// Debug: after `seconds`, dump every registered guest thread and exit(3). Enabled by BB_HANG_SECS in bbgame.
void start_hang_watchdog(unsigned seconds);
// Debug (BB_TRACE_COND): per thread the cond slot it waits on and the signals that slot received since the wait began.
void print_cond_waiters();
// Debug: sample the main guest thread's host RIP for `seconds` and print "PROFILE <exe rva> <count>" lines (BB_PROFILE_SECS).
void start_profiler(unsigned seconds);

// Sleep for `us` microseconds without rounding up to the OS timer tick (guest polling loops depend on short sleeps).
void precise_sleep_us(uint64_t us);

// Call guest code from an HLE function on the current thread: `fn(a0, a1, a2, a3)`, returns rax. Only rsp is restored; the
// callee clobbers caller-saved registers like any guest call.
uint64_t call_guest(Context& c, uint64_t fn, uint64_t a0 = 0, uint64_t a1 = 0, uint64_t a2 = 0, uint64_t a3 = 0);

// printf-style formatting of the guest format string `fmt`; `va_list_addr` is a SysV va_list in guest memory.
struct GuestVaList {
    uint32_t gp_offset, fp_offset;
    uint64_t overflow_arg_area, reg_save_area;
};
std::string format_guest(Context& c, uint64_t fmt, uint64_t va_list_addr);

} // namespace bb::hle
