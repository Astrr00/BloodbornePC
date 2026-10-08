// SPDX-License-Identifier: GPL-3.0-or-later
// bbgame: host program for the fully recompiled eboot (local build, needs your own dump: cmake -DBB_EBOOT=<path>).
// Guest memory is identity-mapped at the load bias (Context::base = 0): [image][import cells][TLS][heap][stack].
// Runs DT_INIT and the entry point; imports without an HLE implementation abort with library, NID and name.
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

#include <atomic>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <string>

#include "core/bbconfig.h"
#include "core/orbis_elf.h"
#include "core/util.h"
#include "hle/hle.h"
#include "gpu/audio_out.h"
#include "gpu/backend.h"
#include "gpu/pad.h"
#include "gpu/present.h"
#include "gpu/vk_context.h"
#include "hle/vfs.h"
#include "runtime/guest.h"
#include "runtime/loader.h"
#include "runtime/process.h"

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // The eboot this binary was recompiled from (hash recorded by CMake at configure time): the launcher matches dumps to builds with it.
    if (argc == 2 && std::string(argv[1]) == "--eboot-hash") {
#ifdef BB_BUILD_EBOOT_SHA256
        std::puts(BB_BUILD_EBOOT_SHA256);
        return 0;
#else
        return 1;
#endif
    }
    // bbconfig.ini (launcher settings): fills BB_* variables that are not already set, before any option is read.
    if (const std::string cfg = bb::load_config_into_env(bb::exe_dir()); !cfg.empty()) std::puts(cfg.c_str());
    if (argc < 2) {
        std::fputs("usage: bbgame <eboot.bin> [--no-run] | --eboot-hash\n", stderr);
        return 2;
    }
    const bool run = !(argc > 2 && std::string(argv[2]) == "--no-run");
    auto file = bb::read_file(argv[1]);
    if (!file) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    auto parsed = bb::elf::parse(*file);
    if (auto* e = std::get_if<bb::elf::Error>(&parsed)) {
        std::fprintf(stderr, "error: %s\n", e->message.c_str());
        return 1;
    }
    const auto& img = std::get<bb::elf::Image>(parsed);
    const uint64_t bias = bb::rt::kLoadBias;

    constexpr uint64_t kHeap = 64ull << 20, kStack = 8ull << 20, kAlign = 0xFFFF;
    const uint64_t cells_hi = bb::rt::import_cells_end(img, bias);
    const uint64_t tls = (cells_hi + kAlign) & ~kAlign;
    const uint64_t heap = (tls + bb::rt::tls_block_size(img) + kAlign) & ~kAlign;
    const uint64_t stack_lo = heap + kHeap, end = stack_lo + kStack;
    void* at = bb::hle::host_map(reinterpret_cast<void*>(bias), end - bias);  // fixed address, committed (VirtualAlloc / mmap MAP_FIXED_NOREPLACE)
    if (at != reinterpret_cast<void*>(bias)) {
        std::fprintf(stderr, "cannot reserve guest memory at 0x%llx (%llu MiB)\n", (unsigned long long)bias,
                     (unsigned long long)((end - bias) >> 20));
        return 1;
    }
    bb::rt::Context c;
    c.base = 0;
    std::string error;
    if (!bb::rt::load_image(c, img, bias, end, error)) {
        std::fprintf(stderr, "load_image: %s\n", error.c_str());
        return 1;
    }
    if (!bb::rt::setup_main_thread(c, img, bias, end - 0x1000, tls, error)) {
        std::fprintf(stderr, "setup_main_thread: %s\n", error.c_str());
        return 1;
    }

    std::printf("image: bias 0x%llx, end 0x%llx, entry 0x%llx (guest 0x%llx)\n", (unsigned long long)bias,
                (unsigned long long)(bb::rt::image_end(img) + bias), (unsigned long long)img.entry,
                (unsigned long long)(img.entry + bias));
    std::printf("guest memory: [0x%llx, 0x%llx) cells..0x%llx tls 0x%llx heap 0x%llx stack top 0x%llx fs 0x%llx\n",
                (unsigned long long)bias, (unsigned long long)end, (unsigned long long)cells_hi,
                (unsigned long long)tls, (unsigned long long)heap, (unsigned long long)c.r[4],
                (unsigned long long)c.fs_base);
    std::printf("recompiled functions: %zu, imports: %zu\n", bb::rt::kFunctionCount, bb::rt::kImportCount);
    bb::hle::install_crash_handler();
    if (const char* prof = std::getenv("BB_PROFILE_SECS")) bb::hle::start_profiler(unsigned(std::atoi(prof)));
    if (const char* hang = std::getenv("BB_HANG_SECS")) bb::hle::start_hang_watchdog(unsigned(std::atoi(hang)));
    // /app0 = the directory holding eboot.bin (the dump's Image0 folder, with dvdroot_ps4 next to it); override with --app0.
    std::filesystem::path app0 = std::filesystem::path(argv[1]).parent_path();
    for (int i = 2; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "--app0") app0 = argv[i + 1];
    bb::hle::vfs_mount("/app0", app0);
    // PKG-extractor layout: Image0/ (app files) next to Sc0/ (system files) -> /app0/sce_sys. An update's Image0/sce_sys/ (keystone,
    // about/) has no param.sfo, so the mount depends on that file, not on the folder.
    if (std::filesystem::is_regular_file(app0.parent_path() / "Sc0" / "param.sfo") && !std::filesystem::is_regular_file(app0 / "sce_sys" / "param.sfo"))
        bb::hle::vfs_mount("/app0/sce_sys", app0.parent_path() / "Sc0");
    bb::hle::register_all();
#ifdef _WIN32
    if (const char* oc = std::getenv("BB_ONE_CORE"))  // diagnosis: run every guest thread on one host core (is a crash concurrency-related?)
        SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1) << std::atoi(oc));
#endif
    bb::hle::process() = {&img, bb::rt::ProcessConfig{}.canary};
    if (!run) return 0;
    std::printf("running DT_INIT + entry ...\n");
    // Generated code recurses on the host stack, once per guest call frame: run on a thread with a large stack.
    const bool headless = std::getenv("BB_HEADLESS") != nullptr;
    std::atomic<bool> guest_done{false};
    const bool no_gpu = std::getenv("BB_NO_GPU") != nullptr;  // PM4 is parsed but nothing is executed
    if (!headless) {
        int win_w = 1280, win_h = 720;  // BB_WINDOW=3840x2160 etc.; the window is resizable, the picture scales to it
        if (const char* ws = std::getenv("BB_WINDOW")) { int w = 0, h = 0; if (std::sscanf(ws, "%dx%d", &w, &h) == 2 && w >= 320 && h >= 200) { win_w = w; win_h = h; } }
        if (!bb::gpu::init(win_w, win_h, "BloodbornePC")) return 1;
        bb::hle::set_pad_provider(+[]() {
            const bb::gpu::PadSnapshot s = bb::gpu::pad_snapshot();
            return bb::hle::PadInput{s.buttons, s.lx, s.ly, s.rx, s.ry, s.l2, s.r2, s.connected};
        });
        if (!std::getenv("BB_BACKGROUND") || std::getenv("BB_AUDIO_DUMP"))  // background test runs stay silent (audio_open only records)
            bb::hle::set_audio_sink({+[](uint32_t rate, uint32_t channels, bool is_float) { return bb::gpu::audio_open(rate, channels, is_float); },
                                     +[](int id, const void* data, uint32_t bytes) { bb::gpu::audio_write(id, data, bytes); },
                                     +[](int id) { bb::gpu::audio_close(id); }});
        bb::hle::set_flip_hook(+[](uint32_t index, uint64_t addr, uint32_t w, uint32_t h) { bb::gpu::request_present(index, addr, w, h); });
    } else if (!no_gpu) {
        if (!bb::gpu::vk_create_instance(nullptr, 0) || !bb::gpu::vk_create_device(VK_NULL_HANDLE)) return 1;
    }
    if (!no_gpu) {
        bb::hle::install_gpu_mem_hook();
        if (!bb::gpu::backend_init()) return 1;
    }
    bb::hle::HostThread main_thread;
    if (!main_thread.start(512ull << 20, [&] {
            bb::rt::current_context = &c;
            bb::rt::register_debug_thread("main", bb::hle::current_os_thread_id());
            bb::rt::run_entry(c, img, bias);
            guest_done = true;
        })) {
        std::fputs("cannot start the main guest thread\n", stderr);
        return 1;
    }
    if (!headless) {
        bb::gpu::run(guest_done);  // returns when the window is closed or the guest ended
        if (!guest_done) std::_Exit(0);  // hard exit by design: window closed, guest threads/Vulkan are torn down by the OS (no orderly shutdown yet)
        bb::gpu::shutdown();
    }
    main_thread.join();
    std::printf("entry returned\n");
    return 0;
}
