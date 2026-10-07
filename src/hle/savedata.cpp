// SPDX-License-Identifier: GPL-3.0-or-later
// libSceSaveData: save directories live on the host as <root>/<user id>/<TITLE_ID>/<dirName>/ and are mounted into the
// guest VFS as /savedata<N>. "Save data memory" (the small fixed-size blob API) is a single file per title.
// Struct layouts follow the SDK headers as documented by shadPS4 (src/core/libraries/save_data/savedata.cpp).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "core/sfo.h"
#include "hle/abi.h"
#include "hle/vfs.h"

namespace bb::hle {
namespace {

namespace fs = std::filesystem;

constexpr int64_t kErrParameter = int32_t(0x809F0000), kErrNotInitialized = int32_t(0x809F0001), kErrNotMounted = int32_t(0x809F0004),
                  kErrExists = int32_t(0x809F0007), kErrNotFound = int32_t(0x809F0008), kErrInternal = int32_t(0x809F000B),
                  kErrMountFull = int32_t(0x809F000C);
constexpr uint32_t kModeRdOnly = 1, kModeRdWr = 2, kModeCreate = 4, kModeCreate2 = 32;
constexpr uint64_t kBlockSize = 32768;

std::mutex g_m;
void trace(const char* fn, Context& c) {
    static const bool on = std::getenv("BB_SAVE_LOG") != nullptr;
    if (on) std::fprintf(stderr, "savedata: %s(0x%llx, 0x%llx, 0x%llx, 0x%llx)\n", fn, (unsigned long long)arg(c, 0), (unsigned long long)arg(c, 1), (unsigned long long)arg(c, 2), (unsigned long long)arg(c, 3));
}
std::string g_root;
bool g_init = false;
struct Slot { bool used = false; std::string dir; fs::path host; uint64_t blocks = 0; };
Slot g_slots[16];

std::string title_id(Context& c) {
    static std::string id;
    if (!id.empty()) return id;
    id = "CUSA00000";
    if (auto p = vfs_resolve("/app0/sce_sys/param.sfo"))
        if (auto sfo = load_sfo(*p))
            if (auto t = sfo->str("TITLE_ID"); t && t->size() == 9) id = *t;
    (void)c;
    return id;
}

fs::path root() {
    if (!g_root.empty()) return g_root;
    if (const char* e = std::getenv("BB_SAVE_DIR")) return e;
    return fs::current_path() / "savedata";
}

fs::path title_dir(Context& c, int32_t user) { return root() / std::to_string(user) / title_id(c); }

std::string cstr(Context& c, uint64_t at, size_t max) {
    if (!at) return {};
    const char* p = ptr<const char>(c, at);
    size_t n = 0;
    while (n < max && p[n]) ++n;
    return std::string(p, n);
}

bool safe_name(const std::string& n) { return !n.empty() && n.find_first_of("/\\") == std::string::npos && n != "." && n != ".."; }

uint64_t dir_bytes(const fs::path& d) {
    uint64_t t = 0;
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(d, ec); !ec && it != fs::recursive_directory_iterator(); it.increment(ec))
        if (it->is_regular_file(ec)) t += it->file_size(ec);
    return t;
}

// sceSaveDataMount(const Mount*, MountResult*): userId@0, titleId*@8, dirName*@16, fingerprint*@24, blocks@32, mode@40
void sd_mount(Context& c) {
    trace("sd_mount", c);
    std::lock_guard lk(g_m);
    if (!g_init) { ret(c, uint64_t(kErrNotInitialized)); return; }
    const uint64_t m = arg(c, 0), res = arg(c, 1);
    if (!m || !res) { ret(c, uint64_t(kErrParameter)); return; }
    const int32_t user = rt::ld<int32_t>(c, m);
    const std::string dir = cstr(c, rt::ld<uint64_t>(c, m + 16), 32);
    const uint64_t blocks = rt::ld<uint64_t>(c, m + 32);
    const uint32_t mode = rt::ld<uint32_t>(c, m + 40);
    if (std::getenv("BB_SAVE_LOG")) std::fprintf(stderr, "savedata: mount user=%d dir='%s' blocks=%llu mode=0x%x\n", user, dir.c_str(), (unsigned long long)blocks, mode);
    if (!safe_name(dir)) { ret(c, uint64_t(kErrParameter)); return; }
    const fs::path host = title_dir(c, user) / dir;
    std::error_code ec;
    const bool exists = fs::is_directory(host, ec);
    uint32_t status = 0;
    if (!exists) {
        if (!(mode & (kModeCreate | kModeCreate2))) { ret(c, uint64_t(kErrNotFound)); return; }
        fs::create_directories(host, ec);
        if (ec) { ret(c, uint64_t(kErrInternal)); return; }
        status = 1;
    } else if ((mode & kModeCreate) && !(mode & kModeCreate2)) {
        ret(c, uint64_t(kErrExists));
        return;
    }
    int slot = -1;
    for (int i = 0; i < 16; ++i)
        if (!g_slots[i].used) { slot = i; break; }
    if (slot < 0) { ret(c, uint64_t(kErrMountFull)); return; }
    g_slots[slot] = {true, dir, host, blocks};
    const std::string mp = "/savedata" + std::to_string(slot);
    vfs_mount(mp, host);
    char out[16] = {};
    std::memcpy(out, mp.c_str(), mp.size());
    std::memcpy(ptr<char>(c, res), out, 16);
    rt::st<uint64_t>(c, res + 16, 0);       // required blocks
    rt::st<uint32_t>(c, res + 24, 0);
    rt::st<uint32_t>(c, res + 28, status);  // 1 = created
    (void)kModeRdOnly; (void)kModeRdWr;
    ret(c, 0);
}

int slot_of(Context& c, uint64_t mount_point) {
    const std::string mp = cstr(c, mount_point, 16);
    for (int i = 0; i < 16; ++i)
        if (g_slots[i].used && mp == "/savedata" + std::to_string(i)) return i;
    return -1;
}

void sd_umount(Context& c) {
    trace("sd_umount", c);
    std::lock_guard lk(g_m);
    const int s = slot_of(c, arg(c, 0));
    if (s < 0) { ret(c, uint64_t(kErrNotMounted)); return; }
    vfs_unmount("/savedata" + std::to_string(s));
    g_slots[s] = {};
    ret(c, 0);
}

void sd_mount_info(Context& c) {
    trace("sd_mount_info", c);  // (const MountPoint*, MountInfo*): blocks@0, freeBlocks@8
    std::lock_guard lk(g_m);
    const int s = slot_of(c, arg(c, 0));
    if (s < 0 || !arg(c, 1)) { ret(c, uint64_t(s < 0 ? kErrNotMounted : kErrParameter)); return; }
    const uint64_t total = std::max<uint64_t>(g_slots[s].blocks, 1024), used = (dir_bytes(g_slots[s].host) + kBlockSize - 1) / kBlockSize;
    rt::st<uint64_t>(c, arg(c, 1), total);
    rt::st<uint64_t>(c, arg(c, 1) + 8, total > used ? total - used : 0);
    ret(c, 0);
}

// OrbisSaveDataParam: title[128]@0 subTitle[128]@128 detail[1024]@256 userParam@1280 mtime@1288; sizeof 1328
constexpr uint64_t kParamSize = 1328;

void sd_set_param(Context& c) {
    trace("sd_set_param", c);  // (mountPoint*, paramType, buf, bufSize)
    std::lock_guard lk(g_m);
    const int s = slot_of(c, arg(c, 0));
    if (s < 0) { ret(c, uint64_t(kErrNotMounted)); return; }
    const uint32_t type = arg32(c, 1);
    const uint64_t buf = arg(c, 2), size = arg(c, 3);
    std::vector<char> param(kParamSize, 0);
    const fs::path file = g_slots[s].host / "sce_sys" / "param.bin";
    {
        std::ifstream f(file, std::ios::binary);
        if (f) f.read(param.data(), std::streamsize(param.size()));
    }
    const char* src = ptr<const char>(c, buf);
    auto put = [&](size_t off, size_t max) { std::memset(&param[off], 0, max); std::memcpy(&param[off], src, std::min<size_t>(size, max)); };
    switch (type) {
        case 0: std::memcpy(param.data(), src, std::min<size_t>(size, kParamSize)); break;
        case 1: put(0, 128); break;
        case 2: put(128, 128); break;
        case 3: put(256, 1024); break;
        case 4: std::memcpy(&param[1280], src, std::min<size_t>(size, 4)); break;
        default: ret(c, uint64_t(kErrParameter)); return;
    }
    const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::memcpy(&param[1288], &now, 8);
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    std::ofstream o(file, std::ios::binary | std::ios::trunc);
    o.write(param.data(), std::streamsize(param.size()));
    ret(c, 0);
}

void sd_save_icon(Context& c) {
    trace("sd_save_icon", c);  // (mountPoint*, Icon*): buf*@0 bufSize@8 dataSize@16
    std::lock_guard lk(g_m);
    const int s = slot_of(c, arg(c, 0));
    if (s < 0 || !arg(c, 1)) { ret(c, uint64_t(s < 0 ? kErrNotMounted : kErrParameter)); return; }
    const uint64_t ic = arg(c, 1);
    const uint64_t buf = rt::ld<uint64_t>(c, ic), size = std::min(rt::ld<uint64_t>(c, ic + 8), rt::ld<uint64_t>(c, ic + 16));
    const fs::path file = g_slots[s].host / "sce_sys" / "icon0.png";
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    std::ofstream o(file, std::ios::binary | std::ios::trunc);
    o.write(ptr<const char>(c, buf), std::streamsize(size));
    ret(c, 0);
}

void sd_dir_search(Context& c) {
    trace("sd_dir_search", c);  // (Cond*, Result*)
    std::lock_guard lk(g_m);
    if (!g_init) { ret(c, uint64_t(kErrNotInitialized)); return; }
    const uint64_t cond = arg(c, 0), res = arg(c, 1);
    if (!cond || !res) { ret(c, uint64_t(kErrParameter)); return; }
    const int32_t user = rt::ld<int32_t>(c, cond);
    std::string pat = cstr(c, rt::ld<uint64_t>(c, cond + 16), 32);
    const uint32_t key = rt::ld<uint32_t>(c, cond + 24), order = rt::ld<uint32_t>(c, cond + 28);
    struct Hit { std::string name; fs::file_time_type mtime; uint64_t bytes; };
    std::vector<Hit> hits;
    std::error_code ec;
    const bool prefix = !pat.empty() && pat.back() == '%';
    if (prefix) pat.pop_back();
    for (auto it = fs::directory_iterator(title_dir(c, user), ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_directory(ec)) continue;
        const std::string n = it->path().filename().string();
        if (n.size() > 31) continue;
        if (!pat.empty() && (prefix ? n.compare(0, pat.size(), pat) != 0 : n != pat)) continue;
        hits.push_back({n, it->last_write_time(ec), dir_bytes(it->path())});
    }
    std::sort(hits.begin(), hits.end(), [&](const Hit& a, const Hit& b) {
        const bool lt = key == 3 ? a.mtime < b.mtime : key == 2 ? a.bytes < b.bytes : a.name < b.name;
        const bool gt = key == 3 ? b.mtime < a.mtime : key == 2 ? b.bytes < a.bytes : b.name < a.name;
        return order == 1 ? gt : lt;
    });
    const uint64_t names = rt::ld<uint64_t>(c, res + 8), params = rt::ld<uint64_t>(c, res + 24), infos = rt::ld<uint64_t>(c, res + 32);
    const uint32_t cap = rt::ld<uint32_t>(c, res + 16);
    uint32_t set = 0;
    for (size_t i = 0; i < hits.size() && names && set < cap; ++i, ++set) {
        char nm[32] = {};
        std::memcpy(nm, hits[i].name.c_str(), hits[i].name.size());
        std::memcpy(ptr<char>(c, names + 32ull * set), nm, 32);
        if (params) {
            std::vector<char> p(kParamSize, 0);
            std::ifstream f(title_dir(c, user) / hits[i].name / "sce_sys" / "param.bin", std::ios::binary);
            if (f) f.read(p.data(), std::streamsize(p.size()));
            std::memcpy(ptr<char>(c, params + kParamSize * set), p.data(), p.size());
        }
        if (infos) {
            rt::st<uint64_t>(c, infos + 48ull * set, 1024);
            rt::st<uint64_t>(c, infos + 48ull * set + 8, 1024 - std::min<uint64_t>(1024, (hits[i].bytes + kBlockSize - 1) / kBlockSize));
        }
    }
    rt::st<uint32_t>(c, res, uint32_t(hits.size()));
    rt::st<uint32_t>(c, res + 20, set);
    ret(c, 0);
}

void sd_delete(Context& c) {
    trace("sd_delete", c);  // (const Delete*): userId@0, titleId*@8, dirName*@16
    std::lock_guard lk(g_m);
    const uint64_t d = arg(c, 0);
    if (!d) { ret(c, uint64_t(kErrParameter)); return; }
    const std::string dir = cstr(c, rt::ld<uint64_t>(c, d + 16), 32);
    if (!safe_name(dir)) { ret(c, uint64_t(kErrParameter)); return; }
    std::error_code ec;
    const fs::path host = title_dir(c, rt::ld<int32_t>(c, d)) / dir;
    if (!fs::is_directory(host, ec)) { ret(c, uint64_t(kErrNotFound)); return; }
    fs::remove_all(host, ec);
    ret(c, ec ? uint64_t(kErrInternal) : 0);
}

// Save data memory: one file per (user, title).
fs::path memory_file(Context& c, int32_t user) { return title_dir(c, user) / "memory.dat"; }

void sd_memory_setup(Context& c) {
    trace("sd_memory_setup", c);  // (userId, memorySize, Param*)
    std::lock_guard lk(g_m);
    const fs::path f = memory_file(c, argi(c, 0));
    std::error_code ec;
    fs::create_directories(f.parent_path(), ec);
    if (!fs::exists(f, ec)) std::ofstream(f, std::ios::binary).put(0);
    if (fs::file_size(f, ec) < arg(c, 1)) fs::resize_file(f, arg(c, 1), ec);
    ret(c, 0);
}

void sd_memory_get(Context& c) {
    trace("sd_memory_get", c);  // (userId, buf, bufSize, offset)
    std::lock_guard lk(g_m);
    std::ifstream f(memory_file(c, argi(c, 0)), std::ios::binary);
    if (!f) { ret(c, uint64_t(kErrNotFound)); return; }
    f.seekg(std::streamoff(arg(c, 3)));
    std::memset(ptr<char>(c, arg(c, 1)), 0, size_t(arg(c, 2)));
    f.read(ptr<char>(c, arg(c, 1)), std::streamsize(arg(c, 2)));
    ret(c, 0);
}

void sd_memory_set(Context& c) {
    trace("sd_memory_set", c);
    std::lock_guard lk(g_m);
    const fs::path path = memory_file(c, argi(c, 0));
    std::error_code ec;
    fs::create_directories(path.parent_path(), ec);
    std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!f) { std::ofstream(path, std::ios::binary).close(); f.open(path, std::ios::binary | std::ios::in | std::ios::out); }
    if (!f) { ret(c, uint64_t(kErrInternal)); return; }
    f.seekp(std::streamoff(arg(c, 3)));
    f.write(ptr<const char>(c, arg(c, 1)), std::streamsize(arg(c, 2)));
    ret(c, 0);
}

} // namespace

void set_save_root(const std::string& dir) { g_root = dir; }

void register_savedata() {
    reg("sceSaveDataInitialize", [](Context& c) { trace("Initialize", c); std::lock_guard lk(g_m); g_init = true; ret(c, 0); });
    reg("sceSaveDataTerminate", [](Context& c) { std::lock_guard lk(g_m); g_init = false; ret(c, 0); });
    reg("sceSaveDataMount", sd_mount);
    reg("sceSaveDataUmount", sd_umount);
    reg("sceSaveDataGetMountInfo", sd_mount_info);
    reg("sceSaveDataSetParam", sd_set_param);
    reg("sceSaveDataSaveIcon", sd_save_icon);
    reg("sceSaveDataDirNameSearch", sd_dir_search);
    reg("sceSaveDataDelete", sd_delete);
    reg("sceSaveDataSetupSaveDataMemory", sd_memory_setup);
    reg("sceSaveDataGetSaveDataMemory", sd_memory_get);
    reg("sceSaveDataSetSaveDataMemory", sd_memory_set);
    // Dialogs: nothing to show; they report FINISHED immediately (status enum: 0 none, 1 initialized, 2 running, 3 finished).
    reg("sceSaveDataDialogInitialize", [](Context& c) { ret(c, 0); });
    reg("sceSaveDataDialogTerminate", [](Context& c) { ret(c, 0); });
    reg("sceSaveDataDialogOpen", [](Context& c) { ret(c, 0); });
    reg("sceSaveDataDialogUpdateStatus", [](Context& c) { ret(c, 3); });
}

} // namespace bb::hle
