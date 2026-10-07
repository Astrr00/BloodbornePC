// SPDX-License-Identifier: GPL-3.0-or-later
// libkernel file I/O (open/read/write/lseek/stat/dir) and Fios2 over the VFS.
#include "hle/vfs.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <vector>

#include "hle/abi.h"
#include "hle/guest_heap.h"

#ifdef _WIN32
#define bb_fseek _fseeki64
#define bb_ftell _ftelli64
#else
#define bb_fseek fseeko
#define bb_ftell ftello
#endif

namespace bb::hle {
namespace fs = std::filesystem;
namespace {

std::map<std::string, fs::path>& mounts() {
    static std::map<std::string, fs::path> m;
    return m;
}

constexpr uint64_t kEnoent = 0x80020002, kEbadf = 0x80020009, kEexist = 0x80020011, kEnotdir = 0x80020014,
                   kEisdir = 0x80020015, kEacces = 0x8002000D, kEio = 0x80020005;

struct OpenFile {
    FILE* file = nullptr;
    fs::path path;
    bool dir = false;
    std::vector<fs::directory_entry> entries;  // snapshot for getdirentries
    size_t next_entry = 0;
};
std::mutex g_fd_mutex;
std::vector<OpenFile> g_fds(3);  // 0..2 reserved (stdio)

OpenFile* get_fd(int64_t fd) {
    std::lock_guard lk(g_fd_mutex);
    return fd >= 3 && size_t(fd) < g_fds.size() && (g_fds[fd].file || g_fds[fd].dir) ? &g_fds[fd] : nullptr;
}
// ponytail: the table only grows; fds are reused through the free slot search, which is linear.
int64_t add_fd(OpenFile&& f) {
    std::lock_guard lk(g_fd_mutex);
    for (size_t i = 3; i < g_fds.size(); ++i)
        if (!g_fds[i].file && !g_fds[i].dir) { g_fds[i] = std::move(f); return int64_t(i); }
    g_fds.push_back(std::move(f));
    return int64_t(g_fds.size() - 1);
}

std::string guest_string(Context& c, uint64_t a) { return a ? std::string(ptr<const char>(c, a)) : std::string(); }
bool trace_io() {
    static const bool t = std::getenv("BB_TRACE_IO") != nullptr;
    return t;
}

// FreeBSD open flags.
constexpr int kOWrOnly = 1, kORdWr = 2, kOAppend = 8, kOCreat = 0x200, kOTrunc = 0x400, kOExcl = 0x800;

void k_open(Context& c) {  // (path, flags, mode)
    const std::string gp = guest_string(c, arg(c, 0));
    const int flags = int(arg(c, 1));
    auto host = vfs_resolve(gp);
    if (trace_io()) {
        static const auto t0 = std::chrono::steady_clock::now();
        std::fprintf(stderr, "io: [%.1fs] open '%s' flags=%x -> %s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), gp.c_str(), flags,
                     host ? host->string().c_str() : "(unmounted)");
    }
    if (!host) return ret(c, kEnoent);
    std::error_code ec;
    const bool is_dir = fs::is_directory(*host, ec);
    if (is_dir) {
        OpenFile f;
        f.dir = true;
        f.path = *host;
        for (auto& e : fs::directory_iterator(*host, ec)) f.entries.push_back(e);
        return ret(c, uint64_t(add_fd(std::move(f))));
    }
    const bool exists = fs::exists(*host, ec);
    const bool write = flags & (kOWrOnly | kORdWr);
    if (!exists && !(flags & kOCreat)) return ret(c, kEnoent);
    if (exists && (flags & kOCreat) && (flags & kOExcl)) return ret(c, kEexist);
    const char* mode = !write ? "rb" : (flags & kOTrunc) || !exists ? "w+b" : "r+b";
    if ((flags & kOAppend) && write) mode = "a+b";
    FILE* fp = std::fopen(host->string().c_str(), mode);
    if (!fp) return ret(c, write ? kEacces : kEnoent);
    OpenFile f;
    f.file = fp;
    f.path = *host;
    ret(c, uint64_t(add_fd(std::move(f))));
}
void k_close(Context& c) {
    std::lock_guard lk(g_fd_mutex);
    const int64_t fd = int64_t(argi(c, 0));
    if (fd < 3 || size_t(fd) >= g_fds.size() || !(g_fds[fd].file || g_fds[fd].dir)) return ret(c, kEbadf);
    if (g_fds[fd].file) std::fclose(g_fds[fd].file);
    g_fds[fd] = OpenFile{};
    ret(c, 0);
}
// Read/write return the byte count (>= 0) or a negative errno-style value (as uint64 two's complement it is the 0x8002xxxx code).
void k_read(Context& c) {  // (fd, buf, n)
    OpenFile* f = get_fd(int64_t(argi(c, 0)));
    if (!f || !f->file) return ret(c, kEbadf);
    ret(c, std::fread(ptr(c, arg(c, 1)), 1, arg(c, 2), f->file));
}
void k_write(Context& c) {
    const int64_t fd = int64_t(argi(c, 0));
    if (fd == 1 || fd == 2) {
        std::fwrite(ptr(c, arg(c, 1)), 1, arg(c, 2), fd == 1 ? stdout : stderr);
        return ret(c, arg(c, 2));
    }
    OpenFile* f = get_fd(fd);
    if (!f || !f->file) return ret(c, kEbadf);
    ret(c, std::fwrite(ptr(c, arg(c, 1)), 1, arg(c, 2), f->file));
}
void k_lseek(Context& c) {  // (fd, offset, whence)
    OpenFile* f = get_fd(int64_t(argi(c, 0)));
    if (!f || !f->file) return ret(c, kEbadf);
    static constexpr int kWhence[3] = {SEEK_SET, SEEK_CUR, SEEK_END};
    if (arg(c, 2) > 2 || bb_fseek(f->file, int64_t(arg(c, 1)), kWhence[arg(c, 2)]) != 0) return ret(c, 0x80020016);
    ret(c, uint64_t(bb_ftell(f->file)));
}

// SceKernelStat (FreeBSD struct stat, 0x78 bytes).
void fill_stat(Context& c, uint64_t out, const fs::path& p) {
    std::memset(ptr(c, out), 0, 0x78);
    std::error_code ec;
    const bool dir = fs::is_directory(p, ec);
    rt::st<uint16_t>(c, out + 8, dir ? 040755 : 0100644);
    rt::st<uint16_t>(c, out + 10, 1);
    const uint64_t size = dir ? 0 : fs::file_size(p, ec);
    rt::st<uint64_t>(c, out + 72, size);
    rt::st<uint64_t>(c, out + 80, (size + 511) / 512);
    rt::st<uint32_t>(c, out + 88, 512);
    auto t = fs::last_write_time(p, ec);
    const int64_t secs = std::chrono::duration_cast<std::chrono::seconds>(t.time_since_epoch()).count();
    for (uint64_t off : {24, 40, 56, 104}) rt::st<int64_t>(c, out + off, secs);
}
void k_stat(Context& c) {  // (path, SceKernelStat*)
    const std::string gp = guest_string(c, arg(c, 0));
    auto host = vfs_resolve(gp);
    std::error_code ec;
    if (trace_io()) std::fprintf(stderr, "io: stat '%s' -> %s\n", gp.c_str(), host ? host->string().c_str() : "(unmounted)");
    if (!host || !fs::exists(*host, ec)) return ret(c, kEnoent);
    fill_stat(c, arg(c, 1), *host);
    ret(c, 0);
}
void k_fstat(Context& c) {
    OpenFile* f = get_fd(int64_t(argi(c, 0)));
    if (!f) return ret(c, kEbadf);
    fill_stat(c, arg(c, 1), f->path);
    ret(c, 0);
}
void k_mkdir(Context& c) {
    auto host = vfs_resolve(guest_string(c, arg(c, 0)));
    std::error_code ec;
    if (!host) return ret(c, kEnoent);
    if (fs::exists(*host, ec)) return ret(c, kEexist);
    ret(c, fs::create_directory(*host, ec) ? 0 : kEnoent);
}
void k_rmdir(Context& c) {
    auto host = vfs_resolve(guest_string(c, arg(c, 0)));
    std::error_code ec;
    ret(c, host && fs::is_directory(*host, ec) && fs::remove(*host, ec) ? 0 : kEnoent);
}
void k_unlink(Context& c) {
    auto host = vfs_resolve(guest_string(c, arg(c, 0)));
    std::error_code ec;
    ret(c, host && fs::exists(*host, ec) && fs::remove(*host, ec) ? 0 : kEnoent);
}
void k_rename(Context& c) {
    auto a = vfs_resolve(guest_string(c, arg(c, 0))), b = vfs_resolve(guest_string(c, arg(c, 1)));
    std::error_code ec;
    if (!a || !b) return ret(c, kEnoent);
    fs::rename(*a, *b, ec);
    ret(c, ec ? kEnoent : 0);
}
void k_ftruncate(Context& c) {
    OpenFile* f = get_fd(int64_t(argi(c, 0)));
    if (!f || !f->file) return ret(c, kEbadf);
    std::fflush(f->file);
    std::error_code ec;
    fs::resize_file(f->path, arg(c, 1), ec);
    ret(c, ec ? kEio : 0);
}
void k_truncate(Context& c) {
    auto host = vfs_resolve(guest_string(c, arg(c, 0)));
    std::error_code ec;
    if (!host) return ret(c, kEnoent);
    fs::resize_file(*host, arg(c, 1), ec);
    ret(c, ec ? kEnoent : 0);
}
// nbytes is u64 per shadPS4, basep is output only (the cursor lives in the fd). Orbis dirent (verified against shadPS4 OrbisKernelDirent, GPL-2.0+ source read for facts only; 8-byte header, name[256]): u32 fileno, u16 reclen, u8 type, u8 namlen, name[] (NUL padded to 4). Returns bytes written, 0 at end.
void k_getdirentries(Context& c) {  // (fd, buf, nbytes, int64* basep)
    OpenFile* f = get_fd(int64_t(argi(c, 0)));
    if (!f || !f->dir) return ret(c, kEnotdir);
    uint64_t used = 0;
    const int64_t cookie = int64_t(f->next_entry);  // basep (shadPS4 NormalDirectory::getdents): offset before this call
    while (f->next_entry < f->entries.size()) {
        const auto& e = f->entries[f->next_entry];
        const std::string name = e.path().filename().string();
        const uint64_t reclen = (8 + name.size() + 1 + 3) & ~uint64_t(3);
        if (used + reclen > arg(c, 2)) break;
        const uint64_t at = arg(c, 1) + used;
        std::memset(ptr(c, at), 0, reclen);
        rt::st<uint32_t>(c, at, uint32_t(f->next_entry + 1));
        rt::st<uint16_t>(c, at + 4, uint16_t(reclen));
        std::error_code ec;
        rt::st<uint8_t>(c, at + 6, e.is_directory(ec) ? 4 : 8);  // DT_DIR / DT_REG
        rt::st<uint8_t>(c, at + 7, uint8_t(name.size()));
        std::memcpy(ptr(c, at + 8), name.c_str(), name.size());
        used += reclen;
        ++f->next_entry;
    }
    if (arg(c, 3)) rt::st<int64_t>(c, arg(c, 3), cookie);
    if (trace_io())
        std::fprintf(stderr, "io: getdirentries fd %d -> %llu bytes, entry %zu of %zu\n", argi(c, 0), (unsigned long long)used, f->next_entry, f->entries.size());
    ret(c, used);
}

// ---- C stdio over the VFS ---------------------------------------------------------------------------------------------
// A guest FILE* is a small low-arena block holding an index into the host FILE table. Index 0 is host stdout: the loader's zeroed
// data cells _Stdout/_Stderr/_Stdin read as index 0, so the guest's standard streams go to the host log instead of the first fopen'd file.
std::mutex g_file_mutex;
std::vector<FILE*> g_files{stdout};

FILE* host_file(Context& c, uint64_t guest) {
    if (!guest) return nullptr;
    const uint64_t idx = rt::ld<uint64_t>(c, guest);
    std::lock_guard lk(g_file_mutex);
    return idx < g_files.size() ? g_files[idx] : nullptr;
}
void c_fopen(Context& c) {  // (path, mode)
    const std::string gp = guest_string(c, arg(c, 0));
    auto host = vfs_resolve(gp);
    if (trace_io()) std::fprintf(stderr, "io: fopen '%s' -> %s\n", gp.c_str(), host ? host->string().c_str() : "(unmounted)");
    if (!host) { *guest_errno() = 2; return ret(c, 0); }
    FILE* fp = std::fopen(host->string().c_str(), ptr<const char>(c, arg(c, 1)));
    if (!fp) { *guest_errno() = 2; return ret(c, 0); }
    auto* blk = static_cast<uint64_t*>(guest_alloc(16));
    std::lock_guard lk(g_file_mutex);
    g_files.push_back(fp);
    *blk = g_files.size() - 1;
    ret(c, reinterpret_cast<uintptr_t>(blk) - c.base);
}
void c_fclose(Context& c) {
    if (arg(c, 0) && rt::ld<uint64_t>(c, arg(c, 0)) == 0) return ret(c, 0);  // a standard stream (loader cell, not a heap block): keep it open
    FILE* f = host_file(c, arg(c, 0));
    if (!f) return ret(c, uint64_t(-1));
    std::fclose(f);
    {
        std::lock_guard lk(g_file_mutex);
        g_files[rt::ld<uint64_t>(c, arg(c, 0))] = nullptr;
    }
    guest_free(ptr(c, arg(c, 0)));
    ret(c, 0);
}
void c_fread(Context& c) {  // (buf, size, count, FILE*)
    FILE* f = host_file(c, arg(c, 3));
    ret(c, f ? std::fread(ptr(c, arg(c, 0)), arg(c, 1), arg(c, 2), f) : 0);
}
void c_fwrite(Context& c) {
    FILE* f = host_file(c, arg(c, 3));
    ret(c, f ? std::fwrite(ptr(c, arg(c, 0)), arg(c, 1), arg(c, 2), f) : 0);
}
void c_fseek(Context& c) {  // (FILE*, offset, whence)
    FILE* f = host_file(c, arg(c, 0));
    ret(c, f ? uint64_t(int64_t(bb_fseek(f, int64_t(arg(c, 1)), int(arg(c, 2))))) : uint64_t(-1));
}
void c_ftell(Context& c) { FILE* f = host_file(c, arg(c, 0)); ret(c, f ? uint64_t(bb_ftell(f)) : uint64_t(-1)); }
void c_fgets(Context& c) {  // (buf, n, FILE*)
    FILE* f = host_file(c, arg(c, 2));
    char* r = f ? std::fgets(ptr<char>(c, arg(c, 0)), int(arg(c, 1)), f) : nullptr;
    ret(c, r ? arg(c, 0) : 0);
}
void c_feof(Context& c) { FILE* f = host_file(c, arg(c, 0)); ret(c, f ? std::feof(f) != 0 : 1); }
void c_fflush(Context& c) { FILE* f = host_file(c, arg(c, 0)); ret(c, f ? uint64_t(std::fflush(f)) : 0); }
// Output: count (fprintf) / non-negative (fputs) on success, EOF (-1) without a stream.
constexpr uint64_t kEof = uint64_t(-1);
uint64_t put_text(FILE* f, const std::string& s) { return f && std::fwrite(s.data(), 1, s.size(), f) == s.size() ? s.size() : kEof; }
void c_fputs(Context& c) { ret(c, put_text(host_file(c, arg(c, 1)), guest_string(c, arg(c, 0))) == kEof ? kEof : 0); }  // (s, FILE*)
void c_fputc(Context& c) {  // (ch, FILE*) -> ch as unsigned char
    const unsigned char ch = arg8(c, 0);
    FILE* f = host_file(c, arg(c, 1));
    ret(c, f && std::fputc(ch, f) != EOF ? ch : kEof);
}
void c_fprintf(Context& c) { ret(c, put_text(host_file(c, arg(c, 0)), format_variadic(c, 1))); }  // (FILE*, fmt, ...)
void c_vfprintf(Context& c) { ret(c, put_text(host_file(c, arg(c, 0)), format_guest(c, arg(c, 1), arg(c, 2)))); }  // (FILE*, fmt, va_list)
void c_puts(Context& c) { ret(c, put_text(stdout, guest_string(c, arg(c, 0)) + "\n") == kEof ? kEof : 0); }
void c_putchar(Context& c) { ret(c, std::fputc(arg8(c, 0), stdout) != EOF ? arg8(c, 0) : kEof); }
void c_perror(Context& c) {  // "s: message\n" to stderr; FreeBSD errno 1..34 match the host CRT's texts
    const std::string s = arg(c, 0) ? guest_string(c, arg(c, 0)) : "";
    std::fprintf(stderr, "%s%s%s\n", s.c_str(), s.empty() ? "" : ": ", std::strerror(*guest_errno()));
}

// POSIX stat: -1 with errno instead of the Orbis error code.
void p_stat(Context& c) {
    k_stat(c);
    if (c.r[0]) { *guest_errno() = 2; c.r[0] = uint64_t(-1); }
}
// POSIX wrappers over the sceKernel* variants: Orbis kernel errors are 0x80020000 + errno.
template <void (*K)(Context&)> void posix(Context& c) {
    K(c);
    if (c.r[0]) { *guest_errno() = int(c.r[0] & 0xFFFF); c.r[0] = uint64_t(-1); }
}
void p_remove(Context& c) {  // C remove(): a file, or an empty directory
    auto host = vfs_resolve(guest_string(c, arg(c, 0)));
    std::error_code ec;
    if (host && fs::is_directory(*host, ec)) return posix<k_rmdir>(c);
    posix<k_unlink>(c);
}

} // namespace

std::mutex g_mounts_mutex;

void vfs_mount(const std::string& guest_prefix, const fs::path& host_dir) {
    std::lock_guard lk(g_mounts_mutex);
    mounts()[guest_prefix] = host_dir;
}
void vfs_unmount(const std::string& guest_prefix) {
    std::lock_guard lk(g_mounts_mutex);
    mounts().erase(guest_prefix);
}

std::optional<fs::path> vfs_resolve(const std::string& gp) {
    std::lock_guard lk(g_mounts_mutex);
    for (auto it = mounts().rbegin(); it != mounts().rend(); ++it) {  // reverse order: "/app0/sce_sys" before "/app0"
        auto& [prefix, dir] = *it;
        if (gp.compare(0, prefix.size(), prefix) != 0) continue;
        if (gp.size() > prefix.size() && gp[prefix.size()] != '/') continue;
        std::string rest = gp.substr(prefix.size());
        fs::path rel = fs::path(rest).relative_path();
        for (auto& part : rel)
            if (part == "..") return std::nullopt;
        return dir / rel;
    }
    return std::nullopt;
}

void register_vfs() {
    reg("sceKernelOpen", k_open);
    reg("sceKernelClose", k_close);
    reg("sceKernelRead", k_read);
    reg("sceKernelWrite", k_write);
    reg("sceKernelLseek", k_lseek);
    reg("sceKernelStat", k_stat);
    reg("sceKernelFstat", k_fstat);
    reg("sceKernelMkdir", k_mkdir);
    reg("sceKernelRmdir", k_rmdir);
    reg("sceKernelUnlink", k_unlink);
    reg("sceKernelRename", k_rename);
    reg("sceKernelFtruncate", k_ftruncate);
    reg("sceKernelTruncate", k_truncate);
    reg("sceKernelGetdirentries", k_getdirentries);
    reg("rewind", [](Context& c) { if (FILE* f = host_file(c, arg(c, 0))) bb_fseek(f, 0, SEEK_SET); });
    reg("fopen", c_fopen);
    reg("fclose", c_fclose);
    reg("fread", c_fread);
    reg("fwrite", c_fwrite);
    reg("fseek", c_fseek);
    reg("ftell", c_ftell);
    reg("fgets", c_fgets);
    reg("feof", c_feof);
    reg("fflush", c_fflush);
    reg("fputs", c_fputs);
    reg("fputc", c_fputc);
    reg("fprintf", c_fprintf);
    reg("vfprintf", c_vfprintf);
    reg("puts", c_puts);
    reg("putchar", c_putchar);
    reg("perror", c_perror);
    reg("stat", p_stat);
    reg("mkdir", posix<k_mkdir>);  // (path, mode)
    reg("rmdir", posix<k_rmdir>);
    reg("rename", posix<k_rename>);
    reg("remove", p_remove);
}

} // namespace bb::hle
