// GCN (GFX7) -> SPIR-V translator. One SPIR-V invocation models one GCN lane; scalar registers are per-invocation
// copies (wave-uniform by construction). SGPR values that only depend on user data and guest memory are tracked
// symbolically so the host can re-evaluate descriptor chains per draw without recompiling (see ScalarVal / SLoad).
#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace bb::gpu {

enum class ShStage : uint8_t { VS, PS, CS };

// A u32 the host can compute without running the shader: ((base + add) & mask).
struct ScalarVal {
    enum Kind : uint8_t { User, Const, LoadWord } kind = Const;
    uint32_t a = 0;  // User: register index; Const: value; LoadWord: SLoad index
    uint32_t b = 0;  // LoadWord: dword within the load result
    uint32_t add = 0, mask = ~0u;
};

// s_load_dwordN: address = lo | (hi << 32), then + offset (bytes). Evaluated by the host per draw.
struct SLoad {
    ScalarVal lo, hi;
    uint32_t offset = 0;
    uint32_t dwords = 0;
};

// Where a resource descriptor (V#/T#/S#) lives.
struct DescRef {
    int32_t load = -1;  // -1: user-data SGPRs starting at `word`; else SLoad index (word = dword within result)
    uint32_t word = 0;
    bool operator==(const DescRef&) const = default;
};

struct Resource {
    // Buffer, Image, StorageImage, Sampler: descriptor-backed (dwords = 4 or 8, ref points at the descriptor).
    // LdsBuffer, TessIndices: tessellation emulation buffers, ref stays {-1, 0}, dwords = 0, all words[] zero - the host binds
    // its own buffer. They are appended after every shader-derived resource: LdsBuffer is always the last resource, in an LS
    // the TessIndices buffer is the one before it. Written by the shader means what it says (an LS writes the LDS buffer).
    //   LdsBuffer:     uint[] workgroup memory of the emulated LS pass, indexed by byte address / 4; read-write in an LS,
    //                  read-only in a DS.
    //   TessIndices:   uint count at dword 0, uint ids[] from dword 1 (read-only); an LS reads ids[gl_GlobalInvocationID.x].
    //   GdsBuffer:     uint[] global data share (64 KiB, byte address / 4), one host buffer for the whole run; compute only.
    enum Type : uint8_t { Buffer, Image, StorageImage, Sampler, LdsBuffer, TessIndices, GdsBuffer } type = Buffer;
    DescRef ref;
    uint32_t binding = 0;        // descriptor binding in set 0: 2 * resource index + GcnEnv::binding_base
    uint32_t dwords = 0;         // descriptor size (4 or 8)
    uint32_t words[8] = {};      // descriptor contents seen at translation time (shape; part of the cache key)
    bool written = false;
    bool load_data = false;      // Buffer: the dwords an s_load result holds (ref.load = SLoad index); read as a shader value at run time
    bool scalar = false;         // Buffer: read by s_buffer_load, i.e. a constant buffer (not clamped to the records a draw fetches)
    uint8_t vertex_use = 0;      // Buffer: bit0 = loaded with the vertex id (vaddr v0, idxen, no offen), bit1 = any other MUBUF use (instance streams, computed offsets)
    int16_t sampler = -1;        // Image: index of the sampler resource it is sampled with (-1: none, only loads; -2: several)
};

struct GcnEnv {
    ShStage stage = ShStage::PS;
    uint32_t binding_base = 0;               // 0 (VS/CS) or 1 (PS): both stages' resources share one descriptor set without colliding
    uint32_t push_offset = 0;                // byte offset of this stage's user-data block in the push constants
    // BB_RES_SCALE (1 = off): with s > 1 a u32 follows the user data in the push constants: bit i = image resource i is s times the guest
    // size, bit 31 (PS) = the render targets are. The shader then reads FragCoord in guest pixels and maps guest texel coordinates of
    // integer accesses (image_load/store) and the padded-target correction to the scaled image.
    uint32_t res_scale = 1;
    uint32_t user_sgprs = 0;                 // SGPRs preloaded from user data
    uint32_t user[16] = {};                  // user data at translation time
    std::function<bool(uint64_t addr, uint32_t* dst, uint32_t dwords)> read_mem;  // guest memory
    // LDS (compute/vertex): size of the workgroup-shared memory in bytes (COMPUTE_PGM_RSRC2.LDS_SIZE), 0 = none
    uint32_t lds_bytes = 0;
    // PS
    uint32_t ps_input_addr = 0;              // SPI_PS_INPUT_ADDR (VGPR layout)
    // SPI_PS_INPUT_CNTL_n: PS input n reads VS param OFFSET[4:0]; bit 5 = constant DEFAULT_VAL[9:8] (0: 0000, 1: 0001, 2: 1110,
    // 3: 1111), bit 10 = flat. Without ps_input_map input n reads param n.
    bool ps_input_map = false;
    uint32_t ps_input_cntl[32] = {};
    bool ps_bary = false;                    // interpolate as P0 + i*(P1-P0) + j*(P2-P0) from per-vertex inputs (VK_KHR_fragment_shader_barycentric)
    // VS
    uint32_t vs_out_cntl = 0;                // PA_CL_VS_OUT_CNTL: what the POS1-3 exports carry (misc vector, clip/cull distances)
    // CS
    uint32_t cs_local[3] = {64, 1, 1};
    uint32_t cs_tgid_en = 0;                 // bit0..2: tgid x/y/z SGPRs follow the user SGPRs
    uint32_t cs_tidig_comps = 0;             // 0: v0=x, 1: +v1=y, 2: +v2=z
    // Tessellation emulation (the shaders are given in the VS slot; the host runs the LS as a compute pass and the DS as a
    // vertex pass over a generated (N+1)x(N+1) grid per patch).
    uint32_t tess_ls = 0;        // 1: translate as LS: GLCompute, one invocation per control point, LDS becomes a buffer
    uint32_t tess_ds_level = 0;  // N > 0: translate as DS: Vertex, an N x N cell grid per patch, LDS reads a buffer
};

struct Translation {
    std::vector<uint32_t> spirv;
    std::vector<SLoad> loads;
    std::vector<Resource> resources;
    uint32_t vs_params = 0;                  // bitmask of exported param slots
    uint32_t ps_attrs = 0;                   // bitmask of interpolated attribute slots read
    uint32_t ps_mrts = 0;                    // bitmask of exported MRTs
    bool ps_kill = false;                    // PS: the final export kills lanes whose EXEC bit is off (alpha test)
    uint64_t fetch_addr = 0;                 // VS only: inlined fetch shader at translation time (0 = none)
    ScalarVal fetch_lo, fetch_hi;            // its address as user data (a draw's own fetch shader: fetch_address)
    std::vector<uint32_t> fetch_code;        // its code: part of the cache key (compared at the draw's fetch_address)
    std::string error;                       // non-empty: translation failed
    uint32_t tess_ls = 0;                    // copy of GcnEnv::tess_ls (part of the shader identity)
    uint32_t tess_ds_level = 0;              // copy of GcnEnv::tess_ds_level (0 = not a domain shader)
};

Translation translate(const uint32_t* code, size_t dwords, const GcnEnv& env);

// Re-evaluates the descriptor chains of `t` against live state. words[i] receives resource i's descriptor dwords.
// Returns false if guest memory is unreadable.
bool eval_resources(const Translation& t, const uint32_t user[16],
                    const std::function<bool(uint64_t, uint32_t*, uint32_t)>& read_mem, std::vector<std::array<uint32_t, 8>>& words);

// Bits of descriptor dword `i` that are baked into the SPIR-V (stride, formats, image type...). A live descriptor with
// equal masked words can reuse the translation.
uint32_t shape_mask(Resource::Type type, unsigned i);

// Address of the fetch shader a draw with user data `user` calls (t.fetch_addr != 0).
uint64_t fetch_address(const Translation& t, const uint32_t user[16]);

}  // namespace bb::gpu
