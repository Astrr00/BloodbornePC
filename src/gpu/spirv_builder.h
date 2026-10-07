// Minimal SPIR-V module builder (types/constants deduplicated, one function body at a time).
#pragma once
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <spirv/unified1/spirv.hpp11>

namespace bb::gpu {

class SpvBuilder {
public:
    using Id = uint32_t;

    Id new_id() { return ++bound_; }

    // ---- module-level -------------------------------------------------------------------------------------------
    void capability(spv::Capability c) { cap_.insert(c); }
    void extension(const char* name) { emit(ext_, spv::Op::OpExtension, str(name)); }
    Id import_glsl() {
        if (!glsl_) { glsl_ = new_id(); std::vector<uint32_t> w{glsl_}; auto s = str("GLSL.std.450"); w.insert(w.end(), s.begin(), s.end()); emit(imports_, spv::Op::OpExtInstImport, w); }
        return glsl_;
    }
    void decorate(Id target, spv::Decoration d, std::initializer_list<uint32_t> args = {}) {
        std::vector<uint32_t> w{target, uint32_t(d)};
        w.insert(w.end(), args);
        emit(annot_, spv::Op::OpDecorate, w);
    }
    void member_decorate(Id st, uint32_t member, spv::Decoration d, std::initializer_list<uint32_t> args = {}) {
        std::vector<uint32_t> w{st, member, uint32_t(d)};
        w.insert(w.end(), args);
        emit(annot_, spv::Op::OpMemberDecorate, w);
    }
    void entry_point(spv::ExecutionModel m, Id fn, const char* name, const std::vector<Id>& interface) {
        std::vector<uint32_t> w{uint32_t(m), fn};
        auto s = str(name);
        w.insert(w.end(), s.begin(), s.end());
        w.insert(w.end(), interface.begin(), interface.end());
        emit(entries_, spv::Op::OpEntryPoint, w);
    }
    void exec_mode(Id fn, spv::ExecutionMode m, std::initializer_list<uint32_t> args = {}) {
        std::vector<uint32_t> w{fn, uint32_t(m)};
        w.insert(w.end(), args);
        emit(modes_, spv::Op::OpExecutionMode, w);
    }

    // ---- types / constants (deduplicated) ------------------------------------------------------------------------
    Id t_void() { return type(spv::Op::OpTypeVoid, {}); }
    Id t_bool() { return type(spv::Op::OpTypeBool, {}); }
    Id t_u32() { return type(spv::Op::OpTypeInt, {32, 0}); }
    Id t_i32() { return type(spv::Op::OpTypeInt, {32, 1}); }
    Id t_u64() { capability(spv::Capability::Int64); return type(spv::Op::OpTypeInt, {64, 0}); }
    Id t_f32() { return type(spv::Op::OpTypeFloat, {32}); }
    Id t_vec(Id comp, uint32_t n) { return type(spv::Op::OpTypeVector, {comp, n}); }
    Id t_ptr(spv::StorageClass sc, Id pointee) { return type(spv::Op::OpTypePointer, {uint32_t(sc), pointee}); }
    Id t_array(Id elem, Id length_const) { return type(spv::Op::OpTypeArray, {elem, length_const}); }
    Id t_rtarray(Id elem) { return type(spv::Op::OpTypeRuntimeArray, {elem}); }
    Id t_struct(const std::vector<Id>& members) { return type(spv::Op::OpTypeStruct, members); }
    Id t_fn(Id ret, const std::vector<Id>& args = {}) { std::vector<uint32_t> k{ret}; k.insert(k.end(), args.begin(), args.end()); return type(spv::Op::OpTypeFunction, k); }
    Id t_image(Id sampled_type, spv::Dim dim, bool arrayed, uint32_t sampled /*1 sampled, 2 storage*/, spv::ImageFormat fmt = spv::ImageFormat::Unknown) {
        return type(spv::Op::OpTypeImage, {sampled_type, uint32_t(dim), 0, arrayed ? 1u : 0u, 0, sampled, uint32_t(fmt)});
    }
    Id t_sampler() { return type(spv::Op::OpTypeSampler, {}); }
    Id t_sampled_image(Id image) { return type(spv::Op::OpTypeSampledImage, {image}); }

    Id c_u32(uint32_t v) { return constant(t_u32(), v); }
    Id c_i32(int32_t v) { return constant(t_i32(), uint32_t(v)); }
    Id c_f32(float f) { uint32_t b; std::memcpy(&b, &f, 4); return constant(t_f32(), b); }
    Id c_bool(bool v) { return type(v ? spv::Op::OpConstantTrue : spv::Op::OpConstantFalse, {t_bool()}, true); }
    Id c_composite(Id type_id, const std::vector<Id>& parts) {
        std::vector<uint32_t> k{type_id};
        k.insert(k.end(), parts.begin(), parts.end());
        return type(spv::Op::OpConstantComposite, k, true);
    }

    // Global variable (Input/Output/Uniform/StorageBuffer/...).
    Id global_var(Id ptr_type, spv::StorageClass sc) {
        Id id = new_id();
        emit(globals_, spv::Op::OpVariable, {ptr_type, id, uint32_t(sc)});
        return id;
    }

    // ---- function bodies -----------------------------------------------------------------------------------------
    // begin_function emits OpFunction/OpLabel; locals must be created right after via local_var().
    Id begin_function(Id ret, Id fn_type) {
        Id id = new_id();
        emit(fn_head_, spv::Op::OpFunction, {ret, id, 0u, fn_type});
        Id label = new_id();
        emit(fn_head_, spv::Op::OpLabel, {label});
        cur_label_ = label;
        in_fn_ = true;
        return id;
    }
    Id local_var_init(Id pointee, Id init) {  // Function-storage variables must sit at the top of the first block
        Id id = new_id();
        emit(fn_locals_, spv::Op::OpVariable, {t_ptr(spv::StorageClass::Function, pointee), id, uint32_t(spv::StorageClass::Function), init});
        return id;
    }
    void end_function() {
        emit(body_, spv::Op::OpFunctionEnd, {});
        fn_.insert(fn_.end(), fn_head_.begin(), fn_head_.end());
        fn_.insert(fn_.end(), fn_locals_.begin(), fn_locals_.end());
        fn_.insert(fn_.end(), body_.begin(), body_.end());
        fn_head_.clear(); fn_locals_.clear(); body_.clear();
        in_fn_ = false;
    }

    Id label() { return new_id(); }
    void place_label(Id l) { emit(body_, spv::Op::OpLabel, {l}); cur_label_ = l; }
    Id current_label() const { return cur_label_; }

    // Result-producing instruction in the current block.
    Id op(spv::Op o, Id result_type, const std::vector<uint32_t>& operands) {
        Id r = new_id();
        std::vector<uint32_t> w{result_type, r};
        w.insert(w.end(), operands.begin(), operands.end());
        emit(body_, o, w);
        return r;
    }
    // No-result instruction in the current block.
    void op0(spv::Op o, const std::vector<uint32_t>& operands) { emit(body_, o, operands); }
    Id glsl(Id result_type, uint32_t inst, const std::vector<uint32_t>& args) {
        std::vector<uint32_t> w{import_glsl(), inst};
        w.insert(w.end(), args.begin(), args.end());
        return op(spv::Op::OpExtInst, result_type, w);
    }

    std::vector<uint32_t> finish() const {
        std::vector<uint32_t> out{spv::MagicNumber, 0x00010500u, 0, bound_ + 1, 0};  // SPIR-V 1.5
        for (auto c : cap_) { out.push_back((2u << 16) | uint32_t(spv::Op::OpCapability)); out.push_back(uint32_t(c)); }
        for (const auto* sec : {&ext_, &imports_}) out.insert(out.end(), sec->begin(), sec->end());
        out.push_back((3u << 16) | uint32_t(spv::Op::OpMemoryModel));
        out.push_back(uint32_t(spv::AddressingModel::Logical));
        out.push_back(uint32_t(spv::MemoryModel::GLSL450));
        for (const auto* sec : {&entries_, &modes_, &annot_, &types_, &globals_, &fn_}) out.insert(out.end(), sec->begin(), sec->end());
        return out;
    }

private:
    static std::vector<uint32_t> str(const char* s) {
        std::vector<uint32_t> w((std::strlen(s) + 4) / 4, 0);
        std::memcpy(w.data(), s, std::strlen(s));
        return w;
    }
    static void emit(std::vector<uint32_t>& sec, spv::Op o, const std::vector<uint32_t>& w) {
        sec.push_back((uint32_t(w.size() + 1) << 16) | uint32_t(o));
        sec.insert(sec.end(), w.begin(), w.end());
    }

    // Types and constants share one section; OpType* have no result type, constants have type as first operand.
    Id type(spv::Op o, const std::vector<uint32_t>& ops, bool has_result_type = false) {
        auto key = std::make_pair(uint32_t(o), ops);
        auto it = dedup_.find(key);
        if (it != dedup_.end()) return it->second;
        Id id = new_id();
        std::vector<uint32_t> w;
        if (has_result_type) { w.push_back(ops[0]); w.push_back(id); w.insert(w.end(), ops.begin() + 1, ops.end()); }
        else { w.push_back(id); w.insert(w.end(), ops.begin(), ops.end()); }
        emit(types_, o, w);
        dedup_[key] = id;
        return id;
    }
    Id constant(Id type_id, uint32_t bits) { return type(spv::Op::OpConstant, {type_id, bits}, true); }

    uint32_t bound_ = 0, glsl_ = 0, cur_label_ = 0;
    bool in_fn_ = false;
    std::set<spv::Capability> cap_;
    std::vector<uint32_t> ext_, imports_, entries_, modes_, annot_, types_, globals_, fn_, fn_head_, fn_locals_, body_;
    std::map<std::pair<uint32_t, std::vector<uint32_t>>, Id> dedup_;
};

}  // namespace bb::gpu
