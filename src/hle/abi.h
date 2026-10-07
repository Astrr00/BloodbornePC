// SPDX-License-Identifier: GPL-3.0-or-later
// wrap<F>: adapts a host function with plain C parameters to a guest SysV call. Integral/enum arguments come from
// rdi, rsi, rdx, rcx, r8, r9 (then the stack), float/double from xmm0..7; pointers are guest addresses translated
// through Context::base. The result goes to rax / xmm0. Parameter types fix the widths (use int64_t for `long`).
#pragma once
#include <cstring>
#include <tuple>
#include <type_traits>
#include <utility>

#include "hle/hle.h"

namespace bb::hle {

template <class> struct FnTraits;
template <class R, class... A> struct FnTraits<R (*)(A...)> {
    using Ret = R;
    using Args = std::tuple<A...>;
};

struct ArgCursor {
    Context& c;
    int ints = 0, floats = 0;
    int stack = 0;
    template <class T> T next() {
        if constexpr (std::is_floating_point_v<T>) {
            T v;
            if (floats < 8) std::memcpy(&v, c.ymm[floats++], sizeof v);
            else v = rt::ld<T>(c, c.r[4] + 8 + 8 * uint64_t(stack++));
            return v;
        } else {
            uint64_t raw = ints < 6 ? arg(c, ints++) : rt::ld<uint64_t>(c, c.r[4] + 8 + 8 * uint64_t(stack++));
            if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(c.base + raw);
            else return static_cast<T>(raw);
        }
    }
};

template <auto F> void wrap(Context& c) {
    using Tr = FnTraits<decltype(+F)>;
    ArgCursor cur{c};
    auto args = [&]<size_t... I>(std::index_sequence<I...>) {
        // braced init guarantees left-to-right evaluation
        return typename Tr::Args{cur.next<std::tuple_element_t<I, typename Tr::Args>>()...};
    }(std::make_index_sequence<std::tuple_size_v<typename Tr::Args>>{});
    using R = typename Tr::Ret;
    if constexpr (std::is_void_v<R>) {
        std::apply(+F, args);
    } else if constexpr (std::is_floating_point_v<R>) {
        R v = std::apply(+F, args);
        std::memset(c.ymm[0], 0, 16);
        std::memcpy(c.ymm[0], &v, sizeof v);
    } else if constexpr (std::is_pointer_v<R>) {
        c.r[0] = reinterpret_cast<uintptr_t>(std::apply(+F, args)) - c.base;
    } else if constexpr (std::is_signed_v<R>) {
        c.r[0] = uint64_t(int64_t(std::apply(+F, args)));  // sign-extend like a 64-bit return of int
    } else {
        c.r[0] = uint64_t(std::apply(+F, args));
    }
}

} // namespace bb::hle
