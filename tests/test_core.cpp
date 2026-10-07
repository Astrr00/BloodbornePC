// SPDX-License-Identifier: GPL-3.0-or-later
// Known-answer checks for hashes and NIDs (vectors: FIPS 180-4 examples; NIDs from public PS4 symbol lists).
#include <cstdio>
#include <string>
#include <string_view>

#include "core/hash.h"
#include "core/nid.h"

static int failures = 0;
#define CHECK(c)                                                    \
    do {                                                            \
        if (!(c)) {                                                 \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                             \
        }                                                           \
    } while (0)

static std::span<const uint8_t> bytes(std::string_view s) { return {reinterpret_cast<const uint8_t*>(s.data()), s.size()}; }

template <typename H>
static std::string hex(std::string_view s) {
    H h;
    h.update(bytes(s));
    auto d = h.finish();
    return bb::to_hex(d);
}

int main() {
    CHECK(hex<bb::Sha256>("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(hex<bb::Sha256>("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    // 56 bytes: forces padding into a second block.
    CHECK(hex<bb::Sha256>("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(hex<bb::Sha1>("abc") == "a9993e364706816aba3e25717850c26c9cd0d89d");

    // Streaming in odd-sized chunks must equal one-shot.
    std::string big(100000, 'x');
    bb::Sha256 a;
    for (size_t i = 0; i < big.size(); i += 777) a.update(bytes(std::string_view(big).substr(i, 777)));
    auto da = a.finish();
    CHECK(bb::to_hex(da) == hex<bb::Sha256>(big));

    CHECK(bb::generate_nid("__error") == "9BcDykPmo1I");
    CHECK(bb::generate_nid("sceKernelUsleep") == "1jfXLRVzisc");
    CHECK(bb::generate_nid("__stack_chk_guard") == "f7uOxY9mM1U");
    CHECK(bb::decode_nid_id("A") == 0u);
    CHECK(bb::decode_nid_id("B") == 1u);
    CHECK(bb::decode_nid_id("BA") == 64u);
    CHECK(!bb::decode_nid_id("!"));

    std::printf(failures ? "%d failure(s)\n" : "all core checks passed\n", failures);
    return failures ? 1 : 0;
}
