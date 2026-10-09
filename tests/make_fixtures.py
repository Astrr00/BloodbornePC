# SPDX-License-Identifier: GPL-3.0-or-later
"""Builds synthetic, content-free fixtures: a minimal Orbis ELF, SELF wrappers, a fake dump and DLC folder."""
import os
import struct
import sys

B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"

PT_LOAD, PT_DYNAMIC = 1, 2
PT_GNU_EH_FRAME = 0x6474E550
PT_SCE_DYNLIBDATA, PT_SCE_PROCPARAM = 0x61000000, 0x61000001
DT = dict(NEEDED_MODULE=0x6100000F, MODULE_INFO=0x6100000D, IMPORT_LIB=0x61000015, EXPORT_LIB=0x61000013,
          FINGERPRINT=0x61000007, ORIGINAL_FILENAME=0x61000009, STRTAB=0x61000035, STRSZ=0x61000037,
          SYMTAB=0x61000039, SYMTABSZ=0x6100003F, RELA=0x6100002F, RELASZ=0x61000031, JMPREL=0x61000029,
          PLTRELSZ=0x6100002D)


def uleb(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


RECOMP_FUNCS = {}
RECOMP_TESTS = [
    ("sum", bytes.fromhex(              # sum 1..n (loop, flags, jcc)
        "31c0"                          # 0:  xor eax,eax
        "85ff"                          # 2:  test edi,edi
        "7e06"                          # 4:  jle 12
        "01f8"                          # 6:  add eax,edi
        "ffcf"                          # 8:  dec edi
        "75fa"                          # 10: jnz 6
        "c3")),                         # 12: ret
    ("fact", bytes.fromhex(             # recursive factorial (call/ret, push/pop, imul)
        "83ff01"                        # 0:  cmp edi,1
        "7f06"                          # 3:  jg 11
        "b801000000"                    # 5:  mov eax,1
        "c3"                            # 10: ret
        "53"                            # 11: push rbx
        "89fb"                          # 12: mov ebx,edi
        "8d7fff"                        # 14: lea edi,[rdi-1]
        "e8eaffffff"                    # 17: call 0
        "0fafc3"                        # 22: imul eax,ebx
        "5b"                            # 25: pop rbx
        "c3")),                         # 26: ret
    ("sel", bytes.fromhex(              # switch: n<3 ? 10*n : -1 (jump table, table without cmp on index reg)
        "83ff02"                        # 0:  cmp edi,2
        "7724"                          # 3:  ja 41
        "89f8"                          # 5:  mov eax,edi
        "488d0d22000000"                # 7:  lea rcx,[rip+34] -> 48
        "48630481"                      # 14: movsxd rax,dword [rcx+rax*4]
        "4801c8"                        # 18: add rax,rcx
        "ffe0"                          # 21: jmp rax
        "b800000000c3"                  # 23: mov eax,0; ret
        "b80a000000c3"                  # 29: mov eax,10; ret
        "b814000000c3"                  # 35: mov eax,20; ret
        "b8ffffffffc3"                  # 41: mov eax,-1; ret
        "cc")                           # 47: pad
        + struct.pack("<iii", 23 - 48, 29 - 48, 35 - 48)),  # 48: table
    ("mem", bytes.fromhex(              # memory, movzx, partial register al
        "8937"                          # 0:  mov [rdi],esi
        "0fb607"                        # 2:  movzx eax,byte [rdi]
        "0307"                          # 5:  add eax,[rdi]
        "884704"                        # 7:  mov [rdi+4],al
        "0fb64f04"                      # 10: movzx ecx,byte [rdi+4]
        "01c8"                          # 14: add eax,ecx
        "c3")),                         # 16: ret
    ("max", bytes.fromhex(              # signed max + (a<b)<<8 (cmovl, setl, shl, or)
        "89f8"                          # 0:  mov eax,edi
        "39f7"                          # 2:  cmp edi,esi
        "0f4cc6"                        # 4:  cmovl eax,esi
        "0f9cc1"                        # 7:  setl cl
        "0fb6c9"                        # 10: movzx ecx,cl
        "c1e108"                        # 13: shl ecx,8
        "09c8"                          # 16: or eax,ecx
        "c3")),                         # 18: ret
    # Flag fixtures return RFLAGS right after the operation (pushfq; pop rax; ret).
    ("flags_add", bytes.fromhex("89f8" "01f0" "9c58c3")),            # mov eax,edi; add eax,esi
    ("flags_sub", bytes.fromhex("89f8" "29f0" "9c58c3")),            # mov eax,edi; sub eax,esi
    ("flags_shl", bytes.fromhex("89f8" "89f1" "d3e0" "9c58c3")),     # mov eax,edi; mov ecx,esi; shl eax,cl
    ("flags_sar", bytes.fromhex("89f8" "89f1" "d3f8" "9c58c3")),     # ...; sar eax,cl
    ("flags_inc", bytes.fromhex("89f8" "01f0" "ffc0" "9c58c3")),     # add then inc: CF must survive inc
    ("vec_math", bytes.fromhex(         # out[8..11] = (in[0..3] + in[4..7])^2 as float (rdi = float buffer)
        "c5f81007"                      # vmovups xmm0,[rdi]
        "c5f8584710"                    # vaddps  xmm0,xmm0,[rdi+16]
        "c5f859c0"                      # vmulps  xmm0,xmm0,xmm0
        "c5f8114720"                    # vmovups [rdi+32],xmm0
        "c3")),
    ("vec_less", bytes.fromhex(         # eax = (float[rdi] < float[rdi+4]) or unordered (vucomiss + setb)
        "c5fa1007"                      # vmovss xmm0,[rdi]
        "c5fa104f04"                    # vmovss xmm1,[rdi+4]
        "31c0"                          # xor eax,eax
        "c5f82ec1"                      # vucomiss xmm0,xmm1
        "0f92c0"                        # setb al
        "c3")),
    ("vec_shuf", bytes.fromhex(         # reverse dwords of [rdi] into [rdi+16]; eax = new lane 0
        "c5f970071b"                    # vpshufd xmm0,[rdi],0x1b
        "c5fa7f4710"                    # vmovdqu [rdi+16],xmm0
        "c5f97ec0"                      # vmovd eax,xmm0
        "c3")),
    ("atom_xadd", bytes.fromhex(        # eax = old [rdi]; [rdi] += esi
        "89f0"                          # mov eax,esi
        "f00fc107"                      # lock xadd [rdi],eax
        "c3")),
    ("atom_cas", bytes.fromhex(         # if [rdi]==5: [rdi]=esi; returns eax | ZF<<16
        "b805000000"                    # mov eax,5
        "f00fb137"                      # lock cmpxchg [rdi],esi
        "0f94c1"                        # setz cl
        "0fb6c9"                        # movzx ecx,cl
        "c1e110"                        # shl ecx,16
        "09c8"                          # or eax,ecx
        "c3")),
    # OF=SF=1 first (0x7fffffff+1), then a scalar float compare; return RFLAGS. Settles which flags (v)comiss and
    # (v)ucomiss clear on real hardware (native differential run in test_recomp).
    ("vec_comiss", bytes.fromhex("b8ffffff7f" "83c001" "c5fa1007" "c5fa104f04" "c5f82fc1" "9c58c3")),
    ("vec_ucomiss", bytes.fromhex("b8ffffff7f" "83c001" "c5fa1007" "c5fa104f04" "c5f82ec1" "9c58c3")),
    # Integer long tail; native differential runs in test_recomp provide the ground truth.
    ("int_div", bytes.fromhex("89f8" "31d2" "f7f6" "48c1e220" "4809d0" "c3")),     # eax/esi -> rem<<32 | quot
    ("int_idiv", bytes.fromhex("89f8" "99" "f7fe" "48c1e220" "4809d0" "c3")),      # signed edx:eax/esi
    ("int_mul", bytes.fromhex("4889f8" "48f7e6" "4831d0" "c3")),                   # rdx:rax = rdi*rsi; rax ^= rdx
    ("int_imul1", bytes.fromhex("4889f8" "48f7ee" "4831d0" "c3")),                 # signed one-operand imul
    ("int_adcsbb", bytes.fromhex("4889f8" "4801f0" "4811f0" "4819f8" "c3")),       # add; adc; sbb chain
    ("int_adcflags", bytes.fromhex("89f8" "01f0" "11f0" "9c58c3")),                # flags after adc
    ("int_rol", bytes.fromhex("89f8" "89f1" "d3c0" "c3")),                         # rol eax,cl
    ("int_ror1", bytes.fromhex("89f8" "d1c8" "9c58c3")),                           # ror eax,1 -> flags
    ("int_bt", bytes.fromhex("89f8" "0fa3f0" "0f92c0" "c3")),                      # bt eax,esi; setc al
    ("int_bextr", bytes.fromhex("c4e248f7c7" "c3")),                               # bextr eax,edi,esi
    ("int_andn", bytes.fromhex("c4e240f2c6" "c3")),                                # andn eax,edi,esi
    ("int_lzcnt", bytes.fromhex("f30fbdc7" "c3")),                                 # lzcnt eax,edi
    ("int_tzcnt", bytes.fromhex("f30fbcc7" "c3")),                                 # tzcnt eax,edi
    ("int_popcnt", bytes.fromhex("f30fb8c7" "c3")),                                # popcnt eax,edi
    ("int_movbe", bytes.fromhex("0f38f007" "c3")),                                 # movbe eax,[rdi]
    # Long tail (coverage census leftovers): string ops, scalar odds and ends, AVX/SSE/MMX/x87 forms. Bytes were
    # assembled with keystone and checked by a capstone round trip; test_recomp diffs each against the host CPU
    # (ours-only cases are marked) so the Zydis-decoded semantics are what the hardware does.
    ("str_movs", bytes.fromhex(        # rep/single movs b/w/d/q, backward (std), overlapping forward copies; returns rdi/rsi/rcx deltas
        "4889fa488db700040000b90d000000f3a4b905000000f366a5b903000000f3a5b902000000f348a5a466a5a548a5fd488db2"
        "20050000488dba20030000b909000000f3a4b904000000f366a5b903000000f3a5b902000000f348a548a5fc488db2000600"
        "00488dba01060000b909000000f3a4488db240060000488dba44060000b905000000f3a54889f84829d04829d648c1e61448"
        "09f048c1e1284809c8c3")),
    ("str_stos", bytes.fromhex(        # rep/single stos b/w/d/q (rax = rsi), backward; returns rdi delta
        "4889fa4889f0b907000000f3aab903000000f366abb902000000f3abb902000000f348abaa66abab48abb946000000f348ab"
        "fd488dba00070000b905000000f3abaafc4829d74889f848c1e1284809c8c3")),
    ("str_lods", bytes.fromhex(        # lods b/w/d/q (partial rax writes), rep lods with rcx=0 and >0, backward
        "4889fa4889fe4531c048c7c0ffffffffac4931c048c7c0ffffffff66ad4931c048c7c0ffffffffad4931c048c7c0ffffffff"
        "48ad4931c048c7c03412000031c9f3ac4931c0b903000000f3ac4931c0b902000000f366ad4931c0fdadfc4931c04829d648"
        "c1e6304931f04c89c0c3")),
    ("str_cmps_e", bytes.fromhex(        # repe cmpsb over 32 bytes; returns flags | rdi delta << 32 | rcx << 48
        "4889fa488db700010000b920000000f3a69c584829d748c1e72048c1e1304809f84809c8c3")),
    ("str_cmps_ne", bytes.fromhex(        # repne cmpsw over 16 words; same return layout
        "4889fa488db700010000b910000000f266a79c584829d748c1e72048c1e1304809f84809c8c3")),
    ("str_cmps_one", bytes.fromhex(        # single cmpsb/w/d/q, flags of each packed at bit 0/16/32/48
        "488db700010000a69c415866a79c415949c1e110a79c415a49c1e22048a79c415b49c1e3304d09c84d09d04d09d84c89c0c3")),
    ("str_cmps_d", bytes.fromhex(        # repne cmpsd + repe cmpsq backward
        "4889fa488db700010000b908000000f2a79c4158fd488db280010000488dba80000000b906000000f348a7fc9c5848c1e010"
        "4c09c04829d748c1e72848c1e1344809f84809c8c3")),
    ("str_scas_ne", bytes.fromhex(        # repne scasb for al = sil over 64 bytes; flags | rdi delta << 32 | rcx << 48
        "4889fa89f0b940000000f2ae9c41584829d748c1e72048c1e1304909f84909c84c89c0c3")),
    ("str_scas_e", bytes.fromhex(        # repe scasw for ax = si over 32 words, same layout
        "4889fa89f0b920000000f366af9c41584829d748c1e72048c1e1304909f84909c84c89c0c3")),
    ("str_scas_one", bytes.fromhex(        # single scasb/w/d/q (rax = rsi), flags packed at bit 0/16/32/48
        "4889f0ae9c415866af9c415949c1e110af9c415a49c1e22048af9c415b49c1e3304d09c84d09d04d09d84c89c0c3")),
    ("int_bswap", bytes.fromhex(        # bswap r64 / r32
        "4889f8"                                # mov rax, rdi
        "480fc8"                                # bswap rax
        "89f2"                                  # mov edx, esi
        "0fca"                                  # bswap edx
        "4831d0"                                # xor rax, rdx
        "c3")),                                 # ret
    ("int_cstc", bytes.fromhex(        # cmc/stc/clc after an add, flags pushed back to back; flags after each at bit 0/16/32
        "89f801f0f59cf99cf89c5848c1e0205948c1e1105a4809c84809d0c3")),
    ("int_cwd", bytes.fromhex(        # cbw / cwde / cwd with preserved upper bits
        "89f866984189c089f8984189c1baffffffff89f0669948c1e2104931d049c1e1204d31c84c89c0c3")),
    ("int_lahf", bytes.fromhex(        # lahf after add (AF excluded by the test mask)
        "89f8"                                  # mov eax, edi
        "01f0"                                  # add eax, esi
        "9f"                                    # lahf
        "0fb6c4"                                # movzx eax, ah
        "c3")),                                 # ret
    ("int_sahf", bytes.fromhex(        # sahf from ah of edi after xor (OF = 0), returns RFLAGS
        "31c9"                                  # xor ecx, ecx
        "89f8"                                  # mov eax, edi
        "9e"                                    # sahf
        "9c"                                    # pushfq
        "58"                                    # pop rax
        "c3")),                                 # ret
    ("int_xlat", bytes.fromhex(        # xlatb: al = [rbx + al] (rdi = table, rsi = index)
        "53"                                    # push rbx
        "4889fb"                                # mov rbx, rdi
        "89f0"                                  # mov eax, esi
        "d7"                                    # xlatb
        "5b"                                    # pop rbx
        "c3")),                                 # ret
    ("int_bsf", bytes.fromhex(        # bsf r32 with preloaded destination; zero source leaves it; ZF at bit 32
        "b8682457130fbcc70f94c1480fb6c948c1e1204809c8c3")),
    ("int_bsr", bytes.fromhex(        # bsr r32 with preloaded destination; ZF at bit 32
        "b8682457130fbdc70f94c1480fb6c948c1e1204809c8c3")),
    ("int_bs64", bytes.fromhex(        # bsf r64 / bsr r64 on non-zero sources
        "480fbcc7"                              # bsf rax, rdi
        "480fbdce"                              # bsr rcx, rsi
        "48c1e107"                              # shl rcx, 7
        "4831c8"                                # xor rax, rcx
        "c3")),                                 # ret
    ("int_blsi", bytes.fromhex(        # blsi r32: result | RFLAGS << 32 (PF excluded by the test mask)
        "c4e278f3df"                            # blsi eax, edi
        "9c"                                    # pushfq
        "5a"                                    # pop rdx
        "48c1e220"                              # shl rdx, 0x20
        "4809d0"                                # or rax, rdx
        "c3")),                                 # ret
    ("int_blsr", bytes.fromhex(        # blsr r32: result | RFLAGS << 32 (PF excluded by the test mask)
        "c4e278f3cf"                            # blsr eax, edi
        "9c"                                    # pushfq
        "5a"                                    # pop rdx
        "48c1e220"                              # shl rdx, 0x20
        "4809d0"                                # or rax, rdx
        "c3")),                                 # ret
    ("int_blsmsk", bytes.fromhex(        # blsmsk r32: result | RFLAGS << 32 (PF excluded by the test mask)
        "c4e278f3d7"                            # blsmsk eax, edi
        "9c"                                    # pushfq
        "5a"                                    # pop rdx
        "48c1e220"                              # shl rdx, 0x20
        "4809d0"                                # or rax, rdx
        "c3")),                                 # ret
    ("int_bls64", bytes.fromhex(        # blsi/blsr/blsmsk r64
        "c4e2f8f3df"                            # blsi rax, rdi
        "c4e2f0f3cf"                            # blsr rcx, rdi
        "c4e2e8f3d7"                            # blsmsk rdx, rdi
        "4831c8"                                # xor rax, rcx
        "4831d0"                                # xor rax, rdx
        "c3")),                                 # ret
    ("int_rcl8", bytes.fromhex(        # cmp sets CF; rcl al, cl; result | RFLAGS << 32
        "89f889f139f7d2d09c5a48c1e2204809d0c3")),
    ("int_rcr16", bytes.fromhex(        # rcr ax, cl
        "89f889f139f766d3d89c5a48c1e2204809d0c3")),
    ("int_rcl32", bytes.fromhex(        # rcl eax, cl
        "89f889f139f7d3d09c5a48c1e2204809d0c3")),
    ("int_rcr1", bytes.fromhex(        # rcr eax, 1 (D1 form: OF defined)
        "89f839f7d1d89c5a48c1e2204809d0c3")),
    ("int_rcl64", bytes.fromhex(        # rcl rax, cl; rcr rdx, cl chain; flags folded
        "4889f84889f289f14839f748d3d048d3da9c5983e10148c1e13f4831d04831c8c3")),
    ("int_shld32", bytes.fromhex(        # shld eax, edx, cl; result | RFLAGS << 32
        "89f869d7b179379e89f10fa5d09c5948c1e1204809c8c3")),
    ("int_shrd32", bytes.fromhex(        # shrd eax, edx, cl
        "89f869d7b179379e89f10fadd09c5948c1e1204809c8c3")),
    ("int_shld64", bytes.fromhex(        # shld rax, rdx, cl / shrd rdx, rax, 5 (OF/AF masked in code)
        "4889f84889fa48c1c20d89f1480fa5d09c41584181e0c7000000480facc2059c41594181e1c700000049c1e03449c1e12848"
        "31d04c31c04c31c8c3")),
    ("int_loop", bytes.fromhex(        # loop: sum of ecx..1
        "89f9"                                  # mov ecx, edi
        "31c0"                                  # xor eax, eax
        "01c8"                                  # add eax, ecx
        "e2fc"                                  # loop 4
        "c3")),                                 # ret
    ("int_loope", bytes.fromhex(        # loope / loopne walk on the low bits of eax; returns eax | rcx << 32 (rdi = count)
        "89f989f0ffc0a803e1fa89ca48c1e2204809d089f94189f041ffc041f6c003e0f748c1e1344809c84c31c0c3")),
    ("int_jrcxz", bytes.fromhex(        # jrcxz / jecxz (67 prefix) on 64-bit rdi
        "31c04889f9e30383c80167e30383c802c3")),
    ("int_leave", bytes.fromhex(        # frame via push rbp/mov/sub, value through the frame, leave
        "554889e54883ec3048897c2408488b442408c9c3")),
    ("int_enter", bytes.fromhex(        # enter 32,0 and enter 16,3: rbp-rsp distances in bits 0 and 8
        "554889e54883ec40c82000004889e84829e0c9c81000034889e94829e1c948c1e1084809c8c9c3")),
    ("int_fence", bytes.fromhex(        # mfence/sfence/lfence are no-ops for the result
        "89f8"                                  # mov eax, edi
        "0faef0"                                # mfence
        "0faef8"                                # sfence
        "0faee8"                                # lfence
        "c3")),                                 # ret
    ("int_cpuid", bytes.fromhex(        # cpuid leaf edi, returns ecx (ours-only: fixed Jaguar identity)
        "5389f831c90fa289c85bc3")),
    ("int_rdtscp", bytes.fromhex(        # rdtscp: ecx = 0 (ours-only)
        "0f01f9"                                # rdtscp
        "89c8"                                  # mov eax, ecx
        "c3")),                                 # ret
    ("int_sreg", bytes.fromhex(        # mov r32, ds/cs and mov ds, ax (ours-only: constant selectors)
        "8cd8"                                  # mov eax, ds
        "8cca"                                  # mov edx, cs
        "c1e210"                                # shl edx, 0x10
        "09d0"                                  # or eax, edx
        "668ed8"                                # mov ds, ax
        "c3")),                                 # ret
    ("v_ps", bytes.fromhex(        # ymm single-precision math, shuffles, blends, compares, conversions (rdi = 96 random bytes in)
        "c5fc1007c5fc104f20c5fc105f40c5fc58d1c5fc119700010000c5fc59d1c5fc119720010000c5fc5cd1c5fc119740010000"
        "c5fc5ed1c5fc119760010000c5fc5dd1c5fc119780010000c5fc5fd1c5fc1197a0010000c5fc54d1c5fc1197c0010000c5fc"
        "56d1c5fc1197e0010000c5fc55d1c5fc119700020000c5fc57d1c5fc119720020000c5ff7cd1c5fc119740020000c5fc14d1"
        "c5fc119760020000c5fc15d1c5fc119780020000c5fcc6d11bc5fc1197a0020000c4e37d0cd1a5c5fc1197c0020000c5fcc2"
        "d101c5fc1197e0020000c5fcc2d104c5fc119700030000c5fc51d0c5fc119720030000c4e37d4ad130c5fc119740030000c5"
        "fc5bd0c5fc119760030000c5fd5bd0c5fc119780030000c5fe5bd0c5fc1197a0030000c5fe16d0c5fc1197c0030000c5fe12"
        "d0c5fc1197e0030000c5ff12d0c5fc119700040000c4e27d185704c5fc119720040000c5fc5ad0c5fc119740040000c5fee6"
        "d0c5fc119760040000c4e37d40d1f1c5fc119780040000c4e37d08d00ac5fc1197a0040000c5fd5ad0c5f81197c0040000c5"
        "fde6d0c5f81197d0040000c5f95ad0c5f81197e0040000c5f9e6d1c5f81197f0040000c5fc2b8700050000c5fc50c0c5f877"
        "c3")),
    ("v_pd", bytes.fromhex(        # ymm and xmm double-precision math, blends, compares, dot/hadd, movemask
        "c5fd1007c5fd104f20c5fd105f40c5fd58d1c5fd119700010000c5fd5cd1c5fd119720010000c5fd59d1c5fd119740010000"
        "c5fd5ed1c5fd119760010000c5fd5dd1c5fd119780010000c5fd5fd1c5fd1197a0010000c5fd54d1c5fd1197c0010000c5fd"
        "56d1c5fd1197e0010000c5fd55d1c5fd119700020000c5fd57d1c5fd119720020000c5fd7cd1c5fd119740020000c5fdd0d1"
        "c5fd119760020000c5fd14d1c5fd119780020000c5fd15d1c5fd1197a0020000c5fdc6d106c5fd1197c0020000c4e37d0dd1"
        "09c5fd1197e0020000c4e37d4bd130c5fd119700030000c5fdc2d104c5fd119720030000c5fd51d0c5fd119740030000c4e3"
        "7d09d00ac5fd119760030000c4e37d09d009c5fd119780030000c5f97cd1c5f91197a0030000c4e37941d131c5f91197b003"
        "0000c5f9c2d102c5f91197c0030000c5f9c6d101c5f91197d0030000c4e3790dd102c5f91197e0030000c5f954d1c5f91197"
        "f0030000c5f95fd1c5f9119700040000c5f9565720c5f9119710040000c5f951d0c5f9119720040000c4e379415720ffc5f9"
        "119730040000c5f97c5720c5f9119740040000c5fd50c1c5f950c8c1e10809c8c3")),
    ("v_scalar", bytes.fromhex(        # scalar rounding/sqrt/min/max/compare, vcvtsd2si/vcvtss2si, dup/movehdup on xmm
        "c5f81007c5f8104f10c5f8105720c4e3790ad909c5f8119f00010000c4e3790a5f100ac5f8119f10010000c4e3790bd90bc5"
        "f8119f20010000c4e3790b5f1004c5f8119f30010000c4e37908d80ac5f8119f40010000c4e37909d809c5f8119f50010000"
        "c5fb51d9c5f8119f60010000c5fb515f10c5f8119f70010000c5fb5dd9c5f8119f80010000c5fb5fd9c5f8119f90010000c5"
        "fbc2d901c5f8119fa0010000c5fbc25f100ec5f8119fb0010000c5fb12d8c5f8119fc0010000c5fb125f08c5f8119fd00100"
        "00c5fa16d8c5f8119fe0010000c5fa121fc5f8119ff0010000c5fb2dc0898700020000c4e1fb2dc948898f08020000c5fa2d"
        "d0899710020000c461fa2dc14c898718020000c5fb2dc1898720020000c5f877c3")),
    ("v_fma", bytes.fromhex(        # vfnmadd132sd xmm2 = -(xmm2*xmm1)+xmm0 (needs host FMA)
        "c5f81007c5f8104f10c5f8105720c4e2f99dd1c5f8119700010000c4e2f99d5710c5f8119710010000c3")),
    ("v_int_a", bytes.fromhex(        # integer add/sub/mul/compare/pack/min/max/avg on xmm (register and memory forms)
        "c5f81007c5f8104f10c5f9fbd1c5f8119700010000c5f9f8d1c5f8119710010000c5f9f9d1c5f8119720010000c5f9fad1c5"
        "f8119730010000c5f9fcd1c5f8119740010000c5f9fdd1c5f8119750010000c5f9ddd1c5f8119760010000c5f9e8d1c5f811"
        "9770010000c4e27940d1c5f8119780010000c5f9f4d1c5f8119790010000c4e27929d1c5f81197a0010000c4e27937d1c5f8"
        "1197b0010000c5f975d1c5f81197c0010000c5f965d1c5f81197d0010000c5f960d1c5f81197e0010000c5f968d1c5f81197"
        "f0010000c5f96bd1c5f8119700020000c5f967d1c5f8119710020000c5f9f6d1c5f8119720020000c4e2793bd1c5f8119730"
        "020000c4e2793fd1c5f8119740020000c5f9ead1c5f8119750020000c5f9eed1c5f8119760020000c4e27902d1c5f8119770"
        "020000c4e2790ad1c5f8119780020000c5f9e0d1c5f8119790020000c5f9ebd1c5f81197a0020000c4e3790ed15ac5f81197"
        "b0020000c5f9f85720c5f81197c0020000c4e279405720c5f81197d0020000c4e279295720c5f81197e0020000c5f9605720"
        "c5f81197f0020000c5f96b5720c5f8119700030000c5f9f65720c5f8119710030000c5f9f45720c5f8119720030000c3")),
    ("v_int_b", bytes.fromhex(        # integer widen (pmovsx/zx), pabs, shuffles, insert/extract, shifts by immediate/xmm/m128 count
        "c5f81007c5f8104f108b473083e03fc5f96ed8c4e2791ed0c5f8119700010000c5fb70d01bc5f8119710010000c5fa70d0b1"
        "c5f8119720010000c4e27920d0c5f8119730010000c4e27921d0c5f8119740010000c4e27922d0c5f8119750010000c4e279"
        "23d0c5f8119760010000c4e27924d0c5f8119770010000c4e27925d0c5f8119780010000c4e27930d0c5f8119790010000c4"
        "e27931d0c5f81197a0010000c4e27932d0c5f81197b0010000c4e27933d0c5f81197c0010000c4e27934d0c5f81197d00100"
        "00c4e27935d0c5f81197e0010000c4e279235708c5f81197f0010000c4e279325703c5f8119700020000c4e279315705c5f8"
        "119710020000c4e279205707c5f8119720020000c4e279355709c5f8119730020000c5f9c4d003c5f8119740020000c5f9c4"
        "572a06c5f8119750020000c4e37920d009c5f8119760020000c4e37920572c0fc5f8119770020000c5f9c5c805898f800200"
        "00c4e37915878402000002c4e37914c10b898f88020000c4e37914878c02000004c4e37914c90f898f90020000c5e971d003"
        "c5f81197a0020000c5e971f005c5f81197b0020000c5e971e007c5f81197c0020000c5e972d009c5f81197d0020000c5e972"
        "f00bc5f81197e0020000c5e972e00dc5f81197f0020000c5e973f003c5f8119700030000c5e973d006c5f8119710030000c5"
        "f9d1d3c5f8119720030000c5f9f1d3c5f8119730030000c5f9e1d3c5f8119740030000c5f9d2d3c5f8119750030000c5f9f2"
        "d3c5f8119760030000c5f9e2d3c5f8119770030000c5f9f3d3c5f8119780030000c5f9d3d3c5f8119790030000c5f9d1d1c5"
        "f81197a0030000c5f9e1d1c5f81197b0030000c5f9f2d1c5f81197c0030000c5f9d35730c5f81197d0030000c3")),
    ("v_avx2", bytes.fromhex(        # ymm integer ops (AVX2 encodings, split into SSE halves; needs host AVX2)
        "c5fc1007c5fc104f20c5fdf8d1c5fc119700010000c5fdf9d1c5fc119720010000c5fdfad1c5fc119740010000c5fde8d1c5"
        "fc119760010000c5fde0d1c5fc119780010000c5fdebd1c5fc1197a0010000c5fdf85740c5fc1197c0010000c3")),
    ("v_misc", bytes.fromhex(        # legacy movss/movsd merge+load+store, (v)maskmovdqu, non-temporal moves
        "c5f81007c5f8104f10c5f8105720f20f10d0c5f8119700010000c5f8105720f30f10d1c5f8119710010000c5f8105f30f20f"
        "105f08c5f8119f20010000c5f8105f30f30f105f04c5f8119f30010000f20f118740010000f30f118f500100004889fa488d"
        "ba00020000c5f9f7c1488dba40020000660ff7c84889d7c5f9e787c0020000c4e2792a6f10c5f811afd00200000f2b8fe002"
        "0000660fe787f0020000c3")),
    ("v_zeroupper", bytes.fromhex(        # vzeroupper clears bits 255:128
        "c5fc1007"                              # vmovups ymm0, ymmword ptr [rdi]
        "c5f877"                                # vzeroupper
        "c4e37d19c101"                          # vextractf128 xmm1, ymm0, 1
        "c5f8118f00010000"                      # vmovups xmmword ptr [rdi + 0x100], xmm1
        "c5f8118710010000"                      # vmovups xmmword ptr [rdi + 0x110], xmm0
        "c3")),                                 # ret
    ("v_mxcsr", bytes.fromhex(        # (v)stmxcsr stores the power-on MXCSR; (v)ldmxcsr accepted (ours-only)
        "c5f8ae1f"                              # vstmxcsr dword ptr [rdi]
        "c5f8ae17"                              # vldmxcsr dword ptr [rdi]
        "0fae5f04"                              # stmxcsr dword ptr [rdi + 4]
        "0fae5704"                              # ldmxcsr dword ptr [rdi + 4]
        "c3")),                                 # ret
    ("mmx_ops", bytes.fromhex(        # MMX pcmpeqd/paddw/por/pinsrw/pextrw/emms
        "0f76c00ffd070feb470889f00fc4c0020fc5c0000fc5d0010fc5c802440fc5c003c1e21009d048c1e1204809c849c1e0304c"
        "09c00f77c3")),
    ("x87_arith", bytes.fromhex(        # x87 add/sub/subr/mul/div/divr in st0,sti / sti,st0 / popping forms and m32/m64/m16int/m32int operands
        "4883ec18d93c2466c74424087f02d96c2408dd07dd4708d8c1dd9f00010000dd9f08010000dd07dd4708dcc1dd9f10010000"
        "dd9f18010000dd07dd4708dec1dd9f20010000dd07dd4708d8e1dd9f28010000dd9f30010000dd07dd4708dce9dd9f380100"
        "00dd9f40010000dd07dd4708dee9dd9f48010000dd07dd4708d8e9dd9f50010000dd9f58010000dd07dd4708dce1dd9f6001"
        "0000dd9f68010000dd07dd4708dee1dd9f70010000dd07dd4708d8c9dd9f78010000dd9f80010000dd07dd4708dcc9dd9f88"
        "010000dd9f90010000dd07dd4708dec9dd9f98010000dd07dd4708d8f1dd9fa0010000dd9fa8010000dd07dd4708dcf9dd9f"
        "b0010000dd9fb8010000dd07dd4708def9dd9fc0010000dd07dd4708d8f9dd9fc8010000dd9fd0010000dd07dd4708dcf1dd"
        "9fd8010000dd9fe0010000dd07dd4708def1dd9fe8010000dd07d84710dd9ff0010000dd07dc4708dd9ff8010000dd07d867"
        "10dd9f00020000dd07dc6708dd9f08020000dd07d86f10dd9f10020000dd07dc6f08dd9f18020000dd07d84f10dd9f200200"
        "00dd07dc4f08dd9f28020000dd07d87710dd9f30020000dd07dc7708dd9f38020000dd07d87f10dd9f40020000dd07dc7f08"
        "dd9f48020000dd07de4718dd9f50020000dd07da471cdd9f58020000dd07de6718dd9f60020000dd07da671cdd9f68020000"
        "dd07de6f18dd9f70020000dd07da6f1cdd9f78020000dd07de4f18dd9f80020000dd07da4f1cdd9f88020000dd07de7718dd"
        "9f90020000dd07da771cdd9f98020000dd07de7f18dd9fa0020000dd07da7f1cdd9fa8020000d92c244883c418c3")),
    ("x87_cmp", bytes.fromhex(        # fcom/fucom/fcomp/fucomp/fcompp/fucompp/m32/m64/ficom: C0|C2|C3 folded 7 bits apart
        "4883ec18d93c2466c74424087f02d96c24084531c0dd4708dd07d8d1dfe0c1e80883e04548c1e0004909c0ddd8ddd8dd4708"
        "dd07dde1dfe0c1e80883e04548c1e0074909c0ddd8ddd8dd4708dd07d8d9dfe0c1e80883e04548c1e00e4909c0ddd8dd4708"
        "dd07dde9dfe0c1e80883e04548c1e0154909c0ddd8dd4708dd07ded9dfe0c1e80883e04548c1e01c4909c0dd4708dd07dae9"
        "dfe0c1e80883e04548c1e0234909c0dd07dc5708dfe0c1e80883e04548c1e02a4909c0ddd8dd07d85f10dfe0c1e80883e045"
        "48c1e0314909c0dd07de5718dfe0c1e80883e04548c1e0384909c0ddd8dd07da5f1cdfe0c1e80883e04548c1e03f4909c0dd"
        "07d8d0dfe0c1e80883e04548c1e0464909c0ddd84c89c0d92c244883c418c3")),
    ("x87_comi", bytes.fromhex(        # fcomi/fucomi/fcomip/fucomip flags, then a chain of all eight fcmovcc on those flags
        "4883ec18d93c2466c74424087f02d96c24084531c0dd4708dd07dbf19c5883e04548c1e0004909c0ddd8ddd8dd4708dd07db"
        "e99c5883e04548c1e0074909c0ddd8ddd8dd4708dd07dff19c5883e04548c1e00e4909c0ddd8dd4708dd07dfe99c5883e045"
        "48c1e0154909c0ddd8dd07dd4708dd4710dbf1dac2dd9700010000dbc1dd9708010000dacadd9710010000dbc9dd97180100"
        "00dad2dd9720010000dbd1dd9728010000dadadd9730010000dbd9dd9738010000ddd8ddd8ddd84c89c0d92c244883c418c3")),
    ("x87_int", bytes.fromhex(        # fild/fist/fistp/fisttp/fbstp under a dynamic rounding mode (esi), fld1/fldz/fscale/fnstcw/ffreep
        "4883ec18d93c2466c74424087f02d96c240889f083e003c1e00a0d7f0200006689442410d96c2410d9bf00030000dd07df9f"
        "00010000dd07db9f08010000dd07dfbf10010000dd07db9718010000ddd8dd07df9720010000ddd8dd07df8f28010000dd07"
        "db8f30010000dd07dd8f38010000d94710db8f40010000d94710dfbf48010000dd4708dfb750010000df4718dd9f60010000"
        "db471cdd9f68010000df6f20dd9f70010000d9e8d9eedd9f78010000dd9f80010000dd4728dd07d9fddd9f88010000ddd8d9"
        "bf90010000d9e8dd07dfc0dd9f98010000d92c244883c418c3")),
    ("x87_trig", bytes.fromhex(        # fsin/fcos/fsincos (ours-only: libm vs x87 microcode differ in the last ulps)
        "dd07d9fedd9f00010000dd07d9ffdd9f08010000dd07d9fbdd9f10010000dd9f18010000c3")),
    ("x87_env", bytes.fromhex(        # fnstenv [rdi] (ours-only: native stores instruction pointers)
        "d937"                                  # fnstenv [rdi]
        "c3")),                                 # ret
    # Imports: GOT slot 0x402010 is the JUMP_SLOT of symbol 2 (sceKernelUsleep, NID 1jfXLRVzisc).
    ("plt", lambda a: b"\xff\x25" + struct.pack("<i", 0x402010 - (a + 6)) + b"\x68\0\0\0\0" + b"\xe9\0\0\0\0"),
    ("imp_plt", lambda a: b"\xe8" + struct.pack("<i", RECOMP_FUNCS["plt"] - (a + 5)) + b"\xc3"),  # call plt; ret
    ("imp_got", lambda a: b"\xff\x15" + struct.pack("<i", 0x402010 - (a + 6)) + b"\xc3"),  # call [rip+GOT]; ret
]


def build_text(base):
    """A (FDE) calls B (no FDE); C (FDE); S (FDE) is a 3-case switch with an in-body int32 table.
    .eh_frame: one CIE 'zR' pcrel|sdata4, FDEs for A, C, S; plus .eh_frame_hdr. Returns (blob, hdr_off)."""
    func = b"\x55\x48\x89\xE5\x5D\xC3\xCC\xCC"  # push rbp; mov rbp,rsp; pop rbp; ret; int3 x2
    switch = (b"\x83\xFF\x02"                    # 0:  cmp edi,2
              b"\x77\x13"                        # 3:  ja +0x13 -> 24 (default)
              b"\x48\x8D\x0D\x10\x00\x00\x00"    # 5:  lea rcx,[rip+16] -> 28 (table)
              b"\x48\x63\x04\xB9"                # 12: movsxd rax,dword [rcx+rdi*4]
              b"\x48\x01\xC8"                    # 16: add rax,rcx
              b"\xFF\xE0"                        # 19: jmp rax
              b"\xC3\xC3\xC3\xC3"                # 21-23: cases 0..2, 24: default
              b"\xCC\xCC\xCC"                    # 25: pad to 28
              + struct.pack("<iii", 21 - 28, 22 - 28, 23 - 28))
    code = b"\xE8\x03\x00\x00\x00\xC3\xCC\xCC" + func + func + switch  # A: call A+8 (=B); ret
    funcs = [(base, 8), (base + 16, 8), (base + 24, len(switch))]
    # Recompiler end-to-end functions (SysV: args edi, esi; result eax). Addresses exported via RECOMP_FUNCS.
    for name, body in RECOMP_TESTS:
        start = len(code)
        if callable(body):  # position-dependent fixtures get their own address
            body = body(base + start)
        code += body
        funcs.append((base + start, len(body)))
        RECOMP_FUNCS[name] = base + start
        code = code.ljust((len(code) + 15) & ~15, b"\xCC")
    cie_body = struct.pack("<IB", 0, 1) + b"zR\0" + uleb(1) + bytes([0x78]) + bytes([16]) + uleb(1) + bytes([0x1B])
    cie_body = cie_body.ljust((len(cie_body) + 3) & ~3, b"\0")
    eh = bytearray(struct.pack("<I", len(cie_body)) + cie_body)
    eh_base = base + len(code)
    fdes = []
    for start, size in funcs:
        fde_off = len(eh)
        cie_ptr = fde_off + 4  # distance from the CIE-pointer field back to the CIE at offset 0
        pc_field = eh_base + fde_off + 8
        body = struct.pack("<IiI", cie_ptr, start - pc_field, size) + uleb(0)
        body = body.ljust((len(body) + 3) & ~3, b"\0")
        eh += struct.pack("<I", len(body)) + body
        fdes.append((start, eh_base + fde_off))
    eh += struct.pack("<I", 0)
    hdr_off = len(code) + len(eh)
    hdr_va = base + hdr_off
    hdr = bytes([1, 0x1B, 0x03, 0x3B]) + struct.pack("<iI", eh_base - (hdr_va + 4), len(fdes))
    hdr += b"".join(struct.pack("<ii", s - hdr_va, f - hdr_va) for s, f in fdes)
    return code + bytes(eh) + hdr, hdr_off


def build_elf():
    strtab = bytearray(b"\0")

    def s(name):
        off = len(strtab)
        strtab.extend(name.encode() + b"\0")
        return off

    n_kernel, n_eboot, n_file = s("libkernel"), s("eboot"), s("eboot.bin")
    syms = [(0, 0, 0, 0)]  # (name, info, shndx, value)
    syms.append((s("9BcDykPmo1I#B#B"), 0x12, 0, 0))        # __error, import lib 1 / module 1
    syms.append((s("1jfXLRVzisc#B#B"), 0x12, 0, 0))        # sceKernelUsleep
    syms.append((s("AAAAAAAAAAA#A#A"), 0x12, 1, 0x401000))  # export from lib 0
    symtab = b"".join(struct.pack("<IBBHQQ", n, i, 0, sh, v, 0) for n, i, sh, v in syms)
    rela = struct.pack("<QQq", 0x402000, 8, 0x401000) + struct.pack("<QQq", 0x402008, (1 << 32) | 6, 0)
    jmprel = struct.pack("<QQq", 0x402010, (2 << 32) | 7, 0)
    fingerprint = bytes(range(20))

    dynlib = bytearray()

    def put(blob):
        off = len(dynlib)
        dynlib.extend(blob)
        return off

    o_fp, o_str, o_sym = put(fingerprint), put(bytes(strtab)), put(symtab)
    o_rela, o_jmp = put(rela), put(jmprel)
    dyn = [
        (DT["FINGERPRINT"], o_fp), (DT["ORIGINAL_FILENAME"], n_file),
        (DT["MODULE_INFO"], n_eboot | (1 << 40)),
        (DT["NEEDED_MODULE"], n_kernel | (1 << 32) | (1 << 40) | (1 << 48)),
        (DT["EXPORT_LIB"], n_eboot | (1 << 32)),
        (DT["IMPORT_LIB"], n_kernel | (1 << 32) | (1 << 48)),
        (DT["STRTAB"], o_str), (DT["STRSZ"], len(strtab)), (DT["SYMTAB"], o_sym), (DT["SYMTABSZ"], len(symtab)),
        (DT["RELA"], o_rela), (DT["RELASZ"], len(rela)), (DT["JMPREL"], o_jmp), (DT["PLTRELSZ"], len(jmprel)),
        (0, 0),
    ]
    dynamic = b"".join(struct.pack("<qQ", t, v) for t, v in dyn)
    procparam = struct.pack("<QIIQ", 0x40, 0x4942524F, 1, 0x04508101).ljust(0x40, b"\0")
    text, hdr_off = build_text(0x400000)

    nph = 6
    off = 64 + 56 * nph
    layout = []
    for blob in (text, dynamic, bytes(dynlib), procparam):
        layout.append(off)
        off += len(blob)
    phdrs = [
        (PT_LOAD, 5, layout[0], 0x400000, len(text), len(text), 0x4000),
        (PT_LOAD, 6, layout[1], 0x402000, 0x18, 0x1000, 0x4000),  # data covering the reloc targets
        (PT_DYNAMIC, 4, layout[1], 0, len(dynamic), len(dynamic), 8),
        (PT_SCE_DYNLIBDATA, 4, layout[2], 0, len(dynlib), 0, 16),
        (PT_SCE_PROCPARAM, 4, layout[3], 0x403000, len(procparam), len(procparam), 8),
        (PT_GNU_EH_FRAME, 4, layout[0] + hdr_off, 0x400000 + hdr_off, len(text) - hdr_off, len(text) - hdr_off, 4),
    ]
    ident = b"\x7fELF" + bytes([2, 1, 1, 9]) + bytes(8)
    ehdr = ident + struct.pack("<HHIQQQIHHHHHH", 0xFE10, 0x3E, 1, 0x400000, 64, 0, 0, 64, 56, nph, 64, 0, 0)
    ph = b"".join(struct.pack("<IIQQQQQQ", t, f, o, v, v, fs, ms, a) for t, f, o, v, fs, ms, a in phdrs)
    return ehdr + ph + text + dynamic + bytes(dynlib) + procparam, phdrs


def build_self(elf, phdrs, encrypted):
    """fSELF: header, one blocked segment entry per phdr with file data, then the ELF headers."""
    count = len(phdrs)
    elf_hdr = elf[: 64 + 56 * count]
    data_start = 0x20 + 32 * count + len(elf_hdr)
    segs, body = b"", b""
    for i, (_, _, off, _, fsz, _, _) in enumerate(phdrs):
        flags = 0x800 | (i << 20) | (0x2 if encrypted else 0)
        segs += struct.pack("<QQQQ", flags, data_start + len(body), fsz, fsz)
        body += elf[off:off + fsz]
    header = struct.pack("<IBBBBIHHQHHI", 0x1D3D154F, 0, 1, 1, 0x12, 0x101, 0x20, 0, 0, count, 0x22, 0)
    return header + segs + elf_hdr + body


def build_sfo(entries):
    keys, data, index = bytearray(), bytearray(), bytearray()
    for k, v in entries:
        if isinstance(v, int):
            raw, fmt = struct.pack("<I", v), 0x0404
        else:
            raw, fmt = v.encode() + b"\0", 0x0204
        cap = (len(raw) + 3) & ~3
        index += struct.pack("<HHIII", len(keys), fmt, len(raw), cap, len(data))
        keys += k.encode() + b"\0"
        data += raw.ljust(cap, b"\0")
    keys = bytes(keys).ljust((len(keys) + 3) & ~3, b"\0")
    key_start = 0x14 + len(index)
    return struct.pack("<IIIII", 0x46535000, 0x101, key_start, key_start + len(keys), len(entries)) + index + keys + data


def write(path, blob):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)


def main(out):
    elf, phdrs = build_elf()
    write(os.path.join(out, "eboot.elf"), elf)
    write(os.path.join(out, "recomp_addrs.h"), "".join(
        f"#define RECOMP_{name.upper()} 0x{addr:x}ull\n" for name, addr in RECOMP_FUNCS.items()).encode())
    write(os.path.join(out, "eboot.self"), build_self(elf, phdrs, encrypted=False))
    write(os.path.join(out, "eboot.encrypted.self"), build_self(elf, phdrs, encrypted=True))
    write(os.path.join(out, "names.txt"), b"# test names\n__error\nsceKernelUsleep\n")

    game = os.path.join(out, "dump")
    write(os.path.join(game, "eboot.bin"), elf)
    write(os.path.join(game, "sce_sys", "param.sfo"),
          build_sfo([("APP_VER", "01.09"), ("CATEGORY", "gd"), ("TITLE_ID", "CUSA00207"), ("VERSION", "01.00"),
                     ("ATTRIBUTE", 0)]))
    write(os.path.join(game, "dvdroot_ps4", "data.bin"), b"synthetic data")
    write(os.path.join(out, "dlc", "sce_sys", "param.sfo"),
          build_sfo([("CATEGORY", "ac"), ("CONTENT_ID", "EP9000-CUSA00207_00-TESTDLC000000000"),
                     ("TITLE_ID", "CUSA00207")]))
    write(os.path.join(out, "old_version", "sce_sys", "param.sfo"),
          build_sfo([("APP_VER", "01.00"), ("CATEGORY", "gd"), ("TITLE_ID", "CUSA00207")]))
    write(os.path.join(out, "fake.pkg"), b"\x7fCNT" + bytes(60))
    merged = os.path.join(out, "merged")
    write(os.path.join(merged, "eboot.bin"), elf)
    write(os.path.join(merged, "sce_sys", "param.sfo"),
          build_sfo([("APP_VER", "01.09"), ("CATEGORY", "gp"), ("TITLE_ID", "CUSA00207")]))
    write(os.path.join(merged, "dvdroot_ps4", "data.bin"), b"synthetic data")
    # PKG-extractor layout: Image0/ (app files) + Sc0/ (param.sfo etc.)
    ext = os.path.join(out, "extracted")
    write(os.path.join(ext, "Image0", "eboot.bin"), elf)
    write(os.path.join(ext, "Image0", "dvdroot_ps4", "data.bin"), b"synthetic data")
    write(os.path.join(ext, "Sc0", "param.sfo"),
          build_sfo([("APP_VER", "01.09"), ("CATEGORY", "gp"), ("TITLE_ID", "CUSA00207")]))
    # entitlement-only DLC package as extracted: Sc0/ only (no Image0/)
    write(os.path.join(out, "dlc_sc0", "Sc0", "param.sfo"),
          build_sfo([("CATEGORY", "ac"), ("CONTENT_ID", "EP9000-CUSA00207_00-TESTDLC000000000"),
                     ("TITLE_ID", "CUSA00207")]))
    # update merge: PKG-extractor base 01.00 (eboot, a.bin, b.bin) + update 01.09 (b.bin overrides, d.bin is new, param.sfo 01.09 gp)
    def sfo(ver, cat, tid="CUSA00207"):
        return build_sfo([("APP_VER", ver), ("CATEGORY", cat), ("TITLE_ID", tid)])
    ub = os.path.join(out, "upd_base")
    write(os.path.join(ub, "Image0", "eboot.bin"), elf)
    write(os.path.join(ub, "Image0", "dvdroot_ps4", "a.bin"), b"base a")
    write(os.path.join(ub, "Image0", "dvdroot_ps4", "b.bin"), b"base b")
    write(os.path.join(ub, "Sc0", "param.sfo"), sfo("01.00", "gd"))
    for name, ver, tid in (("upd", "01.09", "CUSA00207"), ("upd_wrong_title", "01.09", "CUSA99999"), ("upd_old", "01.05", "CUSA00207")):
        u = os.path.join(out, name)
        write(os.path.join(u, "Image0", "dvdroot_ps4", "b.bin"), b"update b")
        write(os.path.join(u, "Image0", "dvdroot_ps4", "d.bin"), b"update d")
        write(os.path.join(u, "Sc0", "param.sfo"), sfo(ver, "gp", tid))
    # console-layout base + PKG-layout update (the layouts differ): the update's Sc0/ replaces sce_sys files of the base
    cb = os.path.join(out, "upd_cbase")
    write(os.path.join(cb, "eboot.bin"), elf)
    write(os.path.join(cb, "sce_sys", "param.sfo"), sfo("01.00", "gd"))
    write(os.path.join(cb, "sce_sys", "extra.dat"), b"base extra")
    write(os.path.join(cb, "dvdroot_ps4", "a.bin"), b"base a")
    cu = os.path.join(out, "upd_pkglayout")
    write(os.path.join(cu, "Image0", "dvdroot_ps4", "a.bin"), b"update a")
    write(os.path.join(cu, "Sc0", "param.sfo"), sfo("01.09", "gp"))
    write(os.path.join(cu, "Sc0", "extra.dat"), b"update extra")
    write(os.path.join(out, "upd_empty", "readme.txt"), b"nothing here")
    # half-extracted game: Sc0/ without Image0/
    write(os.path.join(out, "no_image0", "Sc0", "param.sfo"),
          build_sfo([("APP_VER", "01.09"), ("CATEGORY", "gp"), ("TITLE_ID", "CUSA00207")]))


if __name__ == "__main__":
    main(sys.argv[1])
