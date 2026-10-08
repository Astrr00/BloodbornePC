cmake_minimum_required(VERSION 3.24)
project(BloodbornePC LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Zydis (MIT) - x86-64 decoder for function discovery and later the recompiler.
include(FetchContent)
set(ZYDIS_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(ZYDIS_BUILD_DOXYGEN OFF CACHE BOOL "" FORCE)
FetchContent_Declare(zydis
    GIT_REPOSITORY https://github.com/zyantific/zydis.git
    GIT_TAG v4.1.1
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(zydis)

# SPIR-V headers (Apache-2.0/MIT) for the GCN->SPIR-V translator; SPIRV-Tools (Apache-2.0) only to validate its output.
FetchContent_Declare(spirv_headers GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers.git GIT_TAG vulkan-sdk-1.4.363.0 GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(spirv_headers)
option(BB_SPIRV_VALIDATE "Link SPIRV-Tools into bbspv/test_spirv to validate generated SPIR-V" ON)
if(BB_SPIRV_VALIDATE)
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
    set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
    set(SPIRV-Headers_SOURCE_DIR ${spirv_headers_SOURCE_DIR} CACHE PATH "" FORCE)
    FetchContent_Declare(spirv_tools GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools.git GIT_TAG vulkan-sdk-1.4.363.0 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(spirv_tools)
endif()

# LibAtrac9 (MIT, Alex Barney) decodes the ATRAC9 streams the engine feeds through libSceAjm. Third-party C code: warnings silenced, no DLL export decoration needed.
FetchContent_Declare(libatrac9 GIT_REPOSITORY https://github.com/Thealexbarney/LibAtrac9.git GIT_TAG efca2e3af35562a09a9bb6deed90e45b4b824dc4 SOURCE_SUBDIR no_cmake)  # sources only: no add_subdirectory
FetchContent_MakeAvailable(libatrac9)
file(GLOB BB_ATRAC9_SOURCES ${libatrac9_SOURCE_DIR}/C/src/*.c)
add_library(bbatrac9 STATIC ${BB_ATRAC9_SOURCES})
target_compile_definitions(bbatrac9 PRIVATE COMPILING_DLL)
if(MSVC)
    target_compile_options(bbatrac9 PRIVATE /W0)
else()
    target_compile_options(bbatrac9 PRIVATE -w)
endif()

if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/W4> $<$<COMPILE_LANGUAGE:C,CXX>:/permissive->
                        $<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra)
endif()

# Image parsing only (ELF/SELF, NIDs, hashes): shared by tools and the game runtime, no disassembler.
add_library(bbimage STATIC
    src/core/hash.cpp
    src/core/nid.cpp
    src/core/orbis_elf.cpp
    src/core/sfo.cpp)
target_include_directories(bbimage PUBLIC src)

# Analysis + recompiler (tools only).
add_library(bbcore STATIC
    src/core/functions.cpp
    src/core/dump.cpp
    src/recomp/emitter.cpp)
target_link_libraries(bbcore PUBLIC bbimage Zydis)

add_library(bbrt STATIC src/runtime/guest.cpp src/runtime/loader.cpp src/runtime/hle_registry.cpp
                        src/runtime/process.cpp)
target_link_libraries(bbrt PUBLIC bbimage)

# Host implementations of PS4 system libraries (HLE).
add_library(bbhle STATIC src/hle/hle.cpp src/hle/libc.cpp src/hle/kernel.cpp src/hle/thread.cpp src/hle/libc_wrap.cpp src/hle/crash.cpp src/hle/system.cpp src/hle/guest_heap.cpp src/hle/net.cpp src/hle/np.cpp src/hle/audio.cpp src/hle/vfs.cpp src/hle/gnm.cpp src/hle/input.cpp src/hle/services.cpp src/hle/savedata.cpp src/hle/ajm.cpp src/hle/ime.cpp src/hle/sleep.cpp src/hle/avplayer.cpp)
add_library(bbgcn STATIC src/gpu/gcn.cpp src/gpu/gcn_spirv.cpp)
target_include_directories(bbgcn PUBLIC src)
target_link_libraries(bbgcn PUBLIC SPIRV-Headers::SPIRV-Headers)
target_link_libraries(bbhle PUBLIC bbrt bbgcn bbatrac9)
if(WIN32)
    target_link_libraries(bbhle PRIVATE mfplat mfreadwrite mfuuid ole32)  # AvPlayer movies: OS Media Foundation (H.264/AAC)
endif()

add_executable(bbinstall tools/bbinstall/main.cpp)
target_link_libraries(bbinstall PRIVATE bbcore)

add_executable(bbelf tools/bbelf/main.cpp)
target_link_libraries(bbelf PRIVATE bbcore)

if(BB_SPIRV_VALIDATE)
    add_executable(bbspv tools/bbspv/main.cpp)
    target_link_libraries(bbspv PRIVATE bbgcn SPIRV-Tools)
endif()

add_executable(bbrecomp tools/bbrecomp/main.cpp)
target_link_libraries(bbrecomp PRIVATE bbcore)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_core tests/test_core.cpp)
    target_link_libraries(test_core PRIVATE bbcore)
    add_test(NAME core COMMAND test_core)

    add_executable(test_process tests/test_process.cpp)
    target_link_libraries(test_process PRIVATE bbrt)
    add_test(NAME process COMMAND test_process)

    add_executable(test_gcn tests/test_gcn.cpp)
    target_link_libraries(test_gcn PRIVATE bbgcn)
    add_test(NAME gcn COMMAND test_gcn)

    if(WIN32)
        add_executable(test_guest_heap tests/test_guest_heap.cpp src/hle/guest_heap.cpp)
        target_include_directories(test_guest_heap PRIVATE src)
        add_test(NAME guest_heap COMMAND test_guest_heap)
    endif()

    if(BB_SPIRV_VALIDATE)
        add_executable(test_spirv tests/test_spirv.cpp)
        target_link_libraries(test_spirv PRIVATE bbgcn SPIRV-Tools)
        add_test(NAME spirv COMMAND test_spirv)
    endif()

    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_FOUND)
        add_test(NAME cli
            COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py
                    $<TARGET_FILE_DIR:bbelf> ${CMAKE_CURRENT_BINARY_DIR}/fixtures
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

        # Recompiler end-to-end: fixture ELF -> bbrecomp --emit -> compile generated C++ -> run.
        set(RECOMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/recomp_fixture)
        add_custom_command(
            OUTPUT ${RECOMP_DIR}/eboot.elf ${RECOMP_DIR}/recomp_addrs.h
            COMMAND Python3::Interpreter ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_fixtures.py ${RECOMP_DIR}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/make_fixtures.py)
        # Same test twice: default bias (0 for this fixture) and a non-zero bias, which moves every absolute guest
        # address (return addresses, switch cases, RIP-relative data, dispatch table, loader relocations).
        foreach(variant IN ITEMS default bias)
            if(variant STREQUAL "bias")
                set(BIAS_ARGS --bias 100000)
                set(TEST_NAME test_recomp_bias)
            else()
                set(BIAS_ARGS)
                set(TEST_NAME test_recomp)
            endif()
            add_custom_command(
                OUTPUT ${RECOMP_DIR}/generated_${variant}.cpp
                COMMAND bbrecomp ${RECOMP_DIR}/eboot.elf --emit ${RECOMP_DIR}/generated_${variant}.cpp ${BIAS_ARGS}
                DEPENDS bbrecomp ${RECOMP_DIR}/eboot.elf)
            add_executable(${TEST_NAME} tests/test_recomp.cpp ${RECOMP_DIR}/generated_${variant}.cpp
                                        ${RECOMP_DIR}/recomp_addrs.h)
            target_include_directories(${TEST_NAME} PRIVATE ${RECOMP_DIR})
            target_link_libraries(${TEST_NAME} PRIVATE bbrt)
            # Guest code is AVX (Jaguar); generated intrinsics need the host compiler to target AVX too.
            if(MSVC)
                target_compile_options(${TEST_NAME} PRIVATE /arch:AVX)
            else()
                target_compile_options(${TEST_NAME} PRIVATE -mavx)
            endif()
            add_test(NAME recomp_${variant} COMMAND ${TEST_NAME} ${RECOMP_DIR}/eboot.elf)
        endforeach()
    endif()
endif()

# Local differential test against the user's own eboot (never committed): cmake -DBB_EBOOT=<path to eboot.bin>
set(BB_EBOOT "" CACHE FILEPATH "eboot.bin of your own dump, enables the bbdiff target")
set(BB_DIFF_FUNCTIONS 3000 CACHE STRING "number of leaf functions bbdiff samples")
if(BB_EBOOT AND MSVC)
    enable_language(ASM_MASM)
    set(LEAVES ${CMAKE_CURRENT_BINARY_DIR}/eboot_leaves.cpp)
    add_custom_command(
        OUTPUT ${LEAVES}
        COMMAND bbrecomp ${BB_EBOOT} --emit-leaves ${LEAVES} ${BB_DIFF_FUNCTIONS}
        DEPENDS bbrecomp ${BB_EBOOT})
    add_executable(bbdiff tools/bbdiff/main.cpp tools/bbdiff/native_call_win64.asm ${LEAVES})
    target_link_libraries(bbdiff PRIVATE bbrt)
    target_compile_options(bbdiff PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX>)
    set_source_files_properties(${LEAVES} PROPERTIES COMPILE_OPTIONS "/bigobj")
elseif(BB_EBOOT)
    message(WARNING "bbdiff currently needs MSVC/clang-cl (MASM trampoline); a SysV/GAS variant is not written yet.")
endif()

# --- full eboot build (local, BB_EBOOT) ---
# cmake -DBB_EBOOT=<eboot.bin> [-DBB_GAME_OPT=O2|Od] ; cmake --build <dir> --target bbgame
# Shards are generated at build time into <build>/gen (never committed).
set(BB_GAME_OPT "Od" CACHE STRING "optimisation for generated shards: O2 or Od")
set(BB_SHARD_FUNCTIONS 2000 CACHE STRING "functions per generated shard")
# Optional NID name list (e.g. from ps4libdoc; none is bundled): only adds import names to logs and stub reports.
set(BB_NAMES "" CACHE FILEPATH "optional NID name list for bbrecomp --names")
set(BB_NAMES_ARGS)
if(BB_NAMES AND EXISTS "${BB_NAMES}")
    set(BB_NAMES_ARGS --names ${BB_NAMES})
endif()
if(BB_EBOOT)
    set(GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/gen)
    set(GEN_STAMP ${GEN_DIR}/sources.cmake)
    if(NOT EXISTS ${GEN_STAMP})  # first configure: empty list; build target bbgen, the next build re-configures
        file(MAKE_DIRECTORY ${GEN_DIR})
        file(WRITE ${GEN_STAMP} "set(BB_GENERATED_SOURCES)\n")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${GEN_STAMP})
    include(${GEN_STAMP})
    set_source_files_properties(${BB_GENERATED_SOURCES} PROPERTIES GENERATED TRUE)
    add_custom_command(
        OUTPUT ${GEN_DIR}/emitted.stamp
        COMMAND bbrecomp ${BB_EBOOT} --emit-dir ${GEN_DIR} ${BB_SHARD_FUNCTIONS} ${BB_NAMES_ARGS}
        COMMAND ${CMAKE_COMMAND} -E touch ${GEN_DIR}/emitted.stamp
        DEPENDS bbrecomp ${BB_EBOOT} BYPRODUCTS ${GEN_STAMP} ${BB_GENERATED_SOURCES} VERBATIM)
    add_custom_target(bbgen DEPENDS ${GEN_DIR}/emitted.stamp)
    # Presentation: SDL3 (zlib) window + Vulkan via volk (MIT) and the Khronos headers (Apache-2.0).
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(sdl3 GIT_REPOSITORY https://github.com/libsdl-org/SDL.git GIT_TAG release-3.2.8 GIT_SHALLOW TRUE)
    FetchContent_Declare(vulkan_headers GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git GIT_TAG v1.4.365 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(sdl3 vulkan_headers)
    FetchContent_Declare(volk GIT_REPOSITORY https://github.com/zeux/volk.git GIT_TAG vulkan-sdk-1.4.363.0 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(volk)
    # VulkanMemoryAllocator (MIT), header-only; private to bbgpu so bbgame's compile flags stay untouched.
    FetchContent_Declare(vma GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git GIT_TAG v3.3.0 GIT_SHALLOW TRUE SOURCE_SUBDIR no_cmake)  # header-only: no add_subdirectory
    FetchContent_MakeAvailable(vma)
    add_library(bbgpu STATIC src/gpu/present.cpp src/gpu/vk_context.cpp src/gpu/backend.cpp src/gpu/pad.cpp src/gpu/nav.cpp src/gpu/cheats.cpp src/gpu/audio_out.cpp src/gpu/write_watch.cpp)
    option(BB_DEV_TOOLS "Scripted-play dev tools in src/gpu/nav.cpp (telemetry, navigation, BB_DEV_GODMODE, BB_DEV_NOHIT, BB_DEV_TELEPORT, live 'tp'/'hwwatch'); never in a release" OFF)
    set_source_files_properties(src/gpu/nav.cpp PROPERTIES COMPILE_DEFINITIONS BB_DEV_TOOLS=$<BOOL:${BB_DEV_TOOLS}>)
    target_include_directories(bbgpu PRIVATE ${vma_SOURCE_DIR}/include)
    target_include_directories(bbgpu PUBLIC src)
    target_link_libraries(bbgpu PUBLIC SDL3::SDL3-static volk::volk_headers)
    target_compile_definitions(bbgpu PUBLIC VK_NO_PROTOTYPES)
    target_link_libraries(bbgpu PRIVATE volk bbgcn)
    add_executable(bbgame EXCLUDE_FROM_ALL src/app/main.cpp ${BB_GENERATED_SOURCES})
    add_dependencies(bbgame bbgen)
    target_link_libraries(bbgame PRIVATE bbhle bbgpu)
    if(MSVC)
        target_link_options(bbgame PRIVATE /MAP)
    endif()
    option(BB_TRACE_CALLS "record the last 64 guest function entries per thread (crash reports)" ON)
    if(BB_TRACE_CALLS)
        target_compile_definitions(bbgame PRIVATE BB_TRACE_CALLS)
    endif()
    if(MSVC)
        target_compile_options(bbgame PRIVATE /arch:AVX /bigobj /${BB_GAME_OPT} /W1)
        set_source_files_properties(src/app/main.cpp PROPERTIES COMPILE_OPTIONS "/W4")
        # MSVC silently compiles very large functions (76 shard functions, e.g. the draw-command emitter f_26bc830) without
        # optimization (C4883, off at /W1). The flag optimizes them anyway (main thread -2.5 ms/frame); /w14883 reports any that
        # still are not. Costs build time (full shard rebuild about 2x). Existing build dirs keep their cached value: pass -D...=ON.
        option(BB_OPTIMIZE_HUGE_FUNCTIONS "MSVC: optimize huge generated functions (/d2OptimizeHugeFunctions)" ON)
        if(BB_OPTIMIZE_HUGE_FUNCTIONS)
            target_compile_options(bbgame PRIVATE /d2OptimizeHugeFunctions /w14883)
        endif()
    else()
        # Guest code is AVX (Jaguar); the generated shards are machine output, so no warnings there. -O2/-O0 like the MSVC /O2//Od switch.
        string(REPLACE "O2" "-O2" BB_GCC_OPT "${BB_GAME_OPT}")
        string(REPLACE "Od" "-O0" BB_GCC_OPT "${BB_GCC_OPT}")
        target_compile_options(bbgame PRIVATE -mavx ${BB_GCC_OPT} -w)
        set_source_files_properties(src/app/main.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra")
        target_link_libraries(bbgame PRIVATE pthread dl)
    endif()
endif()
