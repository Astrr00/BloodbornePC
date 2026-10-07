// SPDX-License-Identifier: GPL-3.0-or-later
// Presentation backend (SDL3 window + Vulkan swapchain). The guest's flips are forwarded here from the HLE layer; the
// window loop runs on the process main thread (SDL requires it for event handling on most platforms).
#pragma once
#include <atomic>
#include <cstdint>

namespace bb::gpu {

// Creates the window and the Vulkan swapchain. Call on the main thread before running the guest.
bool init(int width, int height, const char* title);

// Posts a presentation request from any thread (guest flip). Cheap and non-blocking.
void request_present(uint32_t buffer_index, uint64_t guest_addr, uint32_t width, uint32_t height);

// Window/event loop: returns when the window is closed or `stop` becomes true. Main thread only.
void run(const std::atomic<bool>& stop);

void shutdown();

} // namespace bb::gpu
