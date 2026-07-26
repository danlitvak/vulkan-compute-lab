#pragma once

#include <vulkan/vulkan.h>

namespace lab {

const char* vkResultString(VkResult result);

[[noreturn]] void vkThrow(VkResult result, const char* expr, const char* file, int line);

} // namespace lab

// Every Vulkan call that can fail goes through this. Vulkan reports errors by
// return code only, so an unchecked call fails silently and shows up much later
// as a corrupt frame or a device loss.
#define VK_CHECK(expr)                                                          \
    do {                                                                        \
        VkResult vkCheckResult_ = (expr);                                       \
        if (vkCheckResult_ != VK_SUCCESS)                                       \
            ::lab::vkThrow(vkCheckResult_, #expr, __FILE__, __LINE__);          \
    } while (false)
