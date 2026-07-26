#pragma once

#include "context.hpp"

#include <vector>

namespace lab {

// A present-only swapchain. The compute shader never touches these images
// directly — it writes to an offscreen storage image that gets blitted here —
// so no image views and no render pass are needed at all.
class Swapchain {
public:
    void create(Context& ctx, uint32_t width, uint32_t height);
    void destroy(Context& ctx);

    VkSwapchainKHR handle() const { return handle_; }
    VkFormat format() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    const std::vector<VkImage>& images() const { return images_; }
    uint32_t imageCount() const { return static_cast<uint32_t>(images_.size()); }

private:
    VkSwapchainKHR handle_{};
    VkFormat format_{VK_FORMAT_UNDEFINED};
    VkExtent2D extent_{};
    std::vector<VkImage> images_;
};

} // namespace lab
