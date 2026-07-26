#include "swapchain.hpp"

#include <algorithm>
#include <stdexcept>

namespace lab {
namespace {

VkSurfaceFormatKHR chooseFormat(const std::vector<VkSurfaceFormatKHR>& available) {
    // Prefer a plain UNORM format: the compute shader writes 8-bit sRGB-ish
    // values directly, and a UNORM target means the blit copies them verbatim
    // instead of applying an extra sRGB encode.
    for (const auto& format : available)
        if ((format.format == VK_FORMAT_B8G8R8A8_UNORM || format.format == VK_FORMAT_R8G8B8A8_UNORM) &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            return format;
    return available.front();
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& available) {
    for (VkPresentModeKHR mode : available)
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
    return VK_PRESENT_MODE_FIFO_KHR; // always supported
}

} // namespace

void Swapchain::create(Context& ctx, uint32_t width, uint32_t height) {
    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(ctx.physicalDevice(), ctx.surface(), &caps));

    uint32_t formatCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice(), ctx.surface(), &formatCount, nullptr));
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(ctx.physicalDevice(), ctx.surface(), &formatCount, formats.data()));

    uint32_t modeCount = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice(), ctx.surface(), &modeCount, nullptr));
    std::vector<VkPresentModeKHR> modes(modeCount);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(ctx.physicalDevice(), ctx.surface(), &modeCount, modes.data()));

    if (formats.empty() || modes.empty()) throw std::runtime_error("surface reports no formats or present modes");

    // Blitting into the swapchain requires TRANSFER_DST; every desktop driver
    // supports it, but a wrong assumption here fails as a confusing validation
    // error much later, so check up front.
    if ((caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0)
        throw std::runtime_error("surface does not support TRANSFER_DST — cannot blit compute output to it");

    const VkSurfaceFormatKHR surfaceFormat = chooseFormat(formats);

    if (caps.currentExtent.width != UINT32_MAX) {
        extent_ = caps.currentExtent;
    } else {
        extent_.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent_.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    VkSwapchainCreateInfoKHR info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    info.surface = ctx.surface();
    info.minImageCount = imageCount;
    info.imageFormat = surfaceFormat.format;
    info.imageColorSpace = surfaceFormat.colorSpace;
    info.imageExtent = extent_;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.preTransform = caps.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode = choosePresentMode(modes);
    info.clipped = VK_TRUE;
    info.oldSwapchain = handle_;

    VkSwapchainKHR created = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSwapchainKHR(ctx.device(), &info, nullptr, &created));

    if (handle_) vkDestroySwapchainKHR(ctx.device(), handle_, nullptr);
    handle_ = created;
    format_ = surfaceFormat.format;

    uint32_t count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(ctx.device(), handle_, &count, nullptr));
    images_.resize(count);
    VK_CHECK(vkGetSwapchainImagesKHR(ctx.device(), handle_, &count, images_.data()));
}

void Swapchain::destroy(Context& ctx) {
    if (handle_) vkDestroySwapchainKHR(ctx.device(), handle_, nullptr);
    handle_ = VK_NULL_HANDLE;
    images_.clear();
}

} // namespace lab
