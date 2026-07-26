#pragma once

#include "context.hpp"
#include "push_constants.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace lab {

// Progressive accumulation, selected by a `//!accumulate` directive: one
// dispatch per frame traces a single new sample per pixel, adds it to a
// persistent float image, and writes the running mean to the display target.
//
// One sample per frame rather than N samples per dispatch is what keeps the
// window alive. A path tracer that loops internally hands the GPU a multi-second
// dispatch, which on Windows means the watchdog kills the device; this way the
// first frame is noisy but immediate and the image refines while you watch, and
// flying the camera stays responsive because every frame is still a frame.
//
// Bindings, matching the shader:
//   0  rgba8    target      the tonemapped mean, blitted to the swapchain
//   1  rgba32f  accumImage  the running sum, read *and* written
//
// fp32 for the sum, not fp16: a half-float running total stops growing after a
// few thousand samples, because the increment falls below the ULP of the sum and
// every further sample rounds away to nothing.
//
// The host owns the sample counter, and the shader divides by sampleIndex + 1
// after adding its own sample. Keeping the count on the host rather than in the
// image is what makes a reset a single assignment instead of a readback.
class Accumulator {
public:
    static constexpr uint32_t kWorkgroupSize = 16; // matches local_size_x/y in the shader

    bool create(Context& ctx, const std::filesystem::path& shader,
                const std::vector<VkImageView>& targetViews, VkExtent2D extent);
    void destroy(Context& ctx);

    // Rebuilds only the pipeline, keeping the resources. A shader edit changes
    // what a sample *means*, so the samples already summed are no longer
    // comparable with the ones about to be traced: this resets too.
    bool reloadPipeline(Context& ctx, const std::filesystem::path& shader);

    // Swapchain resize: the accumulation image follows the framebuffer, and its
    // contents cannot be carried across a different pixel grid.
    bool resize(Context& ctx, const std::vector<VkImageView>& targetViews, VkExtent2D extent);

    // Records the clear (if a reset is pending) and the single dispatch. Stamps
    // the live sample count into the block it pushes, so the caller cannot hand
    // over a count that a reset has since invalidated.
    void record(VkCommandBuffer cmd, uint32_t frameSlot, PushConstants push);

    // Throws the sum away: the next frame starts from sample 0 again.
    void requestReset();

    // How many samples are already in the buffer, which is what the shader must
    // divide by after adding its own — hence the + 1 on the shader side.
    uint32_t sampleIndex() const { return sampleIndex_; }

private:
    bool buildPipeline(const std::filesystem::path& shader, VkPipeline& out) const;
    bool createImage(Context& ctx, VkExtent2D extent);
    void destroyImage();
    void writeDescriptors(const std::vector<VkImageView>& targetViews);

    VkDevice device_{};
    VkExtent2D extent_{};

    // ONE image shared by every frame, not one per frame slot: the whole point
    // is that it survives from frame to frame. That is also why this mode runs
    // with a single frame in flight — see App::setFramesInFlight.
    VkImage image_{};
    VkDeviceMemory memory_{};
    VkImageView view_{};

    VkDescriptorSetLayout setLayout_{};
    VkPipelineLayout pipelineLayout_{};
    VkDescriptorPool descriptorPool_{};
    std::vector<VkDescriptorSet> descriptors_; // one per frame slot; differ only in the target view
    VkPipeline pipeline_{};

    uint32_t sampleIndex_{0};
    // Set, not acted on immediately: the clear is recorded into the next frame's
    // command buffer. Resetting from a queue submit of its own would block the
    // main loop on a fence every time the camera moves, which is every frame of
    // a drag.
    bool clearPending_{true};
};

} // namespace lab
