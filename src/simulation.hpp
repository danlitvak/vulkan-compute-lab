#pragma once

#include "context.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

namespace lab {

struct PushConstants;

// Push block for simulation shaders: the standard fields plus the particle
// count. Physics constants deliberately live in the shader as `const`s so they
// reload with it — the host only owns what determines buffer sizes.
struct SimPushConstants {
    float resolution[2];
    float mouse[2];
    float center[2];
    float zoom;
    float time;
    float deltaTime;
    uint32_t frame;
    uint32_t particleCount;
};

static_assert(sizeof(SimPushConstants) == 44, "shader Push block must match this layout");

// An N-body particle simulation running entirely on the GPU.
//
// Four passes compiled from one source file with -D:
//   SEED       one invocation per particle, writes initial conditions
//   INTEGRATE  one per particle, all-pairs force sum with shared-memory tiling
//   RENDER     one per particle, atomic scatter into a count buffer
//   TONEMAP    one per pixel, count -> colour, and zeroes the counts again
//
// Nothing is uploaded from the host: seeding is a dispatch, so the initial
// conditions live in the shader and reload with everything else.
class Simulation {
public:
    static constexpr uint32_t kWorkgroupSize = 256; // must match local_size_x in the shader

    bool create(Context& ctx, const std::filesystem::path& shader, uint32_t particleCount,
                const std::vector<VkImageView>& targetViews, VkExtent2D extent);
    void destroy(Context& ctx);

    // Rebuilds only the pipelines, keeping particle state across a shader edit.
    bool reloadPipelines(Context& ctx, const std::filesystem::path& shader);

    // Swapchain resize: the accumulation images follow the framebuffer.
    bool resize(Context& ctx, const std::vector<VkImageView>& targetViews, VkExtent2D extent);

    void record(VkCommandBuffer cmd, uint32_t frameSlot, const SimPushConstants& push);

    void requestSeed() { seedPending_ = true; }
    void setPaused(bool paused) { paused_ = paused; }
    bool paused() const { return paused_; }
    uint32_t particleCount() const { return particleCount_; }

private:
    struct Buffer {
        VkBuffer handle{};
        VkDeviceMemory memory{};
    };
    struct AccumImage {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
    };

    // Builds all four passes into `out`, cleaning up after itself on failure so a
    // bad edit never leaves half a pipeline set behind.
    bool buildPipelines(const std::filesystem::path& shader, VkPipeline out[4]) const;
    void destroyPipelines();
    bool createAccumImages(Context& ctx, VkExtent2D extent);
    void destroyAccumImages(Context& ctx);
    void writeDescriptors(Context& ctx, const std::vector<VkImageView>& targetViews);
    void clearAccumImages(Context& ctx);

    VkDevice device_{};
    uint32_t particleCount_{0};
    uint32_t frameSlots_{0};
    VkExtent2D extent_{};

    Buffer particles_[2]{};
    std::vector<AccumImage> accum_;

    VkDescriptorSetLayout setLayout_{};
    VkPipelineLayout pipelineLayout_{};
    VkDescriptorPool descriptorPool_{};
    // [frame slot][parity]: parity selects which particle buffer is read.
    std::vector<VkDescriptorSet> descriptors_;

    VkPipeline seedPipeline_{};
    VkPipeline integratePipeline_{};
    VkPipeline renderPipeline_{};
    VkPipeline tonemapPipeline_{};

    uint32_t parity_{0};
    bool seedPending_{true};
    bool paused_{false};
};

} // namespace lab
