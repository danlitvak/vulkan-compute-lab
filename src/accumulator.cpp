#include "accumulator.hpp"

#include "shader_compiler.hpp"

#include <array>
#include <cstdio>

namespace fs = std::filesystem;

namespace lab {
namespace {

// Must match `rgba32f` in the shader. R32G32B32A32_SFLOAT is one of the formats
// Vulkan requires to support STORAGE_IMAGE with load *and* store, so no feature
// check is needed before read-modify-writing it.
constexpr VkFormat kAccumFormat = VK_FORMAT_R32G32B32A32_SFLOAT;

uint32_t groupCount(uint32_t threads, uint32_t local) {
    return (threads + local - 1) / local;
}

} // namespace

bool Accumulator::create(Context& ctx, const fs::path& shader,
                         const std::vector<VkImageView>& targetViews, VkExtent2D extent) {
    device_ = ctx.device();

    if (!createImage(ctx, extent)) return false;

    // 0 = the rgba8 display target, 1 = this mode's rgba32f sum.
    const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    }};

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &setLayout_));

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    // One set per frame slot even though accumulation only ever uses slot 0. The
    // sets are nearly free, and this way nothing here has to know which slot the
    // frame loop happens to be on.
    const uint32_t setCount = static_cast<uint32_t>(targetViews.size());
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, setCount * 2};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = setCount;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));

    std::vector<VkDescriptorSetLayout> layouts(setCount, setLayout_);
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = setCount;
    allocInfo.pSetLayouts = layouts.data();
    descriptors_.resize(setCount);
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, descriptors_.data()));

    writeDescriptors(targetViews);

    VkPipeline built = VK_NULL_HANDLE;
    if (!buildPipeline(shader, built)) return false;
    pipeline_ = built;

    requestReset();
    return true;
}

bool Accumulator::createImage(Context& ctx, VkExtent2D extent) {
    extent_ = extent;

    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = kAccumFormat;
    info.extent = {extent.width, extent.height, 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    // TRANSFER_DST is what lets a reset be a vkCmdClearColorImage recorded into
    // the frame, rather than a dispatch or a host upload.
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device_, &info, nullptr, &image_));

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device_, image_, &requirements);
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    alloc.allocationSize = requirements.size;
    alloc.memoryTypeIndex =
        ctx.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device_, &alloc, nullptr, &memory_));
    VK_CHECK(vkBindImageMemory(device_, image_, memory_, 0));

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = kAccumFormat;
    viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &view_));
    return true;
}

void Accumulator::destroyImage() {
    if (view_) vkDestroyImageView(device_, view_, nullptr);
    if (image_) vkDestroyImage(device_, image_, nullptr);
    if (memory_) vkFreeMemory(device_, memory_, nullptr);
    view_ = VK_NULL_HANDLE;
    image_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
}

void Accumulator::writeDescriptors(const std::vector<VkImageView>& targetViews) {
    for (size_t slot = 0; slot < descriptors_.size() && slot < targetViews.size(); ++slot) {
        VkDescriptorImageInfo targetInfo{};
        targetInfo.imageView = targetViews[slot];
        targetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // the only layout storage images can use

        VkDescriptorImageInfo accumInfo{};
        accumInfo.imageView = view_;
        accumInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        std::array<VkWriteDescriptorSet, 2> writes{};
        for (VkWriteDescriptorSet& write : writes) {
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptors_[slot];
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        }
        writes[0].dstBinding = 0;
        writes[0].pImageInfo = &targetInfo;
        writes[1].dstBinding = 1;
        writes[1].pImageInfo = &accumInfo;

        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

bool Accumulator::buildPipeline(const fs::path& shader, VkPipeline& out) const {
    out = VK_NULL_HANDLE;

    // No -D: unlike the simulation this is a single pass, so the file compiles
    // once and means one thing.
    const CompileResult compiled = compileComputeShader(shader);
    if (!compiled.ok) {
        std::fprintf(stderr, "[lab] shader compile failed:\n%s\n", compiled.message.c_str());
        return false;
    }
    if (!compiled.message.empty()) std::fprintf(stderr, "%s", compiled.message.c_str());

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = compiled.spirv.size() * sizeof(uint32_t);
    moduleInfo.pCode = compiled.spirv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device_, &moduleInfo, nullptr, &module));

    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = pipelineLayout_;

    const VkResult created = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &out);
    vkDestroyShaderModule(device_, module, nullptr); // the pipeline owns a copy now
    if (created != VK_SUCCESS) {
        std::fprintf(stderr, "[lab] pipeline creation failed: %s\n", vkResultString(created));
        return false;
    }
    return true;
}

bool Accumulator::reloadPipeline(Context&, const fs::path& shader) {
    // Build into a temporary first: a failed edit must leave the running pipeline
    // untouched, exactly like the other two paths.
    VkPipeline rebuilt = VK_NULL_HANDLE;
    if (!buildPipeline(shader, rebuilt)) return false;

    vkDeviceWaitIdle(device_); // the old pipeline may still be in flight
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    pipeline_ = rebuilt;

    requestReset();
    return true;
}

bool Accumulator::resize(Context& ctx, const std::vector<VkImageView>& targetViews,
                         VkExtent2D extent) {
    // A swapchain recreation does not imply a new pixel grid. VK_SUBOPTIMAL_KHR
    // alone triggers one, and some drivers report it steadily without the extent
    // ever changing — which would throw the accumulated sum away every frame and
    // leave a path trace permanently stuck at one sample. Only the descriptors
    // need refreshing when the size is the same, since the target views were
    // recreated underneath us.
    if (extent.width == extent_.width && extent.height == extent_.height && image_ != VK_NULL_HANDLE) {
        writeDescriptors(targetViews);
        return true;
    }

    // The caller has already waited for the device to go idle to recreate the
    // swapchain, so tearing the image down here is safe.
    destroyImage();
    if (!createImage(ctx, extent)) return false;
    writeDescriptors(targetViews);
    requestReset();
    return true;
}

void Accumulator::requestReset() {
    sampleIndex_ = 0;
    clearPending_ = true;
}

void Accumulator::record(VkCommandBuffer cmd, uint32_t frameSlot, PushConstants push) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image_;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

    if (clearPending_) {
        clearPending_ = false;

        // oldLayout UNDEFINED rather than GENERAL: a clear overwrites every
        // pixel, so there is nothing to preserve, and discarding lets this one
        // path also serve the image's very first use without a separate initial
        // transition submitted at creation time.
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);

        const VkClearColorValue zero{};
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd, image_, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);

        // The dispatch must see zeroes, not whatever the clear had reached.
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &barrier);
    } else {
        // Order this frame's read-modify-write against the previous frame's. The
        // barrier reaches back across the submission because a barrier's first
        // scope covers everything already submitted to the queue, and with a
        // single frame in flight the previous frame is always already submitted.
        // Without it the load could observe a partially written sum.
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &barrier);
    }

    // Stamped here rather than trusted from the caller: a reset between filling
    // the block and recording would otherwise leave the shader dividing a
    // freshly cleared sum by a stale sample count.
    push.sampleIndex = sampleIndex_;

    const VkDescriptorSet set = descriptors_[frameSlot];
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &set, 0,
                            nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, groupCount(extent_.width, kWorkgroupSize),
                  groupCount(extent_.height, kWorkgroupSize), 1);

    // Saturate instead of wrapping. At 60 fps this takes two years to reach, but
    // wrapping to 0 would make the shader divide a huge sum by one and flash the
    // whole image white, which is a far worse failure than the image quietly
    // stopping its refinement.
    if (sampleIndex_ < UINT32_MAX) ++sampleIndex_;
}

void Accumulator::destroy(Context&) {
    if (pipeline_) vkDestroyPipeline(device_, pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;

    destroyImage();

    if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    if (setLayout_) vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
    descriptorPool_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    setLayout_ = VK_NULL_HANDLE;
    descriptors_.clear();
}

} // namespace lab
