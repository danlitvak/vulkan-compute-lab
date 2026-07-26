#include "simulation.hpp"

#include "shader_compiler.hpp"

#include <array>
#include <cstdio>

namespace fs = std::filesystem;

namespace lab {
namespace {

// vec2 pos, vec2 vel, float mass, float pad. std430 puts the array stride at 24:
// the struct is 24 bytes and its alignment is 8, so there is no tail padding.
constexpr VkDeviceSize kParticleStride = 24;
constexpr VkFormat kAccumFormat = VK_FORMAT_R32_UINT;
constexpr uint32_t kTonemapGroup = 16; // matches local_size in the TONEMAP pass

uint32_t groupCount(uint32_t threads, uint32_t local) {
    return (threads + local - 1) / local;
}

void bufferBarrier(VkCommandBuffer cmd, VkBuffer buffer, VkAccessFlags src, VkAccessFlags dst) {
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = src;
    barrier.dstAccessMask = dst;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void accumBarrier(VkCommandBuffer cmd, VkImage image) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

bool Simulation::create(Context& ctx, const fs::path& shader, uint32_t particleCount,
                        const std::vector<VkImageView>& targetViews, VkExtent2D extent) {
    device_ = ctx.device();
    particleCount_ = particleCount;
    frameSlots_ = static_cast<uint32_t>(targetViews.size());
    extent_ = extent;

    // --- particle buffers, ping-ponged -------------------------------------
    const VkDeviceSize size = VkDeviceSize(particleCount_) * kParticleStride;
    for (Buffer& buffer : particles_) {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(device_, &info, nullptr, &buffer.handle));

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, buffer.handle, &requirements);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex =
            ctx.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_, &alloc, nullptr, &buffer.memory));
        VK_CHECK(vkBindBufferMemory(device_, buffer.handle, buffer.memory, 0));
    }

    if (!createAccumImages(ctx, extent)) return false;

    // --- descriptors --------------------------------------------------------
    const std::array<VkDescriptorSetLayoutBinding, 4> bindings{{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
    }};

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &setLayout_));

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(SimPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    // Two sets per frame slot: one per ping-pong parity.
    const uint32_t setCount = frameSlots_ * 2;
    const std::array<VkDescriptorPoolSize, 2> poolSizes{{
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, setCount * 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, setCount * 2},
    }};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = setCount;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_));

    std::vector<VkDescriptorSetLayout> layouts(setCount, setLayout_);
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = setCount;
    allocInfo.pSetLayouts = layouts.data();
    descriptors_.resize(setCount);
    VK_CHECK(vkAllocateDescriptorSets(device_, &allocInfo, descriptors_.data()));

    writeDescriptors(ctx, targetViews);

    VkPipeline built[4]{};
    if (!buildPipelines(shader, built)) return false;
    seedPipeline_ = built[0];
    integratePipeline_ = built[1];
    renderPipeline_ = built[2];
    tonemapPipeline_ = built[3];

    clearAccumImages(ctx);
    seedPending_ = true;
    return true;
}

bool Simulation::createAccumImages(Context& ctx, VkExtent2D extent) {
    extent_ = extent;
    accum_.resize(frameSlots_);
    for (AccumImage& target : accum_) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = kAccumFormat;
        info.extent = {extent.width, extent.height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(device_, &info, nullptr, &target.image));

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, target.image, &requirements);
        VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex =
            ctx.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(device_, &alloc, nullptr, &target.memory));
        VK_CHECK(vkBindImageMemory(device_, target.image, target.memory, 0));

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = target.image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kAccumFormat;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(device_, &viewInfo, nullptr, &target.view));
    }
    return true;
}

void Simulation::destroyAccumImages(Context&) {
    for (AccumImage& target : accum_) {
        if (target.view) vkDestroyImageView(device_, target.view, nullptr);
        if (target.image) vkDestroyImage(device_, target.image, nullptr);
        if (target.memory) vkFreeMemory(device_, target.memory, nullptr);
    }
    accum_.clear();
}

// The tonemap pass zeroes the counts as it reads them, so this only has to run
// once per image: after that every frame leaves it clean for the next.
void Simulation::clearAccumImages(Context& ctx) {
    VkCommandBufferAllocateInfo allocInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = ctx.commandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device_, &allocInfo, &cmd));

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkClearColorValue zero{};
    for (AccumImage& target : accum_) {
        VkImageMemoryBarrier toGeneral{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        toGeneral.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toGeneral.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toGeneral.image = target.image;
        toGeneral.subresourceRange = range;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &toGeneral);
        vkCmdClearColorImage(cmd, target.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
    }
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device_, &fenceInfo, nullptr, &fence));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(ctx.queue(), 1, &submit, fence));
    VK_CHECK(vkWaitForFences(device_, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(device_, fence, nullptr);
    vkFreeCommandBuffers(device_, ctx.commandPool(), 1, &cmd);
}

void Simulation::writeDescriptors(Context&, const std::vector<VkImageView>& targetViews) {
    for (uint32_t slot = 0; slot < frameSlots_; ++slot) {
        for (uint32_t parity = 0; parity < 2; ++parity) {
            VkDescriptorImageInfo targetInfo{};
            targetInfo.imageView = targetViews[slot];
            targetInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorImageInfo accumInfo{};
            accumInfo.imageView = accum_[slot].view;
            accumInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            VkDescriptorBufferInfo inInfo{particles_[parity].handle, 0, VK_WHOLE_SIZE};
            VkDescriptorBufferInfo outInfo{particles_[1 - parity].handle, 0, VK_WHOLE_SIZE};

            const VkDescriptorSet set = descriptors_[slot * 2 + parity];
            std::array<VkWriteDescriptorSet, 4> writes{};
            for (VkWriteDescriptorSet& write : writes) {
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = set;
                write.descriptorCount = 1;
            }
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[0].pImageInfo = &targetInfo;
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[1].pBufferInfo = &inInfo;
            writes[2].dstBinding = 2;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[2].pBufferInfo = &outInfo;
            writes[3].dstBinding = 3;
            writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[3].pImageInfo = &accumInfo;

            vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                                   nullptr);
        }
    }
}

bool Simulation::buildPipelines(const fs::path& shader, VkPipeline out[4]) const {
    static constexpr std::array<const char*, 4> kDefines{"PASS_SEED", "PASS_INTEGRATE", "PASS_RENDER",
                                                         "PASS_TONEMAP"};
    for (int i = 0; i < 4; ++i) out[i] = VK_NULL_HANDLE;

    for (int i = 0; i < 4; ++i) {
        const CompileResult compiled = compileComputeShader(shader, {kDefines[i]});
        if (!compiled.ok) {
            std::fprintf(stderr, "[lab] %s failed to compile:\n%s\n", kDefines[i],
                         compiled.message.c_str());
            for (int done = 0; done < i; ++done) vkDestroyPipeline(device_, out[done], nullptr);
            return false;
        }

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

        const VkResult created =
            vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info, nullptr, &out[i]);
        vkDestroyShaderModule(device_, module, nullptr);
        if (created != VK_SUCCESS) {
            std::fprintf(stderr, "[lab] %s pipeline creation failed: %s\n", kDefines[i],
                         vkResultString(created));
            for (int done = 0; done < i; ++done) vkDestroyPipeline(device_, out[done], nullptr);
            return false;
        }
    }
    return true;
}

void Simulation::destroyPipelines() {
    for (VkPipeline* pipeline :
         {&seedPipeline_, &integratePipeline_, &renderPipeline_, &tonemapPipeline_}) {
        if (*pipeline) vkDestroyPipeline(device_, *pipeline, nullptr);
        *pipeline = VK_NULL_HANDLE;
    }
}

bool Simulation::reloadPipelines(Context&, const fs::path& shader) {
    // Build into temporaries first: a failed edit must leave the running
    // pipelines untouched, exactly like the single-pass path.
    VkPipeline rebuilt[4]{};
    if (!buildPipelines(shader, rebuilt)) return false;

    vkDeviceWaitIdle(device_); // the old pipelines may still be in flight
    destroyPipelines();
    seedPipeline_ = rebuilt[0];
    integratePipeline_ = rebuilt[1];
    renderPipeline_ = rebuilt[2];
    tonemapPipeline_ = rebuilt[3];
    return true;
}

bool Simulation::resize(Context& ctx, const std::vector<VkImageView>& targetViews, VkExtent2D extent) {
    destroyAccumImages(ctx);
    if (!createAccumImages(ctx, extent)) return false;
    writeDescriptors(ctx, targetViews);
    clearAccumImages(ctx);
    return true;
}

void Simulation::record(VkCommandBuffer cmd, uint32_t frameSlot, const SimPushConstants& push) {
    const VkDescriptorSet set = descriptors_[frameSlot * 2 + parity_];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &set, 0,
                            nullptr);
    vkCmdPushConstants(cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    const uint32_t particleGroups = groupCount(particleCount_, kWorkgroupSize);
    VkBuffer readBuffer = particles_[parity_].handle;
    VkBuffer writeBuffer = particles_[1 - parity_].handle;

    if (seedPending_) {
        seedPending_ = false;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, seedPipeline_);
        vkCmdDispatch(cmd, particleGroups, 1, 1);
        // Seed fills both buffers, so both need to be visible to what follows.
        bufferBarrier(cmd, readBuffer, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        bufferBarrier(cmd, writeBuffer, VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    if (!paused_) {
        // The previous frame read this buffer; do not overwrite it until that read
        // has completed.
        bufferBarrier(cmd, writeBuffer, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, integratePipeline_);
        vkCmdDispatch(cmd, particleGroups, 1, 1);
        bufferBarrier(cmd, writeBuffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    // Render reads the buffer integrate just wrote. When paused nothing was
    // written, and that same buffer still holds the last state.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, renderPipeline_);
    vkCmdDispatch(cmd, particleGroups, 1, 1);

    accumBarrier(cmd, accum_[frameSlot].image);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tonemapPipeline_);
    vkCmdDispatch(cmd, groupCount(extent_.width, kTonemapGroup),
                  groupCount(extent_.height, kTonemapGroup), 1);

    if (!paused_) parity_ ^= 1;
}

void Simulation::destroy(Context& ctx) {
    destroyPipelines();
    destroyAccumImages(ctx);

    if (descriptorPool_) vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
    if (setLayout_) vkDestroyDescriptorSetLayout(device_, setLayout_, nullptr);
    for (Buffer& buffer : particles_) {
        if (buffer.handle) vkDestroyBuffer(device_, buffer.handle, nullptr);
        if (buffer.memory) vkFreeMemory(device_, buffer.memory, nullptr);
        buffer = {};
    }

    descriptorPool_ = VK_NULL_HANDLE;
    pipelineLayout_ = VK_NULL_HANDLE;
    setLayout_ = VK_NULL_HANDLE;
    descriptors_.clear();
    particleCount_ = 0;
}

} // namespace lab
