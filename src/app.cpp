#include "app.hpp"

#include "png.hpp"
#include "shader_compiler.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

namespace fs = std::filesystem;

namespace lab {
namespace {

constexpr VkFormat kTargetFormat = VK_FORMAT_R8G8B8A8_UNORM; // matches `rgba8` in the shader
constexpr uint32_t kWorkgroupSize = 16;                      // matches local_size_x/y in the shader
constexpr uint32_t kReloadPollInterval = 20;                 // frames between shader mtime checks

uint32_t groupCount(uint32_t threads, uint32_t local) {
    return (threads + local - 1) / local;
}

void imageBarrier(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags srcStage,
                  VkPipelineStageFlags dstStage, VkAccessFlags srcAccess, VkAccessFlags dstAccess,
                  VkImageLayout oldLayout, VkImageLayout newLayout) {
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

App::App(Options options) : options_(std::move(options)), shaderPath_(options_.shader) {}

void App::run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
}

// ---------------------------------------------------------------- window ----

void App::initWindow() {
    if (!glfwInit()) throw std::runtime_error("glfwInit failed");
    if (!glfwVulkanSupported()) throw std::runtime_error("no Vulkan loader found on this system");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // no OpenGL context, this is Vulkan
    window_ = glfwCreateWindow(1280, 720, "vulkan-compute-lab", nullptr, nullptr);
    if (!window_) throw std::runtime_error("failed to create window");

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, onFramebufferResize);
    glfwSetKeyCallback(window_, onKey);
}

void App::onFramebufferResize(GLFWwindow* window, int, int) {
    static_cast<App*>(glfwGetWindowUserPointer(window))->framebufferResized_ = true;
}

void App::onKey(GLFWwindow* window, int key, int, int action, int) {
    if (action != GLFW_PRESS) return;
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_R) app->reloadRequested_ = true;
    if (key == GLFW_KEY_F12) app->screenshotRequested_ = true;
}

// ---------------------------------------------------------------- vulkan ----

void App::initVulkan() {
#ifdef LAB_VALIDATION
    const bool validation = true;
#else
    const bool validation = false;
#endif
    ctx_.init(window_, validation);
    std::printf("[lab] device: %s (Vulkan %u.%u.%u)\n", ctx_.properties().deviceName,
                VK_API_VERSION_MAJOR(ctx_.properties().apiVersion),
                VK_API_VERSION_MINOR(ctx_.properties().apiVersion),
                VK_API_VERSION_PATCH(ctx_.properties().apiVersion));

    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    swapchain_.create(ctx_, static_cast<uint32_t>(width), static_cast<uint32_t>(height));

    createDescriptorResources();
    createFrames();
    createFrameTargets();
    createPresentSemaphores();

    if (!reloadShader(true)) throw std::runtime_error("initial shader compilation failed");
}

void App::createDescriptorResources() {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    VK_CHECK(vkCreateDescriptorSetLayout(ctx_.device(), &layoutInfo, nullptr, &setLayout_));

    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.offset = 0;
    range.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &setLayout_;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(ctx_.device(), &pipelineLayoutInfo, nullptr, &pipelineLayout_));

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kFramesInFlight};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kFramesInFlight;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx_.device(), &poolInfo, nullptr, &descriptorPool_));
}

void App::createFrames() {
    frames_.resize(kFramesInFlight);

    std::vector<VkDescriptorSetLayout> layouts(kFramesInFlight, setLayout_);
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = kFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();
    std::vector<VkDescriptorSet> sets(kFramesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device(), &allocInfo, sets.data()));

    VkCommandBufferAllocateInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdInfo.commandPool = ctx_.commandPool();
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = kFramesInFlight;
    std::vector<VkCommandBuffer> buffers(kFramesInFlight);
    VK_CHECK(vkAllocateCommandBuffers(ctx_.device(), &cmdInfo, buffers.data()));

    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // so the first wait returns immediately

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        frames_[i].descriptor = sets[i];
        frames_[i].cmd = buffers[i];
        VK_CHECK(vkCreateSemaphore(ctx_.device(), &semaphoreInfo, nullptr, &frames_[i].imageAvailable));
        VK_CHECK(vkCreateFence(ctx_.device(), &fenceInfo, nullptr, &frames_[i].inFlight));
    }
}

void App::createFrameTargets() {
    const VkExtent2D extent = swapchain_.extent();

    for (Frame& frame : frames_) {
        VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = kTargetFormat;
        imageInfo.extent = {extent.width, extent.height, 1};
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VK_CHECK(vkCreateImage(ctx_.device(), &imageInfo, nullptr, &frame.target));

        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(ctx_.device(), frame.target, &requirements);

        VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocInfo.allocationSize = requirements.size;
        allocInfo.memoryTypeIndex =
            ctx_.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkAllocateMemory(ctx_.device(), &allocInfo, nullptr, &frame.targetMemory));
        VK_CHECK(vkBindImageMemory(ctx_.device(), frame.target, frame.targetMemory, 0));

        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = frame.target;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = kTargetFormat;
        viewInfo.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(ctx_.device(), &viewInfo, nullptr, &frame.targetView));

        VkDescriptorImageInfo descriptorImage{};
        descriptorImage.imageView = frame.targetView;
        descriptorImage.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // the only layout storage images can use

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = frame.descriptor;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &descriptorImage;
        vkUpdateDescriptorSets(ctx_.device(), 1, &write, 0, nullptr);
    }
}

void App::destroyFrameTargets() {
    for (Frame& frame : frames_) {
        if (frame.targetView) vkDestroyImageView(ctx_.device(), frame.targetView, nullptr);
        if (frame.target) vkDestroyImage(ctx_.device(), frame.target, nullptr);
        if (frame.targetMemory) vkFreeMemory(ctx_.device(), frame.targetMemory, nullptr);
        frame.targetView = VK_NULL_HANDLE;
        frame.target = VK_NULL_HANDLE;
        frame.targetMemory = VK_NULL_HANDLE;
    }
}

void App::createPresentSemaphores() {
    // One per swapchain image, not per frame in flight: present consumes the
    // semaphore asynchronously, and reusing a frame-indexed one can have it
    // still pending when the same frame slot comes around again.
    destroyPresentSemaphores();
    presentReady_.resize(swapchain_.imageCount());
    VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (VkSemaphore& semaphore : presentReady_)
        VK_CHECK(vkCreateSemaphore(ctx_.device(), &info, nullptr, &semaphore));
}

void App::destroyPresentSemaphores() {
    for (VkSemaphore semaphore : presentReady_)
        if (semaphore) vkDestroySemaphore(ctx_.device(), semaphore, nullptr);
    presentReady_.clear();
}

// ------------------------------------------------------- shader pipeline ----

bool App::reloadShader(bool initial) {
    const CompileResult result = compileComputeShader(shaderPath_);
    if (!result.ok) {
        std::fprintf(stderr, "[lab] shader compile failed:\n%s\n", result.message.c_str());
        if (!initial) std::fprintf(stderr, "[lab] keeping the previous pipeline\n");
        return false;
    }
    if (!result.message.empty()) std::fprintf(stderr, "%s", result.message.c_str());

    VkShaderModuleCreateInfo moduleInfo{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = result.spirv.size() * sizeof(uint32_t);
    moduleInfo.pCode = result.spirv.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(ctx_.device(), &moduleInfo, nullptr, &module));

    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.layout = pipelineLayout_;

    VkPipeline built = VK_NULL_HANDLE;
    const VkResult created =
        vkCreateComputePipelines(ctx_.device(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &built);
    vkDestroyShaderModule(ctx_.device(), module, nullptr); // the pipeline owns a copy now

    if (created != VK_SUCCESS) {
        std::fprintf(stderr, "[lab] pipeline creation failed: %s\n", vkResultString(created));
        return false;
    }

    if (pipeline_) {
        vkDeviceWaitIdle(ctx_.device()); // the old pipeline may still be in flight
        vkDestroyPipeline(ctx_.device(), pipeline_, nullptr);
    }
    pipeline_ = built;

    std::error_code ec;
    shaderWriteTime_ = fs::last_write_time(shaderPath_, ec);
    std::printf("[lab] %s %s\n", initial ? "loaded" : "reloaded", shaderPath_.filename().string().c_str());
    return true;
}

void App::destroyPipeline() {
    if (pipeline_) vkDestroyPipeline(ctx_.device(), pipeline_, nullptr);
    pipeline_ = VK_NULL_HANDLE;
}

void App::pollShaderFile() {
    if (reloadRequested_) {
        reloadRequested_ = false;
        reloadShader(false);
        return;
    }
    if (frameCounter_ % kReloadPollInterval != 0) return;

    std::error_code ec;
    const auto written = fs::last_write_time(shaderPath_, ec);
    if (ec || written == shaderWriteTime_) return;

    shaderWriteTime_ = written;
    reloadShader(false);
}

// ------------------------------------------------------------ frame loop ----

void App::mainLoop() {
    startTime_ = glfwGetTime();
    lastFrameTime_ = startTime_;
    lastTitleUpdate_ = startTime_;

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        pollShaderFile();
        // Capture only after a frame that actually rendered, so the target image
        // is guaranteed to be in TRANSFER_SRC_OPTIMAL with real contents.
        if (drawFrame()) handleCapture();
    }
    vkDeviceWaitIdle(ctx_.device());
}

bool App::drawFrame() {
    Frame& frame = frames_[frameIndex_];
    VK_CHECK(vkWaitForFences(ctx_.device(), 1, &frame.inFlight, VK_TRUE, UINT64_MAX));

    uint32_t imageIndex = 0;
    const VkResult acquired = vkAcquireNextImageKHR(ctx_.device(), swapchain_.handle(), UINT64_MAX,
                                                    frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return false;
    }
    if (acquired != VK_SUCCESS && acquired != VK_SUBOPTIMAL_KHR) VK_CHECK(acquired);

    VK_CHECK(vkResetFences(ctx_.device(), 1, &frame.inFlight));

    const double now = glfwGetTime();
    const auto deltaTime = static_cast<float>(now - lastFrameTime_);
    lastFrameTime_ = now;

    const VkExtent2D extent = swapchain_.extent();

    // Only follow the cursor while dragging. Otherwise the mouse parks wherever
    // it happens to be — including outside the window — and every shader that
    // reads it starts off-centre for no visible reason.
    double mouseX = extent.width * 0.5;
    double mouseY = extent.height * 0.5;
    if (glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
        glfwGetCursorPos(window_, &mouseX, &mouseY);
    PushConstants push{};
    push.resolution[0] = static_cast<float>(extent.width);
    push.resolution[1] = static_cast<float>(extent.height);
    push.mouse[0] = static_cast<float>(mouseX);
    push.mouse[1] = static_cast<float>(mouseY);
    push.time = static_cast<float>(now - startTime_);
    push.deltaTime = deltaTime;
    push.frame = frameCounter_;

    VK_CHECK(vkResetCommandBuffer(frame.cmd, 0));
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(frame.cmd, &beginInfo));

    // UNDEFINED discards last frame's contents, which is what we want for a
    // shader that writes every pixel. A ping-pong effect would keep the old
    // layout here instead.
    imageBarrier(frame.cmd, frame.target, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, VK_ACCESS_SHADER_WRITE_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1,
                            &frame.descriptor, 0, nullptr);
    vkCmdPushConstants(frame.cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(frame.cmd, groupCount(extent.width, kWorkgroupSize),
                  groupCount(extent.height, kWorkgroupSize), 1);

    imageBarrier(frame.cmd, frame.target, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                 VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    VkImage swapchainImage = swapchain_.images()[imageIndex];
    imageBarrier(frame.cmd, swapchainImage, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // Blit rather than copy: the storage image is RGBA8 and the swapchain is
    // usually BGRA8, and only a blit performs the format conversion.
    VkImageBlit region{};
    region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.srcOffsets[1] = {static_cast<int32_t>(extent.width), static_cast<int32_t>(extent.height), 1};
    region.dstSubresource = region.srcSubresource;
    region.dstOffsets[1] = region.srcOffsets[1];
    vkCmdBlitImage(frame.cmd, frame.target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, swapchainImage,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);

    imageBarrier(frame.cmd, swapchainImage, VK_PIPELINE_STAGE_TRANSFER_BIT,
                 VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(frame.cmd));

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT; // first use of the acquired image
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &frame.imageAvailable;
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &frame.cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &presentReady_[imageIndex];
    VK_CHECK(vkQueueSubmit(ctx_.queue(), 1, &submit, frame.inFlight));

    VkSwapchainKHR swapchainHandle = swapchain_.handle();
    VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &presentReady_[imageIndex];
    present.swapchainCount = 1;
    present.pSwapchains = &swapchainHandle;
    present.pImageIndices = &imageIndex;

    const VkResult presented = vkQueuePresentKHR(ctx_.queue(), &present);
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || framebufferResized_) {
        framebufferResized_ = false;
        recreateSwapchain();
    } else if (presented != VK_SUCCESS) {
        VK_CHECK(presented);
    }

    frameIndex_ = (frameIndex_ + 1) % kFramesInFlight;
    ++frameCounter_;
    updateTitle(now);
    return true;
}

// --------------------------------------------------------------- capture ----

void App::handleCapture() {
    if (screenshotRequested_) {
        screenshotRequested_ = false;
        char name[64];
        std::snprintf(name, sizeof(name), "capture-%03u.png", ++screenshotCounter_);
        captureFrame(name);
    }

    const bool autoCapture = !options_.capturePath.empty() && !autoCaptureDone_ &&
                             frameCounter_ >= options_.captureAfterFrames;
    if (autoCapture) {
        autoCaptureDone_ = true;
        captureFrame(options_.capturePath);
        if (options_.exitAfterCapture) glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
}

bool App::captureFrame(const fs::path& path) {
    // The most recently submitted frame, whose target still holds its pixels.
    const uint32_t last = (frameIndex_ + kFramesInFlight - 1) % kFramesInFlight;
    Frame& frame = frames_[last];
    VK_CHECK(vkWaitForFences(ctx_.device(), 1, &frame.inFlight, VK_TRUE, UINT64_MAX));

    const VkExtent2D extent = swapchain_.extent();
    const VkDeviceSize size = VkDeviceSize(extent.width) * extent.height * 4;

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer staging = VK_NULL_HANDLE;
    VK_CHECK(vkCreateBuffer(ctx_.device(), &bufferInfo, nullptr, &staging));

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(ctx_.device(), staging, &requirements);
    VkMemoryAllocateInfo allocInfo{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = requirements.size;
    allocInfo.memoryTypeIndex = ctx_.findMemoryType(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateMemory(ctx_.device(), &allocInfo, nullptr, &stagingMemory));
    VK_CHECK(vkBindBufferMemory(ctx_.device(), staging, stagingMemory, 0));

    VkCommandBufferAllocateInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdInfo.commandPool = ctx_.commandPool();
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(ctx_.device(), &cmdInfo, &cmd));

    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, frame.target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1, &copy);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(ctx_.device(), &fenceInfo, nullptr, &fence));

    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(ctx_.queue(), 1, &submit, fence));
    VK_CHECK(vkWaitForFences(ctx_.device(), 1, &fence, VK_TRUE, UINT64_MAX));

    void* mapped = nullptr;
    VK_CHECK(vkMapMemory(ctx_.device(), stagingMemory, 0, size, 0, &mapped));
    const bool written = writePng(path, extent.width, extent.height,
                                  static_cast<const uint8_t*>(mapped), size_t(extent.width) * 4);
    vkUnmapMemory(ctx_.device(), stagingMemory);

    vkDestroyFence(ctx_.device(), fence, nullptr);
    vkFreeCommandBuffers(ctx_.device(), ctx_.commandPool(), 1, &cmd);
    vkFreeMemory(ctx_.device(), stagingMemory, nullptr);
    vkDestroyBuffer(ctx_.device(), staging, nullptr);

    if (written)
        std::printf("[lab] captured %ux%u -> %s\n", extent.width, extent.height, path.string().c_str());
    else
        std::fprintf(stderr, "[lab] failed to write %s\n", path.string().c_str());
    return written;
}

void App::recreateSwapchain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0) { // minimised — nothing to render
        glfwGetFramebufferSize(window_, &width, &height);
        glfwWaitEvents();
        if (glfwWindowShouldClose(window_)) return;
    }

    vkDeviceWaitIdle(ctx_.device());

    destroyFrameTargets();
    swapchain_.create(ctx_, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    createFrameTargets();
    createPresentSemaphores();
}

void App::updateTitle(double now) {
    ++framesSinceTitleUpdate_;
    if (now - lastTitleUpdate_ < 0.5) return;

    const double fps = framesSinceTitleUpdate_ / (now - lastTitleUpdate_);
    const VkExtent2D extent = swapchain_.extent();
    char title[256];
    std::snprintf(title, sizeof(title), "vulkan-compute-lab — %s — %ux%u — %.0f fps",
                  shaderPath_.filename().string().c_str(), extent.width, extent.height, fps);
    glfwSetWindowTitle(window_, title);

    lastTitleUpdate_ = now;
    framesSinceTitleUpdate_ = 0;
}

void App::cleanup() {
    destroyPipeline();
    destroyPresentSemaphores();
    destroyFrameTargets();

    for (Frame& frame : frames_) {
        if (frame.imageAvailable) vkDestroySemaphore(ctx_.device(), frame.imageAvailable, nullptr);
        if (frame.inFlight) vkDestroyFence(ctx_.device(), frame.inFlight, nullptr);
    }
    frames_.clear();

    if (descriptorPool_) vkDestroyDescriptorPool(ctx_.device(), descriptorPool_, nullptr);
    if (pipelineLayout_) vkDestroyPipelineLayout(ctx_.device(), pipelineLayout_, nullptr);
    if (setLayout_) vkDestroyDescriptorSetLayout(ctx_.device(), setLayout_, nullptr);

    swapchain_.destroy(ctx_);
    ctx_.destroy();

    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

} // namespace lab
