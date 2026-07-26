#include "app.hpp"

#include "png.hpp"
#include "shader_compiler.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>
#include <string>
#include <utility>

#ifndef LAB_PROJECT_DIR
#define LAB_PROJECT_DIR "."
#endif

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

App::App(Options options) : options_(std::move(options)), shaderPath_(options_.shader) {
    // Seed both the target and the smoothed value, so a requested starting view
    // is already in place on frame zero instead of easing into it.
    nav_.centerX = nav_.smoothCenterX = options_.centerX;
    nav_.centerY = nav_.smoothCenterY = options_.centerY;
    nav_.scale = nav_.smoothScale = 1.0 / std::max(options_.zoom, 1e-9);
}

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
    glfwSetMouseButtonCallback(window_, onMouseButton);
    glfwSetScrollCallback(window_, onScroll);
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
    if (app->sim_) {
        if (key == GLFW_KEY_SPACE) app->sim_->requestSeed();
        if (key == GLFW_KEY_P) app->sim_->setPaused(!app->sim_->paused());
    }
    // Space means "start over" in both modes: reseed the particles, or throw away
    // the accumulated samples and re-converge from the current view. Useful on
    // its own for a shader whose sampling changed with `time` rather than with
    // the camera, which nothing else here can detect.
    if (app->accum_ && key == GLFW_KEY_SPACE) app->accum_->requestReset();
    if (key == GLFW_KEY_HOME) { // ease back to the default view
        if (app->cameraMode_) {
            app->resetCamera();
        } else {
            app->nav_.centerX = app->nav_.centerY = 0.0;
            app->nav_.scale = 1.0;
        }
    }
}

void App::onMouseButton(GLFWwindow* window, int button, int action, int) {
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));

    if (action == GLFW_PRESS) {
        // Re-anchor on press. Without this the first frame of a drag applies the
        // distance the cursor travelled since the last drag ended, which is the
        // jump this design exists to avoid.
        glfwGetCursorPos(window, &app->nav_.lastCursorX, &app->nav_.lastCursorY);
        app->nav_.dragging = true;
    } else if (action == GLFW_RELEASE) {
        app->nav_.dragging = false;
    }
}

// The deepest zoom at which fp32 shader coordinates still resolve individual
// pixels. The shader computes p = centre + uv * scale, so adjacent pixels differ
// by scale/height; once that step falls below the float32 ULP of p itself, the
// whole neighbourhood collapses onto one representable coordinate.
//
//   step = scale / height        must exceed        ulp(p) ~= |p| * 2^-23
//
// |p| depends on the shader (the shipped ones stay within ~0.3 uv units of the
// origin), which is why this is a default rather than a law: --max-zoom
// overrides it, and a negative value removes the limit entirely.
double App::minScale() const {
    if (options_.maxZoom > 0.0) return 1.0 / options_.maxZoom;
    if (options_.maxZoom < 0.0) return 0.0; // explicitly unlimited

    const VkExtent2D extent = swapchain_.extent();
    const double height = extent.height > 0 ? extent.height : 1.0;
    constexpr double kCoordinateMagnitude = 0.3;
    constexpr double kFloat32Ulp = 1.0 / 8388608.0; // 2^-23
    return height * kCoordinateMagnitude * kFloat32Ulp;
}

void App::onScroll(GLFWwindow* window, double, double yOffset) {
    auto* app = static_cast<App*>(glfwGetWindowUserPointer(window));

    // With a 3D camera the wheel trims fly speed rather than zooming: you close
    // distance by flying, and a separate zoom would just be a second, confusing
    // way to change apparent scale.
    if (app->cameraMode_) {
        app->camera_.speed = std::clamp(app->camera_.speed * std::exp(yOffset * 0.2), 0.004, 20.0);
        return;
    }

    Navigation& nav = app->nav_;

    const VkExtent2D extent = app->swapchain_.extent();
    if (extent.height == 0) return;

    double cursorX = 0.0, cursorY = 0.0;
    glfwGetCursorPos(window, &cursorX, &cursorY);
    const double height = extent.height;
    const double uvX = (cursorX - 0.5 * extent.width) / height;
    const double uvY = (cursorY - 0.5 * height) / height;

    // Anchor on the *displayed* view, not the target. Both ends of the ease then
    // satisfy centre == anchor - uv * scale for the same anchor, so the point
    // under the cursor stays pinned on every intermediate frame.
    const double anchorX = nav.smoothCenterX + uvX * nav.smoothScale;
    const double anchorY = nav.smoothCenterY + uvY * nav.smoothScale;

    // Compound from the target so fast scrolling accumulates.
    double scale = nav.scale * std::exp(-yOffset * 0.15);

    // Clamp before deriving the centre, or the anchor would be computed for a
    // scale the view never reaches.
    const double floorScale = app->minScale();
    if (floorScale > 0.0 && scale < floorScale) {
        scale = floorScale;
        if (!app->zoomClampReported_) {
            app->zoomClampReported_ = true;
            std::printf("[lab] zoom limited to %.0fx: past this, fp32 shader coordinates "
                        "quantize and the view snaps between lattice steps instead of "
                        "moving. --max-zoom overrides, --max-zoom -1 removes the limit.\n",
                        1.0 / floorScale);
        }
    }

    nav.scale = scale;
    nav.centerX = anchorX - uvX * scale;
    nav.centerY = anchorY - uvY * scale;
}

void App::resetCamera() {
    camera_ = Camera{};
}

// Pulled straight out of the push block rather than recomputed from camera_, so
// what is compared is byte-for-byte what the shader was given. Anything the
// shader cannot see must not be able to invalidate the accumulated sum.
App::AccumView App::accumViewOf(const PushConstants& push) {
    AccumView view;
    view.fovScale = push.fovScale;
    std::memcpy(view.camPos, push.camPos, sizeof(view.camPos));
    std::memcpy(view.camRight, push.camRight, sizeof(view.camRight));
    std::memcpy(view.camUp, push.camUp, sizeof(view.camUp));
    std::memcpy(view.camForward, push.camForward, sizeof(view.camForward));
    return view;
}

// Right-handed basis from yaw/pitch, with world +z as up.
void App::cameraBasis(float right[3], float up[3], float forward[3]) const {
    const double cp = std::cos(camera_.pitch), sp = std::sin(camera_.pitch);
    const double cy = std::cos(camera_.yaw), sy = std::sin(camera_.yaw);

    const double fx = cp * cy, fy = cp * sy, fz = sp;

    // right = normalize(forward x worldUp). Pitch is clamped away from vertical
    // in updateCamera, so forward is never parallel to worldUp and this cross
    // product cannot collapse.
    double rx = fy * 1.0 - fz * 0.0;
    double ry = fz * 0.0 - fx * 1.0;
    double rz = fx * 0.0 - fy * 0.0;
    const double rlen = std::sqrt(rx * rx + ry * ry + rz * rz);
    rx /= rlen; ry /= rlen; rz /= rlen;

    // up = right x forward, already unit length given the two are orthonormal.
    const double ux = ry * fz - rz * fy;
    const double uy = rz * fx - rx * fz;
    const double uz = rx * fy - ry * fx;

    right[0] = float(rx);   right[1] = float(ry);   right[2] = float(rz);
    up[0] = float(ux);      up[1] = float(uy);      up[2] = float(uz);
    forward[0] = float(fx); forward[1] = float(fy); forward[2] = float(fz);
}

void App::updateCamera(float deltaTime) {
    // Look: the same drag-accumulates-deltas rule as the 2D path, so pressing
    // the button contributes nothing and the view cannot jump.
    double cursorX = 0.0, cursorY = 0.0;
    glfwGetCursorPos(window_, &cursorX, &cursorY);
    if (nav_.dragging) {
        // Grab-the-world, not FPS look: dragging right moves the scene right,
        // which means the camera turns left. This matches the 2D pan, where the
        // image follows the cursor. FPS-style look (turn right when dragging
        // right) reads as inverted to anyone coming from the other mode.
        constexpr double kLookSpeed = 0.0035; // radians per pixel
        camera_.yaw += (cursorX - nav_.lastCursorX) * kLookSpeed;
        camera_.pitch += (cursorY - nav_.lastCursorY) * kLookSpeed;
        constexpr double kPitchLimit = 1.5533; // 89 degrees
        camera_.pitch = std::clamp(camera_.pitch, -kPitchLimit, kPitchLimit);
    }
    nav_.lastCursorX = cursorX;
    nav_.lastCursorY = cursorY;

    float right[3], up[3], forward[3];
    cameraBasis(right, up, forward);

    // Movement is polled, not event-driven: key callbacks fire on press and
    // repeat, which would make held-key motion stutter at the OS repeat rate.
    const auto held = [this](int key) { return glfwGetKey(window_, key) == GLFW_PRESS; };
    double moveF = 0.0, moveR = 0.0, moveU = 0.0;
    if (held(GLFW_KEY_W)) moveF += 1.0;
    if (held(GLFW_KEY_S)) moveF -= 1.0;
    if (held(GLFW_KEY_D)) moveR += 1.0;
    if (held(GLFW_KEY_A)) moveR -= 1.0;
    if (held(GLFW_KEY_E)) moveU += 1.0;
    if (held(GLFW_KEY_Q)) moveU -= 1.0;

    double speed = camera_.speed;
    if (held(GLFW_KEY_LEFT_SHIFT) || held(GLFW_KEY_RIGHT_SHIFT)) speed *= 4.0;
    if (held(GLFW_KEY_LEFT_CONTROL) || held(GLFW_KEY_RIGHT_CONTROL)) speed *= 0.25;

    double desiredX = forward[0] * moveF + right[0] * moveR;
    double desiredY = forward[1] * moveF + right[1] * moveR;
    double desiredZ = forward[2] * moveF + right[2] * moveR + moveU;

    // Normalise so diagonals are not faster than the axes.
    const double length = std::sqrt(desiredX * desiredX + desiredY * desiredY + desiredZ * desiredZ);
    if (length > 1e-9) {
        desiredX = desiredX / length * speed;
        desiredY = desiredY / length * speed;
        desiredZ = desiredZ / length * speed;
    } else {
        desiredX = desiredY = desiredZ = 0.0;
    }

    const double alpha = 1.0 - std::exp(-12.0 * double(deltaTime));
    camera_.velX += (desiredX - camera_.velX) * alpha;
    camera_.velY += (desiredY - camera_.velY) * alpha;
    camera_.velZ += (desiredZ - camera_.velZ) * alpha;

    // Come to an exact stop once nothing is held and the eased velocity is
    // negligible. Exponential decay never actually reaches zero, so without this
    // the camera creeps by a vanishing amount every frame forever — invisible on
    // screen, but accumulation compares the camera against the one the last
    // sample was traced with and would throw the image away on every frame. The
    // cut-off is four orders of magnitude below fly speed, so the snap itself
    // cannot move the view by a fraction of a pixel.
    if (length <= 1e-9) {
        const double stopped = camera_.speed * 1e-4;
        if (std::abs(camera_.velX) < stopped && std::abs(camera_.velY) < stopped &&
            std::abs(camera_.velZ) < stopped)
            camera_.velX = camera_.velY = camera_.velZ = 0.0;
    }

    camera_.x += camera_.velX * deltaTime;
    camera_.y += camera_.velY * deltaTime;
    camera_.z += camera_.velZ * deltaTime;
}

void App::updateNavigation(float deltaTime) {
    const VkExtent2D extent = swapchain_.extent();
    const double height = extent.height > 0 ? extent.height : 1.0;

    double cursorX = 0.0, cursorY = 0.0;
    glfwGetCursorPos(window_, &cursorX, &cursorY);

    if (nav_.dragging) {
        // Deltas, not absolute position. Scaling by `scale` is what keeps a drag
        // moving the image 1:1 with the cursor at every zoom level, and keeps the
        // increment proportional to what a pixel is currently worth instead of
        // shrinking into the noise floor as you zoom in.
        nav_.centerX -= (cursorX - nav_.lastCursorX) / height * nav_.scale;
        nav_.centerY -= (cursorY - nav_.lastCursorY) / height * nav_.scale;
    }
    nav_.lastCursorX = cursorX;
    nav_.lastCursorY = cursorY;

    // Frame-rate independent exponential smoothing, so the feel does not change
    // with frame rate the way a plain per-frame lerp would. Centre and scale
    // must ease with the same factor — see the note on Navigation.
    const double alpha = 1.0 - std::exp(-18.0 * double(deltaTime));
    nav_.smoothCenterX += (nav_.centerX - nav_.smoothCenterX) * alpha;
    nav_.smoothCenterY += (nav_.centerY - nav_.smoothCenterY) * alpha;
    nav_.smoothScale += (nav_.scale - nav_.smoothScale) * alpha;
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

    // The starting view goes through the same ceiling as scrolling does; the
    // swapchain has to exist first, since the limit depends on the height.
    const double floorScale = minScale();
    if (floorScale > 0.0 && nav_.scale < floorScale) {
        std::printf("[lab] requested zoom %.0fx exceeds the fp32 limit, clamped to %.0fx "
                    "(--max-zoom -1 to override)\n",
                    1.0 / nav_.scale, 1.0 / floorScale);
        nav_.scale = nav_.smoothScale = floorScale;
    }

    std::printf("[lab] F12 screenshots -> %s\n", screenshotDirectory().string().c_str());
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

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kMaxFramesInFlight};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = kMaxFramesInFlight;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(ctx_.device(), &poolInfo, nullptr, &descriptorPool_));
}

void App::createFrames() {
    // Every slot is built up front even though accumulation only ever uses the
    // first. Two command buffers, fences and images are cheap, and it means the
    // frame count can change on a shader reload without touching any of this.
    frames_.resize(kMaxFramesInFlight);

    std::vector<VkDescriptorSetLayout> layouts(kMaxFramesInFlight, setLayout_);
    VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = kMaxFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();
    std::vector<VkDescriptorSet> sets(kMaxFramesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(ctx_.device(), &allocInfo, sets.data()));

    VkCommandBufferAllocateInfo cmdInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cmdInfo.commandPool = ctx_.commandPool();
    cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdInfo.commandBufferCount = kMaxFramesInFlight;
    std::vector<VkCommandBuffer> buffers(kMaxFramesInFlight);
    VK_CHECK(vkAllocateCommandBuffers(ctx_.device(), &cmdInfo, buffers.data()));

    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // so the first wait returns immediately

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
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

std::vector<VkImageView> App::targetViews() const {
    std::vector<VkImageView> views;
    views.reserve(frames_.size());
    for (const Frame& frame : frames_) views.push_back(frame.targetView);
    return views;
}

// A shader declaring //!nbody runs the multi-pass simulation path instead of the
// single dispatch. Editing such a shader rebuilds its pipelines but keeps the
// particle state, so you can retune gravity mid-flight without restarting.
bool App::reloadSimulation(uint32_t particleCount, bool initial) {
    if (sim_ && sim_->particleCount() == particleCount) {
        return sim_->reloadPipelines(ctx_, shaderPath_);
    }

    auto rebuilt = std::make_unique<Simulation>();
    if (!rebuilt->create(ctx_, shaderPath_, particleCount, targetViews(), swapchain_.extent())) {
        rebuilt->destroy(ctx_);
        if (!initial) std::fprintf(stderr, "[lab] keeping the previous simulation\n");
        return false;
    }

    if (sim_) {
        vkDeviceWaitIdle(ctx_.device());
        sim_->destroy(ctx_);
    }
    sim_ = std::move(rebuilt);
    std::printf("[lab] simulating %u particles (%.1f M interactions per step)\n", particleCount,
                double(particleCount) * particleCount / 1e6);
    return true;
}

// A shader declaring //!accumulate keeps a running sum across frames. Editing it
// rebuilds the pipeline and resets the sum, because samples traced by the old
// code cannot be averaged with samples traced by the new one.
bool App::reloadAccumulator(bool initial) {
    if (accum_) return accum_->reloadPipeline(ctx_, shaderPath_);

    auto rebuilt = std::make_unique<Accumulator>();
    if (!rebuilt->create(ctx_, shaderPath_, targetViews(), swapchain_.extent())) {
        rebuilt->destroy(ctx_);
        if (!initial) std::fprintf(stderr, "[lab] keeping the previous pipeline\n");
        return false;
    }
    accum_ = std::move(rebuilt);
    std::printf("[lab] accumulating: one sample per frame, Space restarts\n");
    return true;
}

void App::discardSimulation() {
    if (!sim_) return;
    vkDeviceWaitIdle(ctx_.device()); // its pipelines may still be in flight
    sim_->destroy(ctx_);
    sim_.reset();
}

void App::discardAccumulator() {
    if (!accum_) return;
    vkDeviceWaitIdle(ctx_.device());
    accum_->destroy(ctx_);
    accum_.reset();
}

// Accumulation must run with a single frame in flight: two frames both
// read-modify-writing the one shared sum have no ordering between them, since
// the second is recorded and submitted before the first has finished, so its
// sample count is stale and its imageLoad may or may not see the other's store.
// Draining first is what makes dropping a slot safe — its command buffer may
// still be executing, and nothing will ever wait on its fence again.
void App::setFramesInFlight(uint32_t count) {
    if (count == framesInFlight_) return;
    vkDeviceWaitIdle(ctx_.device());
    framesInFlight_ = count;
    frameIndex_ = 0;
}

bool App::reloadShader(bool initial) {
    const ShaderDirectives directives = readShaderDirectives(shaderPath_);

    // A simulation is framed by a 3D view too, and readShaderDirectives already
    // folds //!accumulate into camera3d. Applied only once a mode has been built
    // successfully: a failed reload keeps the previous pipeline running, and the
    // input routing has to keep matching whatever is actually on screen.
    const bool wantsCamera = directives.camera3d || directives.particleCount > 0;

    const auto noteLoaded = [&] {
        std::error_code ec;
        shaderWriteTime_ = fs::last_write_time(shaderPath_, ec);
        std::printf("[lab] %s %s\n", initial ? "loaded" : "reloaded",
                    shaderPath_.filename().string().c_str());
    };

    if (directives.particleCount > 0) {
        const uint32_t count =
            options_.particleCount > 0 ? options_.particleCount : directives.particleCount;
        if (!reloadSimulation(count, initial)) return false;
        discardAccumulator(); // only after the new mode is known to work
        setFramesInFlight(kMaxFramesInFlight);
        cameraMode_ = wantsCamera;
        noteLoaded();
        return true;
    }

    if (directives.accumulate) {
        if (!reloadAccumulator(initial)) return false;
        discardSimulation();
        setFramesInFlight(1);
        cameraMode_ = wantsCamera;
        noteLoaded();
        return true;
    }

    // Plain one-dispatch-per-pixel path. Note that the modes it might be
    // replacing are only torn down once this has succeeded, so a bad edit that
    // deletes a directive leaves the running simulation or accumulation alone
    // instead of dropping to a blank window.
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

    discardSimulation();
    discardAccumulator();
    setFramesInFlight(kMaxFramesInFlight);
    cameraMode_ = wantsCamera; // //!camera3d without either special mode
    noteLoaded();
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
    // Exactly one navigation model runs per frame. Both read the cursor and both
    // write nav_.lastCursor, so letting both run would double-count a drag.
    if (cameraMode_) {
        updateCamera(deltaTime);
    } else {
        updateNavigation(deltaTime);
    }

    // One block for every mode. The camera fields are filled unconditionally
    // rather than per mode: a pixel shader declares only the prefix it uses, so
    // the extra bytes cost one memcpy into the command buffer and save a branch.
    PushConstants push{};
    push.resolution[0] = static_cast<float>(extent.width);
    push.resolution[1] = static_cast<float>(extent.height);
    push.mouse[0] = extent.width > 0 ? static_cast<float>(nav_.lastCursorX / extent.width) : 0.0f;
    push.mouse[1] = extent.height > 0 ? static_cast<float>(nav_.lastCursorY / extent.height) : 0.0f;
    push.center[0] = static_cast<float>(nav_.smoothCenterX);
    push.center[1] = static_cast<float>(nav_.smoothCenterY);
    push.zoom = static_cast<float>(1.0 / nav_.smoothScale);
    push.time = static_cast<float>(now - startTime_);
    push.deltaTime = deltaTime;
    push.frame = frameCounter_;
    push.particleCount = sim_ ? sim_->particleCount() : 0;

    constexpr float kFovDegrees = 60.0f;
    push.fovScale = std::tan(kFovDegrees * 0.5f * 3.14159265f / 180.0f);
    push.camPos[0] = float(camera_.x);
    push.camPos[1] = float(camera_.y);
    push.camPos[2] = float(camera_.z);
    cameraBasis(push.camRight, push.camUp, push.camForward);

    if (accum_) {
        // Camera moved => every sample in the buffer was traced through a
        // different lens, so averaging the next one in would smear the image.
        // Compared against what the *last sample* used, not against the previous
        // frame, so a reset that has already happened is not repeated.
        const AccumView view = accumViewOf(push);
        if (!(view == lastAccumView_)) {
            lastAccumView_ = view;
            accum_->requestReset();
        }
    }

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

    if (sim_) {
        sim_->record(frame.cmd, frameIndex_, push);
    } else if (accum_) {
        // The accumulator stamps in the live sample count itself, so nothing here
        // has to stay in step with a reset it may have just performed.
        accum_->record(frame.cmd, frameIndex_, push);
    } else {
        vkCmdBindPipeline(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(frame.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1,
                                &frame.descriptor, 0, nullptr);
        vkCmdPushConstants(frame.cmd, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);
        vkCmdDispatch(frame.cmd, groupCount(extent.width, kWorkgroupSize),
                      groupCount(extent.height, kWorkgroupSize), 1);
    }

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
    bool recreated = false;
    if (presented == VK_ERROR_OUT_OF_DATE_KHR || presented == VK_SUBOPTIMAL_KHR || framebufferResized_) {
        framebufferResized_ = false;
        recreateSwapchain();
        recreated = true;
    } else if (presented != VK_SUCCESS) {
        VK_CHECK(presented);
    }

    frameIndex_ = (frameIndex_ + 1) % framesInFlight_;
    ++frameCounter_;
    updateTitle(now);

    // Report "did not render" when the swapchain was just rebuilt. Recreating the
    // frame targets leaves every one of them in VK_IMAGE_LAYOUT_UNDEFINED, but
    // captureFrame copies from the last submitted target assuming
    // TRANSFER_SRC_OPTIMAL — so allowing a capture after a recreate would read an
    // image in the wrong layout.
    return !recreated;
}

// --------------------------------------------------------------- capture ----

void App::handleCapture() {
    if (screenshotRequested_) {
        screenshotRequested_ = false;
        captureFrame(screenshotPath());
    }

    const bool autoCapture = !options_.capturePath.empty() && !autoCaptureDone_ &&
                             frameCounter_ >= options_.captureAfterFrames;
    if (autoCapture) {
        autoCaptureDone_ = true;
        captureFrame(options_.capturePath);
        if (options_.exitAfterCapture) glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
}

// screenshots/<shader>/, created eagerly at startup rather than on the first
// capture, so the destination is visible before you go looking for it.
fs::path App::screenshotDirectory() const {
    // make_preferred: LAB_PROJECT_DIR arrives from CMake with forward slashes,
    // which would otherwise log as a mix of both separators on Windows.
    const fs::path directory =
        (fs::path(LAB_PROJECT_DIR) / "screenshots" / shaderPath_.stem()).make_preferred();

    std::error_code ec;
    fs::create_directories(directory, ec);
    if (ec) {
        std::fprintf(stderr, "[lab] cannot create %s (%s); using the working directory\n",
                     directory.string().c_str(), ec.message().c_str());
        return fs::current_path();
    }
    return directory;
}

// screenshots/<shader>/<shader>-<timestamp>.png, so captures from different
// shaders never pile up together. Nothing is drawn into the frame to indicate a
// capture — the image is the deliverable, and an overlay would be baked into it.
fs::path App::screenshotPath() const {
    const std::string stem = shaderPath_.stem().string();
    const fs::path directory = screenshotDirectory();

    const std::time_t now = std::time(nullptr);
    const std::tm* local = std::localtime(&now);
    char stamp[32] = "unknown-time";
    if (local) std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S", local);

    // Two captures inside the same second would otherwise overwrite each other.
    const std::string base = stem + "-" + stamp;
    fs::path candidate = directory / (base + ".png");
    for (int suffix = 2; fs::exists(candidate); ++suffix)
        candidate = directory / (base + "-" + std::to_string(suffix) + ".png");
    return candidate;
}

bool App::captureFrame(const fs::path& path) {
    // The most recently submitted frame, whose target still holds its pixels. In
    // accumulation mode that is slot 0 every time, and its target holds the mean
    // of every sample taken so far — so --capture --frames N writes the image as
    // it stood after N samples, not a single noisy one.
    const uint32_t last = (frameIndex_ + framesInFlight_ - 1) % framesInFlight_;
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

    // The simulation's count images track the framebuffer, and its descriptors
    // point at target views that were just recreated.
    if (sim_) sim_->resize(ctx_, targetViews(), swapchain_.extent());
    // Same for the accumulation image, which additionally has to start over: a
    // sum computed for one pixel grid means nothing on another.
    if (accum_) accum_->resize(ctx_, targetViews(), swapchain_.extent());
}

void App::updateTitle(double now) {
    ++framesSinceTitleUpdate_;
    if (now - lastTitleUpdate_ < 0.5) return;

    const double fps = framesSinceTitleUpdate_ / (now - lastTitleUpdate_);
    const VkExtent2D extent = swapchain_.extent();
    char detail[128] = "";
    if (sim_)
        std::snprintf(detail, sizeof(detail), " — %u bodies%s — speed %.3f", sim_->particleCount(),
                      sim_->paused() ? " (paused)" : "", camera_.speed);
    else if (accum_)
        // The sample count is the only feedback that says whether the image is
        // still refining or has been reset by a camera nudge you did not notice.
        std::snprintf(detail, sizeof(detail), " — %u samples — speed %.3f", accum_->sampleIndex(),
                      camera_.speed);
    else if (cameraMode_)
        std::snprintf(detail, sizeof(detail), " — speed %.3f", camera_.speed);
    char title[320];
    std::snprintf(title, sizeof(title), "vulkan-compute-lab — %s%s — %ux%u — %.0f fps",
                  shaderPath_.filename().string().c_str(), detail, extent.width, extent.height, fps);
    glfwSetWindowTitle(window_, title);

    lastTitleUpdate_ = now;
    framesSinceTitleUpdate_ = 0;
}

void App::cleanup() {
    if (sim_) {
        sim_->destroy(ctx_);
        sim_.reset();
    }
    if (accum_) {
        accum_->destroy(ctx_);
        accum_.reset();
    }
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
