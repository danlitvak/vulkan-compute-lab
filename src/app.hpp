#pragma once

#include "context.hpp"
#include "swapchain.hpp"

#include <cstdint>
#include <filesystem>
#include <vector>

struct GLFWwindow;

namespace lab {

// Mirrored by the `Push` block in every .comp shader. Keep the two in sync:
// std430 push-constant rules put these at the same offsets as the C++ struct.
struct PushConstants {
    float resolution[2];
    float mouse[2];
    float time;
    float deltaTime;
    uint32_t frame;
};

struct Options {
    std::filesystem::path shader;
    std::filesystem::path capturePath;      // empty: no automatic capture
    uint32_t captureAfterFrames{60};        // let the animation get somewhere first
    bool exitAfterCapture{false};
};

class App {
public:
    explicit App(Options options);
    void run();

private:
    static constexpr uint32_t kFramesInFlight = 2;

    // One storage image per frame in flight. Sharing a single target would let
    // frame N's dispatch overwrite frame N-1's pixels while they are still
    // being blitted.
    struct Frame {
        VkCommandBuffer cmd{};
        VkSemaphore imageAvailable{};
        VkFence inFlight{};
        VkImage target{};
        VkDeviceMemory targetMemory{};
        VkImageView targetView{};
        VkDescriptorSet descriptor{};
    };

    void initWindow();
    void initVulkan();
    void createDescriptorResources();
    void createFrames();
    void createFrameTargets();
    void destroyFrameTargets();
    void createPresentSemaphores();
    void destroyPresentSemaphores();

    bool reloadShader(bool initial);
    void destroyPipeline();

    void mainLoop();
    bool drawFrame(); // false when the frame was skipped to rebuild the swapchain
    void handleCapture();
    bool captureFrame(const std::filesystem::path& path);
    void recreateSwapchain();
    void pollShaderFile();
    void updateTitle(double now);
    void cleanup();

    static void onFramebufferResize(GLFWwindow* window, int width, int height);
    static void onKey(GLFWwindow* window, int key, int scancode, int action, int mods);

    Options options_;
    std::filesystem::path shaderPath_;
    std::filesystem::file_time_type shaderWriteTime_{};

    GLFWwindow* window_{};
    Context ctx_;
    Swapchain swapchain_;

    VkDescriptorSetLayout setLayout_{};
    VkPipelineLayout pipelineLayout_{};
    VkPipeline pipeline_{};
    VkDescriptorPool descriptorPool_{};

    std::vector<Frame> frames_;
    std::vector<VkSemaphore> presentReady_; // one per swapchain image

    uint32_t frameIndex_{0};
    uint32_t frameCounter_{0};
    bool framebufferResized_{false};
    bool reloadRequested_{false};
    bool screenshotRequested_{false};
    bool autoCaptureDone_{false};
    uint32_t screenshotCounter_{0};

    double startTime_{0.0};
    double lastFrameTime_{0.0};
    double lastTitleUpdate_{0.0};
    uint32_t framesSinceTitleUpdate_{0};
};

} // namespace lab
