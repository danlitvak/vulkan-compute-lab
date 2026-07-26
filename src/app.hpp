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
//
// `mouse` is the raw pointer and `pan`/`zoom` are navigation state. Keeping them
// separate is what stops a click from jumping the image: absolute values are
// only ever read, never re-anchored, and navigation only ever accumulates
// deltas, so pressing a button contributes exactly zero.
struct PushConstants {
    float resolution[2];
    float mouse[2]; // normalised 0..1, y down, always live
    float pan[2];   // accumulated drag, in units of screen height
    float zoom;     // 1.0 at rest, grows as you scroll in
    float time;
    float deltaTime;
    uint32_t frame;
};

static_assert(sizeof(PushConstants) == 40, "shader Push block must match this layout");

struct Options {
    std::filesystem::path shader;
    std::filesystem::path capturePath;      // empty: no automatic capture
    uint32_t captureAfterFrames{60};        // let the animation get somewhere first
    bool exitAfterCapture{false};
    // Starting view, so a capture can be framed reproducibly instead of having
    // to be dragged there by hand.
    double panX{0.0};
    double panY{0.0};
    double zoom{1.0};
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
    static void onMouseButton(GLFWwindow* window, int button, int action, int mods);
    static void onScroll(GLFWwindow* window, double xOffset, double yOffset);

    void updateNavigation(float deltaTime);

    // Drag-to-pan, scroll-to-zoom. Targets are what input writes; the smoothed
    // values are what the shader sees, so motion eases instead of snapping.
    struct Navigation {
        double panX{0.0}, panY{0.0};
        double smoothPanX{0.0}, smoothPanY{0.0};
        double zoomExponent{0.0}; // zoom == exp(zoomExponent)
        double smoothZoomExponent{0.0};
        double lastCursorX{0.0}, lastCursorY{0.0};
        bool dragging{false};
    } nav_;

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
