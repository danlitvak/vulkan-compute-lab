#pragma once

#include "context.hpp"
#include "simulation.hpp"
#include "swapchain.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
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
    float mouse[2];  // normalised 0..1, y down, always live
    float center[2]; // view centre, in screen-height units at zoom 1
    float zoom;      // 1.0 at rest, grows as you scroll in
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
    double centerX{0.0};
    double centerY{0.0};
    double zoom{1.0};
    // Deepest zoom allowed. 0 derives it from the framebuffer height; negative
    // removes the limit and lets the view quantize.
    double maxZoom{0.0};
    // Overrides a simulation shader's //!nbody directive. 0 keeps the directive.
    uint32_t particleCount{0};
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
    bool reloadSimulation(uint32_t particleCount, bool initial);
    void destroyPipeline();
    std::vector<VkImageView> targetViews() const;

    void mainLoop();
    bool drawFrame(); // false when the frame was skipped to rebuild the swapchain
    void handleCapture();
    bool captureFrame(const std::filesystem::path& path);
    std::filesystem::path screenshotDirectory() const; // creates it if missing
    std::filesystem::path screenshotPath() const;
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
    //
    // The view is stored as (centre, scale) with scale == 1/zoom, and both are
    // eased linearly with the same factor. That pairing is what makes zoom
    // anchoring exact: the anchor condition centre == anchor - uv * scale is
    // affine in scale, so any interpolation that moves both together preserves
    // it at every intermediate frame, not just at the endpoints.
    struct Navigation {
        double centerX{0.0}, centerY{0.0};
        double smoothCenterX{0.0}, smoothCenterY{0.0};
        double scale{1.0}; // world units per unit of uv; zoom == 1/scale
        double smoothScale{1.0};
        double lastCursorX{0.0}, lastCursorY{0.0};
        bool dragging{false};
    } nav_;

    double minScale() const; // the deepest zoom fp32 coordinates still resolve
    bool zoomClampReported_{false};

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

    // Non-null only when the shader declares //!nbody. In that mode the
    // single-pass pipeline above is unused and the simulation records the frame.
    std::unique_ptr<Simulation> sim_;

    std::vector<Frame> frames_;
    std::vector<VkSemaphore> presentReady_; // one per swapchain image

    uint32_t frameIndex_{0};
    uint32_t frameCounter_{0};
    bool framebufferResized_{false};
    bool reloadRequested_{false};
    bool screenshotRequested_{false};
    bool autoCaptureDone_{false};

    double startTime_{0.0};
    double lastFrameTime_{0.0};
    double lastTitleUpdate_{0.0};
    uint32_t framesSinceTitleUpdate_{0};
};

} // namespace lab
