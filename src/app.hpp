#pragma once

#include "accumulator.hpp"
#include "context.hpp"
#include "push_constants.hpp"
#include "simulation.hpp"
#include "swapchain.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

struct GLFWwindow;

namespace lab {

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
    // Slots are always allocated; how many of them the loop actually cycles
    // through is framesInFlight_, because accumulation has to run with one.
    static constexpr uint32_t kMaxFramesInFlight = 2;

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
    bool reloadAccumulator(bool initial);
    void destroyPipeline();
    void discardSimulation();
    void discardAccumulator();
    void setFramesInFlight(uint32_t count);
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

    // Free-fly camera, used whenever the shader asked for a 3D view (//!camera3d,
    // //!nbody, //!accumulate). 2D pixel shaders keep the centre/zoom navigation
    // above; the two never apply at once — see cameraMode_.
    //
    // World is z-up with the galactic disk in the xy plane, which is the usual
    // convention for a disk galaxy and makes "height above the disk" just z.
    struct Camera {
        double x{0.0}, y{-0.95}, z{0.42};
        double yaw{1.5707963};   // radians, measured in the xy plane
        double pitch{-0.42};     // radians, clamped away from straight up/down
        double speed{0.35};      // world units per second
        double velX{0.0}, velY{0.0}, velZ{0.0}; // smoothed, for a little inertia
    } camera_;

    void updateCamera(float deltaTime);
    void cameraBasis(float right[3], float up[3], float forward[3]) const;
    void resetCamera();

    // The view a sample was traced with, as the exact float32 values the shader
    // received rather than the doubles behind them. That is the honest test for
    // "did the camera move": a change too small to alter the push constants
    // cannot alter the image either, so it must not throw the sum away.
    struct AccumView {
        float fovScale{};
        float camPos[4]{};
        float camRight[4]{};
        float camUp[4]{};
        float camForward[4]{};
        bool operator==(const AccumView&) const = default;
    };
    static AccumView accumViewOf(const PushConstants& push);
    AccumView lastAccumView_{};

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

    // At most one of these is ever non-null, chosen by the shader's directives.
    // When either is live the single-pass pipeline above is unused and that
    // object records the frame instead.
    std::unique_ptr<Simulation> sim_;   // //!nbody
    std::unique_ptr<Accumulator> accum_; // //!accumulate

    // //!camera3d, or implied by //!nbody or //!accumulate. Routes input to the
    // fly camera; otherwise the 2D pan/zoom above applies. Never both at once.
    bool cameraMode_{false};

    std::vector<Frame> frames_;
    std::vector<VkSemaphore> presentReady_; // one per swapchain image

    uint32_t framesInFlight_{kMaxFramesInFlight};
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
