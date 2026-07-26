#include "app.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <string>

#ifndef LAB_SHADER_DIR
#define LAB_SHADER_DIR "shaders"
#endif

namespace {

void printUsage() {
    std::printf(
        "usage: lab [shader.comp] [options]\n"
        "\n"
        "  shader.comp             compute shader to run (default: mandelbrot.comp)\n"
        "  --capture <file.png>    save a frame to file.png\n"
        "  --frames <n>            frames to render before capturing (default 60)\n"
        "  --exit-after-capture    quit once the capture is written\n"
        "  --pan <x> <y>           starting pan, in units of screen height\n"
        "  --zoom <z>              starting zoom (1.0 = default view)\n"
        "  --help\n"
        "\n"
        "mouse: drag to pan, scroll to zoom (about the cursor)\n"
        "keys:  R reload shader, Home reset view, F12 screenshot, Esc quit\n"
        "the shader also reloads on its own whenever the file changes on disk\n");
}

} // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    // Unbuffered: when stdout is a pipe or file, MSVC buffers fully, so a run
    // that is killed rather than closed loses its whole log.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    lab::Options options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool hasValue = i + 1 < argc;

        if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else if (arg == "--capture" && hasValue) {
            options.capturePath = argv[++i];
        } else if (arg == "--frames" && hasValue) {
            options.captureAfterFrames = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else if (arg == "--exit-after-capture") {
            options.exitAfterCapture = true;
        } else if (arg == "--pan" && i + 2 < argc) {
            options.panX = std::strtod(argv[++i], nullptr);
            options.panY = std::strtod(argv[++i], nullptr);
        } else if (arg == "--zoom" && hasValue) {
            options.zoom = std::strtod(argv[++i], nullptr);
        } else if (!arg.empty() && arg.front() == '-') {
            std::fprintf(stderr, "[lab] unknown option: %s\n", arg.c_str());
            printUsage();
            return 2;
        } else if (options.shader.empty()) {
            options.shader = arg;
        } else {
            std::fprintf(stderr, "[lab] unexpected argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (options.shader.empty()) options.shader = fs::path(LAB_SHADER_DIR) / "mandelbrot.comp";

    try {
        lab::App app(options);
        app.run();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[lab] fatal: %s\n", error.what());
        return 1;
    }
    return 0;
}
