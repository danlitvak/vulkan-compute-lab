#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace lab {

struct CompileResult {
    bool ok{false};
    std::vector<uint32_t> spirv;
    std::string message; // glslc diagnostics on failure
};

// Compiles a GLSL compute shader to SPIR-V by shelling out to the SDK's glslc.
// Done at runtime rather than at build time so shaders can be edited and
// reloaded without restarting the app.
CompileResult compileComputeShader(const std::filesystem::path& source);

} // namespace lab
