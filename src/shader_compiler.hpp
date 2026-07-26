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
//
// `defines` are passed as -D. A multi-pass effect keeps every pass in one file
// behind #ifdef guards and compiles that file once per pass, so the whole thing
// still reloads as a unit when you save.
CompileResult compileComputeShader(const std::filesystem::path& source,
                                   const std::vector<std::string>& defines = {});

// Reads a `//!nbody <count>` directive from the top of a shader. Returns 0 when
// absent, which is what selects the plain one-dispatch-per-pixel path.
uint32_t readParticleDirective(const std::filesystem::path& source);

} // namespace lab
