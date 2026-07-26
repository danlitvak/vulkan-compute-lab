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

// Host-side switches a shader sets for itself, written as `//!name [args]` in a
// comment near the top of the file. They live in the shader rather than on the
// command line so a .comp file is self-describing: pointing the app at one is
// enough to get the mode it needs.
//
// `//!alias <names>` also exists but is consumed by run.ps1 when it builds its
// shader catalogue, so nothing here looks at it.
struct ShaderDirectives {
    uint32_t particleCount{0}; // //!nbody <count>; 0 selects the plain pixel path
    bool accumulate{false};    // //!accumulate; progressive path-tracer mode
    bool camera3d{false};      // //!camera3d; input drives the fly camera
};

// Scans the first 40 lines of a shader for every directive at once. Absent
// directives keep their defaults, so a plain pixel shader needs none.
ShaderDirectives readShaderDirectives(const std::filesystem::path& source);

} // namespace lab
