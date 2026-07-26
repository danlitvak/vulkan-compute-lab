#include "shader_compiler.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace lab {
namespace {

fs::path glslcPath() {
#ifdef LAB_GLSLC_PATH
    const fs::path configured{LAB_GLSLC_PATH};
    if (!configured.empty() && fs::exists(configured)) return configured;
#endif
    if (const char* sdk = std::getenv("VULKAN_SDK")) {
        const fs::path fromSdk = fs::path(sdk) / "Bin" / "glslc.exe";
        if (fs::exists(fromSdk)) return fromSdk;
    }
    return fs::path("glslc"); // last resort: hope it is on PATH
}

std::string readTextFile(const fs::path& path) {
    std::ifstream file(path);
    if (!file) return {};
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// glslc does not skip a UTF-8 byte order mark, so #version goes unrecognised and
// the diagnostic that comes back is about built-in declarations rather than the
// real problem. Strip it into a temporary copy instead of letting that through.
bool stripBom(const fs::path& source, const fs::path& destination) {
    std::ifstream in(source, std::ios::binary);
    if (!in) return false;
    std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (bytes.size() < 3) return false;
    std::ofstream out(destination, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(bytes.data() + 3, static_cast<std::streamsize>(bytes.size() - 3));
    return out.good();
}

bool hasUtf8Bom(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    unsigned char head[3]{};
    in.read(reinterpret_cast<char*>(head), 3);
    return in.gcount() == 3 && head[0] == 0xEF && head[1] == 0xBB && head[2] == 0xBF;
}

std::vector<uint32_t> readSpirv(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto size = static_cast<std::streamsize>(file.tellg());
    if (size <= 0 || size % 4 != 0) return {};
    std::vector<uint32_t> words(static_cast<size_t>(size) / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(words.data()), size);
    return words;
}

// Matches `name` against the directive opening `line` at `start`, returning a
// pointer just past the name (so the caller can parse arguments) or nullptr.
//
// Two rules, both load-bearing:
//   * the directive must OPEN the line, because prose that merely mentions one
//     must not act as one — galaxy.comp's header both describes its `//!nbody`
//     directive in English and carries `//!alias gravity grav nbody stars`, and
//     neither line may be what switches the simulation path on;
//   * the name must end at the match, or `//!camera3dish` would pass for
//     `//!camera3d` and a typo would silently change modes instead of being
//     ignored.
const char* directiveArgument(const std::string& line, size_t start, const char* name) {
    const size_t length = std::strlen(name);
    if (line.compare(start, length, name) != 0) return nullptr;

    const size_t after = start + length;
    if (after < line.size()) {
        const char next = line[after];
        // \r is here for a file with mixed line endings: text-mode getline only
        // strips the \r that came with a \n.
        if (next != ' ' && next != '\t' && next != '\r') return nullptr;
    }
    return line.c_str() + after;
}

} // namespace

ShaderDirectives readShaderDirectives(const fs::path& source) {
    ShaderDirectives directives;

    std::ifstream file(source);
    if (!file) return directives;

    std::string line;
    for (int scanned = 0; scanned < 40 && std::getline(file, line); ++scanned) {
        const size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;

        if (const char* argument = directiveArgument(line, start, "//!nbody")) {
            char* end = nullptr;
            const unsigned long count = std::strtoul(argument, &end, 10);
            // No number after the name: treat the line as prose and keep
            // scanning, rather than failing the whole load over a half-typed
            // directive in a file that is being edited live.
            if (end != argument && count > 0 && directives.particleCount == 0)
                directives.particleCount = static_cast<uint32_t>(count);
            continue;
        }
        if (directiveArgument(line, start, "//!accumulate")) {
            directives.accumulate = true;
            continue;
        }
        if (directiveArgument(line, start, "//!camera3d")) {
            directives.camera3d = true;
            continue;
        }
    }

    // Accumulation implies the fly camera: a path tracer is framed by a 3D view
    // by construction, and there is no 2D pan/zoom that would mean anything to
    // it. Folding it in here keeps every caller from having to remember.
    if (directives.accumulate) directives.camera3d = true;
    return directives;
}

CompileResult compileComputeShader(const fs::path& source, const std::vector<std::string>& defines) {
    CompileResult result;

    if (!fs::exists(source)) {
        result.message = "shader not found: " + source.string();
        return result;
    }

    // The output name has to vary with the defines, or the passes of a
    // multi-pass effect overwrite each other's SPIR-V.
    std::string variant;
    for (const std::string& define : defines) variant += "_" + define;

    const fs::path temp = fs::temp_directory_path();
    const fs::path spvPath = temp / ("vulkan_compute_lab" + variant + ".spv");
    const fs::path logPath = temp / ("vulkan_compute_lab" + variant + ".log");

    std::error_code ec;
    fs::remove(spvPath, ec);

    fs::path input = source;
    if (hasUtf8Bom(source)) {
        const fs::path stripped = temp / ("vulkan_compute_lab_nobom" + source.extension().string());
        if (stripBom(source, stripped)) input = stripped;
    }

    std::ostringstream command;
    command << '"' << glslcPath().string() << '"'
            << " -O --target-env=vulkan1.3 -fshader-stage=compute";
    for (const std::string& define : defines) command << " -D" << define;
    command << " \"" << input.string() << "\""
            << " -o \"" << spvPath.string() << "\""
            << " > \"" << logPath.string() << "\" 2>&1";

    // cmd.exe strips the outermost pair of quotes from the command line, so the
    // whole thing has to be wrapped again for the quoted paths to survive.
    const std::string wrapped = "\"" + command.str() + "\"";
    const int exitCode = std::system(wrapped.c_str());

    std::string log = readTextFile(logPath);

    // Diagnostics should name the file the user edits, not the temporary copy.
    if (input != source) {
        const std::string from = input.string();
        const std::string to = source.string();
        for (size_t at = log.find(from); at != std::string::npos; at = log.find(from, at + to.size()))
            log.replace(at, from.size(), to);
    }

    if (exitCode != 0 || !fs::exists(spvPath)) {
        result.message = log.empty() ? "glslc failed (exit " + std::to_string(exitCode) + ")" : log;
        return result;
    }

    result.spirv = readSpirv(spvPath);
    if (result.spirv.empty()) {
        result.message = "glslc produced an empty or malformed SPIR-V module";
        return result;
    }

    result.ok = true;
    result.message = log; // may carry warnings even on success
    return result;
}

} // namespace lab
