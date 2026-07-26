#include "shader_compiler.hpp"

#include <cstdio>
#include <cstdlib>
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

} // namespace

CompileResult compileComputeShader(const fs::path& source) {
    CompileResult result;

    if (!fs::exists(source)) {
        result.message = "shader not found: " + source.string();
        return result;
    }

    const fs::path temp = fs::temp_directory_path();
    const fs::path spvPath = temp / "vulkan_compute_lab.spv";
    const fs::path logPath = temp / "vulkan_compute_lab.log";

    std::error_code ec;
    fs::remove(spvPath, ec);

    fs::path input = source;
    if (hasUtf8Bom(source)) {
        const fs::path stripped = temp / ("vulkan_compute_lab_nobom" + source.extension().string());
        if (stripBom(source, stripped)) input = stripped;
    }

    std::ostringstream command;
    command << '"' << glslcPath().string() << '"'
            << " -O --target-env=vulkan1.3 -fshader-stage=compute"
            << " \"" << input.string() << "\""
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
