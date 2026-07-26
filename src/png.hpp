#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace lab {

// Writes 8-bit RGBA pixels as a PNG. Self-contained: the deflate stream uses
// stored (uncompressed) blocks, which are valid zlib and keep the whole writer
// dependency-free at the cost of file size. Fine for screenshots.
bool writePng(const std::filesystem::path& path, uint32_t width, uint32_t height,
              const uint8_t* rgba, size_t rowPitch);

} // namespace lab
