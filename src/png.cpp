#include "png.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <vector>

namespace lab {
namespace {

uint32_t crcTable(uint32_t index) {
    uint32_t c = index;
    for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
    return c;
}

uint32_t crc32(const uint8_t* data, size_t size, uint32_t crc = 0xFFFFFFFFu) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) t[i] = crcTable(i);
        return t;
    }();
    for (size_t i = 0; i < size; ++i) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc;
}

void pushBigEndian32(std::vector<uint8_t>& out, uint32_t value) {
    out.push_back(static_cast<uint8_t>(value >> 24));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value));
}

void pushChunk(std::vector<uint8_t>& out, const char type[5], const std::vector<uint8_t>& data) {
    pushBigEndian32(out, static_cast<uint32_t>(data.size()));
    const size_t crcStart = out.size();
    out.insert(out.end(), type, type + 4);
    out.insert(out.end(), data.begin(), data.end());
    const uint32_t crc = crc32(out.data() + crcStart, out.size() - crcStart) ^ 0xFFFFFFFFu;
    pushBigEndian32(out, crc);
}

uint32_t adler32(const std::vector<uint8_t>& data) {
    uint32_t a = 1, b = 0;
    for (uint8_t byte : data) {
        a = (a + byte) % 65521;
        b = (b + a) % 65521;
    }
    return (b << 16) | a;
}

} // namespace

bool writePng(const std::filesystem::path& path, uint32_t width, uint32_t height,
              const uint8_t* rgba, size_t rowPitch) {
    if (!rgba || width == 0 || height == 0) return false;

    // Scanlines, each prefixed by filter type 0 (none).
    std::vector<uint8_t> raw;
    raw.reserve(static_cast<size_t>(height) * (static_cast<size_t>(width) * 4 + 1));
    for (uint32_t y = 0; y < height; ++y) {
        raw.push_back(0);
        const uint8_t* row = rgba + static_cast<size_t>(y) * rowPitch;
        raw.insert(raw.end(), row, row + static_cast<size_t>(width) * 4);
    }

    // zlib container around stored deflate blocks.
    std::vector<uint8_t> zlib;
    zlib.reserve(raw.size() + raw.size() / 65535 * 5 + 16);
    zlib.push_back(0x78); // CM=8 (deflate), CINFO=7 (32K window)
    zlib.push_back(0x01); // FCHECK so that (0x78<<8 | 0x01) % 31 == 0, no dictionary
    for (size_t offset = 0; offset < raw.size();) {
        const size_t blockSize = std::min<size_t>(65535, raw.size() - offset);
        const bool isFinal = offset + blockSize >= raw.size();
        zlib.push_back(isFinal ? 1 : 0); // BFINAL, BTYPE=00 (stored)
        zlib.push_back(static_cast<uint8_t>(blockSize & 0xFF));
        zlib.push_back(static_cast<uint8_t>(blockSize >> 8));
        zlib.push_back(static_cast<uint8_t>(~blockSize & 0xFF));
        zlib.push_back(static_cast<uint8_t>((~blockSize >> 8) & 0xFF));
        zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(offset),
                    raw.begin() + static_cast<std::ptrdiff_t>(offset + blockSize));
        offset += blockSize;
    }
    pushBigEndian32(zlib, adler32(raw));

    std::vector<uint8_t> ihdr;
    pushBigEndian32(ihdr, width);
    pushBigEndian32(ihdr, height);
    ihdr.push_back(8); // bit depth
    ihdr.push_back(6); // colour type: truecolour with alpha
    ihdr.push_back(0); // deflate
    ihdr.push_back(0); // adaptive filtering
    ihdr.push_back(0); // no interlace

    std::vector<uint8_t> file{0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    pushChunk(file, "IHDR", ihdr);
    pushChunk(file, "IDAT", zlib);
    pushChunk(file, "IEND", {});

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
    return out.good();
}

} // namespace lab
