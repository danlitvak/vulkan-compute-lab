#pragma once

#include <cstddef>
#include <cstdint>

namespace lab {

// One push-constant block for every mode: plain pixel shader, N-body simulation,
// progressive accumulation. Mirrored by the `Push` block in every .comp shader.
//
// A shader declares only the prefix it uses — mandelbrot.comp stops at `frame`,
// galaxy.comp at `camForward` — which is legal because leaving a trailing field
// out cannot move the offset of any field before it. That is the whole reason
// for one shared struct instead of one per mode: the host pushes the same bytes
// no matter which mode is running, and a shader looks at whatever it cares
// about. Adding a field is only ever safe at the *end* for the same reason.
//
// `mouse` is the raw pointer position while `center`/`zoom` are navigation
// state. Keeping them separate is what stops a click from jumping the image:
// absolute values are only ever read, never re-anchored, and navigation only
// ever accumulates deltas, so pressing a button contributes exactly zero.
//
// The camera basis is vec4 rather than vec3 on purpose. std430 gives a vec3
// 16-byte alignment *and* 16 bytes of stride anyway, so declaring vec3 on the
// shader side would silently pad and desynchronise the two structs. vec4 makes
// the padding explicit and keeps every member on an offset both sides agree on.
//
// 128 bytes is exactly the push-constant budget every Vulkan implementation must
// provide (maxPushConstantsSize >= 128), so this is the largest block that needs
// no capability check before it can be used.
struct alignas(16) PushConstants {
    float resolution[2];    //   0  framebuffer size in pixels
    float mouse[2];         //   8  normalised 0..1, y down, always live
    float center[2];        //  16  2D view centre, pixel shaders only
    float zoom;             //  24  1.0 at rest, grows as you scroll in
    float time;             //  28  seconds since launch
    float deltaTime;        //  32  seconds since the previous frame
    uint32_t frame;         //  36  frames rendered since launch
    uint32_t particleCount; //  40  0 when this is not a simulation
    float fovScale;         //  44  tan(fov / 2)
    float camPos[4];        //  48  xyz + pad, 16-byte aligned
    float camRight[4];      //  64
    float camUp[4];         //  80
    float camForward[4];    //  96
    uint32_t sampleIndex;   // 112  accumulation: samples ALREADY in the buffer
    uint32_t pad[3];        // 116  spelled out so `{}` zeroes it, rather than
};                          // 128  leaving the tail to an alignment quirk

static_assert(sizeof(PushConstants) == 128, "shader Push block must match this layout");
static_assert(alignof(PushConstants) == 16, "vec4 members need 16-byte alignment");
static_assert(offsetof(PushConstants, camPos) == 48, "camera basis must stay 16-byte aligned");
static_assert(offsetof(PushConstants, sampleIndex) == 112, "accumulation counter moved");

} // namespace lab
