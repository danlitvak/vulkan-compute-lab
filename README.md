# vulkan-compute-lab

A Vulkan harness for writing GPU **compute** shaders and seeing them immediately.
Edit a `.comp` file, save, and the window updates — no rebuild, no restart.

There is no graphics pipeline anywhere in this project. Compute writes pixels to
an offscreen storage image, and that image is blitted straight to the swapchain.
No render pass, no framebuffers, no vertex or fragment stages.

![mandelbrot](docs/mandelbrot.png)
![domain-warped noise](docs/warp.png)

*Both frames captured by the app itself with `--capture`.*

## Run it

```powershell
.\build.ps1 -Run
```

`build.ps1` locates VS Build Tools, imports the MSVC environment, and builds with
the CMake and Ninja that ship inside it — nothing needs to be on `PATH`.

```powershell
.\build.ps1 -Run -Shader shaders\warp.comp    # a different shader
.\build.ps1 -Config Release -Run              # no validation layers
```

| Key | |
|---|---|
| `R` | force a shader reload |
| `F12` | screenshot to `capture-NNN.png` |
| drag | feed the cursor to the shader's `mouse` uniform |
| `Esc` | quit |

Saving the shader file reloads it on its own; `R` is only for when you want to
re-run the compile without touching the file.

```
lab.exe [shader.comp] [--capture out.png] [--frames N] [--exit-after-capture]
```

`--capture` is how the images in this README were made, and how the renderer gets
verified without a human looking at a window.

The PNG writer in `src/png.cpp` has no dependencies, which it pays for by using
stored (uncompressed) deflate blocks — a 1280x720 capture lands around 3.7 MB.
Correct PNG, just large; recompress anything you intend to share.

## Writing a shader

Copy `shaders/mandelbrot.comp` and change the body. The contract is three things:

```glsl
layout(local_size_x = 16, local_size_y = 16) in;      // 256 threads per workgroup
layout(binding = 0, rgba8) uniform writeonly image2D target;

layout(push_constant) uniform Push {
    vec2  resolution;
    vec2  mouse;      // pixels, top-left origin; window centre unless dragging
    float time;       // seconds since start
    float deltaTime;
    uint  frame;
} pc;
```

Guard against the overhang — the dispatch is rounded up to whole workgroups, so
the last row and column of invocations sit outside the image:

```glsl
ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
if (pixel.x >= int(pc.resolution.x) || pixel.y >= int(pc.resolution.y)) return;
```

A shader that fails to compile prints the `glslc` diagnostic and leaves the
previous pipeline running, so a typo does not kill the session.

## How a frame works

```
wait frame fence
acquire swapchain image                         -> semaphore
  storage image  UNDEFINED     -> GENERAL       (barrier)
  dispatch  ceil(w/16) x ceil(h/16) workgroups
  storage image  GENERAL       -> TRANSFER_SRC  (barrier)
  swapchain img  UNDEFINED     -> TRANSFER_DST  (barrier)
  blit  storage image -> swapchain image
  swapchain img  TRANSFER_DST  -> PRESENT_SRC   (barrier)
submit (waits on acquire at TRANSFER)           -> semaphore
present
```

Decisions worth knowing about, because they are the ones that bite:

- **One queue** for compute, transfer and present. No queue-family ownership
  transfers to reason about.
- **A storage image per frame in flight.** Sharing one target lets frame N's
  dispatch overwrite pixels that frame N-1 is still blitting.
- **Present semaphores are per swapchain image, not per frame in flight.**
  Present consumes them asynchronously; a frame-indexed semaphore can still be
  pending when that frame slot comes around again.
- **`UNDEFINED` as the storage image's old layout** each frame, which discards
  the previous contents. That is right for a shader that writes every pixel and
  wrong for a ping-pong effect — those need the old layout preserved.
- **Blit, not copy.** The target is `R8G8B8A8_UNORM` and the swapchain is usually
  `B8G8R8A8_UNORM`; only a blit converts between them. A copy would swap R and B.
- **A non-sRGB swapchain format** is preferred, so the 8-bit values the shader
  writes reach the screen unchanged instead of picking up an extra encode.

## Layout

```
src/
  context.cpp          instance, debug messenger, device, queue, command pool
  swapchain.cpp        surface format/present mode choice, present-only swapchain
  shader_compiler.cpp  runs glslc at runtime; this is what makes reload work
  app.cpp              storage images, descriptors, pipeline, frame loop, capture
  png.cpp              dependency-free PNG writer for screenshots
shaders/
  mandelbrot.comp      smooth-iteration Mandelbrot with a breathing zoom
  warp.comp            domain-warped value noise
```

Shaders are compiled at **runtime**, not build time — that is the whole point of
the reload loop. It means the Vulkan SDK's `glslc` has to be present when the app
runs, not just when it builds.

## Requirements

- A Vulkan 1.3 GPU and driver
- [LunarG Vulkan SDK](https://vulkan.lunarg.com/) — headers, `glslc`, validation layers
- MSVC with the C++ workload (VS 2022 or Build Tools), CMake 3.24+, Ninja
- GLFW is fetched automatically by CMake; there are no vendored dependencies

Debug builds enable `VK_LAYER_KHRONOS_validation` and print warnings and errors
to stderr. Development happens with them on.

## Next

- [ ] Storage buffers: an N-body particle system, dispatch over particles rather than pixels
- [ ] Ping-pong images for stateful simulation (reaction-diffusion, fluid)
- [ ] A raymarched SDF scene with soft shadows
- [ ] Timestamp queries, to report actual GPU dispatch time instead of frame rate
- [ ] Shared-memory tiling, to show `local_size` and cache behaviour mattering
