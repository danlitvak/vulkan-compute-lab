cmake_minimum_required(VERSION 3.24)
project(vulkan_compute_lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Vulkan SDK: headers, vulkan-1.lib, and glslc (used at runtime for hot reload).
find_package(Vulkan REQUIRED)

# GLFW is the only third-party dependency. Fetched at configure time so the repo
# stays source-only; the download is cached in the build directory.
include(FetchContent)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw.git
  GIT_TAG        3.4
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glfw)

add_executable(lab
  src/main.cpp
  src/vk_common.cpp
  src/context.cpp
  src/swapchain.cpp
  src/shader_compiler.cpp
  src/png.cpp
  src/simulation.cpp
  src/app.cpp
)

target_link_libraries(lab PRIVATE Vulkan::Vulkan glfw)

# Warnings are set per-target, not globally: GLFW builds with its own /W3 and a
# global /W4 only produces D9025 override noise.
if(MSVC)
  target_compile_options(lab PRIVATE /W4 /permissive- /utf-8)
  target_compile_definitions(lab PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Shaders are compiled at *runtime* (that is what makes hot reload work), so the
# app needs to know where they live and where glslc is.
target_compile_definitions(lab PRIVATE
  LAB_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
  LAB_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  LAB_GLSLC_PATH="${Vulkan_GLSLC_EXECUTABLE}"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
  target_compile_definitions(lab PRIVATE LAB_VALIDATION=1)
endif()
