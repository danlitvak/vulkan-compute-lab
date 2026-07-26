#pragma once

#include "vk_common.hpp"

#include <cstdint>

struct GLFWwindow;

namespace lab {

// Everything that lives for the whole run: instance, debug messenger, surface,
// device, and the one queue used for compute, transfer and present.
//
// A single queue family keeps the frame loop honest — no ownership transfers,
// no cross-queue semaphores. Essentially every desktop GPU exposes a graphics
// family that also does compute, transfer and present, so this costs nothing.
class Context {
public:
    void init(GLFWwindow* window, bool enableValidation);
    void destroy();

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physicalDevice_; }
    VkDevice device() const { return device_; }
    VkSurfaceKHR surface() const { return surface_; }
    VkQueue queue() const { return queue_; }
    uint32_t queueFamily() const { return queueFamily_; }
    VkCommandPool commandPool() const { return commandPool_; }
    const VkPhysicalDeviceProperties& properties() const { return properties_; }

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const;

private:
    void createInstance(bool enableValidation);
    void createDebugMessenger();
    void pickPhysicalDevice();
    void createDevice();

    VkInstance instance_{};
    VkDebugUtilsMessengerEXT debugMessenger_{};
    VkSurfaceKHR surface_{};
    VkPhysicalDevice physicalDevice_{};
    VkPhysicalDeviceProperties properties_{};
    VkPhysicalDeviceMemoryProperties memoryProperties_{};
    VkDevice device_{};
    uint32_t queueFamily_{UINT32_MAX};
    VkQueue queue_{};
    VkCommandPool commandPool_{};
    bool validation_{false};
};

} // namespace lab
