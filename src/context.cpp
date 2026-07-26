#include "context.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace lab {
namespace {

constexpr const char* kValidationLayer = "VK_LAYER_KHRONOS_validation";
constexpr const char* kSwapchainExtension = VK_KHR_SWAPCHAIN_EXTENSION_NAME;

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void*) {
    const char* tag = severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT     ? "ERROR"
                      : severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT ? "WARN "
                                                                                   : "INFO ";
    std::fprintf(stderr, "[validation %s] %s\n", tag, data->pMessage);
    return VK_FALSE;
}

bool layerAvailable(const char* name) {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const auto& layer : layers)
        if (std::strcmp(layer.layerName, name) == 0) return true;
    return false;
}

bool deviceHasExtension(VkPhysicalDevice device, const char* name) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data());
    for (const auto& extension : extensions)
        if (std::strcmp(extension.extensionName, name) == 0) return true;
    return false;
}

VkDebugUtilsMessengerCreateInfoEXT debugMessengerInfo() {
    VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
    return info;
}

} // namespace

void Context::init(GLFWwindow* window, bool enableValidation) {
    validation_ = enableValidation && layerAvailable(kValidationLayer);
    if (enableValidation && !validation_)
        std::fprintf(stderr, "[lab] validation layers requested but not available\n");

    createInstance(validation_);
    if (validation_) createDebugMessenger();

    VK_CHECK(glfwCreateWindowSurface(instance_, window, nullptr, &surface_));

    pickPhysicalDevice();
    createDevice();

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamily_;
    VK_CHECK(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_));
}

void Context::createInstance(bool enableValidation) {
    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "vulkan-compute-lab";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "lab";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    if (!glfwExtensions) throw std::runtime_error("GLFW could not report the required Vulkan instance extensions");

    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (enableValidation) extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    info.pApplicationInfo = &appInfo;
    info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    info.ppEnabledExtensionNames = extensions.data();

    // Chaining the messenger info here catches errors raised during instance
    // creation and destruction, which the standalone messenger cannot see.
    VkDebugUtilsMessengerCreateInfoEXT messengerInfo = debugMessengerInfo();
    if (enableValidation) {
        info.enabledLayerCount = 1;
        info.ppEnabledLayerNames = &kValidationLayer;
        info.pNext = &messengerInfo;
    }

    VK_CHECK(vkCreateInstance(&info, nullptr, &instance_));
}

void Context::createDebugMessenger() {
    auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (!create) return;
    VkDebugUtilsMessengerCreateInfoEXT info = debugMessengerInfo();
    VK_CHECK(create(instance_, &info, nullptr, &debugMessenger_));
}

void Context::pickPhysicalDevice() {
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    if (count == 0) throw std::runtime_error("no Vulkan-capable physical device found");
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));

    int bestScore = -1;
    for (VkPhysicalDevice candidate : devices) {
        if (!deviceHasExtension(candidate, kSwapchainExtension)) continue;

        uint32_t familyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
        std::vector<VkQueueFamilyProperties> families(familyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());

        uint32_t family = UINT32_MAX;
        for (uint32_t i = 0; i < familyCount; ++i) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(candidate, i, surface_, &present);
            const bool compute = (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
            if (compute && present) {
                family = i;
                break;
            }
        }
        if (family == UINT32_MAX) continue;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(candidate, &props);
        const int score = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU     ? 3
                          : props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2
                                                                                       : 1;
        if (score > bestScore) {
            bestScore = score;
            physicalDevice_ = candidate;
            properties_ = props;
            queueFamily_ = family;
        }
    }

    if (physicalDevice_ == VK_NULL_HANDLE)
        throw std::runtime_error("no device with a queue family supporting both compute and present");

    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memoryProperties_);
}

void Context::createDevice() {
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queueFamily_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    info.queueCreateInfoCount = 1;
    info.pQueueCreateInfos = &queueInfo;
    info.enabledExtensionCount = 1;
    info.ppEnabledExtensionNames = &kSwapchainExtension;
    info.pEnabledFeatures = &features;

    VK_CHECK(vkCreateDevice(physicalDevice_, &info, nullptr, &device_));
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
}

uint32_t Context::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const {
    for (uint32_t i = 0; i < memoryProperties_.memoryTypeCount; ++i) {
        const bool typeAllowed = (typeBits & (1u << i)) != 0;
        const bool hasFlags = (memoryProperties_.memoryTypes[i].propertyFlags & required) == required;
        if (typeAllowed && hasFlags) return i;
    }
    throw std::runtime_error("no memory type satisfies the requested properties");
}

void Context::destroy() {
    if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
    if (device_) vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (debugMessenger_) {
        auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy) destroy(instance_, debugMessenger_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);

    commandPool_ = VK_NULL_HANDLE;
    device_ = VK_NULL_HANDLE;
    surface_ = VK_NULL_HANDLE;
    debugMessenger_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
}

} // namespace lab
