// test_fuchsia_link.cpp -- Phase 2.1 link gate for the vendored Fuchsia
// vk_radix_sort library.
//
// Goal: prove that
//   (a) the vendored static library `vk-radix-sort` builds and links into the
//       harmonyos_3dgs test binary, and
//   (b) `radix_sort_vk_create` + `radix_sort_vk_destroy` succeed at runtime on
//       a Vulkan 1.2 device that exposes the four core features the SM35
//       target needs: shaderInt64, bufferDeviceAddress, vulkanMemoryModel,
//       vulkanMemoryModelDeviceScope.
//
// We do NOT exercise sort dispatch here — that is the wrapper's job in
// Phase 2.2. Phase 2.1 is purely "the bits are wired up and pipelines compile
// against this device."
//
// Setup pattern mirrors spike/fuchsia_radix_sort/bench_radix.cpp (which
// validated this exact API on Tegra Thor on 2026-04-25), but trimmed to just
// create+destroy. We do NOT use harmonyos_3dgs's VulkanContext because that
// helper does not enable the Vulkan-1.2 features Fuchsia requires; that
// hookup is the Phase-2.2 wrapper's responsibility, not this gate's.
//
// Skip semantics: GTEST_SKIP on "no instance", "no compute-capable device",
// or "device lacks the required core features". A FAIL is reserved for the
// case where the device claims the features but radix_sort_vk_create still
// returns NULL — that is a real integration bug.

#include <gtest/gtest.h>

#include <vulkan/vulkan.h>

#include "radix_sort/platforms/vk/radix_sort_vk.h"

#include <cstdint>
#include <vector>

namespace {

struct VkOwnedDevice {
    VkInstance       instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys     = VK_NULL_HANDLE;
    VkDevice         device   = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};

    ~VkOwnedDevice() {
        if (device   != VK_NULL_HANDLE) vkDestroyDevice(device, nullptr);
        if (instance != VK_NULL_HANDLE) vkDestroyInstance(instance, nullptr);
    }
};

// Returns a code/message describing why setup failed, or empty string on
// success. On success, `out` owns instance+device and `out.props` is valid.
// Fills `out_target` with the auto-detected radix_sort_vk_target_t.
std::string setup_vk_for_fuchsia(VkOwnedDevice & out,
                                 radix_sort_vk_target_t const ** out_target) {
    // ---- Instance (Vulkan 1.2 — Fuchsia targets vulkan1.2 SPIR-V).
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "test_fuchsia_link";
    app.apiVersion       = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &out.instance) != VK_SUCCESS)
        return "vkCreateInstance failed (no Vulkan loader / no driver)";

    // ---- Pick first physical device that has a compute-capable queue
    //      family AND exposes all four required core features.
    uint32_t pdc = 0;
    if (vkEnumeratePhysicalDevices(out.instance, &pdc, nullptr) != VK_SUCCESS || pdc == 0)
        return "no physical devices";
    std::vector<VkPhysicalDevice> pds(pdc);
    vkEnumeratePhysicalDevices(out.instance, &pdc, pds.data());

    int qf_idx = -1;
    VkPhysicalDeviceFeatures2 chosen_f2{};
    VkPhysicalDeviceVulkan12Features chosen_f12{};
    for (auto pd : pds) {
        VkPhysicalDeviceProperties pdp{};
        vkGetPhysicalDeviceProperties(pd, &pdp);
        if (pdp.apiVersion < VK_API_VERSION_1_2) continue;

        // Probe features.
        VkPhysicalDeviceVulkan12Features f12{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
        VkPhysicalDeviceFeatures2 f2{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f12;
        vkGetPhysicalDeviceFeatures2(pd, &f2);
        if (!f2.features.shaderInt64) continue;
        if (!f12.bufferDeviceAddress) continue;
        if (!f12.vulkanMemoryModel) continue;
        if (!f12.vulkanMemoryModelDeviceScope) continue;

        // Find a compute-capable queue family.
        uint32_t qfc = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, nullptr);
        std::vector<VkQueueFamilyProperties> qfp(qfc);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, qfp.data());
        int local_qf = -1;
        for (uint32_t i = 0; i < qfc; ++i) {
            if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                local_qf = static_cast<int>(i);
                break;
            }
        }
        if (local_qf < 0) continue;

        out.phys  = pd;
        out.props = pdp;
        qf_idx    = local_qf;
        chosen_f2  = f2;
        chosen_f12 = f12;
        break;
    }
    if (out.phys == VK_NULL_HANDLE)
        return "no compute device with shaderInt64 + bufferDeviceAddress + vulkanMemoryModel + vulkanMemoryModelDeviceScope";

    // ---- Auto-detect Fuchsia target (keyval_dwords=2 -> u64 keyvals).
    *out_target = radix_sort_vk_target_auto_detect(&out.props,
                                                   /*keyval_dwords=*/2);
    if (!*out_target)
        return "radix_sort_vk_target_auto_detect returned NULL for this device";

    // ---- Discover required device extensions.
    VkPhysicalDeviceFeatures         f10_req{};
    VkPhysicalDeviceVulkan11Features f11_req{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features f12_req{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    radix_sort_vk_target_requirements_t req{};
    req.pdf   = &f10_req;
    req.pdf11 = &f11_req;
    req.pdf12 = &f12_req;
    // First call discovers ext_name_count.
    radix_sort_vk_target_get_requirements(*out_target, &req);
    std::vector<const char *> ext_names(req.ext_name_count);
    req.ext_names = ext_names.empty() ? nullptr : ext_names.data();
    if (!radix_sort_vk_target_get_requirements(*out_target, &req))
        return "radix_sort_vk_target_get_requirements failed";

    // ---- Create logical device with the merged feature chain the target
    //      requires (the target zeroed any features it does not need; we
    //      keep the values it set).
    float prio = 1.0f;
    VkDeviceQueueCreateInfo dqci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    dqci.queueFamilyIndex = static_cast<uint32_t>(qf_idx);
    dqci.queueCount       = 1;
    dqci.pQueuePriorities = &prio;

    f12_req.pNext = nullptr;
    f11_req.pNext = &f12_req;
    VkPhysicalDeviceFeatures2 f2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext    = &f11_req;
    f2.features = f10_req;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext                   = &f2;
    dci.queueCreateInfoCount    = 1;
    dci.pQueueCreateInfos       = &dqci;
    dci.enabledExtensionCount   = req.ext_name_count;
    dci.ppEnabledExtensionNames = ext_names.empty() ? nullptr : ext_names.data();

    if (vkCreateDevice(out.phys, &dci, nullptr, &out.device) != VK_SUCCESS)
        return "vkCreateDevice failed with the target's required features";

    return std::string{};  // success
}

}  // namespace

// -----------------------------------------------------------------------------
// Phase 2.1 link gate: vendored Fuchsia static lib is present, headers
// resolve, target auto-detect succeeds, radix_sort_vk_create + destroy
// round-trip cleanly on this device.
// -----------------------------------------------------------------------------
TEST(FuchsiaLink, CreateAndDestroyRoundTrip) {
    VkOwnedDevice                  vk;
    radix_sort_vk_target_t const * target = nullptr;
    std::string                    why    = setup_vk_for_fuchsia(vk, &target);
    if (!why.empty()) {
        GTEST_SKIP() << "skip: " << why;
    }

    radix_sort_vk_t * rs =
        radix_sort_vk_create(vk.device, /*ac=*/nullptr, /*pc=*/VK_NULL_HANDLE,
                             target);
    ASSERT_NE(rs, nullptr)
        << "radix_sort_vk_create returned NULL on device '" << vk.props.deviceName
        << "' (vendor=0x" << std::hex << vk.props.vendorID
        << " device=0x" << vk.props.deviceID << std::dec
        << ") — vendored library link is up but pipeline creation failed";

    radix_sort_vk_destroy(rs, vk.device, /*ac=*/nullptr);
    SUCCEED() << "Fuchsia vk_radix_sort: create+destroy ok on '"
              << vk.props.deviceName << "'";
}
