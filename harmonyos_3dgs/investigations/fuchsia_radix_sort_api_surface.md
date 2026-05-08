# Fuchsia Vulkan Radix Sort API Surface

## Vendored C API

Header: `third_party/fuchsia_vk_radix_sort/platforms/vk/include/radix_sort/platforms/vk/radix_sort_vk.h`.

- `radix_sort_vk_target_auto_detect(const VkPhysicalDeviceProperties*, uint32_t keyval_dwords) -> const radix_sort_vk_target_t*`
  - Selects a prebuilt vendor/device target. The profiling integration uses `keyval_dwords=2` for 64-bit key/value entries.
- `radix_sort_vk_target_get_requirements(const radix_sort_vk_target_t*, radix_sort_vk_target_requirements_t*) -> bool`
  - Reports required device extensions/features before device creation.
- `radix_sort_vk_create(VkDevice, const VkAllocationCallbacks*, VkPipelineCache, const radix_sort_vk_target_t*) -> radix_sort_vk_t*`
  - Creates pipelines for the selected target. Returns null on failure.
- `radix_sort_vk_destroy(radix_sort_vk_t*, VkDevice, const VkAllocationCallbacks*)`
  - Destroys the instance with the same device/allocator.
- `radix_sort_vk_get_memory_requirements(const radix_sort_vk_t*, uint32_t count, radix_sort_vk_memory_requirements_t*)`
  - Reports keyval/internal/indirect buffer sizes and alignments.
- `radix_sort_vk_sort(const radix_sort_vk_t*, const radix_sort_vk_sort_info_t*, VkDevice, VkCommandBuffer, VkDescriptorBufferInfo* keyvals_sorted)`
  - Records direct-dispatch sort work into a caller-owned command buffer and returns which ping-pong keyval buffer contains sorted output.

## Local C++ wrapper

Header: `include/vulkan/radix_sort_fuchsia.h`.

- `RadixSortFuchsia(VulkanContext&, uint32_t max_keyvals)` auto-detects target and creates the C library instance.
- `~RadixSortFuchsia()` destroys the C library instance.
- `memory_requirements(uint32_t count)` returns byte sizes and alignments for buffers.
- `record(VkCommandBuffer, VkBuffer keyvals_in, VkBuffer keyvals_scratch, VkBuffer internal_scratch, uint32_t count, uint32_t key_bits)` appends sort commands.
- `sorted_buffer_after_sort(...)` returns the sorted buffer captured during the latest `record()` call.
- `pick_target(VulkanContext&)` is exposed for wrapper tests.
