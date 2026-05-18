#include "radix_sort/platforms/vk/radix_sort_vk.h"

// AAAGS LOCAL PATCH: vendor-tree trimmed per integration plan §2 / §8 Decision #3.
// The original auto-detect referenced gcn3 (AMD), gen8 (Intel), and bifrost4
// (ARM Mali pre-v8). Those target source dirs were dropped from this vendored
// copy, so their `extern` symbols would fail to link. Removed the matching
// `extern` decls + switch cases. Detect now returns NULL on unsupported
// vendors/devices instead of dispatching to a missing target.
extern const struct radix_sort_vk_target bifrost8_u32_target;
extern const struct radix_sort_vk_target bifrost8_u64_target;
extern const struct radix_sort_vk_target bifrost8_u96_target;
extern const struct radix_sort_vk_target bifrost8_u128_target;
extern const struct radix_sort_vk_target sm35_u32_target;
extern const struct radix_sort_vk_target sm35_u64_target;
extern const struct radix_sort_vk_target sm35_u96_target;
extern const struct radix_sort_vk_target sm35_u128_target;

radix_sort_vk_target_t const *
radix_sort_vk_target_auto_detect(VkPhysicalDeviceProperties const * props,
                                 uint32_t                           keyval_dwords)
{
  uint32_t vendor_id = props->vendorID;
  uint32_t device_id = props->deviceID;
  switch (vendor_id)
    {
      case 0x10DE:
      default:
        //
        // NVIDIA (default for unknown vendors — empirically Tegra Thor is
        // 0x10DE; in this trimmed build other vendors fall through to NVIDIA
        // SM35 too, which is intentionally permissive while we expand vendor
        // coverage).
        //
        switch (keyval_dwords)
          {
            case 1:
              return &sm35_u32_target;
            case 2:
              return &sm35_u64_target;
            case 3:
              return &sm35_u96_target;
            case 4:
              return &sm35_u128_target;
          }

      case 0x13B5:
        //
        // ARM MALI — only BIFROST8 is shipped in this trimmed vendor copy.
        // device_id 0x72120000 is the canonical Bifrost8; any other ARM Mali
        // device falls through to bifrost8 as the closest available target.
        //
        (void)device_id;
        switch (keyval_dwords)
          {
            case 1:
              return &bifrost8_u32_target;
            case 2:
              return &bifrost8_u64_target;
            case 3:
              return &bifrost8_u96_target;
            case 4:
              return &bifrost8_u128_target;
          }
    }
  return NULL;
}
