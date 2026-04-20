// test_prefix_scan_pass_vk.cpp -- SP-2 T10 TDD gate for PrefixScanPass.
//
// Verifies the 3-phase Blelloch exclusive scan against std::exclusive_scan on
// three shapes that each exercise a different piece of the shader:
//
//   * N=64  - single workgroup, phase-0 only meaningfully active (phase-1
//             scans a 1-element workgroup_sums, phase-2 adds back zero).
//             Input is 1..64, sanity-check the basic exclusive-prefix contract.
//   * N=256 - exactly one full workgroup. Tests the phase-0 boundary: the
//             final Blelloch down-sweep writes s_data[255]=total-last, and
//             if lid 255 is not handled correctly, output[255] diverges.
//             Input is all-ones -> output[i] = i.
//   * N=4096 - 16 workgroups, exercising phase-1 (scan of 16 WG sums) and
//             phase-2 (add-back). Input alternates {1,2,1,2,...} so adjacent
//             WG offsets differ, catching off-by-one workgroup indexing bugs.

#include "vulkan/tile_binner_passes.h"
#include "vulkan/vk_context.h"
#include "vulkan/vk_buffer.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <vector>

namespace {

// Run one exclusive scan on the GPU and compare with std::exclusive_scan.
// Returns the GPU output so individual tests can inspect it if they want.
std::vector<uint32_t> run_and_check(VulkanContext& ctx,
                                    const std::vector<uint32_t>& input) {
    const uint32_t N      = static_cast<uint32_t>(input.size());
    const uint32_t num_wg = (N + 255u) / 256u;
    const uint32_t ws_n   = num_wg == 0u ? 1u : num_wg;

    VulkanBuffer in_buf (ctx, N      * sizeof(uint32_t));
    VulkanBuffer out_buf(ctx, N      * sizeof(uint32_t));
    VulkanBuffer ws_buf (ctx, ws_n   * sizeof(uint32_t));

    in_buf.upload(input.data(), N * sizeof(uint32_t));
    // Pre-fill output with a sentinel so we can detect phase-2 not running.
    std::vector<uint32_t> sentinel(N, 0xDEADBEEFu);
    out_buf.upload(sentinel.data(), N * sizeof(uint32_t));

    PrefixScanPass scan(ctx);
    scan.bind_buffers(in_buf.handle(), out_buf.handle(), ws_buf.handle());
    scan.scan_sync(N);

    std::vector<uint32_t> got(N, 0u);
    out_buf.download(got.data(), N * sizeof(uint32_t));

    std::vector<uint32_t> expected(N, 0u);
    std::exclusive_scan(input.begin(), input.end(), expected.begin(),
                        static_cast<uint32_t>(0));

    EXPECT_EQ(got.size(), expected.size());
    // Use EXPECT_EQ (not ASSERT_EQ) so this helper returns `got` normally;
    // ASSERT_* returns void from the enclosing function, which conflicts with
    // the std::vector<uint32_t> return type here.
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(got[i], expected[i])
            << "scan mismatch at i=" << i
            << " input[i]=" << (i < input.size() ? input[i] : 0u);
    }
    return got;
}

}  // namespace

TEST(PrefixScanPassVk, Small64_Incrementing) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    std::vector<uint32_t> in(64);
    for (uint32_t i = 0; i < 64u; ++i) in[i] = i + 1u;   // 1..64

    auto got = run_and_check(ctx, in);
    // A concrete spot-check: sum of 1..63 = 63*64/2 = 2016.
    EXPECT_EQ(got[63], 2016u);
}

TEST(PrefixScanPassVk, Exactly256_AllOnes) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    std::vector<uint32_t> in(256u, 1u);
    auto got = run_and_check(ctx, in);
    // For all-ones exclusive prefix: output[i] = i.
    EXPECT_EQ(got[0], 0u);
    EXPECT_EQ(got[255], 255u);
}

TEST(PrefixScanPassVk, MultiWorkgroup4096_Alternating) {
    VulkanContext ctx;
    ASSERT_TRUE(ctx.init()) << "Vulkan init failed — no compute device?";

    std::vector<uint32_t> in(4096u);
    for (uint32_t i = 0; i < in.size(); ++i) in[i] = (i & 1u) ? 2u : 1u;

    auto got = run_and_check(ctx, in);
    // Spot-check: after 4096 elements the total is 1.5 * 4096 = 6144; the
    // exclusive prefix at the end is total - in[4095] = 6144 - 2 = 6142.
    EXPECT_EQ(got[4095], 6142u);
}
