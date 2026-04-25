// test_config_loader.cpp — Path A defensive config unification.
//
// 1. Load `AAA-Gaussians/configs/aaa.json` and check the values match what
//    the VK shaders are hard-coded against.
// 2. Construct RasterizerVulkan(ctx, settings) — must succeed.
// 3. Mutate `sort_mode = GLOBAL` and verify the constructor throws.
// 4. Same for queue_sizes / new_aabb / culling flags / load_balancing /
//    proper_ewa_scaling.
//
// REPO_ROOT_DIR is defined by gs3d_vk_tests' compile definitions and points
// at the harmonyos_3dgs/.. directory, so the JSON path is
// REPO_ROOT_DIR "/AAA-Gaussians/configs/aaa.json".

#include "splatting_settings.h"
#include "rasterizer.h"            // PreprocessOutput / BinningOutput / RenderConfig (via types.h)
#include "vulkan/rasterizer_vulkan.h"
#include "vulkan/vk_context.h"

#include <gtest/gtest.h>

#include <vector>

#include <filesystem>
#include <stdexcept>
#include <string>

namespace {
// Resolve aaa.json path. We prefer the canonical submodule path, falling
// back to a vendored copy under tests/test_data/ for worktrees where the
// AAA-Gaussians submodule is not checked out (pure-VK CI environments).
//
// BOTH copies must agree — if the submodule path exists and the schema /
// values diverge from the bundled fixture, that is a bug to fix in
// tests/test_data/aaa.json (or in the source-of-truth, depending on which
// changed). Keeping the fallback file in sync is part of the "Update =
// Rewrite" discipline.
std::string aaa_json_path() {
    namespace fs = std::filesystem;
    std::string canonical = std::string(REPO_ROOT_DIR) + "/AAA-Gaussians/configs/aaa.json";
    if (fs::exists(canonical)) return canonical;
    return std::string(TEST_DATA_DIR) + "/test_data/aaa.json";
}
}  // namespace

TEST(ConfigLoader, AaaJsonMatchesVkHardcoded) {
    // The file must exist — if not, the worktree is missing the AAA submodule.
    ASSERT_TRUE(std::filesystem::exists(aaa_json_path()))
        << "aaa.json not found at " << aaa_json_path();

    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);

    // Spot-check loaded values against the snapshot in CLAUDE.md / Path A spec.
    EXPECT_EQ(s.sort_settings.sort_mode,  splatting::HIERARCHICAL);
    EXPECT_EQ(s.sort_settings.sort_order, splatting::PER_TILE_DEPTH_MAXPOS);
    EXPECT_EQ(s.sort_settings.queue_sizes.per_pixel, 4);
    EXPECT_EQ(s.sort_settings.queue_sizes.tile_2x2,  8);
    EXPECT_EQ(s.sort_settings.queue_sizes.tile_4x4,  64);
    EXPECT_TRUE(s.culling_settings.rect_bounding);
    EXPECT_TRUE(s.culling_settings.tight_opacity_bounding);
    EXPECT_TRUE(s.culling_settings.tile_based_culling);
    EXPECT_TRUE(s.culling_settings.hierarchical_4x4_culling);
    EXPECT_TRUE(s.load_balancing);
    EXPECT_TRUE(s.proper_ewa_scaling);
    EXPECT_TRUE(s.eval_3D);
    EXPECT_TRUE(s.new_aabb);

    // All values match VK hard-coded — validator should NOT throw.
    EXPECT_NO_THROW(splatting::validate_vk_supported(s));
}

// Y1: GLOBAL is now a supported sort_mode (routes to the existing HEAD_W=8
// fallback path). The validator must accept it.
TEST(ConfigLoader, ValidatorAcceptsGlobalSortMode) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.sort_mode = splatting::GLOBAL;
    EXPECT_NO_THROW(splatting::validate_vk_supported(s));
}

// HIERARCHICAL must continue to be accepted (regression guard for the
// current 54.3 dB / cascade-trace baseline).
TEST(ConfigLoader, ValidatorAcceptsHierarchicalSortMode) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.sort_mode = splatting::HIERARCHICAL;
    EXPECT_NO_THROW(splatting::validate_vk_supported(s));
}

// PER_PIXEL_FULL / PER_PIXEL_KBUFFER are NOT implemented in VK and must be
// rejected.
TEST(ConfigLoader, ValidatorRejectsPerPixelFull) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.sort_mode = splatting::PER_PIXEL_FULL;
    EXPECT_THROW(splatting::validate_vk_supported(s), std::runtime_error);
}

TEST(ConfigLoader, ValidatorRejectsPerPixelKbuffer) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.sort_mode = splatting::PER_PIXEL_KBUFFER;
    EXPECT_THROW(splatting::validate_vk_supported(s), std::runtime_error);
}

TEST(ConfigLoader, ValidatorRejectsUnsupportedSortOrder) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.sort_order = splatting::VIEWSPACE_Z;
    EXPECT_THROW(splatting::validate_vk_supported(s), std::runtime_error);
}

TEST(ConfigLoader, ValidatorRejectsWrongQueueSizes) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.queue_sizes.per_pixel = 16;  // VK hard-codes 4
    EXPECT_THROW(splatting::validate_vk_supported(s), std::runtime_error);
}

TEST(ConfigLoader, ValidatorRejectsNewAabbFalse) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.new_aabb = false;
    EXPECT_THROW(splatting::validate_vk_supported(s), std::runtime_error);
}

TEST(ConfigLoader, ValidatorRejectsCullingFlagFalse) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.culling_settings.hierarchical_4x4_culling = false;
    EXPECT_THROW(splatting::validate_vk_supported(s), std::runtime_error);
}

TEST(ConfigLoader, ValidatorRejectsLoadBalancingFalse) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.load_balancing = false;
    EXPECT_THROW(splatting::validate_vk_supported(s), std::runtime_error);
}

TEST(ConfigLoader, ValidatorRejectsProperEwaFalse) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.proper_ewa_scaling = false;
    EXPECT_THROW(splatting::validate_vk_supported(s), std::runtime_error);
}

// eval_3D and near_clipping are runtime-honoured; toggling them must NOT
// trigger an assertion.
TEST(ConfigLoader, ValidatorAcceptsEval3DBothPolarities) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.eval_3D = false;
    EXPECT_NO_THROW(splatting::validate_vk_supported(s));
    s.eval_3D = true;
    EXPECT_NO_THROW(splatting::validate_vk_supported(s));
}

TEST(ConfigLoader, ValidatorAcceptsNearClippingBothPolarities) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.near_clipping = false;
    EXPECT_NO_THROW(splatting::validate_vk_supported(s));
    s.near_clipping = true;
    EXPECT_NO_THROW(splatting::validate_vk_supported(s));
}

// Constructor-level assertion: the defensive ctor must accept aaa.json.
TEST(ConfigLoader, RasterizerVulkanCtorAcceptsAaaJson) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device — skipping.";
    EXPECT_NO_THROW({
        RasterizerVulkan rast(ctx, s);
        (void)rast;
    });
}

// Constructor-level: Y1 — GLOBAL is now ACCEPTED (was rejected pre-Y1).
// The ctor builds a RasterizePass with spec_sort_mode=0 and the shader's
// HEAD_W=8 fallback path runs.
TEST(ConfigLoader, RasterizerVulkanCtorAcceptsGlobalSortMode) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.sort_mode = splatting::GLOBAL;

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device — skipping.";
    EXPECT_NO_THROW({
        RasterizerVulkan rast(ctx, s);
        (void)rast;
    });
}

// Constructor-level: PER_PIXEL_FULL must still be rejected (no VK path).
TEST(ConfigLoader, RasterizerVulkanCtorRejectsPerPixelFull) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.sort_mode = splatting::PER_PIXEL_FULL;

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device — skipping.";
    EXPECT_THROW({ RasterizerVulkan rast(ctx, s); }, std::runtime_error);
}

// Y1: smoke-test that a single rasterize() call on a sort_mode=GLOBAL
// rasterizer does not throw or crash. Uses the empty-scene fast path
// (binning.values_sorted == nullptr ⇒ host-side bg fill, no GPU dispatch),
// which still exercises the full ctor + spec-constant pipeline build.
TEST(ConfigLoader, RasterizerVulkanGlobalRenderEmptySceneNoThrow) {
    splatting::SplattingSettings s;
    splatting::load_from_json(aaa_json_path(), s);
    s.sort_settings.sort_mode = splatting::GLOBAL;

    VulkanContext ctx;
    if (!ctx.init()) GTEST_SKIP() << "No Vulkan compute device — skipping.";

    RasterizerVulkan rast(ctx, s);

    PreprocessOutput pre{};   // all nullptr
    BinningOutput   bin{};    // total_pairs=0, values_sorted=nullptr
    bin.total_pairs   = 0;
    bin.values_sorted = nullptr;
    bin.tile_ranges   = nullptr;
    bin.num_tiles     = 0;

    Camera cam{};
    cam.width  = 16;
    cam.height = 16;

    RenderConfig cfg{};
    cfg.bg_color[0] = 0.1f;
    cfg.bg_color[1] = 0.2f;
    cfg.bg_color[2] = 0.3f;
    cfg.tile_w = 16;
    cfg.tile_h = 16;

    const int HW = cam.width * cam.height;
    std::vector<float> out_image(static_cast<size_t>(3) * HW, -1.0f);

    EXPECT_NO_THROW({
        rast.rasterize(pre, bin, cam, cfg,
                       out_image.data(), /*output_depth=*/nullptr,
                       /*cache=*/nullptr, /*allocator=*/nullptr);
    });

    // Empty-scene fast path writes bg_color to every pixel — verify so we know
    // the code at least reached the host-side fast path.
    EXPECT_FLOAT_EQ(out_image[0],          0.1f);  // R plane[0]
    EXPECT_FLOAT_EQ(out_image[HW],         0.2f);  // G plane[0]
    EXPECT_FLOAT_EQ(out_image[2 * HW],     0.3f);  // B plane[0]
}
