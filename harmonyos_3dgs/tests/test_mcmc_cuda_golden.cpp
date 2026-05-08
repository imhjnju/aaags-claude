#include "mcmc_densification.h"
#include "golden/compare.h"
#include "golden/npy_reader.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string fixture_root() {
    return std::string(TEST_DATA_DIR) + "/../golden/mcmc_cuda";
}

void require_shape(const NpyArray& a,
                   const std::vector<size_t>& expected,
                   const std::string& path) {
    if (a.shape != expected)
        throw std::runtime_error("fixture shape mismatch: " + path);
}

std::vector<float> copy_f32(const NpyArray& a, const std::string& path) {
    if (a.dtype != NpyDtype::float32)
        throw std::runtime_error("expected float32 fixture: " + path);
    std::vector<float> out(a.numel());
    std::memcpy(out.data(), a.raw.data(), a.raw.size());
    return out;
}

std::vector<float> load_f32_shape(const std::string& path,
                                  const std::vector<size_t>& expected_shape) {
    NpyArray a = load_npy(path);
    require_shape(a, expected_shape, path);
    return copy_f32(a, path);
}

std::vector<int> load_i32(const std::string& path) {
    NpyArray a = load_npy(path);
    if (a.dtype != NpyDtype::int32)
        throw std::runtime_error("expected int32 fixture: " + path);
    if (a.shape.size() != 1u)
        throw std::runtime_error("expected rank-1 int32 fixture: " + path);
    std::vector<int> out(a.numel());
    const int32_t* p = a.i32();
    for (size_t i = 0; i < a.numel(); ++i) out[i] = static_cast<int>(p[i]);
    return out;
}

struct McmcGoldenCase {
    std::string dir;
    int N = 0;
    int max_coeffs = 0;
    OwnedRawParams params;
    mcmc::DensifySamplePlan plan;
    std::vector<float> expected_positions;
    std::vector<float> expected_scales;
    std::vector<float> expected_rotations;
    std::vector<float> expected_opacities;
    std::vector<float> expected_sh;
    std::vector<int> expected_modified_sources;
    std::vector<int> expected_replaced_dsts;
};

McmcGoldenCase load_case(const std::string& name) {
    McmcGoldenCase c;
    c.dir = fixture_root() + "/" + name;

    NpyArray pos = load_npy(c.dir + "/input_raw_positions.npy");
    NpyArray sh = load_npy(c.dir + "/input_raw_sh.npy");
    if (pos.dtype != NpyDtype::float32 || sh.dtype != NpyDtype::float32)
        throw std::runtime_error("expected float32 input position/SH fixtures");
    if (pos.shape.size() != 2u || sh.shape.size() != 2u)
        throw std::runtime_error("expected rank-2 input position/SH fixtures");
    if (pos.shape[1] != 3u || sh.shape[0] != pos.shape[0] || sh.shape[1] % 3u != 0u)
        throw std::runtime_error("invalid input position/SH fixture dimensions");
    c.N = static_cast<int>(pos.shape[0]);
    c.max_coeffs = static_cast<int>(sh.shape[1] / 3);
    const size_t N = static_cast<size_t>(c.N);
    const size_t mc3 = static_cast<size_t>(c.max_coeffs) * 3;

    c.params.sh_degree = 0;
    c.params.max_coeffs = c.max_coeffs;
    c.params.positions = load_f32_shape(c.dir + "/input_raw_positions.npy", {N, 3});
    c.params.scales = load_f32_shape(c.dir + "/input_raw_scales.npy", {N, 3});
    c.params.rotations = load_f32_shape(c.dir + "/input_raw_rotations.npy", {N, 4});
    c.params.opacities = load_f32_shape(c.dir + "/input_raw_opacities.npy", {N});
    c.params.sh_coeffs = load_f32_shape(c.dir + "/input_raw_sh.npy", {N, mc3});

    c.plan.relocate_sources = load_i32(c.dir + "/relocate_src_indices.npy");
    c.plan.add_sources = load_i32(c.dir + "/add_src_indices.npy");

    NpyArray expected_opacity = load_npy(c.dir + "/expected_raw_opacities.npy");
    if (expected_opacity.dtype != NpyDtype::float32 || expected_opacity.shape.size() != 1u)
        throw std::runtime_error("expected rank-1 float32 expected opacity fixture");
    const size_t final_N = expected_opacity.shape[0];
    c.expected_opacities = copy_f32(expected_opacity, c.dir + "/expected_raw_opacities.npy");
    c.expected_positions = load_f32_shape(c.dir + "/expected_raw_positions.npy", {final_N, 3});
    c.expected_scales = load_f32_shape(c.dir + "/expected_raw_scales.npy", {final_N, 3});
    c.expected_rotations = load_f32_shape(c.dir + "/expected_raw_rotations.npy", {final_N, 4});
    c.expected_sh = load_f32_shape(c.dir + "/expected_raw_sh.npy", {final_N, mc3});
    c.expected_modified_sources = load_i32(c.dir + "/expected_modified_sources.npy");
    c.expected_replaced_dsts = load_i32(c.dir + "/expected_replaced_dsts.npy");
    return c;
}

void expect_vec_eq(const std::vector<int>& got,
                   const std::vector<int>& expected,
                   const char* label) {
    ASSERT_EQ(got.size(), expected.size()) << label;
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(got[i], expected[i]) << label << " i=" << i;
    }
}

void expect_f32_close(const std::vector<float>& got,
                      const std::vector<float>& expected,
                      float abs_tol,
                      float rel_tol,
                      const char* label) {
    CompareResult r = compare_f32(got, expected, abs_tol, rel_tol);
    EXPECT_TRUE(r.passed)
        << label << " max_abs=" << r.max_abs_err
        << " max_rel=" << r.max_rel_err
        << " first_bad=" << r.first_bad_index
        << " num_bad=" << r.num_bad;
}

void run_case(const std::string& name, int cap_max) {
    McmcGoldenCase c = load_case(name);
    mcmc::DensifyResult result = mcmc::densify_with_samples(c.params, 0.005f, cap_max, c.plan);

    EXPECT_EQ(result.final_count, static_cast<int>(c.expected_opacities.size()));
    expect_vec_eq(result.modified_source_indices, c.expected_modified_sources, "modified_sources");
    expect_vec_eq(result.replaced_destination_indices, c.expected_replaced_dsts, "replaced_dsts");

    expect_f32_close(c.params.positions, c.expected_positions, 1e-6f, 1e-6f, "positions");
    expect_f32_close(c.params.rotations, c.expected_rotations, 1e-6f, 1e-6f, "rotations");
    expect_f32_close(c.params.sh_coeffs, c.expected_sh, 1e-6f, 1e-6f, "sh");
    expect_f32_close(c.params.opacities, c.expected_opacities, 2e-5f, 2e-5f, "opacities");
    expect_f32_close(c.params.scales, c.expected_scales, 2e-5f, 2e-5f, "scales");
}

}  // namespace

TEST(McmcCudaGolden, RelocateRepeatedSourcesMatchesReference) {
    run_case("relocate_repeated_sources", 12);
}

TEST(McmcCudaGolden, AddOnlyFivePercentMatchesReference) {
    run_case("add_only_5_percent", 200);
}

TEST(McmcCudaGolden, CombinedRelocateAddMatchesReference) {
    run_case("combined_relocate_add", 80);
}

TEST(McmcCudaGolden, NoopEdgesMatchReference) {
    run_case("noop_edges", 10);
}
