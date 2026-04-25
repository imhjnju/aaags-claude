// splatting_settings.cpp — JSON loader + VK-support validator. See header
// for the policy. This is the only file that depends on nlohmann_json so the
// rest of gs3d_vk_core stays JSON-free.

#include "splatting_settings.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace splatting {

namespace {

const char* sort_mode_to_string(SortMode m) {
    switch (m) {
        case GLOBAL:            return "GLOBAL";
        case PER_PIXEL_FULL:    return "PER_PIXEL_FULL";
        case PER_PIXEL_KBUFFER: return "PER_PIXEL_KBUFFER";
        case HIERARCHICAL:      return "HIERARCHICAL";
    }
    return "<unknown>";
}

const char* sort_order_to_string(GlobalSortOrder o) {
    switch (o) {
        case VIEWSPACE_Z:           return "VIEWSPACE_Z";
        case DISTANCE:              return "DISTANCE";
        case PER_TILE_DEPTH_CENTER: return "PER_TILE_DEPTH_CENTER";
        case PER_TILE_DEPTH_MAXPOS: return "PER_TILE_DEPTH_MAXPOS";
    }
    return "<unknown>";
}

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("[splatting_settings] " + msg);
}

}  // namespace

void load_from_json(const std::string& path, SplattingSettings& out) {
    std::ifstream f(path);
    if (!f.is_open()) {
        fail("cannot open config file: " + path);
    }
    nlohmann::json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        fail(std::string("parse error in ") + path + ": " + e.what());
    }

    try {
        // sort_settings
        if (!j.contains("sort_settings")) fail("missing 'sort_settings'");
        const auto& js = j["sort_settings"];
        int sm = js.at("sort_mode").get<int>();
        int so = js.at("sort_order").get<int>();
        if (sm < 0 || sm > 3) fail("sort_mode out of range: " + std::to_string(sm));
        if (so < 0 || so > 3) fail("sort_order out of range: " + std::to_string(so));
        out.sort_settings.sort_mode  = static_cast<SortMode>(sm);
        out.sort_settings.sort_order = static_cast<GlobalSortOrder>(so);
        const auto& jq = js.at("queue_sizes");
        out.sort_settings.queue_sizes.tile_4x4  = jq.at("tile_4x4").get<int>();
        out.sort_settings.queue_sizes.tile_2x2  = jq.at("tile_2x2").get<int>();
        out.sort_settings.queue_sizes.per_pixel = jq.at("per_pixel").get<int>();

        // culling_settings
        if (!j.contains("culling_settings")) fail("missing 'culling_settings'");
        const auto& jc = j["culling_settings"];
        out.culling_settings.rect_bounding            = jc.at("rect_bounding").get<bool>();
        out.culling_settings.tight_opacity_bounding   = jc.at("tight_opacity_bounding").get<bool>();
        out.culling_settings.tile_based_culling       = jc.at("tile_based_culling").get<bool>();
        out.culling_settings.hierarchical_4x4_culling = jc.at("hierarchical_4x4_culling").get<bool>();

        // top-level flags
        out.load_balancing     = j.at("load_balancing").get<bool>();
        out.proper_ewa_scaling = j.at("proper_ewa_scaling").get<bool>();
        // eval_3D / near_clipping / new_aabb are optional in CUDA; default to
        // the same defaults as the CUDA struct.
        if (j.contains("eval_3D"))       out.eval_3D       = j["eval_3D"].get<bool>();
        if (j.contains("near_clipping")) out.near_clipping = j["near_clipping"].get<bool>();
        if (j.contains("new_aabb"))      out.new_aabb      = j["new_aabb"].get<bool>();
    } catch (const std::runtime_error&) {
        throw;  // already prefixed
    } catch (const std::exception& e) {
        fail(std::string("schema mismatch in ") + path + ": " + e.what());
    }
}

void validate_vk_supported(const SplattingSettings& s) {
    using namespace vk_hardcoded;

    // Y1: GLOBAL and HIERARCHICAL are both implemented (GLOBAL via the
    // HEAD_W=8 fallback path; HIERARCHICAL via the cascade). The shader
    // dispatches on spec_sort_mode + spec_trace_enabled.
    if (s.sort_settings.sort_mode != GLOBAL &&
        s.sort_settings.sort_mode != HIERARCHICAL) {
        std::ostringstream oss;
        oss << "VK implements sort_mode in {GLOBAL (=0), HIERARCHICAL (=3)}; "
            << "got " << sort_mode_to_string(s.sort_settings.sort_mode)
            << " (=" << static_cast<int>(s.sort_settings.sort_mode) << ")";
        fail(oss.str());
    }
    if (s.sort_settings.sort_order != SORT_ORDER) {
        std::ostringstream oss;
        oss << "VK only implements sort_order=" << sort_order_to_string(SORT_ORDER)
            << " (=" << static_cast<int>(SORT_ORDER) << "); got "
            << sort_order_to_string(s.sort_settings.sort_order)
            << " (=" << static_cast<int>(s.sort_settings.sort_order) << ")";
        fail(oss.str());
    }
    if (s.sort_settings.queue_sizes.per_pixel != QUEUE_SIZE_PER_PIXEL ||
        s.sort_settings.queue_sizes.tile_2x2  != QUEUE_SIZE_TILE_2X2  ||
        s.sort_settings.queue_sizes.tile_4x4  != QUEUE_SIZE_TILE_4X4) {
        std::ostringstream oss;
        oss << "VK queue_sizes are hard-coded to {per_pixel="
            << QUEUE_SIZE_PER_PIXEL << ", tile_2x2=" << QUEUE_SIZE_TILE_2X2
            << ", tile_4x4=" << QUEUE_SIZE_TILE_4X4 << "}; got {per_pixel="
            << s.sort_settings.queue_sizes.per_pixel
            << ", tile_2x2=" << s.sort_settings.queue_sizes.tile_2x2
            << ", tile_4x4=" << s.sort_settings.queue_sizes.tile_4x4 << "}";
        fail(oss.str());
    }

    if (!s.new_aabb) {
        fail("VK only implements new_aabb=true; the compute_aabb_screen path "
             "was never ported (see preprocess.comp:1030 comment).");
    }

    // The VK shaders apply rect_bounding / tight_opacity_bounding /
    // tile_based_culling / hierarchical_4x4_culling and load_balancing
    // unconditionally — there is no runtime gate. So if the JSON says
    // `false` we cannot honour it; refuse rather than render incorrectly.
    if (!s.culling_settings.rect_bounding) {
        fail("VK preprocess.comp applies rect_bounding unconditionally "
             "(per-axis AABB from cov2D diagonal). JSON says false, which "
             "would require a code path that does not exist.");
    }
    if (!s.culling_settings.tight_opacity_bounding) {
        fail("VK preprocess.comp applies tight_opacity_bounding "
             "unconditionally (extent = sqrt(2*log(opacity/ALPHA_THRESHOLD))). "
             "JSON says false, which would require a code path that does not exist.");
    }
    if (!s.culling_settings.tile_based_culling) {
        fail("VK scatter.comp applies tile_based_culling unconditionally. "
             "JSON says false, which would require a code path that does not exist.");
    }
    if (!s.culling_settings.hierarchical_4x4_culling) {
        fail("VK rasterize.comp's cascade always uses 4x4 sub-tile alpha-cull "
             "(max_contrib_gaussian_frustum_3D<3,3>). JSON says false, "
             "which would require a non-cascade rasterize path.");
    }
    if (!s.load_balancing) {
        fail("VK tile binner / scatter expand each Gaussian over its touched "
             "tiles equivalent to load_balancing=true. JSON says false, which "
             "would require a single-thread-per-Gaussian path that does not exist.");
    }

    // proper_ewa_scaling: the eval_3D path scales opacity by dilation_factor
    // unconditionally (preprocess.comp:961). The 2D path also applies the
    // convolution_scaling_factor unconditionally. Both correspond to
    // proper_ewa_scaling=true. JSON=false would silently render wrong.
    if (!s.proper_ewa_scaling) {
        fail("VK preprocess.comp applies proper_ewa_scaling unconditionally "
             "(opacity *= dilation_factor in eval_3D path; convolution "
             "scaling in 2D path). JSON says false, which would require a "
             "code path that does not exist.");
    }

    // eval_3D and near_clipping are runtime-honoured (eval_3D via spec
    // constant; near_clipping is gated in the 2D path). No assertion.
}

}  // namespace splatting
