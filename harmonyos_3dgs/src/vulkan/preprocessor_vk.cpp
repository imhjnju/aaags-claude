#include "vulkan/preprocessor_vk.h"

#include "vulkan/vk_buffer.h"

#include <cstring>
#include <stdexcept>

namespace {

// Matches the std430 Camera block in preprocess.comp.
struct CameraSSBO {
    float view_matrix[16];
    float viewproj_matrix[16];
    float cam_pos[3];
    float _pad_cam_pos;
    float tan_fovx;
    float tan_fovy;
    float width;
    float height;
};
static_assert(sizeof(CameraSSBO) == 160,
              "CameraSSBO must be 160 bytes to match GLSL std430");

// Matches layout(push_constant) in preprocess.comp.
struct PushConstants {
    uint32_t count;
    float    scale_modifier;
};

// Bindings: keep in sync with preprocess.comp.
//   0: positions (in)
//   1: camera    (in)
//   2: depths    (out)
//   3: active    (out)
//   4: means2D   (out)
//   5: p_hom_w   (out)
//   6: scales    (in)
//   7: rotations (in)
//   8: opacities (in)
//   9: conics    (out)
//  10: opacities_2d (out)
constexpr uint32_t kNumSSBO = 11;

}  // namespace

PreprocessorVK::PreprocessorVK(VulkanContext& ctx, const std::string& shader_dir)
    : ctx_(ctx) {
    shader_ = std::make_unique<VulkanShader>(ctx_, shader_dir + "/preprocess.spv");
    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *shader_,
        /*num_ssbo_bindings=*/kNumSSBO,
        /*push_constant_bytes=*/sizeof(PushConstants));
}

PreprocessorVK::~PreprocessorVK() = default;

PreprocessorVK::Output
PreprocessorVK::process(const GaussianData& g, const Camera& cam,
                        const RenderConfig& cfg) {
    Output out;
    if (g.count <= 0) return out;
    if (!g.positions || !g.scales || !g.rotations || !g.opacities)
        throw std::runtime_error(
            "PreprocessorVK::process: positions/scales/rotations/opacities "
            "must all be non-null");

    const int N = g.count;
    const VkDeviceSize f3  = VkDeviceSize(N) * 3 * sizeof(float);
    const VkDeviceSize f4  = VkDeviceSize(N) * 4 * sizeof(float);
    const VkDeviceSize f1  = VkDeviceSize(N) * sizeof(float);
    const VkDeviceSize u1  = VkDeviceSize(N) * sizeof(uint32_t);
    const VkDeviceSize f2  = VkDeviceSize(N) * 2 * sizeof(float);
    const VkDeviceSize c3  = VkDeviceSize(N) * 3 * sizeof(float);

    VulkanBuffer pos_buf  (ctx_, f3);
    VulkanBuffer cam_buf  (ctx_, sizeof(CameraSSBO));
    VulkanBuffer depth_buf(ctx_, f1);
    VulkanBuffer active_buf(ctx_, u1);
    VulkanBuffer means2D_buf(ctx_, f2);
    VulkanBuffer phomw_buf (ctx_, f1);
    VulkanBuffer scale_buf (ctx_, f3);
    VulkanBuffer rot_buf   (ctx_, f4);
    VulkanBuffer opac_buf  (ctx_, f1);
    VulkanBuffer conic_buf (ctx_, c3);
    VulkanBuffer opac2d_buf(ctx_, f1);

    // Pack Camera.
    CameraSSBO cssbo{};
    std::memcpy(cssbo.view_matrix,     cam.view_matrix,     sizeof(cssbo.view_matrix));
    std::memcpy(cssbo.viewproj_matrix, cam.viewproj_matrix, sizeof(cssbo.viewproj_matrix));
    cssbo.cam_pos[0] = cam.cam_pos[0];
    cssbo.cam_pos[1] = cam.cam_pos[1];
    cssbo.cam_pos[2] = cam.cam_pos[2];
    cssbo.tan_fovx   = cam.tan_fovx;
    cssbo.tan_fovy   = cam.tan_fovy;
    cssbo.width      = static_cast<float>(cam.width);
    cssbo.height     = static_cast<float>(cam.height);

    pos_buf .upload(g.positions, f3);
    cam_buf .upload(&cssbo,      sizeof(cssbo));
    scale_buf.upload(g.scales,   f3);
    rot_buf  .upload(g.rotations, f4);
    opac_buf .upload(g.opacities, f1);

    VkDescriptorSet dset = pipeline_->allocateDescriptorSet({
        pos_buf   .handle(),  // 0
        cam_buf   .handle(),  // 1
        depth_buf .handle(),  // 2
        active_buf.handle(),  // 3
        means2D_buf.handle(), // 4
        phomw_buf .handle(),  // 5
        scale_buf .handle(),  // 6
        rot_buf   .handle(),  // 7
        opac_buf  .handle(),  // 8
        conic_buf .handle(),  // 9
        opac2d_buf.handle(),  // 10
    });

    PushConstants pc{};
    pc.count          = static_cast<uint32_t>(N);
    pc.scale_modifier = cfg.scale_modifier;

    VkCommandBuffer cmd = ctx_.allocatePrimary();
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_->handle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_->layout(), 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cmd, pipeline_->layout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (uint32_t(N) + 63u) / 64u, 1, 1);
    VK_CHECK(vkEndCommandBuffer(cmd));
    ctx_.submitAndWait(cmd);
    ctx_.freePrimary(cmd);

    out.depths      .resize(N);
    out.active      .resize(N);
    out.means2D     .resize(N * 2);
    out.p_hom_w     .resize(N);
    out.conics      .resize(N * 3);
    out.opacities_2d.resize(N);
    depth_buf  .download(out.depths      .data(), f1);
    active_buf .download(out.active      .data(), u1);
    means2D_buf.download(out.means2D     .data(), f2);
    phomw_buf  .download(out.p_hom_w     .data(), f1);
    conic_buf  .download(out.conics      .data(), c3);
    opac2d_buf .download(out.opacities_2d.data(), f1);
    return out;
}
