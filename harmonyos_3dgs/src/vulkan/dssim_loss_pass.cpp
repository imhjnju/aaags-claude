#include "vulkan/dssim_loss_pass.h"

#include "dssim_target_blur5_h_spv.h"
#include "dssim_target_blur5_v_spv.h"
#include "dssim_blur5_h_spv.h"
#include "dssim_blur5_v_terms_spv.h"
#include "dssim_transpose3_h_grad_spv.h"
#include "dssim_transpose3_v_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kLocalSize = 256;

void update_ssbos(VulkanComputePipeline& pipeline, VkDescriptorSet set, const VkBuffer* buffers, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) {
        pipeline.update_ssbo(set, i, buffers[i]);
    }
}
}

DssimLossPass::DssimLossPass(VulkanContext& ctx)
    : ctx_(ctx) {
    target_blur5_h_shader_ = std::make_unique<VulkanShader>(
        ctx_, static_cast<const uint8_t*>(dssim_target_blur5_h_spv),
        static_cast<std::size_t>(dssim_target_blur5_h_spv_len));
    target_blur5_v_shader_ = std::make_unique<VulkanShader>(
        ctx_, static_cast<const uint8_t*>(dssim_target_blur5_v_spv),
        static_cast<std::size_t>(dssim_target_blur5_v_spv_len));
    blur5_h_shader_ = std::make_unique<VulkanShader>(
        ctx_, static_cast<const uint8_t*>(dssim_blur5_h_spv),
        static_cast<std::size_t>(dssim_blur5_h_spv_len));
    blur5_v_terms_shader_ = std::make_unique<VulkanShader>(
        ctx_, static_cast<const uint8_t*>(dssim_blur5_v_terms_spv),
        static_cast<std::size_t>(dssim_blur5_v_terms_spv_len));
    transpose3_v_shader_ = std::make_unique<VulkanShader>(
        ctx_, static_cast<const uint8_t*>(dssim_transpose3_v_spv),
        static_cast<std::size_t>(dssim_transpose3_v_spv_len));
    transpose3_h_grad_shader_ = std::make_unique<VulkanShader>(
        ctx_, static_cast<const uint8_t*>(dssim_transpose3_h_grad_spv),
        static_cast<std::size_t>(dssim_transpose3_h_grad_spv_len));

    std::vector<VkDescriptorType> target_blur5_h_types(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    std::vector<VkDescriptorType> target_blur5_v_types(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    std::vector<VkDescriptorType> blur5_h_types(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    std::vector<VkDescriptorType> blur5_v_terms_types(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    std::vector<VkDescriptorType> transpose3_v_types(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    std::vector<VkDescriptorType> transpose3_h_grad_types(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    target_blur5_h_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *target_blur5_h_shader_, target_blur5_h_types, sizeof(DssimLossPushConstants), 1);
    target_blur5_v_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *target_blur5_v_shader_, target_blur5_v_types, sizeof(DssimLossPushConstants), 1);
    blur5_h_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *blur5_h_shader_, blur5_h_types, sizeof(DssimLossPushConstants), 1);
    blur5_v_terms_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *blur5_v_terms_shader_, blur5_v_terms_types, sizeof(DssimLossPushConstants), 1);
    transpose3_v_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *transpose3_v_shader_, transpose3_v_types, sizeof(DssimLossPushConstants), 1);
    transpose3_h_grad_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *transpose3_h_grad_shader_, transpose3_h_grad_types, sizeof(DssimLossPushConstants), 1);

    target_blur5_h_descriptor_set_ = target_blur5_h_pipeline_->allocate_empty_descriptor_set();
    target_blur5_v_descriptor_set_ = target_blur5_v_pipeline_->allocate_empty_descriptor_set();
    blur5_h_descriptor_set_ = blur5_h_pipeline_->allocate_empty_descriptor_set();
    blur5_v_terms_descriptor_set_ = blur5_v_terms_pipeline_->allocate_empty_descriptor_set();
    transpose3_v_descriptor_set_ = transpose3_v_pipeline_->allocate_empty_descriptor_set();
    transpose3_h_grad_descriptor_set_ = transpose3_h_grad_pipeline_->allocate_empty_descriptor_set();
    cmd_ = ctx_.allocatePrimary();
}

DssimLossPass::~DssimLossPass() {
    ctx_.freePrimary(cmd_);
}

void DssimLossPass::bind_buffers(VkBuffer rendered,
                                 VkBuffer target,
                                 VkBuffer dL_dpixels,
                                 VkBuffer partials,
                                 VkBuffer alpha,
                                 VkBuffer beta,
                                 VkBuffer gamma,
                                 VkBuffer x2,
                                 VkBuffer y2,
                                 VkBuffer xy,
                                 VkBuffer scratch,
                                 VkBuffer mu1) {
    bound_target_ = target;
    bound_target_x2_ = x2;
    bound_target_y2_ = y2;
    bound_target_scratch_ = scratch;
    bound_target_mu1_ = mu1;
    target_terms_valid_ = false;
    target_terms_W_ = 0;
    target_terms_H_ = 0;

    const VkBuffer target_blur5_h_buffers[3] = {target, x2, y2};
    update_ssbos(*target_blur5_h_pipeline_, target_blur5_h_descriptor_set_, target_blur5_h_buffers, 3);

    const VkBuffer target_blur5_v_buffers[4] = {x2, y2, scratch, mu1};
    update_ssbos(*target_blur5_v_pipeline_, target_blur5_v_descriptor_set_, target_blur5_v_buffers, 4);

    const VkBuffer blur5_h_buffers[5] = {rendered, target, x2, y2, xy};
    update_ssbos(*blur5_h_pipeline_, blur5_h_descriptor_set_, blur5_h_buffers, 5);

    const VkBuffer blur5_v_terms_buffers[12] = {x2, scratch, y2, mu1, xy, rendered,
                                                target, dL_dpixels, partials, alpha, beta, gamma};
    update_ssbos(*blur5_v_terms_pipeline_, blur5_v_terms_descriptor_set_, blur5_v_terms_buffers, 12);

    const VkBuffer transpose3_v_buffers[6] = {alpha, beta, gamma, x2, y2, xy};
    update_ssbos(*transpose3_v_pipeline_, transpose3_v_descriptor_set_, transpose3_v_buffers, 6);

    const VkBuffer transpose3_h_grad_buffers[6] = {x2, y2, xy, rendered, target, dL_dpixels};
    update_ssbos(*transpose3_h_grad_pipeline_, transpose3_h_grad_descriptor_set_, transpose3_h_grad_buffers, 6);
}

void DssimLossPass::precompute_target_terms(uint32_t W, uint32_t H, float lambda_dssim) {
    if (target_blur5_h_descriptor_set_ == VK_NULL_HANDLE || target_blur5_v_descriptor_set_ == VK_NULL_HANDLE ||
        bound_target_ == VK_NULL_HANDLE) {
        throw std::runtime_error("DssimLossPass::precompute_target_terms called before bind_buffers");
    }
    target_terms_valid_ = false;
    target_terms_W_ = 0;
    target_terms_H_ = 0;
    if (W == 0u || H == 0u) return;

    DssimLossPushConstants pc{};
    pc.W = W;
    pc.H = H;
    pc.num_elements = W * H * 3u;
    pc.lambda_dssim = lambda_dssim;

    const uint32_t groups = (pc.num_elements + kLocalSize - 1u) / kLocalSize;
    VK_CHECK(vkResetCommandBuffer(cmd_, 0));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_, &begin));

    target_blur5_h_pipeline_->record(cmd_, target_blur5_h_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
    insert_compute_barrier(cmd_);
    target_blur5_v_pipeline_->record(cmd_, target_blur5_v_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));

    VK_CHECK(vkEndCommandBuffer(cmd_));
    ctx_.submitAndWait(cmd_);
    target_terms_valid_ = true;
    target_terms_W_ = W;
    target_terms_H_ = H;
}

void DssimLossPass::dispatch_sync(uint32_t W, uint32_t H, float lambda_dssim) {
    if (blur5_h_descriptor_set_ == VK_NULL_HANDLE || blur5_v_terms_descriptor_set_ == VK_NULL_HANDLE ||
        transpose3_v_descriptor_set_ == VK_NULL_HANDLE || transpose3_h_grad_descriptor_set_ == VK_NULL_HANDLE ||
        bound_target_ == VK_NULL_HANDLE) {
        throw std::runtime_error("DssimLossPass::dispatch_sync called before bind_buffers");
    }
    if (W == 0u || H == 0u) return;
    if (!target_terms_valid_ || target_terms_W_ != W || target_terms_H_ != H) {
        throw std::runtime_error("DssimLossPass::dispatch_sync called before matching precompute_target_terms");
    }

    DssimLossPushConstants pc{};
    pc.W = W;
    pc.H = H;
    pc.num_elements = W * H * 3u;
    pc.lambda_dssim = lambda_dssim;

    const uint32_t groups = (pc.num_elements + kLocalSize - 1u) / kLocalSize;
    VK_CHECK(vkResetCommandBuffer(cmd_, 0));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_, &begin));

    blur5_h_pipeline_->record(cmd_, blur5_h_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
    insert_compute_barrier(cmd_);
    blur5_v_terms_pipeline_->record(cmd_, blur5_v_terms_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
    insert_compute_barrier(cmd_);
    transpose3_v_pipeline_->record(cmd_, transpose3_v_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
    insert_compute_barrier(cmd_);
    transpose3_h_grad_pipeline_->record(cmd_, transpose3_h_grad_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));

    VK_CHECK(vkEndCommandBuffer(cmd_));
    ctx_.submitAndWait(cmd_);
}
