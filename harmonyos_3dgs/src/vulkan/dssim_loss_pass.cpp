#include "vulkan/dssim_loss_pass.h"

#include "dssim_loss_grad_spv.h"
#include "dssim_loss_terms_spv.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint32_t kLocalSize = 256;
constexpr uint32_t kBindingCount = 7;
}

DssimLossPass::DssimLossPass(VulkanContext& ctx)
    : ctx_(ctx) {
    terms_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(dssim_loss_terms_spv),
        static_cast<std::size_t>(dssim_loss_terms_spv_len));
    grad_shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(dssim_loss_grad_spv),
        static_cast<std::size_t>(dssim_loss_grad_spv_len));

    std::vector<VkDescriptorType> binding_types(kBindingCount, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    terms_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *terms_shader_, binding_types,
        /*push_constant_bytes=*/sizeof(DssimLossPushConstants),
        /*max_descriptor_sets=*/4);
    grad_pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_, *grad_shader_, binding_types,
        /*push_constant_bytes=*/sizeof(DssimLossPushConstants),
        /*max_descriptor_sets=*/4);
    terms_descriptor_set_ = terms_pipeline_->allocate_empty_descriptor_set();
    grad_descriptor_set_ = grad_pipeline_->allocate_empty_descriptor_set();
}

void DssimLossPass::bind_buffers(VkBuffer rendered,
                                 VkBuffer target,
                                 VkBuffer dL_dpixels,
                                 VkBuffer partials,
                                 VkBuffer alpha,
                                 VkBuffer beta,
                                 VkBuffer gamma) {
    VulkanComputePipeline* pipelines[2] = {terms_pipeline_.get(), grad_pipeline_.get()};
    VkDescriptorSet sets[2] = {terms_descriptor_set_, grad_descriptor_set_};
    for (int i = 0; i < 2; ++i) {
        pipelines[i]->update_ssbo(sets[i], dssim_loss_bind::RENDERED, rendered);
        pipelines[i]->update_ssbo(sets[i], dssim_loss_bind::TARGET, target);
        pipelines[i]->update_ssbo(sets[i], dssim_loss_bind::DLDPIX, dL_dpixels);
        pipelines[i]->update_ssbo(sets[i], dssim_loss_bind::PARTIALS, partials);
        pipelines[i]->update_ssbo(sets[i], dssim_loss_bind::ALPHA, alpha);
        pipelines[i]->update_ssbo(sets[i], dssim_loss_bind::BETA, beta);
        pipelines[i]->update_ssbo(sets[i], dssim_loss_bind::GAMMA, gamma);
    }
}

void DssimLossPass::dispatch_sync(uint32_t W, uint32_t H, float lambda_dssim) {
    if (terms_descriptor_set_ == VK_NULL_HANDLE || grad_descriptor_set_ == VK_NULL_HANDLE) {
        throw std::runtime_error("DssimLossPass::dispatch_sync called before bind_buffers");
    }
    if (W == 0u || H == 0u) return;

    DssimLossPushConstants pc{};
    pc.W = W;
    pc.H = H;
    pc.num_elements = W * H * 3u;
    pc.lambda_dssim = lambda_dssim;

    const uint32_t groups = (pc.num_elements + kLocalSize - 1u) / kLocalSize;
    VkCommandBuffer cmd = ctx_.allocatePrimary();

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));

    terms_pipeline_->record(cmd, terms_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));
    insert_compute_barrier(cmd);
    grad_pipeline_->record(cmd, grad_descriptor_set_, groups, 1u, 1u, &pc, sizeof(pc));

    VK_CHECK(vkEndCommandBuffer(cmd));
    ctx_.submitAndWait(cmd);
    ctx_.freePrimary(cmd);
}
