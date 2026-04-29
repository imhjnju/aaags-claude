// vulkan_adam.cpp -- GPU Adam optimizer backed by adam_step.comp.
//
// One VulkanComputePipeline is shared across all parameter groups.
// Each group owns its descriptor set and UBO, allocated at add_group().
// This allows all 6 groups to be recorded into one command buffer without
// descriptor-set aliasing (step_group_record), reducing 6 submit+wait to 1.
//
// Binding layout (matches adam_step.comp):
//   0 STORAGE_BUFFER  params
//   1 STORAGE_BUFFER  grad  (read-only in shader, STORAGE_BUFFER on host)
//   2 STORAGE_BUFFER  m
//   3 STORAGE_BUFFER  v
//   4 UNIFORM_BUFFER  AdamStepUBO

#include "vulkan/vulkan_adam.h"

// xxd-embedded SPIR-V for adam_step.comp.
// Provides: unsigned char adam_step_spv[] and unsigned int adam_step_spv_len.
#include "adam_step_spv.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

VulkanAdam::VulkanAdam(VulkanContext& ctx,
                       float beta1,
                       float beta2,
                       float eps)
    : ctx_(ctx), beta1_(beta1), beta2_(beta2), eps_(eps)
{
    // 1. Load SPIR-V module from embedded bytes.
    shader_ = std::make_unique<VulkanShader>(
        ctx_,
        static_cast<const uint8_t*>(adam_step_spv),
        static_cast<std::size_t>(adam_step_spv_len));

    // 2. Build mixed-binding descriptor layout:
    //    bindings 0-3 = STORAGE_BUFFER, binding 4 = UNIFORM_BUFFER.
    std::vector<VkDescriptorType> binding_types = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // 0: params
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // 1: grad
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // 2: m
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,  // 3: v
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,  // 4: AdamStepUBO
    };

    pipeline_ = std::make_unique<VulkanComputePipeline>(
        ctx_,
        *shader_,
        binding_types,
        /*push_constant_bytes=*/0,
        /*max_descriptor_sets=*/8u);

    // 3. No shared descriptor set or UBO — each group allocates its own at add_group().
}

int VulkanAdam::add_group(uint32_t n, float lr)
{
    // Vulkan spec: VkBuffer size must be > 0.
    // When n==0 (e.g. sh_degree=0 → zero rest coefficients), allocate a 4-byte
    // placeholder. step_group() returns early for n==0 groups without dispatching.
    const std::size_t bytes     = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t alloc_sz  = std::max(bytes, static_cast<std::size_t>(4));

    Group g;
    g.n   = n;
    g.lr  = lr;

    g.m_buf = std::make_unique<VulkanBuffer>(
        ctx_, alloc_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    g.v_buf = std::make_unique<VulkanBuffer>(
        ctx_, alloc_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    g.ubo_buf = std::make_unique<VulkanBuffer>(
        ctx_, sizeof(AdamStepUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    g.descriptor_set = pipeline_->allocate_empty_descriptor_set();

    const uint32_t upload_n = std::max(n, 1u);
    std::vector<float> zeros(upload_n, 0.0f);
    g.m_buf->upload(zeros.data(), alloc_sz);
    g.v_buf->upload(zeros.data(), alloc_sz);

    int idx = static_cast<int>(groups_.size());
    groups_.push_back(std::move(g));
    return idx;
}

void VulkanAdam::reset_groups()
{
    groups_.clear();
    // All per-group descriptor sets are now invalid — reset the pool so the
    // next add_group() calls can reallocate from a fresh pool.
    pipeline_->reset_descriptor_pool();
}

void VulkanAdam::zero_moments()
{
    for (auto& g : groups_) {
        const size_t n_bytes = static_cast<size_t>(g.n) * sizeof(float);
        const std::vector<float> zeros(g.n, 0.0f);
        g.m_buf->upload(zeros.data(), n_bytes);
        g.v_buf->upload(zeros.data(), n_bytes);
    }
}

void VulkanAdam::extend_group(int group_idx, uint32_t added_floats)
{
    if (added_floats == 0) return;
    auto& g = groups_.at(static_cast<std::size_t>(group_idx));
    const uint32_t old_n = g.n;
    const uint32_t new_n = old_n + added_floats;

    std::vector<float> m_cpu(new_n, 0.0f);
    std::vector<float> v_cpu(new_n, 0.0f);
    if (old_n > 0) {
        g.m_buf->download(m_cpu.data(), static_cast<std::size_t>(old_n) * sizeof(float));
        g.v_buf->download(v_cpu.data(), static_cast<std::size_t>(old_n) * sizeof(float));
    }

    const std::size_t new_bytes = static_cast<std::size_t>(new_n) * sizeof(float);
    g.m_buf = std::make_unique<VulkanBuffer>(ctx_, new_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    g.v_buf = std::make_unique<VulkanBuffer>(ctx_, new_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    g.m_buf->upload(m_cpu.data(), new_bytes);
    g.v_buf->upload(v_cpu.data(), new_bytes);
    g.n = new_n;
}

void VulkanAdam::shrink_group(int group_idx, uint32_t new_n)
{
    auto& g = groups_.at(static_cast<std::size_t>(group_idx));
    if (new_n == g.n) return;

    const std::size_t keep_bytes = static_cast<std::size_t>(new_n) * sizeof(float);
    std::vector<float> m_cpu(new_n, 0.0f);
    std::vector<float> v_cpu(new_n, 0.0f);
    if (new_n > 0 && g.n > 0) {
        g.m_buf->download(m_cpu.data(), keep_bytes);
        g.v_buf->download(v_cpu.data(), keep_bytes);
    }

    const std::size_t alloc_bytes = std::max(keep_bytes, static_cast<std::size_t>(4));
    g.m_buf = std::make_unique<VulkanBuffer>(ctx_, alloc_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    g.v_buf = std::make_unique<VulkanBuffer>(ctx_, alloc_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (new_n > 0) {
        g.m_buf->upload(m_cpu.data(), keep_bytes);
        g.v_buf->upload(v_cpu.data(), keep_bytes);
    }
    g.n = new_n;
}

void VulkanAdam::zero_moment_floats(int group_idx,
                                    const std::vector<uint32_t>& float_indices)
{
    if (float_indices.empty()) return;
    auto& g = groups_.at(static_cast<std::size_t>(group_idx));
    if (g.n == 0) return;

    const std::size_t total_bytes = static_cast<std::size_t>(g.n) * sizeof(float);
    std::vector<float> m_cpu(g.n, 0.0f);
    std::vector<float> v_cpu(g.n, 0.0f);
    g.m_buf->download(m_cpu.data(), total_bytes);
    g.v_buf->download(v_cpu.data(), total_bytes);

    for (uint32_t fi : float_indices) {
        if (fi < g.n) {
            m_cpu[fi] = 0.0f;
            v_cpu[fi] = 0.0f;
        }
    }

    g.m_buf->upload(m_cpu.data(), total_bytes);
    g.v_buf->upload(v_cpu.data(), total_bytes);
}

void VulkanAdam::download_moments(int idx, std::vector<float>& out_m, std::vector<float>& out_v) const
{
    download_group_moments(idx, out_m, out_v);
}

void VulkanAdam::download_group_moments(int group_idx,
                                        std::vector<float>& m_out,
                                        std::vector<float>& v_out) const
{
    const Group& grp = groups_.at(static_cast<std::size_t>(group_idx));
    m_out.resize(grp.n);
    v_out.resize(grp.n);
    if (grp.n == 0) return;

    const size_t n_bytes = static_cast<size_t>(grp.n) * sizeof(float);
    grp.m_buf->download(m_out.data(), n_bytes);
    grp.v_buf->download(v_out.data(), n_bytes);
}

void VulkanAdam::step_group(int idx,
                             VkBuffer params_buf,
                             VkBuffer grad_buf,
                             float    lr,
                             uint32_t step)
{
    Group& grp = groups_.at(static_cast<std::size_t>(idx));

    // Use group lr if caller passes negative sentinel.
    float effective_lr = (lr < 0.0f) ? grp.lr : lr;

    // Element count from grp.n — guards against caller overrunning m/v buffers.
    const uint32_t n = grp.n;
    if (n == 0) return;   // no-op: zero-element group (e.g. sh_degree=0 rest coeffs)

    // 1. Fill and upload the per-group UBO.
    AdamStepUBO ubo;
    ubo.beta1 = beta1_;
    ubo.beta2 = beta2_;
    ubo.eps   = eps_;
    ubo.lr    = effective_lr;
    ubo.n     = n;
    ubo.step  = step;
    ubo._pad0 = 0.f;
    ubo._pad1 = 0.f;
    grp.ubo_buf->upload(&ubo, sizeof(ubo));

    // 2. Update all bindings on the per-group descriptor set.
    VkDescriptorSet ds = grp.descriptor_set;
    pipeline_->update_ssbo(ds, 0, params_buf);
    pipeline_->update_ssbo(ds, 1, grad_buf);
    pipeline_->update_ssbo(ds, 2, grp.m_buf->handle());
    pipeline_->update_ssbo(ds, 3, grp.v_buf->handle());
    pipeline_->update_ubo (ds, 4, grp.ubo_buf->handle(), sizeof(AdamStepUBO));

    // 3. Dispatch and wait.
    const uint32_t gx = (n + 255u) / 256u;
    pipeline_->dispatch_sync(ds, gx, 1, 1);
}

void VulkanAdam::step_group_record(VkCommandBuffer cmd,
                                    int idx, VkBuffer params_buf, VkBuffer grad_buf,
                                    float lr, uint32_t step)
{
    Group& grp = groups_.at(static_cast<std::size_t>(idx));
    const float effective_lr = (lr < 0.0f) ? grp.lr : lr;
    const uint32_t n = grp.n;
    if (n == 0) return;

    AdamStepUBO ubo;
    ubo.beta1 = beta1_; ubo.beta2 = beta2_; ubo.eps = eps_;
    ubo.lr    = effective_lr; ubo.n = n; ubo.step = step;
    ubo._pad0 = 0.f; ubo._pad1 = 0.f;
    grp.ubo_buf->upload(&ubo, sizeof(ubo));

    VkDescriptorSet ds = grp.descriptor_set;
    pipeline_->update_ssbo(ds, 0, params_buf);
    pipeline_->update_ssbo(ds, 1, grad_buf);
    pipeline_->update_ssbo(ds, 2, grp.m_buf->handle());
    pipeline_->update_ssbo(ds, 3, grp.v_buf->handle());
    pipeline_->update_ubo (ds, 4, grp.ubo_buf->handle(), sizeof(AdamStepUBO));

    const uint32_t gx = (n + 255u) / 256u;
    pipeline_->record(cmd, ds, gx, 1, 1);
}
