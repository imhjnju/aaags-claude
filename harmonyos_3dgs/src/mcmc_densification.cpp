// mcmc_densification.cpp — MCMC relocate / add for AAA-Gaussians port.
//
// Mirrors AAA-Gaussians/scene/gaussian_model.py:
//   _sample_alives(probs, num, alive_indices?)
//   _update_params(idxs, ratio)
//   relocate_gs(camera_center, dead_mask)
//   add_new_gs(cap_max)
// invoked from train.py:130-134.
//
// All randomness is driven by std::mt19937 seeded from caller-provided
// rng_seed → results are a pure function of inputs (deterministic).

#include "mcmc_densification.h"
#include "relocation.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace mcmc {

namespace {

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float inverse_sigmoid(float p) {
    // logit. Caller must ensure p ∈ (0, 1).
    return std::log(p / (1.0f - p));
}

// Multinomial-with-replacement sampling matching torch.multinomial(replacement=True).
// candidate_indices: pool to sample from (global indices). probs: weights aligned
// with candidate_indices. Returns out_global_idx[k] = candidate_indices[draw_k].
//
// Uses std::discrete_distribution<int> seeded with rng_seed. Deterministic for a
// given standard-library implementation, but not bit-exact with torch.multinomial.
void sample_with_replacement(
    const std::vector<int>& candidate_indices,
    const std::vector<float>& probs,
    int num_samples,
    uint32_t rng_seed,
    std::vector<int>& out_global_idx)
{
    out_global_idx.clear();
    if (num_samples <= 0 || candidate_indices.empty()) return;

    out_global_idx.reserve(static_cast<size_t>(num_samples));
    std::mt19937 rng(rng_seed);
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    for (int s = 0; s < num_samples; ++s) {
        const int local = dist(rng);
        out_global_idx.push_back(candidate_indices[local]);
    }
}

// _update_params for one index. Reads activation-space (opacity, scale) at
// snapshot time, calls compute_relocation with N = ratio + 1, then converts
// the new opacity/scale back to raw (logit/log) storage.
//
// new_opacity is clamped to [0.005, 1 - eps] before inverse_sigmoid (matches
// Python torch.clamp behavior).
struct UpdatedParams {
    float raw_opacity;   // logit(clamped(new_opacity))
    float raw_scale[3];  // log(new_scale_xyz)
};

UpdatedParams update_params_one(float op_old_act, const float sc_old_act[3], int ratio) {
    const int N = ratio + 1;  // ratio counts duplicates; total clones = ratio+1.

    float op_new_act = 0.0f;
    float sc_new_act[3] = {0.0f, 0.0f, 0.0f};
    compute_relocation(op_old_act, sc_old_act, N, op_new_act, sc_new_act);

    // Clamp opacity to [0.005, 1 - epsilon] before inverse_sigmoid.
    const float eps = std::numeric_limits<float>::epsilon();
    const float lo  = 0.005f;
    const float hi  = 1.0f - eps;
    if (op_new_act < lo) op_new_act = lo;
    if (op_new_act > hi) op_new_act = hi;

    UpdatedParams u;
    u.raw_opacity   = inverse_sigmoid(op_new_act);
    // exp(raw) = activation; raw = log(activation). Guard against log(0).
    for (int j = 0; j < 3; ++j) {
        const float s = sc_new_act[j] > 0.0f ? sc_new_act[j] : std::numeric_limits<float>::min();
        u.raw_scale[j] = std::log(s);
    }
    return u;
}

}  // namespace

// ---------------------------------------------------------------------------
// Internal _ex variants that expose modified source indices
// ---------------------------------------------------------------------------

struct RelocateResult {
    int relocated;
    std::vector<int> source_indices;
    std::vector<int> destination_indices;
};

static RelocateResult relocate_gs_ex(OwnedRawParams& params,
                                      float opacity_thresh,
                                      uint32_t rng_seed,
                                      const std::vector<int>* replay_sources = nullptr)
{
    const int N = params.count();
    if (N == 0) {
        if (replay_sources && !replay_sources->empty())
            throw std::invalid_argument("relocate replay sources provided for empty params");
        return {0, {}, {}};
    }

    std::vector<int> dead_indices;
    std::vector<int> alive_indices;
    std::vector<bool> is_alive(static_cast<size_t>(N), false);
    dead_indices.reserve(static_cast<size_t>(N));
    alive_indices.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        if (sigmoid(params.opacities[i]) <= opacity_thresh) {
            dead_indices.push_back(i);
        } else {
            alive_indices.push_back(i);
            is_alive[static_cast<size_t>(i)] = true;
        }
    }

    const int num_dead  = static_cast<int>(dead_indices.size());
    const int num_alive = static_cast<int>(alive_indices.size());
    if (num_dead == 0 || num_alive == 0) {
        if (replay_sources && !replay_sources->empty())
            throw std::invalid_argument("relocate replay sources do not match no-op relocation");
        return {0, {}, {}};
    }

    std::vector<int> reinit_idx;
    if (replay_sources) {
        if (static_cast<int>(replay_sources->size()) != num_dead)
            throw std::invalid_argument("relocate replay source count does not match dead count");
        reinit_idx = *replay_sources;
        for (int src : reinit_idx) {
            if (src < 0 || src >= N || !is_alive[static_cast<size_t>(src)])
                throw std::invalid_argument("relocate replay source is not in alive pool");
        }
    } else {
        std::vector<float> probs(static_cast<size_t>(num_alive));
        for (int a = 0; a < num_alive; ++a) {
            probs[a] = sigmoid(params.opacities[alive_indices[a]]);
        }
        sample_with_replacement(alive_indices, probs, num_dead, rng_seed, reinit_idx);
    }

    std::vector<int> ratio(static_cast<size_t>(N), 0);
    for (int g : reinit_idx) ratio[g] += 1;

    std::vector<UpdatedParams> updates(static_cast<size_t>(N));
    std::vector<bool> updated(static_cast<size_t>(N), false);
    for (int g : reinit_idx) {
        if (updated[g]) continue;
        const float op_old = sigmoid(params.opacities[g]);
        const float sc_old[3] = {
            std::exp(params.scales[static_cast<size_t>(g) * 3 + 0]),
            std::exp(params.scales[static_cast<size_t>(g) * 3 + 1]),
            std::exp(params.scales[static_cast<size_t>(g) * 3 + 2]),
        };
        updates[g] = update_params_one(op_old, sc_old, ratio[g]);
        updated[g] = true;
    }

    const int mc3 = params.max_coeffs * 3;

    for (int k = 0; k < num_dead; ++k) {
        const int dst = dead_indices[k];
        const int src = reinit_idx[k];
        std::memcpy(&params.positions[static_cast<size_t>(dst) * 3],
                    &params.positions[static_cast<size_t>(src) * 3],
                    3 * sizeof(float));
        std::memcpy(&params.rotations[static_cast<size_t>(dst) * 4],
                    &params.rotations[static_cast<size_t>(src) * 4],
                    4 * sizeof(float));
        std::memcpy(&params.sh_coeffs[static_cast<size_t>(dst) * mc3],
                    &params.sh_coeffs[static_cast<size_t>(src) * mc3],
                    static_cast<size_t>(mc3) * sizeof(float));
        if (!params.filter_3D.empty()) {
            params.filter_3D[static_cast<size_t>(dst)] = params.filter_3D[static_cast<size_t>(src)];
        }
    }

    for (int k = 0; k < num_dead; ++k) {
        const int dst = dead_indices[k];
        const int src = reinit_idx[k];
        const UpdatedParams& u = updates[src];
        params.opacities[dst] = u.raw_opacity;
        params.opacities[src] = u.raw_opacity;
        for (int j = 0; j < 3; ++j) {
            params.scales[static_cast<size_t>(dst) * 3 + j] = u.raw_scale[j];
            params.scales[static_cast<size_t>(src) * 3 + j] = u.raw_scale[j];
        }
    }

    // The unique source indices whose params changed are the alive slots in reinit_idx.
    std::vector<int> unique_sources = reinit_idx;
    std::sort(unique_sources.begin(), unique_sources.end());
    unique_sources.erase(std::unique(unique_sources.begin(), unique_sources.end()),
                         unique_sources.end());

    return {num_dead, std::move(unique_sources), std::move(dead_indices)};
}

struct AddResult { int added; std::vector<int> add_idx; };

static AddResult add_new_gs_ex(OwnedRawParams& params,
                                int cap_max,
                                uint32_t rng_seed,
                                const std::vector<int>* replay_sources = nullptr)
{
    const int N = params.count();
    if (N == 0) {
        if (replay_sources && !replay_sources->empty())
            throw std::invalid_argument("add replay sources provided for empty params");
        return {0, {}};
    }

    const int target_num = std::min(cap_max,
                                     static_cast<int>(1.05 * static_cast<double>(N)));
    const int num_add = std::max(0, target_num - N);
    if (num_add == 0) {
        if (replay_sources && !replay_sources->empty())
            throw std::invalid_argument("add replay sources do not match no-op growth");
        return {0, {}};
    }

    std::vector<int> add_idx;
    if (replay_sources) {
        if (static_cast<int>(replay_sources->size()) != num_add)
            throw std::invalid_argument("add replay source count does not match growth count");
        add_idx = *replay_sources;
        for (int src : add_idx) {
            if (src < 0 || src >= N)
                throw std::invalid_argument("add replay source is outside pre-growth range");
        }
    } else {
        std::vector<int>   all_indices(static_cast<size_t>(N));
        std::vector<float> probs(static_cast<size_t>(N));
        for (int i = 0; i < N; ++i) {
            all_indices[i] = i;
            probs[i]       = sigmoid(params.opacities[i]);
        }
        sample_with_replacement(all_indices, probs, num_add, rng_seed, add_idx);
    }

    std::vector<int> ratio(static_cast<size_t>(N), 0);
    for (int g : add_idx) ratio[g] += 1;

    std::vector<UpdatedParams> updates(static_cast<size_t>(N));
    std::vector<bool> updated(static_cast<size_t>(N), false);
    for (int g : add_idx) {
        if (updated[g]) continue;
        const float op_old = sigmoid(params.opacities[g]);
        const float sc_old[3] = {
            std::exp(params.scales[static_cast<size_t>(g) * 3 + 0]),
            std::exp(params.scales[static_cast<size_t>(g) * 3 + 1]),
            std::exp(params.scales[static_cast<size_t>(g) * 3 + 2]),
        };
        updates[g] = update_params_one(op_old, sc_old, ratio[g]);
        updated[g] = true;
    }

    const int mc3 = params.max_coeffs * 3;
    const int new_total = N + num_add;

    std::vector<float> src_pos(static_cast<size_t>(num_add) * 3);
    std::vector<float> src_rot(static_cast<size_t>(num_add) * 4);
    std::vector<float> src_sh (static_cast<size_t>(num_add) * mc3);
    std::vector<float> src_filter;
    if (!params.filter_3D.empty()) src_filter.resize(static_cast<size_t>(num_add));
    for (int k = 0; k < num_add; ++k) {
        const int src = add_idx[k];
        std::memcpy(&src_pos[static_cast<size_t>(k) * 3],
                    &params.positions[static_cast<size_t>(src) * 3],
                    3 * sizeof(float));
        std::memcpy(&src_rot[static_cast<size_t>(k) * 4],
                    &params.rotations[static_cast<size_t>(src) * 4],
                    4 * sizeof(float));
        std::memcpy(&src_sh [static_cast<size_t>(k) * mc3],
                    &params.sh_coeffs[static_cast<size_t>(src) * mc3],
                    static_cast<size_t>(mc3) * sizeof(float));
        if (!params.filter_3D.empty()) {
            src_filter[static_cast<size_t>(k)] = params.filter_3D[static_cast<size_t>(src)];
        }
    }

    params.resize(new_total);

    for (int k = 0; k < num_add; ++k) {
        const int dst = N + k;
        const int src = add_idx[k];
        const UpdatedParams& u = updates[src];
        std::memcpy(&params.positions[static_cast<size_t>(dst) * 3],
                    &src_pos[static_cast<size_t>(k) * 3],
                    3 * sizeof(float));
        std::memcpy(&params.rotations[static_cast<size_t>(dst) * 4],
                    &src_rot[static_cast<size_t>(k) * 4],
                    4 * sizeof(float));
        std::memcpy(&params.sh_coeffs[static_cast<size_t>(dst) * mc3],
                    &src_sh [static_cast<size_t>(k) * mc3],
                    static_cast<size_t>(mc3) * sizeof(float));
        params.opacities[dst] = u.raw_opacity;
        if (!params.filter_3D.empty()) {
            params.filter_3D[static_cast<size_t>(dst)] = src_filter[static_cast<size_t>(k)];
        }
        for (int j = 0; j < 3; ++j)
            params.scales[static_cast<size_t>(dst) * 3 + j] = u.raw_scale[j];
    }

    for (int k = 0; k < num_add; ++k) {
        const int src = add_idx[k];
        const UpdatedParams& u = updates[src];
        params.opacities[src] = u.raw_opacity;
        for (int j = 0; j < 3; ++j)
            params.scales[static_cast<size_t>(src) * 3 + j] = u.raw_scale[j];
    }

    // Unique source indices whose params changed (before the grow).
    std::vector<int> unique_sources = add_idx;
    std::sort(unique_sources.begin(), unique_sources.end());
    unique_sources.erase(std::unique(unique_sources.begin(), unique_sources.end()),
                         unique_sources.end());

    return {num_add, std::move(unique_sources)};
}

// ---------------------------------------------------------------------------
// relocate_gs — thin wrapper over relocate_gs_ex
// ---------------------------------------------------------------------------

int relocate_gs(OwnedRawParams& params, float opacity_thresh, uint32_t rng_seed) {
    return relocate_gs_ex(params, opacity_thresh, rng_seed).relocated;
}

// ---------------------------------------------------------------------------
// add_new_gs — thin wrapper over add_new_gs_ex
// ---------------------------------------------------------------------------

int add_new_gs(OwnedRawParams& params, int cap_max, uint32_t rng_seed) {
    return add_new_gs_ex(params, cap_max, rng_seed).added;
}

// ---------------------------------------------------------------------------
// densify (combined relocate + add)
// ---------------------------------------------------------------------------

int densify(OwnedRawParams& params,
            float opacity_thresh,
            int cap_max,
            uint32_t rng_seed)
{
    // Use distinct sub-seeds for relocate vs add so the two passes don't
    // share the same RNG state (matches having two independent torch.multinomial
    // calls in Python, each draws fresh randomness).
    relocate_gs(params, opacity_thresh, rng_seed);
    add_new_gs(params, cap_max, rng_seed ^ 0x9E3779B9u);
    return params.count();
}

// ---------------------------------------------------------------------------
// densify_ex — like densify but returns modified source indices
// ---------------------------------------------------------------------------

static DensifyResult make_densify_result(OwnedRawParams& params,
                                         RelocateResult&& reloc,
                                         AddResult&& add)
{
    // Union of both source index sets (already sorted and deduplicated individually).
    std::vector<int> all_modified;
    all_modified.reserve(reloc.source_indices.size() + add.add_idx.size());
    all_modified.insert(all_modified.end(), reloc.source_indices.begin(), reloc.source_indices.end());
    all_modified.insert(all_modified.end(), add.add_idx.begin(),          add.add_idx.end());
    std::sort(all_modified.begin(), all_modified.end());
    all_modified.erase(std::unique(all_modified.begin(), all_modified.end()),
                       all_modified.end());

    std::sort(reloc.destination_indices.begin(), reloc.destination_indices.end());
    reloc.destination_indices.erase(std::unique(reloc.destination_indices.begin(),
                                               reloc.destination_indices.end()),
                                    reloc.destination_indices.end());

    return {params.count(), std::move(all_modified), std::move(reloc.destination_indices)};
}

DensifyResult densify_ex(OwnedRawParams& params,
                          float opacity_thresh,
                          int cap_max,
                          uint32_t rng_seed)
{
    auto reloc = relocate_gs_ex(params, opacity_thresh, rng_seed);
    auto add   = add_new_gs_ex(params, cap_max, rng_seed ^ 0x9E3779B9u);
    return make_densify_result(params, std::move(reloc), std::move(add));
}

DensifyResult densify_with_samples(OwnedRawParams& params,
                                   float opacity_thresh,
                                   int cap_max,
                                   const DensifySamplePlan& plan)
{
    auto reloc = relocate_gs_ex(params, opacity_thresh, 0, &plan.relocate_sources);
    auto add   = add_new_gs_ex(params, cap_max, 0, &plan.add_sources);
    return make_densify_result(params, std::move(reloc), std::move(add));
}

}  // namespace mcmc
