#pragma once
#include <cstdint>
#include <vector>

// Temporary CPU Adam optimizer for SP-4 training integration.
// Will be replaced by a Vulkan compute Adam in a future sprint.
//
// Matches Python reference (training.py):
//   beta1=0.9, beta2=0.999, eps=1e-15
//   step: m = b1*m + (1-b1)*g
//         v = b2*v + (1-b2)*g*g
//         p -= lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)

struct AdamGroup {
    float*       params;  // parameter array (modified in-place)
    const float* grad;    // gradient array (read-only)
    float*       m;       // first moment (same size as params)
    float*       v;       // second moment (same size as params)
    int          n;       // element count
    float        lr;      // learning rate for this group
};

class CpuAdam {
public:
    explicit CpuAdam(float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-15f);

    // Add a parameter group. m and v storage is owned by the caller.
    void add_group(AdamGroup g);

    // Run one Adam step. Increments internal step counter.
    void step();

    int step_count() const { return step_; }

private:
    float beta1_, beta2_, eps_;
    int   step_ = 0;
    std::vector<AdamGroup> groups_;
};
