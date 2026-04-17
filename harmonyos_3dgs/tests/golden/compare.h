#pragma once
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <vector>
#include <limits>

// CompareResult diagnostics.
// NOTE: On size-mismatch, compare_f32 returns a default-constructed result
// (passed=false, max_*_err=0). Callers must always check `passed` before
// reading max_*_err / first_bad_index / num_bad.
struct CompareResult {
    bool    passed        = false;
    float   max_abs_err   = 0.0f;
    float   max_rel_err   = 0.0f;
    size_t  first_bad_index = std::numeric_limits<size_t>::max();
    size_t  num_bad       = 0;
};

inline CompareResult compare_f32(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 float abs_tol, float rel_tol) {
    CompareResult r;
    if (a.size() != b.size()) return r;
    for (size_t i = 0; i < a.size(); ++i) {
        bool is_nan = std::isnan(a[i]) || std::isnan(b[i]);
        float abs_err = is_nan ? std::numeric_limits<float>::infinity() : std::fabs(a[i] - b[i]);
        float rel_err = is_nan ? std::numeric_limits<float>::infinity() : abs_err / (std::fabs(b[i]) + 1e-30f);
        bool bad = is_nan || ((abs_err > abs_tol) && (rel_err > rel_tol));
        if (bad) {
            if (r.num_bad == 0) r.first_bad_index = i;
            ++r.num_bad;
        }
        if (abs_err > r.max_abs_err) r.max_abs_err = abs_err;
        if (rel_err > r.max_rel_err) r.max_rel_err = rel_err;
    }
    r.passed = (r.num_bad == 0);
    return r;
}

inline bool compare_u32(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

inline bool compare_u64(const std::vector<uint64_t>& a, const std::vector<uint64_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}
