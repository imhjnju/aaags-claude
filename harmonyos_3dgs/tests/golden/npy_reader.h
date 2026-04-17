#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class NpyDtype { float32, int32, int64, uint32, uint64 };

struct NpyArray {
    std::vector<uint8_t> raw;
    std::vector<size_t>  shape;
    NpyDtype             dtype = NpyDtype::float32;
    size_t numel() const {
        size_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
    float*    f32() { return reinterpret_cast<float*>(raw.data()); }
    int32_t*  i32() { return reinterpret_cast<int32_t*>(raw.data()); }
    uint32_t* u32() { return reinterpret_cast<uint32_t*>(raw.data()); }
    int64_t*  i64() { return reinterpret_cast<int64_t*>(raw.data()); }
    uint64_t* u64() { return reinterpret_cast<uint64_t*>(raw.data()); }
};

// NOT YET IMPLEMENTED — test should fail at link time
NpyArray load_npy(const std::string& path);
