#pragma once
// Minimal NPY v1.0 writer, companion to npy_reader.h.
// Little-endian host only (same assumption as the reader).

#include "npy_reader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

inline const char* npy_descr_str(NpyDtype d) {
    switch (d) {
        case NpyDtype::float32: return "<f4";
        case NpyDtype::int32:   return "<i4";
        case NpyDtype::int64:   return "<i8";
        case NpyDtype::uint32:  return "<u4";
        case NpyDtype::uint64:  return "<u8";
    }
    throw std::runtime_error("npy_writer: unknown dtype");
}

inline void save_npy(const std::string& path,
                     const void* data,
                     const std::vector<size_t>& shape,
                     NpyDtype dtype) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy_writer: cannot open " + path);

    std::string body = "{'descr': '";
    body += npy_descr_str(dtype);
    body += "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        body += std::to_string(shape[i]);
        if (shape.size() == 1 || i + 1 < shape.size()) body += ", ";
    }
    body += "), }";

    // Prefix = 6 magic + 2 version + 2 header_len = 10 bytes.
    // Total (prefix + body + '\n') must be multiple of 64.
    constexpr size_t prefix = 10;
    size_t total = prefix + body.size() + 1;
    size_t pad = (64 - (total % 64)) % 64;
    body.append(pad, ' ');
    body += '\n';

    uint16_t header_len = static_cast<uint16_t>(body.size());
    f.write("\x93NUMPY", 6);
    const char version[2] = {1, 0};
    f.write(version, 2);
    f.write(reinterpret_cast<const char*>(&header_len), 2);
    f.write(body.data(), body.size());

    size_t elems = 1;
    for (auto s : shape) elems *= s;
    size_t bytes = elems * dtype_size(dtype);
    f.write(reinterpret_cast<const char*>(data), bytes);
    if (!f) throw std::runtime_error("npy_writer: write failed for " + path);
}

inline void save_npy_f32(const std::string& path,
                         const float* data,
                         const std::vector<size_t>& shape) {
    save_npy(path, data, shape, NpyDtype::float32);
}

inline void save_npy_u32(const std::string& path,
                         const uint32_t* data,
                         const std::vector<size_t>& shape) {
    save_npy(path, data, shape, NpyDtype::uint32);
}

inline void save_npy_i32(const std::string& path,
                         const int32_t* data,
                         const std::vector<size_t>& shape) {
    save_npy(path, data, shape, NpyDtype::int32);
}
