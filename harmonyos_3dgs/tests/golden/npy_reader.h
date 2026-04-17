#pragma once
#include <algorithm>
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

#include <fstream>
#include <stdexcept>
#include <sstream>
#include <cstring>

inline NpyDtype parse_descr(const std::string& descr) {
    // descr format: "<f4", "<i4", "<i8", "<u4", "<u8" (little-endian assumed)
    if (descr == "<f4") return NpyDtype::float32;
    if (descr == "<i4") return NpyDtype::int32;
    if (descr == "<i8") return NpyDtype::int64;
    if (descr == "<u4") return NpyDtype::uint32;
    if (descr == "<u8") return NpyDtype::uint64;
    throw std::runtime_error("npy_reader: unsupported dtype: " + descr);
}

inline size_t dtype_size(NpyDtype d) {
    switch (d) {
        case NpyDtype::float32: case NpyDtype::int32: case NpyDtype::uint32: return 4;
        case NpyDtype::int64:   case NpyDtype::uint64:                       return 8;
    }
    throw std::runtime_error("npy_reader: unknown dtype size");
}

inline std::string extract_dict_value(const std::string& header, const std::string& key) {
    // Extract value for "'key': VALUE," from Python dict literal header.
    auto pos = header.find("'" + key + "':");
    if (pos == std::string::npos) throw std::runtime_error("npy_reader: key not found: " + key);
    pos += key.size() + 3;  // past "'key':"
    while (pos < header.size() && header[pos] == ' ') ++pos;
    // Collect until next ',' at top level (shape uses parens, descr/fortran_order do not)
    int paren = 0;
    std::string val;
    while (pos < header.size()) {
        char c = header[pos];
        if (c == '(') ++paren;
        else if (c == ')') --paren;
        else if (c == ',' && paren == 0) break;
        val += c; ++pos;
    }
    return val;
}

inline std::vector<size_t> parse_shape(const std::string& shape_str) {
    // shape_str like "(2, 3)" or "(5,)"
    std::vector<size_t> shape;
    size_t i = shape_str.find('(') + 1;
    std::string num;
    while (i < shape_str.size() && shape_str[i] != ')') {
        char c = shape_str[i];
        if (c >= '0' && c <= '9') num += c;
        else if (c == ',' || c == ' ') {
            if (!num.empty()) { shape.push_back(std::stoul(num)); num.clear(); }
        }
        ++i;
    }
    if (!num.empty()) shape.push_back(std::stoul(num));
    return shape;
}

inline NpyArray load_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy_reader: cannot open " + path);
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npy_reader: bad magic at " + path);
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    if (major != 1) throw std::runtime_error("npy_reader: only v1.0 supported, got v"
                                             + std::to_string(major));
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, '\0');
    f.read(&header[0], header_len);

    NpyArray arr;
    std::string descr = extract_dict_value(header, "descr");
    // strip surrounding quotes
    descr.erase(std::remove(descr.begin(), descr.end(), '\''), descr.end());
    // trim
    while (!descr.empty() && descr.front() == ' ') descr.erase(0, 1);
    while (!descr.empty() && descr.back() == ' ')  descr.pop_back();
    arr.dtype = parse_descr(descr);

    std::string shape_val = extract_dict_value(header, "shape");
    arr.shape = parse_shape(shape_val);

    std::string fo = extract_dict_value(header, "fortran_order");
    if (fo.find("True") != std::string::npos)
        throw std::runtime_error("npy_reader: fortran_order=True not supported");

    size_t total_bytes = arr.numel() * dtype_size(arr.dtype);
    arr.raw.resize(total_bytes);
    f.read(reinterpret_cast<char*>(arr.raw.data()), total_bytes);
    if (f.gcount() != static_cast<std::streamsize>(total_bytes))
        throw std::runtime_error("npy_reader: short read at " + path);
    return arr;
}

inline void assert_shape(const NpyArray& a, std::vector<size_t> expected) {
    if (a.shape != expected) {
        std::ostringstream oss;
        oss << "npy_reader: shape mismatch. got [";
        for (auto s : a.shape) oss << s << ",";
        oss << "], expected [";
        for (auto s : expected) oss << s << ",";
        oss << "]";
        throw std::runtime_error(oss.str());
    }
}

inline void assert_dtype(const NpyArray& a, NpyDtype expected) {
    if (a.dtype != expected)
        throw std::runtime_error("npy_reader: dtype mismatch");
}
