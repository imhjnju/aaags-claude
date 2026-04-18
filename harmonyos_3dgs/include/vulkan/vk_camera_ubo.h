// Host-side CameraUBO struct matching shader's std140 layout exactly.
// Spec §4.6: 224 bytes, std140, vec3 padded to vec4.
#pragma once
#include <cstdint>
#include <cstddef>

struct alignas(16) CameraUBO {
    float viewmatrix[16];         // 64B, column-major
    float projmatrix[16];         // 64B
    float inv_viewprojmatrix[16]; // 64B
    float campos_pad[4];          // 16B: xyz=campos, w=0
    float fov_size[4];            // 16B: x=tan_fovx, y=tan_fovy, z=width, w=height
};
static_assert(sizeof(CameraUBO) == 224, "CameraUBO std140 layout mismatch (spec §4.6)");
static_assert(offsetof(CameraUBO, projmatrix)         == 64,  "");
static_assert(offsetof(CameraUBO, inv_viewprojmatrix) == 128, "");
static_assert(offsetof(CameraUBO, campos_pad)         == 192, "");
static_assert(offsetof(CameraUBO, fov_size)           == 208, "");
