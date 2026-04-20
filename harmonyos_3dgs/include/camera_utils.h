#pragma once
#include "types.h"

// Build a look-at view matrix (column-major, matching GLM/CUDA convention)
// eye: camera position, center: look-at target, up: world up vector
void buildLookAt(const float eye[3], const float center[3], const float up[3],
                 float view_matrix[16]);

// Build perspective projection matrix (column-major)
void buildPerspective(float tan_fovx, float tan_fovy, float znear, float zfar,
                      float proj[16]);

// Multiply two 4x4 column-major matrices: out = A * B
void mat4Mul(const float A[16], const float B[16], float out[16]);

// Auto-compute camera from scene bounding box
Camera autoCamera(const GaussianData& g, int width, int height);

// Load camera from cameras.json (3DGS format)
// cameras.json stores: position (world camera center), rotation (W2C 3x3), fx, fy, width, height
Camera loadCameraJson(const char* json_path, int cam_id);
