#pragma once
#include "types.h"

struct LoadedModel {
    GaussianData data;
    void free();  // Release all heap-allocated arrays
};

// Load a 3DGS PLY file with activation transforms.
// Reads: x,y,z, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3
// Applies: exp(scale), sigmoid(opacity), normalize(rotation)
// Reorders SH: channel-first (PLY) -> basis-interleaved (CUDA layout)
LoadedModel loadPly(const char* path);
