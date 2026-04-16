// test_gpu_context.h -- Shared OpenCL context for GPU tests.
// Uses a single static OpenCLContext instance across all test files
// to avoid the OpenCL loader's once_flag issue where function pointers
// are only populated for the first OpenCLFunctions struct passed.

#pragma once

#ifdef ENABLE_OPENCL
#include "gpu/opencl_context.h"

inline OpenCLContext& getTestGPUContext() {
    static OpenCLContext ctx;
    return ctx;
}

inline bool initTestGPUContext() {
    static bool tried = false;
    static bool ok = false;
    if (!tried) {
        ok = getTestGPUContext().init();
        tried = true;
    }
    return ok;
}
#endif
