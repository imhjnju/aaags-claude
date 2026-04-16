// opencl_loader.cpp -- Runtime OpenCL loader implementation
// Tries several well-known library paths used on HarmonyOS / Android ARM64
// devices, then resolves every required CL 1.2 function pointer via dlsym.

#include "gpu/opencl_loader.h"

#include <dlfcn.h>
#include <cstdio>
#include <mutex>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------
static void*          s_libHandle  = nullptr;
static bool           s_loaded     = false;
static std::once_flag s_onceFlag;

// Convenience macro: load one symbol, bail on failure.
#define LOAD_SYM(fn)                                                         \
    do {                                                                     \
        funcs.fn = reinterpret_cast<pfn_##fn>(dlsym(s_libHandle, #fn));      \
        if (!funcs.fn) {                                                     \
            std::fprintf(stderr, "OpenCL loader: missing symbol %s\n", #fn); \
            return false;                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// tryOpen -- attempt dlopen on a single path
// ---------------------------------------------------------------------------
static bool tryOpen(const char* path) {
    s_libHandle = dlopen(path, RTLD_NOW);
    if (s_libHandle) {
        std::fprintf(stderr, "OpenCL loader: opened %s\n", path);
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// loadAllSymbols -- resolve every function pointer we need
// ---------------------------------------------------------------------------
static bool loadAllSymbols(OpenCLFunctions& funcs) {
    // Platform
    LOAD_SYM(clGetPlatformIDs);
    LOAD_SYM(clGetPlatformInfo);

    // Device
    LOAD_SYM(clGetDeviceIDs);
    LOAD_SYM(clGetDeviceInfo);

    // Context
    LOAD_SYM(clCreateContext);
    LOAD_SYM(clReleaseContext);
    LOAD_SYM(clGetContextInfo);

    // Command queue (CL 1.x)
    LOAD_SYM(clCreateCommandQueue);
    LOAD_SYM(clReleaseCommandQueue);

    // Memory objects
    LOAD_SYM(clCreateBuffer);
    LOAD_SYM(clReleaseMemObject);

    // Buffer transfers
    LOAD_SYM(clEnqueueReadBuffer);
    LOAD_SYM(clEnqueueWriteBuffer);
    LOAD_SYM(clEnqueueCopyBuffer);
    LOAD_SYM(clEnqueueFillBuffer);

    // Program
    LOAD_SYM(clCreateProgramWithSource);
    LOAD_SYM(clBuildProgram);
    LOAD_SYM(clReleaseProgram);
    LOAD_SYM(clGetProgramBuildInfo);

    // Kernel
    LOAD_SYM(clCreateKernel);
    LOAD_SYM(clSetKernelArg);
    LOAD_SYM(clReleaseKernel);

    // Execution
    LOAD_SYM(clEnqueueNDRangeKernel);
    LOAD_SYM(clFinish);

    // Events / profiling
    LOAD_SYM(clGetEventProfilingInfo);
    LOAD_SYM(clWaitForEvents);
    LOAD_SYM(clReleaseEvent);

    return true;
}

#undef LOAD_SYM

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool loadOpenCL(OpenCLFunctions& funcs) {
    std::call_once(s_onceFlag, [&funcs]() {
        // Try each candidate path; stop at first success.
        static const char* candidates[] = {
            "libOpenCL.so",
            "/system/lib64/libOpenCL.so",
            "/vendor/lib64/libOpenCL.so",
            "/system/vendor/lib64/libOpenCL.so",
        };

        for (const char* path : candidates) {
            if (tryOpen(path))
                break;
        }

        if (!s_libHandle) {
            std::fprintf(stderr, "OpenCL loader: could not find libOpenCL.so\n");
            return;
        }

        if (loadAllSymbols(funcs)) {
            s_loaded = true;
        } else {
            std::fprintf(stderr, "OpenCL loader: symbol resolution failed\n");
            dlclose(s_libHandle);
            s_libHandle = nullptr;
        }
    });

    return s_loaded;
}

void unloadOpenCL() {
    if (s_libHandle) {
        dlclose(s_libHandle);
        s_libHandle = nullptr;
    }
    s_loaded = false;
}
