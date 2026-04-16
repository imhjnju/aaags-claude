// opencl_context.h -- RAII wrapper around an OpenCL context, queue, and device.
// Provides CL_CHECK macro, device capability query, kernel build helpers,
// buffer operations, and event-based profiling.

#pragma once

#include "gpu/opencl_loader.h"
#include <string>
#include <stdexcept>

// ---------------------------------------------------------------------------
// CL_CHECK -- throw on any non-CL_SUCCESS return code.
// Every CL call that returns cl_int should be wrapped with this macro
// (or use manual error handling for calls that return objects).
// ---------------------------------------------------------------------------
#define CL_CHECK(fn_call) do {                                              \
    cl_int _cl_err = (fn_call);                                             \
    if (_cl_err != CL_SUCCESS)                                              \
        throw std::runtime_error(                                           \
            std::string("OpenCL error ") + std::to_string(_cl_err) +        \
            " at " + __FILE__ + ":" + std::to_string(__LINE__));             \
} while (0)

// ---------------------------------------------------------------------------
// OpenCLContext -- owns platform, device, context, and command queue.
// ---------------------------------------------------------------------------
class OpenCLContext {
public:
    /// Load OpenCL, find GPU, create context + queue, query device caps.
    /// Returns false if OpenCL is unavailable or no GPU device is found.
    bool init();

    /// Release all CL resources (queue, context, library handle).
    void release();

    ~OpenCLContext();

    // -- Kernel management --------------------------------------------------

    /// Compile source into a program, build it, and extract the named kernel.
    /// On build failure the compiler log is included in the thrown exception.
    /// The intermediate cl_program is released; the kernel retains it.
    cl_kernel buildKernel(const char* source, size_t source_len,
                          const char* kernel_name,
                          const char* build_opts = "");

    // -- Buffer operations --------------------------------------------------

    cl_mem createBuffer(size_t size, cl_mem_flags flags,
                        void* host_ptr = nullptr);

    void readBuffer(cl_mem buf, void* dst, size_t size);
    void writeBuffer(cl_mem buf, const void* src, size_t size);

    // -- Execution ----------------------------------------------------------

    void enqueueKernel(cl_kernel k, int dim,
                       const size_t* global, const size_t* local);
    // Enqueue with event for profiling — caller must release event
    void enqueueKernel(cl_kernel k, int dim,
                       const size_t* global, const size_t* local,
                       cl_event* event);
    void finish();

    // -- Device capabilities (populated by init()) --------------------------

    size_t  maxWorkGroupSize() const { return max_wg_size_; }
    size_t  maxLocalMemSize()  const { return max_local_mem_; }
    cl_uint maxComputeUnits()  const { return max_cu_; }
    size_t  maxAllocSize()     const { return max_alloc_; }

    // -- Profiling (queue created with CL_QUEUE_PROFILING_ENABLE) -----------

    /// Return elapsed time in milliseconds between COMMAND_START and
    /// COMMAND_END for the given event.
    double getEventTimeMs(cl_event ev);

    // -- Raw handle access (for advanced / interop use) ---------------------

    const OpenCLFunctions& cl()      const { return cl_; }
    cl_context             context() const { return context_; }
    cl_command_queue       queue()   const { return queue_; }

private:
    OpenCLFunctions  cl_{};
    cl_platform_id   platform_    = nullptr;
    cl_device_id     device_      = nullptr;
    cl_context       context_     = nullptr;
    cl_command_queue queue_       = nullptr;
    bool             initialized_ = false;

    // Device caps
    size_t  max_wg_size_   = 0;
    size_t  max_local_mem_ = 0;
    cl_uint max_cu_        = 0;
    size_t  max_alloc_     = 0;
};
