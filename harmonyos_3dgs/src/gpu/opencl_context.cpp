// opencl_context.cpp -- OpenCLContext implementation
// See opencl_context.h for the public interface.

#include "gpu/opencl_context.h"

#include <cstdio>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// init -- bootstrap the full OpenCL stack
// ---------------------------------------------------------------------------
bool OpenCLContext::init() {
    if (initialized_)
        return true;

    // 1. Load the shared library and resolve function pointers.
    if (!loadOpenCL(cl_)) {
        std::fprintf(stderr, "OpenCLContext: failed to load OpenCL library\n");
        return false;
    }

    // 2. Get the first available platform.
    cl_int err = cl_.clGetPlatformIDs(1, &platform_, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "OpenCLContext: clGetPlatformIDs failed (%d)\n", err);
        return false;
    }

    // 3. Get the first GPU device on that platform.
    err = cl_.clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
    if (err != CL_SUCCESS) {
        std::fprintf(stderr, "OpenCLContext: no GPU device found (%d)\n", err);
        return false;
    }

    // 4. Print device name.
    char device_name[256] = {};
    cl_.clGetDeviceInfo(device_, CL_DEVICE_NAME,
                        sizeof(device_name), device_name, nullptr);
    std::fprintf(stderr, "OpenCLContext: device = %s\n", device_name);

    // 5. Query device capabilities.
    cl_.clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                        sizeof(max_wg_size_), &max_wg_size_, nullptr);
    cl_.clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE,
                        sizeof(max_local_mem_), &max_local_mem_, nullptr);
    cl_.clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                        sizeof(max_cu_), &max_cu_, nullptr);
    cl_.clGetDeviceInfo(device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                        sizeof(max_alloc_), &max_alloc_, nullptr);

    // 6. Create context.
    context_ = cl_.clCreateContext(nullptr, 1, &device_,
                                   nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_) {
        std::fprintf(stderr, "OpenCLContext: clCreateContext failed (%d)\n", err);
        return false;
    }

    // 7. Create command queue with profiling enabled.
    queue_ = cl_.clCreateCommandQueue(context_, device_,
                                      CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS || !queue_) {
        std::fprintf(stderr, "OpenCLContext: clCreateCommandQueue failed (%d)\n", err);
        cl_.clReleaseContext(context_);
        context_ = nullptr;
        return false;
    }

    // 8. Print device capabilities.
    std::fprintf(stderr,
        "OpenCLContext: max_wg_size=%zu  local_mem=%zu  CUs=%u  max_alloc=%zu\n",
        max_wg_size_, max_local_mem_,
        static_cast<unsigned>(max_cu_), max_alloc_);

    // 9. Done.
    initialized_ = true;
    return true;
}

// ---------------------------------------------------------------------------
// release -- tear down in reverse order
// ---------------------------------------------------------------------------
void OpenCLContext::release() {
    if (!initialized_)
        return;

    if (queue_) {
        cl_.clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    if (context_) {
        cl_.clReleaseContext(context_);
        context_ = nullptr;
    }

    platform_ = nullptr;
    device_   = nullptr;

    unloadOpenCL();
    initialized_ = false;
}

OpenCLContext::~OpenCLContext() {
    release();
}

// ---------------------------------------------------------------------------
// buildKernel -- compile + build + extract kernel
// ---------------------------------------------------------------------------
cl_kernel OpenCLContext::buildKernel(const char* source, size_t source_len,
                                    const char* kernel_name,
                                    const char* build_opts) {
    cl_int err;

    // 1. Create program from source.
    cl_program program = cl_.clCreateProgramWithSource(
        context_, 1, &source, &source_len, &err);
    if (err != CL_SUCCESS || !program) {
        throw std::runtime_error(
            std::string("clCreateProgramWithSource failed: ") +
            std::to_string(err));
    }

    // 2. Build program.
    err = cl_.clBuildProgram(program, 1, &device_, build_opts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // Retrieve build log.
        size_t log_size = 0;
        cl_.clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG,
                                  0, nullptr, &log_size);
        std::vector<char> log(log_size + 1, '\0');
        cl_.clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG,
                                  log_size, log.data(), nullptr);
        cl_.clReleaseProgram(program);
        throw std::runtime_error(
            std::string("clBuildProgram failed (") + std::to_string(err) +
            "):\n" + log.data());
    }

    // 3. Extract named kernel.
    cl_kernel kernel = cl_.clCreateKernel(program, kernel_name, &err);
    if (err != CL_SUCCESS || !kernel) {
        cl_.clReleaseProgram(program);
        throw std::runtime_error(
            std::string("clCreateKernel '") + kernel_name +
            "' failed: " + std::to_string(err));
    }

    // 4. Release program (kernel retains it internally).
    cl_.clReleaseProgram(program);

    return kernel;
}

// ---------------------------------------------------------------------------
// Buffer operations
// ---------------------------------------------------------------------------

cl_mem OpenCLContext::createBuffer(size_t size, cl_mem_flags flags,
                                   void* host_ptr) {
    cl_int err;
    cl_mem buf = cl_.clCreateBuffer(context_, flags, size, host_ptr, &err);
    CL_CHECK(err);
    return buf;
}

void OpenCLContext::readBuffer(cl_mem buf, void* dst, size_t size) {
    CL_CHECK(cl_.clEnqueueReadBuffer(
        queue_, buf, CL_TRUE, 0, size, dst, 0, nullptr, nullptr));
}

void OpenCLContext::writeBuffer(cl_mem buf, const void* src, size_t size) {
    CL_CHECK(cl_.clEnqueueWriteBuffer(
        queue_, buf, CL_TRUE, 0, size, src, 0, nullptr, nullptr));
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

void OpenCLContext::enqueueKernel(cl_kernel k, int dim,
                                  const size_t* global,
                                  const size_t* local) {
    CL_CHECK(cl_.clEnqueueNDRangeKernel(
        queue_, k, static_cast<cl_uint>(dim),
        nullptr, global, local, 0, nullptr, nullptr));
}

void OpenCLContext::enqueueKernel(cl_kernel k, int dim,
                                  const size_t* global,
                                  const size_t* local,
                                  cl_event* event) {
    CL_CHECK(cl_.clEnqueueNDRangeKernel(
        queue_, k, static_cast<cl_uint>(dim),
        nullptr, global, local, 0, nullptr, event));
}

void OpenCLContext::finish() {
    CL_CHECK(cl_.clFinish(queue_));
}

// ---------------------------------------------------------------------------
// Profiling
// ---------------------------------------------------------------------------

double OpenCLContext::getEventTimeMs(cl_event ev) {
    cl_ulong start = 0, end = 0;
    CL_CHECK(cl_.clGetEventProfilingInfo(
        ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr));
    CL_CHECK(cl_.clGetEventProfilingInfo(
        ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr));
    return static_cast<double>(end - start) / 1.0e6;
}
