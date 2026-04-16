// opencl_loader.h -- Runtime OpenCL loader via dlopen/dlsym
// Provides function pointers for all OpenCL 1.2 APIs needed by the renderer.
// HarmonyOS NDK does not ship OpenCL headers or libs, so we vendor the Khronos
// headers and resolve symbols at runtime from the device's libOpenCL.so.

#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

// ---------------------------------------------------------------------------
// Function-pointer typedefs matching the OpenCL 1.x C API.
// Names follow the convention: pfn_<apiName>
// ---------------------------------------------------------------------------

// Platform
using pfn_clGetPlatformIDs = cl_int (CL_API_CALL *)(
    cl_uint, cl_platform_id *, cl_uint *);

using pfn_clGetPlatformInfo = cl_int (CL_API_CALL *)(
    cl_platform_id, cl_platform_info, size_t, void *, size_t *);

// Device
using pfn_clGetDeviceIDs = cl_int (CL_API_CALL *)(
    cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);

using pfn_clGetDeviceInfo = cl_int (CL_API_CALL *)(
    cl_device_id, cl_device_info, size_t, void *, size_t *);

// Context
using pfn_clCreateContext = cl_context (CL_API_CALL *)(
    const cl_context_properties *, cl_uint, const cl_device_id *,
    void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
    void *, cl_int *);

using pfn_clReleaseContext = cl_int (CL_API_CALL *)(cl_context);

using pfn_clGetContextInfo = cl_int (CL_API_CALL *)(
    cl_context, cl_context_info, size_t, void *, size_t *);

// Command queue (CL 1.x API -- NOT clCreateCommandQueueWithProperties)
using pfn_clCreateCommandQueue = cl_command_queue (CL_API_CALL *)(
    cl_context, cl_device_id, cl_command_queue_properties, cl_int *);

using pfn_clReleaseCommandQueue = cl_int (CL_API_CALL *)(cl_command_queue);

// Memory objects
using pfn_clCreateBuffer = cl_mem (CL_API_CALL *)(
    cl_context, cl_mem_flags, size_t, void *, cl_int *);

using pfn_clReleaseMemObject = cl_int (CL_API_CALL *)(cl_mem);

// Read / write / copy / fill buffers
using pfn_clEnqueueReadBuffer = cl_int (CL_API_CALL *)(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *,
    cl_uint, const cl_event *, cl_event *);

using pfn_clEnqueueWriteBuffer = cl_int (CL_API_CALL *)(
    cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void *,
    cl_uint, const cl_event *, cl_event *);

using pfn_clEnqueueCopyBuffer = cl_int (CL_API_CALL *)(
    cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t,
    cl_uint, const cl_event *, cl_event *);

using pfn_clEnqueueFillBuffer = cl_int (CL_API_CALL *)(
    cl_command_queue, cl_mem, const void *, size_t, size_t, size_t,
    cl_uint, const cl_event *, cl_event *);

// Program
using pfn_clCreateProgramWithSource = cl_program (CL_API_CALL *)(
    cl_context, cl_uint, const char **, const size_t *, cl_int *);

using pfn_clBuildProgram = cl_int (CL_API_CALL *)(
    cl_program, cl_uint, const cl_device_id *, const char *,
    void (CL_CALLBACK *)(cl_program, void *), void *);

using pfn_clReleaseProgram = cl_int (CL_API_CALL *)(cl_program);

using pfn_clGetProgramBuildInfo = cl_int (CL_API_CALL *)(
    cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);

// Kernel
using pfn_clCreateKernel = cl_kernel (CL_API_CALL *)(
    cl_program, const char *, cl_int *);

using pfn_clSetKernelArg = cl_int (CL_API_CALL *)(
    cl_kernel, cl_uint, size_t, const void *);

using pfn_clReleaseKernel = cl_int (CL_API_CALL *)(cl_kernel);

// Execution
using pfn_clEnqueueNDRangeKernel = cl_int (CL_API_CALL *)(
    cl_command_queue, cl_kernel, cl_uint,
    const size_t *, const size_t *, const size_t *,
    cl_uint, const cl_event *, cl_event *);

using pfn_clFinish = cl_int (CL_API_CALL *)(cl_command_queue);

// Events / profiling
using pfn_clGetEventProfilingInfo = cl_int (CL_API_CALL *)(
    cl_event, cl_profiling_info, size_t, void *, size_t *);

using pfn_clWaitForEvents = cl_int (CL_API_CALL *)(
    cl_uint, const cl_event *);

using pfn_clReleaseEvent = cl_int (CL_API_CALL *)(cl_event);

// ---------------------------------------------------------------------------
// Aggregate struct -- every pointer is nullptr until loadOpenCL() succeeds.
// ---------------------------------------------------------------------------
struct OpenCLFunctions {
    // Platform
    pfn_clGetPlatformIDs        clGetPlatformIDs        = nullptr;
    pfn_clGetPlatformInfo       clGetPlatformInfo       = nullptr;

    // Device
    pfn_clGetDeviceIDs          clGetDeviceIDs          = nullptr;
    pfn_clGetDeviceInfo         clGetDeviceInfo         = nullptr;

    // Context
    pfn_clCreateContext         clCreateContext         = nullptr;
    pfn_clReleaseContext        clReleaseContext        = nullptr;
    pfn_clGetContextInfo        clGetContextInfo        = nullptr;

    // Command queue
    pfn_clCreateCommandQueue    clCreateCommandQueue    = nullptr;
    pfn_clReleaseCommandQueue   clReleaseCommandQueue   = nullptr;

    // Memory objects
    pfn_clCreateBuffer          clCreateBuffer          = nullptr;
    pfn_clReleaseMemObject      clReleaseMemObject      = nullptr;

    // Buffer read/write/copy/fill
    pfn_clEnqueueReadBuffer     clEnqueueReadBuffer     = nullptr;
    pfn_clEnqueueWriteBuffer    clEnqueueWriteBuffer    = nullptr;
    pfn_clEnqueueCopyBuffer     clEnqueueCopyBuffer     = nullptr;
    pfn_clEnqueueFillBuffer     clEnqueueFillBuffer     = nullptr;

    // Program
    pfn_clCreateProgramWithSource clCreateProgramWithSource = nullptr;
    pfn_clBuildProgram          clBuildProgram          = nullptr;
    pfn_clReleaseProgram        clReleaseProgram        = nullptr;
    pfn_clGetProgramBuildInfo   clGetProgramBuildInfo   = nullptr;

    // Kernel
    pfn_clCreateKernel          clCreateKernel          = nullptr;
    pfn_clSetKernelArg          clSetKernelArg          = nullptr;
    pfn_clReleaseKernel         clReleaseKernel         = nullptr;

    // Execution
    pfn_clEnqueueNDRangeKernel  clEnqueueNDRangeKernel  = nullptr;
    pfn_clFinish                clFinish                = nullptr;

    // Events / profiling
    pfn_clGetEventProfilingInfo clGetEventProfilingInfo  = nullptr;
    pfn_clWaitForEvents         clWaitForEvents          = nullptr;
    pfn_clReleaseEvent          clReleaseEvent           = nullptr;
};

// Load all OpenCL function pointers via dlopen + dlsym.
// Returns true if the library was found AND every required symbol resolved.
// Thread-safe (uses std::call_once internally).
bool loadOpenCL(OpenCLFunctions& funcs);

// Close the shared library handle opened by loadOpenCL().
void unloadOpenCL();
