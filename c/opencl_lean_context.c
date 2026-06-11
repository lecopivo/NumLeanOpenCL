#include "opencl_lean.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

NumLeanOpenCLContext g_numleanopencl_ctx = {0};

static const char* cl_error_name(cl_int err) {
    switch (err) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    default: return "UNKNOWN_OPENCL_ERROR";
    }
}

static lean_obj_res io_error(const char* msg) {
    lean_object* s = lean_mk_string(msg);
    return lean_io_result_mk_error(lean_mk_io_user_error(s));
}

static int choose_device(cl_platform_id* platform_out, cl_device_id* device_out, char* err_buf, size_t err_buf_size) {
    cl_uint nplatforms = 0;
    cl_int err = clGetPlatformIDs(0, NULL, &nplatforms);
    if (err != CL_SUCCESS || nplatforms == 0) {
        snprintf(err_buf, err_buf_size, "clGetPlatformIDs failed: %s (%d)", cl_error_name(err), err);
        return 0;
    }

    cl_platform_id* platforms = (cl_platform_id*)calloc(nplatforms, sizeof(cl_platform_id));
    if (!platforms) {
        snprintf(err_buf, err_buf_size, "out of host memory while listing OpenCL platforms");
        return 0;
    }

    err = clGetPlatformIDs(nplatforms, platforms, NULL);
    if (err != CL_SUCCESS) {
        snprintf(err_buf, err_buf_size, "clGetPlatformIDs failed: %s (%d)", cl_error_name(err), err);
        free(platforms);
        return 0;
    }

    const cl_device_type device_types[] = { CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU };
    for (size_t t = 0; t < sizeof(device_types) / sizeof(device_types[0]); t++) {
        for (cl_uint i = 0; i < nplatforms; i++) {
            cl_uint ndevices = 0;
            err = clGetDeviceIDs(platforms[i], device_types[t], 0, NULL, &ndevices);
            if (err != CL_SUCCESS || ndevices == 0) continue;

            cl_device_id* devices = (cl_device_id*)calloc(ndevices, sizeof(cl_device_id));
            if (!devices) {
                snprintf(err_buf, err_buf_size, "out of host memory while listing OpenCL devices");
                free(platforms);
                return 0;
            }

            err = clGetDeviceIDs(platforms[i], device_types[t], ndevices, devices, NULL);
            if (err == CL_SUCCESS) {
                *platform_out = platforms[i];
                *device_out = devices[0];
                free(devices);
                free(platforms);
                return 1;
            }

            free(devices);
        }
    }

    snprintf(err_buf, err_buf_size, "no OpenCL GPU or CPU device found");
    free(platforms);
    return 0;
}

lean_obj_res numlean_opencl_init(lean_obj_arg world) {
    (void)world;

    if (g_numleanopencl_ctx.initialized) {
        return lean_io_result_mk_ok(lean_box(0));
    }

    char err_msg[512];
    if (!choose_device(&g_numleanopencl_ctx.platform, &g_numleanopencl_ctx.device, err_msg, sizeof(err_msg))) {
        return io_error(err_msg);
    }

    cl_int err;
    g_numleanopencl_ctx.context = clCreateContext(
        NULL, 1, &g_numleanopencl_ctx.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        snprintf(err_msg, sizeof(err_msg), "clCreateContext failed: %s (%d)", cl_error_name(err), err);
        return io_error(err_msg);
    }

    g_numleanopencl_ctx.queue = clCreateCommandQueueWithProperties(
        g_numleanopencl_ctx.context, g_numleanopencl_ctx.device, 0, &err);
    if (err != CL_SUCCESS) {
        snprintf(err_msg, sizeof(err_msg), "clCreateCommandQueueWithProperties failed: %s (%d)", cl_error_name(err), err);
        clReleaseContext(g_numleanopencl_ctx.context);
        memset(&g_numleanopencl_ctx, 0, sizeof(g_numleanopencl_ctx));
        return io_error(err_msg);
    }

    clGetPlatformInfo(g_numleanopencl_ctx.platform, CL_PLATFORM_NAME,
                      sizeof(g_numleanopencl_ctx.platform_name),
                      g_numleanopencl_ctx.platform_name, NULL);
    clGetDeviceInfo(g_numleanopencl_ctx.device, CL_DEVICE_NAME,
                    sizeof(g_numleanopencl_ctx.device_name),
                    g_numleanopencl_ctx.device_name, NULL);
    clGetDeviceInfo(g_numleanopencl_ctx.device, CL_DRIVER_VERSION,
                    sizeof(g_numleanopencl_ctx.driver_version),
                    g_numleanopencl_ctx.driver_version, NULL);

    g_numleanopencl_ctx.initialized = 1;
    return lean_io_result_mk_ok(lean_box(0));
}
