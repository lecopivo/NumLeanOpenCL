#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* cl_error_name(cl_int err) {
    switch (err) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    default: return "UNKNOWN_OPENCL_ERROR";
    }
}

#define CHECK_CL(expr) do { \
    cl_int _err = (expr); \
    if (_err != CL_SUCCESS) { \
        fprintf(stderr, "%s failed: %s (%d)\n", #expr, cl_error_name(_err), _err); \
        exit(1); \
    } \
} while (0)

static char* read_file(const char* path, size_t* len_out) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        perror(path);
        exit(1);
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        perror("fseek");
        exit(1);
    }

    long len = ftell(f);
    if (len < 0) {
        perror("ftell");
        exit(1);
    }
    rewind(f);

    char* data = (char*)malloc((size_t)len + 1);
    if (!data) {
        fprintf(stderr, "out of host memory\n");
        exit(1);
    }

    size_t nread = fread(data, 1, (size_t)len, f);
    if (nread != (size_t)len) {
        fprintf(stderr, "short read from %s\n", path);
        exit(1);
    }
    fclose(f);

    data[len] = 0;
    *len_out = (size_t)len;
    return data;
}

static cl_device_id choose_device(cl_platform_id* platform_out) {
    cl_uint nplatforms = 0;
    CHECK_CL(clGetPlatformIDs(0, NULL, &nplatforms));
    if (nplatforms == 0) {
        fprintf(stderr, "no OpenCL platforms found\n");
        exit(1);
    }

    cl_platform_id* platforms = (cl_platform_id*)calloc(nplatforms, sizeof(cl_platform_id));
    CHECK_CL(clGetPlatformIDs(nplatforms, platforms, NULL));

    for (cl_uint want_gpu = 1; want_gpu < 3; want_gpu++) {
        cl_device_type type = want_gpu == 1 ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
        for (cl_uint i = 0; i < nplatforms; i++) {
            cl_uint ndevices = 0;
            cl_int err = clGetDeviceIDs(platforms[i], type, 0, NULL, &ndevices);
            if (err != CL_SUCCESS || ndevices == 0) continue;

            cl_device_id* devices = (cl_device_id*)calloc(ndevices, sizeof(cl_device_id));
            CHECK_CL(clGetDeviceIDs(platforms[i], type, ndevices, devices, NULL));
            cl_device_id device = devices[0];
            free(devices);
            *platform_out = platforms[i];
            free(platforms);
            return device;
        }
    }

    free(platforms);
    fprintf(stderr, "no OpenCL GPU or CPU devices found\n");
    exit(1);
}

int main(void) {
    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = choose_device(&platform);

    char platform_name[256] = {0};
    char device_name[256] = {0};
    char device_version[256] = {0};
    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(device_version), device_version, NULL);

    printf("OpenCL platform: %s\n", platform_name);
    printf("OpenCL device:   %s\n", device_name);
    printf("OpenCL version:  %s\n", device_version);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err);

    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    CHECK_CL(err);

    size_t src_len = 0;
    char* source = read_file("opencl/smoke_add.cl", &src_len);
    const char* sources[] = { source };

    cl_program program = clCreateProgramWithSource(context, 1, sources, &src_len, &err);
    CHECK_CL(err);

    err = clBuildProgram(program, 1, &device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char* log = (char*)malloc(log_len + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = 0;
        fprintf(stderr, "OpenCL build failed: %s (%d)\n%s\n", cl_error_name(err), err, log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "smoke_add", &err);
    CHECK_CL(err);

    enum { N = 1024 };
    float a[N];
    float b[N];
    float out[N];
    for (size_t i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(2 * i);
        out[i] = 0.0f;
    }

    cl_mem a_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(a), a, &err);
    CHECK_CL(err);
    cl_mem b_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(b), b, &err);
    CHECK_CL(err);
    cl_mem out_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(out), NULL, &err);
    CHECK_CL(err);

    cl_ulong n_arg = N;
    CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buf));
    CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buf));
    CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_buf));
    CHECK_CL(clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n_arg));

    size_t local = 64;
    size_t global = ((N + local - 1) / local) * local;
    CHECK_CL(clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL));
    CHECK_CL(clEnqueueReadBuffer(queue, out_buf, CL_TRUE, 0, sizeof(out), out, 0, NULL, NULL));

    for (size_t i = 0; i < N; i++) {
        float expected = a[i] + b[i];
        if (fabsf(out[i] - expected) > 0.0f) {
            fprintf(stderr, "mismatch at %zu: got %g expected %g\n", i, out[i], expected);
            return 1;
        }
    }

    printf("OpenCL smoke test passed: %d float32 additions\n", N);

    clReleaseMemObject(out_buf);
    clReleaseMemObject(b_buf);
    clReleaseMemObject(a_buf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(source);
    return 0;
}
