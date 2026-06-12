#include "opencl_lean.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

NumLeanOpenCLContext g_numleanopencl_ctx = {0};

static int g_profile_enabled = 0;
static NumLeanOpenCLProfileEvent* g_profile_events = NULL;
static size_t g_profile_count = 0;
static size_t g_profile_capacity = 0;
static size_t g_profile_record_calls = 0;
static size_t g_profile_null_events = 0;
static double g_profile_host_t0_us = 0.0;

double numlean_opencl_profile_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000000.0 + (double)ts.tv_nsec / 1000.0;
}

static double profile_host_relative_us(double t) {
    return g_profile_host_t0_us == 0.0 ? 0.0 : t - g_profile_host_t0_us;
}

static NumLeanOpenCLProfileEvent* profile_alloc_event(void) {
    if (g_profile_count == g_profile_capacity) {
        size_t new_capacity = g_profile_capacity == 0 ? 64 : g_profile_capacity * 2;
        NumLeanOpenCLProfileEvent* new_events =
            (NumLeanOpenCLProfileEvent*)realloc(g_profile_events, new_capacity * sizeof(NumLeanOpenCLProfileEvent));
        if (!new_events) lean_internal_panic_out_of_memory();
        g_profile_events = new_events;
        g_profile_capacity = new_capacity;
    }
    return &g_profile_events[g_profile_count++];
}

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

lean_obj_res numlean_opencl_init(void) {
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

    const cl_queue_properties queue_props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    g_numleanopencl_ctx.queue = clCreateCommandQueueWithProperties(
        g_numleanopencl_ctx.context, g_numleanopencl_ctx.device, queue_props, &err);
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

void numlean_opencl_profile_record(const char* label, size_t work_items, size_t bytes, cl_event event) {
    g_profile_record_calls++;
    if (!g_profile_enabled) {
        if (event) clReleaseEvent(event);
        return;
    }

    if (!event) {
        g_profile_null_events++;
        if (g_numleanopencl_ctx.initialized) {
            cl_int err = clEnqueueMarkerWithWaitList(g_numleanopencl_ctx.queue, 0, NULL, &event);
            if (err != CL_SUCCESS || !event) return;
        } else {
            return;
        }
    }

    NumLeanOpenCLProfileEvent* dst = profile_alloc_event();
    dst->is_host = 0;
    snprintf(dst->label, sizeof(dst->label), "%s", label);
    dst->work_items = work_items;
    dst->bytes = bytes;
    dst->event = event;
    dst->host_start_us = 0.0;
    dst->host_end_us = 0.0;
}

void numlean_opencl_profile_record_host(const char* label, double start_us, double end_us) {
    g_profile_record_calls++;
    if (!g_profile_enabled) return;
    NumLeanOpenCLProfileEvent* dst = profile_alloc_event();
    dst->is_host = 1;
    snprintf(dst->label, sizeof(dst->label), "%s", label);
    dst->work_items = 0;
    dst->bytes = 0;
    dst->event = NULL;
    dst->host_start_us = profile_host_relative_us(start_us);
    dst->host_end_us = profile_host_relative_us(end_us);
}

lean_obj_res numlean_opencl_profile_start(void) {
    g_profile_enabled = 1;
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res numlean_opencl_profile_stop(void) {
    g_profile_enabled = 0;
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res numlean_opencl_profile_clear(void) {
    for (size_t i = 0; i < g_profile_count; i++) {
        if (g_profile_events[i].event) clReleaseEvent(g_profile_events[i].event);
    }
    g_profile_count = 0;
    g_profile_record_calls = 0;
    g_profile_null_events = 0;
    g_profile_host_t0_us = numlean_opencl_profile_now_us();
    g_profile_enabled = 1;
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res numlean_opencl_profile_mark(b_lean_obj_arg label) {
    if (!g_numleanopencl_ctx.initialized) {
        lean_object* r = numlean_opencl_init();
        if (lean_io_result_is_error(r)) return r;
        lean_dec(r);
    }
    cl_event event = NULL;
    cl_int err = clEnqueueMarkerWithWaitList(g_numleanopencl_ctx.queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        return io_error("clEnqueueMarkerWithWaitList failed for OpenCL profile mark");
    }
    numlean_opencl_profile_record(lean_string_cstr(label), 0, 0, event);
    return lean_io_result_mk_ok(lean_box(0));
}

lean_obj_res numlean_opencl_profile_dump(void) {
    size_t cap = 4096 + g_profile_count * 256;
    char* out = (char*)malloc(cap);
    if (!out) lean_internal_panic_out_of_memory();

    size_t pos = 0;
    pos += snprintf(out + pos, cap - pos,
        "OpenCL profile enabled: %d\n"
        "OpenCL profile record calls: %zu\n"
        "OpenCL profile null enqueue events: %zu\n"
        "OpenCL profile events: %zu\n"
        "idx,kind,label,work_items,bytes,queued_us,submit_us,start_us,end_us,queue_to_submit_us,submit_to_start_us,duration_us,host_start_us,host_end_us,host_duration_us\n",
        g_profile_enabled, g_profile_record_calls, g_profile_null_events, g_profile_count);

    for (size_t i = 0; i < g_profile_count; i++) {
        NumLeanOpenCLProfileEvent* ev = &g_profile_events[i];
        if (ev->is_host) {
            double duration_us = ev->host_end_us - ev->host_start_us;
            int written = snprintf(out + pos, cap - pos,
                "%zu,host,%s,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
                i, ev->label, ev->work_items, ev->bytes,
                ev->host_start_us, ev->host_start_us, ev->host_start_us, ev->host_end_us,
                0.0, 0.0, duration_us,
                ev->host_start_us, ev->host_end_us, duration_us);
            if (written < 0) break;
            pos += (size_t)written;
            if (pos >= cap) {
                cap *= 2;
                char* resized = (char*)realloc(out, cap);
                if (!resized) {
                    free(out);
                    lean_internal_panic_out_of_memory();
                }
                out = resized;
            }
            continue;
        }

        clWaitForEvents(1, &ev->event);

        cl_ulong queued = 0, submit = 0, start = 0, end = 0;
        clGetEventProfilingInfo(ev->event, CL_PROFILING_COMMAND_QUEUED, sizeof(queued), &queued, NULL);
        clGetEventProfilingInfo(ev->event, CL_PROFILING_COMMAND_SUBMIT, sizeof(submit), &submit, NULL);
        clGetEventProfilingInfo(ev->event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
        clGetEventProfilingInfo(ev->event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

        cl_ulong q0 = g_profile_count > 0 ? 0 : queued;
        if (i == 0) q0 = queued;
        else {
            clGetEventProfilingInfo(g_profile_events[0].event, CL_PROFILING_COMMAND_QUEUED, sizeof(q0), &q0, NULL);
        }

        double queued_us = (double)(queued - q0) / 1000.0;
        double submit_us = (double)(submit - q0) / 1000.0;
        double start_us = (double)(start - q0) / 1000.0;
        double end_us = (double)(end - q0) / 1000.0;
        double queue_to_submit_us = (double)(submit - queued) / 1000.0;
        double submit_to_start_us = (double)(start - submit) / 1000.0;
        double duration_us = (double)(end - start) / 1000.0;

        int written = snprintf(out + pos, cap - pos,
            "%zu,device,%s,%zu,%zu,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n",
            i, ev->label, ev->work_items, ev->bytes,
            queued_us, submit_us, start_us, end_us,
            queue_to_submit_us, submit_to_start_us, duration_us,
            0.0, 0.0, 0.0);
        if (written < 0) break;
        pos += (size_t)written;
        if (pos >= cap) {
            cap *= 2;
            char* resized = (char*)realloc(out, cap);
            if (!resized) {
                free(out);
                lean_internal_panic_out_of_memory();
            }
            out = resized;
        }
    }

    lean_object* s = lean_mk_string(out);
    free(out);
    return lean_io_result_mk_ok(s);
}
