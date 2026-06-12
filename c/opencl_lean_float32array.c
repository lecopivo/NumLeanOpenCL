#include "opencl_lean.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static lean_external_class* g_float32array_class = NULL;
static cl_program g_add_program = NULL;
static cl_kernel g_add_kernel = NULL;
static cl_program g_beq_program = NULL;
static cl_kernel g_beq_kernel = NULL;
static cl_program g_sum_program = NULL;
static cl_kernel g_sum_kernel = NULL;
static cl_program g_blas1_program = NULL;
static cl_kernel g_scal_kernel = NULL;
static cl_kernel g_axpy_kernel = NULL;
static cl_kernel g_swap_kernel = NULL;
static cl_kernel g_rot_kernel = NULL;
static cl_kernel g_dot_kernel = NULL;
static cl_kernel g_asum_kernel = NULL;
static cl_kernel g_nrm2_kernel = NULL;

static NumLeanOpenCLFloat32Array* clone_external_array(NumLeanOpenCLFloat32Array* src);

static void float32array_finalize(void* ptr) {
    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)ptr;
    if (xs) {
        if (xs->buffer) {
            clReleaseMemObject(xs->buffer);
        }
        free(xs);
    }
}

static void float32array_foreach(void* ptr, b_lean_obj_arg fn) {
    (void)ptr;
    (void)fn;
}

static void ensure_float32array_class(void) {
    if (!g_float32array_class) {
        g_float32array_class = lean_register_external_class(float32array_finalize, float32array_foreach);
    }
}

static void ensure_opencl_initialized(void) {
    if (!g_numleanopencl_ctx.initialized) {
        lean_object* r = numlean_opencl_init();
        if (lean_io_result_is_error(r)) {
            lean_object* err = lean_io_result_get_error(r);
            lean_inc(err);
            lean_dec(r);
            lean_internal_panic("OpenCL initialization failed");
        }
        lean_dec(r);
    }
}

static NumLeanOpenCLFloat32Array* alloc_record(size_t size, size_t capacity) {
    ensure_opencl_initialized();
    ensure_float32array_class();

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)malloc(sizeof(NumLeanOpenCLFloat32Array));
    if (!xs) lean_internal_panic_out_of_memory();

    xs->buffer = NULL;
    xs->size = size;
    xs->capacity = capacity;

    if (capacity > 0) {
        cl_int err;
        xs->buffer = clCreateBuffer(
            g_numleanopencl_ctx.context,
            CL_MEM_READ_WRITE,
            capacity * sizeof(float),
            NULL,
            &err);
        if (err != CL_SUCCESS) {
            free(xs);
            lean_internal_panic("clCreateBuffer failed for OpenCL Float32Array");
        }
    }

    return xs;
}

static lean_object* box_record(NumLeanOpenCLFloat32Array* xs) {
    return lean_alloc_external(g_float32array_class, xs);
}

static lean_object* mk_ctor_from_array(lean_obj_arg data) {
    lean_object* obj = lean_alloc_ctor(0, 1, 0);
    lean_ctor_set(obj, 0, data);
    return obj;
}

static lean_object* mk_empty_array(void) {
    return lean_alloc_array(0, 0);
}

static char* read_kernel_file(const char* path, size_t* len_out) {
    FILE* f = fopen(path, "rb");
    if (!f) lean_internal_panic("failed to open OpenCL kernel file");

    if (fseek(f, 0, SEEK_END) != 0) lean_internal_panic("failed to seek OpenCL kernel file");
    long len = ftell(f);
    if (len < 0) lean_internal_panic("failed to read OpenCL kernel file length");
    rewind(f);

    char* src = (char*)malloc((size_t)len + 1);
    if (!src) lean_internal_panic_out_of_memory();

    if (fread(src, 1, (size_t)len, f) != (size_t)len) {
        free(src);
        fclose(f);
        lean_internal_panic("failed to read OpenCL kernel file");
    }
    fclose(f);

    src[len] = 0;
    *len_out = (size_t)len;
    return src;
}

static cl_kernel get_add_kernel(void) {
    ensure_opencl_initialized();

    if (g_add_kernel) return g_add_kernel;

    size_t src_len = 0;
    char* src = read_kernel_file("opencl/float32array_add.cl", &src_len);
    const char* sources[] = { src };

    cl_int err;
    g_add_program = clCreateProgramWithSource(
        g_numleanopencl_ctx.context, 1, sources, &src_len, &err);
    free(src);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateProgramWithSource failed for Float32Array.add");

    err = clBuildProgram(g_add_program, 1, &g_numleanopencl_ctx.device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        clGetProgramBuildInfo(g_add_program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char* log = (char*)malloc(log_len + 1);
        if (!log) lean_internal_panic_out_of_memory();
        clGetProgramBuildInfo(g_add_program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = 0;
        lean_internal_panic(log);
    }

    g_add_kernel = clCreateKernel(g_add_program, "numlean_opencl_float32array_add", &err);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateKernel failed for Float32Array.add");
    return g_add_kernel;
}

static cl_kernel get_beq_kernel(void) {
    ensure_opencl_initialized();

    if (g_beq_kernel) return g_beq_kernel;

    size_t src_len = 0;
    char* src = read_kernel_file("opencl/float32array_beq.cl", &src_len);
    const char* sources[] = { src };

    cl_int err;
    g_beq_program = clCreateProgramWithSource(
        g_numleanopencl_ctx.context, 1, sources, &src_len, &err);
    free(src);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateProgramWithSource failed for Float32Array.beq");

    err = clBuildProgram(g_beq_program, 1, &g_numleanopencl_ctx.device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        clGetProgramBuildInfo(g_beq_program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char* log = (char*)malloc(log_len + 1);
        if (!log) lean_internal_panic_out_of_memory();
        clGetProgramBuildInfo(g_beq_program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = 0;
        lean_internal_panic(log);
    }

    g_beq_kernel = clCreateKernel(g_beq_program, "numlean_opencl_float32array_beq", &err);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateKernel failed for Float32Array.beq");
    return g_beq_kernel;
}

static cl_kernel get_sum_kernel(void) {
    ensure_opencl_initialized();

    if (g_sum_kernel) return g_sum_kernel;

    size_t src_len = 0;
    char* src = read_kernel_file("opencl/float32array_sum.cl", &src_len);
    const char* sources[] = { src };

    cl_int err;
    g_sum_program = clCreateProgramWithSource(
        g_numleanopencl_ctx.context, 1, sources, &src_len, &err);
    free(src);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateProgramWithSource failed for Float32ArrayOpenCL.sum");

    err = clBuildProgram(g_sum_program, 1, &g_numleanopencl_ctx.device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        clGetProgramBuildInfo(g_sum_program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char* log = (char*)malloc(log_len + 1);
        if (!log) lean_internal_panic_out_of_memory();
        clGetProgramBuildInfo(g_sum_program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = 0;
        lean_internal_panic(log);
    }

    g_sum_kernel = clCreateKernel(g_sum_program, "numlean_opencl_float32array_sum", &err);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateKernel failed for Float32ArrayOpenCL.sum");
    return g_sum_kernel;
}

static cl_kernel get_blas1_kernel(const char* kernel_name, cl_kernel* cache) {
    ensure_opencl_initialized();

    if (*cache) return *cache;

    if (!g_blas1_program) {
        double compile_start_us = numlean_opencl_profile_now_us();
        size_t src_len = 0;
        char* src = read_kernel_file("opencl/float32array_blas1.cl", &src_len);
        const char* sources[] = { src };

        cl_int err;
        g_blas1_program = clCreateProgramWithSource(
            g_numleanopencl_ctx.context, 1, sources, &src_len, &err);
        free(src);
        if (err != CL_SUCCESS) lean_internal_panic("clCreateProgramWithSource failed for BLAS1 kernels");

        err = clBuildProgram(g_blas1_program, 1, &g_numleanopencl_ctx.device, "-cl-std=CL1.2", NULL, NULL);
        if (err != CL_SUCCESS) {
            size_t log_len = 0;
            clGetProgramBuildInfo(g_blas1_program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
            char* log = (char*)malloc(log_len + 1);
            if (!log) lean_internal_panic_out_of_memory();
            clGetProgramBuildInfo(g_blas1_program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
            log[log_len] = 0;
            lean_internal_panic(log);
        }
        numlean_opencl_profile_record_host("compile/blas1", compile_start_us, numlean_opencl_profile_now_us());
    }

    cl_int err;
    *cache = clCreateKernel(g_blas1_program, kernel_name, &err);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateKernel failed for BLAS1 kernel");
    return *cache;
}

static cl_kernel compile_map_unsafe_kernel(const char* expr, cl_program* program_out) {
    double compile_start_us = numlean_opencl_profile_now_us();
    const char* prefix =
        "__kernel void numlean_opencl_map_unsafe(__global float* xs, ulong n) {\n"
        "    size_t gid = get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        float x = xs[gid];\n"
        "        xs[gid] = ";
    const char* suffix =
        ";\n"
        "    }\n"
        "}\n";

    size_t len = strlen(prefix) + strlen(expr) + strlen(suffix) + 1;
    char* src = (char*)malloc(len);
    if (!src) lean_internal_panic_out_of_memory();
    snprintf(src, len, "%s%s%s", prefix, expr, suffix);

    const char* sources[] = { src };
    size_t lengths[] = { strlen(src) };
    cl_int err;
    cl_program program = clCreateProgramWithSource(
        g_numleanopencl_ctx.context, 1, sources, lengths, &err);
    free(src);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateProgramWithSource failed for mapUnsafe");

    err = clBuildProgram(program, 1, &g_numleanopencl_ctx.device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        clGetProgramBuildInfo(program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char* log = (char*)malloc(log_len + 1);
        if (!log) lean_internal_panic_out_of_memory();
        clGetProgramBuildInfo(program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = 0;
        lean_internal_panic(log);
    }

    cl_kernel kernel = clCreateKernel(program, "numlean_opencl_map_unsafe", &err);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateKernel failed for mapUnsafe");
    numlean_opencl_profile_record_host("compile/mapUnsafe", compile_start_us, numlean_opencl_profile_now_us());
    *program_out = program;
    return kernel;
}

static cl_kernel compile_map_in_context_unsafe_kernel(const char* expr, size_t ctx_count, cl_program* program_out) {
    double compile_start_us = numlean_opencl_profile_now_us();
    const char* prefix =
        "__kernel void numlean_opencl_map_in_context_unsafe(__global float* xs, ulong n";
    const char* body_prefix =
        ") {\n"
        "    size_t gid = get_global_id(0);\n"
        "    if (gid < n) {\n"
        "        float x = xs[gid];\n"
        "        xs[gid] = ";
    const char* suffix =
        ";\n"
        "    }\n"
        "}\n";

    size_t len = strlen(prefix) + strlen(body_prefix) + strlen(expr) + strlen(suffix) + 1;
    for (size_t i = 0; i < ctx_count; i++) {
        len += 96;
    }

    char* src = (char*)malloc(len);
    if (!src) lean_internal_panic_out_of_memory();
    size_t off = 0;
    off += snprintf(src + off, len - off, "%s", prefix);
    for (size_t i = 0; i < ctx_count; i++) {
        off += snprintf(src + off, len - off,
                        ", __global const float* ctx%zu, ulong ctx%zu_size", i, i);
    }
    snprintf(src + off, len - off, "%s%s%s", body_prefix, expr, suffix);

    const char* sources[] = { src };
    size_t lengths[] = { strlen(src) };
    cl_int err;
    cl_program program = clCreateProgramWithSource(
        g_numleanopencl_ctx.context, 1, sources, lengths, &err);
    free(src);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateProgramWithSource failed for mapInContextUnsafe");

    err = clBuildProgram(program, 1, &g_numleanopencl_ctx.device, "-cl-std=CL1.2", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_len = 0;
        clGetProgramBuildInfo(program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_len);
        char* log = (char*)malloc(log_len + 1);
        if (!log) lean_internal_panic_out_of_memory();
        clGetProgramBuildInfo(program, g_numleanopencl_ctx.device, CL_PROGRAM_BUILD_LOG, log_len, log, NULL);
        log[log_len] = 0;
        lean_internal_panic(log);
    }

    cl_kernel kernel = clCreateKernel(program, "numlean_opencl_map_in_context_unsafe", &err);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateKernel failed for mapInContextUnsafe");
    numlean_opencl_profile_record_host("compile/mapInContextUnsafe", compile_start_us, numlean_opencl_profile_now_us());
    *program_out = program;
    return kernel;
}

static lean_object* mk_pair(lean_object* fst, lean_object* snd) {
    lean_object* pair = lean_alloc_ctor(0, 2, 0);
    lean_ctor_set(pair, 0, fst);
    lean_ctor_set(pair, 1, snd);
    return pair;
}

static size_t min_size(size_t a, size_t b) {
    return a < b ? a : b;
}

static NumLeanOpenCLFloat32Array* exclusive_or_clone_array(lean_obj_arg* obj_io) {
    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(*obj_io);
    if (!lean_is_exclusive(*obj_io)) {
        NumLeanOpenCLFloat32Array* cloned = clone_external_array(xs);
        lean_dec(*obj_io);
        *obj_io = box_record(cloned);
        return cloned;
    }
    return xs;
}

static lean_object* scalar_array_from_float(float x) {
    lean_object* out = lean_alloc_array(0, 1);
    out = lean_array_push(out, lean_box_float32(x));
    return numlean_opencl_float32array_mk(out);
}

static lean_object* scalar_array_from_kernel_unary(lean_obj_arg obj, cl_kernel kernel) {
    if (!lean_is_external(obj)) {
        lean_object* data = numlean_opencl_float32array_data(obj);
        size_t n = lean_array_size(data);
        float acc = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float v = lean_unbox_float32(lean_array_get_core(data, i));
            if (kernel == g_asum_kernel) acc += fabsf(v);
            else if (kernel == g_nrm2_kernel) acc += v * v;
            else acc += v;
        }
        if (kernel == g_nrm2_kernel) acc = sqrtf(acc);
        lean_dec(data);
        lean_dec(obj);
        return scalar_array_from_float(acc);
    }

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    NumLeanOpenCLFloat32Array* out = alloc_record(1, 1);

    cl_ulong n_arg = (cl_ulong)xs->size;
    size_t global = 1;
    size_t local = 1;
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &xs->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for BLAS1 unary reduction arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for BLAS1 unary reduction arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_arg);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for BLAS1 unary reduction arg 2");
    err = clEnqueueNDRangeKernel(g_numleanopencl_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    lean_dec(obj);
    if (err != CL_SUCCESS) {
        float32array_finalize(out);
        lean_internal_panic("clEnqueueNDRangeKernel failed for BLAS1 unary reduction");
    }
    return box_record(out);
}

static NumLeanOpenCLFloat32Array* clone_external_array(NumLeanOpenCLFloat32Array* src) {
    NumLeanOpenCLFloat32Array* dst = alloc_record(src->size, src->capacity);
    if (src->size > 0) {
        cl_int err = clEnqueueCopyBuffer(
            g_numleanopencl_ctx.queue,
            src->buffer,
            dst->buffer,
            0,
            0,
            src->size * sizeof(float),
            0,
            NULL,
            NULL);
        if (err != CL_SUCCESS) {
            float32array_finalize(dst);
            lean_internal_panic("clEnqueueCopyBuffer failed while cloning OpenCL Float32Array");
        }
    }
    return dst;
}

lean_obj_res numlean_opencl_float32array_empty_with_capacity(b_lean_obj_arg capacity_obj) {
    size_t capacity = lean_usize_of_nat(capacity_obj);
    if (capacity == 0) {
        return mk_ctor_from_array(mk_empty_array());
    }
    return box_record(alloc_record(0, capacity));
}

lean_obj_res numlean_opencl_float32array_mk(lean_obj_arg data) {
    size_t n = lean_array_size(data);
    NumLeanOpenCLFloat32Array* xs = alloc_record(n, n);

    if (n > 0) {
        float* tmp = (float*)malloc(n * sizeof(float));
        if (!tmp) lean_internal_panic_out_of_memory();

        for (size_t i = 0; i < n; i++) {
            tmp[i] = lean_unbox_float32(lean_array_get_core(data, i));
        }

        cl_event event = NULL;
        cl_int err = clEnqueueWriteBuffer(
            g_numleanopencl_ctx.queue,
            xs->buffer,
            CL_TRUE,
            0,
            n * sizeof(float),
            tmp,
            0,
            NULL,
            &event);
        free(tmp);
        if (err != CL_SUCCESS) {
            if (event) clReleaseEvent(event);
            float32array_finalize(xs);
            lean_dec(data);
            lean_internal_panic("clEnqueueWriteBuffer failed for OpenCL Float32Array.mk");
        }
        numlean_opencl_profile_record("write/ofArray", 0, n * sizeof(float), event);
    }

    lean_dec(data);
    return box_record(xs);
}

lean_obj_res numlean_opencl_float32array_data(b_lean_obj_arg obj) {
    if (!lean_is_external((lean_object*)obj)) {
        lean_object* data = lean_ctor_get((lean_object*)obj, 0);
        lean_inc(data);
        return data;
    }

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    lean_object* data = lean_alloc_array(xs->size, xs->size);

    if (xs->size == 0) {
        return data;
    }

    float* tmp = (float*)malloc(xs->size * sizeof(float));
    if (!tmp) lean_internal_panic_out_of_memory();

    cl_event event = NULL;
    cl_int err = clEnqueueReadBuffer(
        g_numleanopencl_ctx.queue,
        xs->buffer,
        CL_TRUE,
        0,
        xs->size * sizeof(float),
        tmp,
        0,
        NULL,
        &event);
    if (err != CL_SUCCESS) {
        if (event) clReleaseEvent(event);
        free(tmp);
        lean_dec(data);
        lean_internal_panic("clEnqueueReadBuffer failed for OpenCL Float32Array.data");
    }
    numlean_opencl_profile_record("read/toArray", 0, xs->size * sizeof(float), event);

    for (size_t i = 0; i < xs->size; i++) {
        lean_array_cptr(data)[i] = lean_box_float32(tmp[i]);
    }

    free(tmp);
    return data;
}

lean_obj_res numlean_opencl_float32array_to_array(b_lean_obj_arg obj) {
    return numlean_opencl_float32array_data(obj);
}

lean_obj_res numlean_opencl_float32arrayopencl_slice(b_lean_obj_arg obj, b_lean_obj_arg start_obj, b_lean_obj_arg stop_obj) {
    ensure_opencl_initialized();
    size_t start = lean_usize_of_nat(start_obj);
    size_t stop = lean_usize_of_nat(stop_obj);

    if (!lean_is_external((lean_object*)obj)) {
        lean_object* data = lean_ctor_get((lean_object*)obj, 0);
        size_t n = lean_array_size(data);
        if (start > n) start = n;
        if (stop > n) stop = n;
        if (stop < start) stop = start;

        lean_object* out = lean_alloc_array(0, stop - start);
        for (size_t i = start; i < stop; i++) {
            lean_object* x = lean_array_get_core(data, i);
            lean_inc(x);
            out = lean_array_push(out, x);
        }
        return numlean_opencl_float32array_mk(out);
    }

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    if (start > xs->size) start = xs->size;
    if (stop > xs->size) stop = xs->size;
    if (stop < start) stop = start;

    size_t out_size = stop - start;
    NumLeanOpenCLFloat32Array* out = alloc_record(out_size, out_size);
    if (out_size > 0) {
        cl_event event = NULL;
        cl_int err = clEnqueueCopyBuffer(
            g_numleanopencl_ctx.queue,
            xs->buffer,
            out->buffer,
            start * sizeof(float),
            0,
            out_size * sizeof(float),
            0,
            NULL,
            &event);
        if (err != CL_SUCCESS) {
            if (event) clReleaseEvent(event);
            float32array_finalize(out);
            lean_internal_panic("clEnqueueCopyBuffer failed for OpenCL Float32Array.slice");
        }
        numlean_opencl_profile_record("copy/slice", out_size, out_size * sizeof(float), event);
    }
    return box_record(out);
}

lean_obj_res numlean_opencl_float32array_push(lean_obj_arg obj, float x) {
    if (!lean_is_external(obj)) {
        lean_object* data = lean_ctor_get(obj, 0);
        lean_inc(data);
        lean_object* pushed = lean_array_push(data, lean_box_float32(x));
        lean_dec(obj);
        return numlean_opencl_float32array_mk(pushed);
    }

    NumLeanOpenCLFloat32Array* old = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    size_t new_size = old->size + 1;
    size_t new_capacity = old->capacity;
    if (new_capacity < new_size) {
        new_capacity = old->capacity == 0 ? 1 : old->capacity * 2;
        if (new_capacity < new_size) new_capacity = new_size;
    }

    NumLeanOpenCLFloat32Array* xs = alloc_record(new_size, new_capacity);

    if (old->size > 0) {
        cl_int err = clEnqueueCopyBuffer(
            g_numleanopencl_ctx.queue,
            old->buffer,
            xs->buffer,
            0,
            0,
            old->size * sizeof(float),
            0,
            NULL,
            NULL);
        if (err != CL_SUCCESS) {
            float32array_finalize(xs);
            lean_dec(obj);
            lean_internal_panic("clEnqueueCopyBuffer failed for OpenCL Float32Array.push");
        }
    }

    cl_int err = clEnqueueWriteBuffer(
        g_numleanopencl_ctx.queue,
        xs->buffer,
        CL_TRUE,
        old->size * sizeof(float),
        sizeof(float),
        &x,
        0,
        NULL,
        NULL);
    if (err != CL_SUCCESS) {
        float32array_finalize(xs);
        lean_dec(obj);
        lean_internal_panic("clEnqueueWriteBuffer failed for OpenCL Float32Array.push");
    }

    lean_dec(obj);
    return box_record(xs);
}

lean_obj_res numlean_opencl_float32array_size(b_lean_obj_arg obj) {
    if (!lean_is_external((lean_object*)obj)) {
        lean_object* data = lean_ctor_get((lean_object*)obj, 0);
        return lean_usize_to_nat(lean_array_size(data));
    }

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    return lean_usize_to_nat(xs->size);
}

size_t numlean_opencl_float32array_usize(b_lean_obj_arg obj) {
    if (!lean_is_external((lean_object*)obj)) {
        lean_object* data = lean_ctor_get((lean_object*)obj, 0);
        return lean_array_size(data);
    }

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    return xs->size;
}

float numlean_opencl_float32array_get(b_lean_obj_arg obj, b_lean_obj_arg i_obj) {
    size_t i = lean_usize_of_nat(i_obj);

    if (!lean_is_external((lean_object*)obj)) {
        lean_object* data = lean_ctor_get((lean_object*)obj, 0);
        return lean_unbox_float32(lean_array_get_core(data, i));
    }

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    float value = 0.0f;
    cl_int err = clEnqueueReadBuffer(
        g_numleanopencl_ctx.queue,
        xs->buffer,
        CL_TRUE,
        i * sizeof(float),
        sizeof(float),
        &value,
        0,
        NULL,
        NULL);
    if (err != CL_SUCCESS) {
        lean_internal_panic("clEnqueueReadBuffer failed for OpenCL Float32Array.get");
    }
    return value;
}

float numlean_opencl_float32array_get_bang(b_lean_obj_arg obj, b_lean_obj_arg i_obj) {
    return numlean_opencl_float32array_get(obj, i_obj);
}

uint8_t numlean_opencl_float32array_beq(lean_obj_arg a_obj, lean_obj_arg b_obj) {
    if (!lean_is_external(a_obj) || !lean_is_external(b_obj)) {
        lean_object* a_data = numlean_opencl_float32array_data(a_obj);
        lean_object* b_data = numlean_opencl_float32array_data(b_obj);
        size_t n = lean_array_size(a_data);
        uint8_t result = 1;

        if (n != lean_array_size(b_data)) {
            result = 0;
        } else {
            for (size_t i = 0; i < n; i++) {
                float a = lean_unbox_float32(lean_array_get_core(a_data, i));
                float b = lean_unbox_float32(lean_array_get_core(b_data, i));
                if (a != b) {
                    result = 0;
                    break;
                }
            }
        }

        lean_dec(a_data);
        lean_dec(b_data);
        lean_dec(a_obj);
        lean_dec(b_obj);
        return result;
    }

    NumLeanOpenCLFloat32Array* a = (NumLeanOpenCLFloat32Array*)lean_get_external_data(a_obj);
    NumLeanOpenCLFloat32Array* b = (NumLeanOpenCLFloat32Array*)lean_get_external_data(b_obj);
    if (a->size != b->size) {
        lean_dec(a_obj);
        lean_dec(b_obj);
        return 0;
    }

    if (a->size == 0) {
        lean_dec(a_obj);
        lean_dec(b_obj);
        return 1;
    }

    cl_uint initial = 1;
    cl_int err;
    cl_mem out = clCreateBuffer(
        g_numleanopencl_ctx.context,
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        sizeof(cl_uint),
        &initial,
        &err);
    if (err != CL_SUCCESS) lean_internal_panic("clCreateBuffer failed for Float32Array.beq result");

    cl_kernel kernel = get_beq_kernel();
    cl_ulong n_arg = (cl_ulong)a->size;
    size_t local = 256;
    size_t global = ((a->size + local - 1) / local) * local;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32Array.beq arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32Array.beq arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &out);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32Array.beq arg 2");
    err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n_arg);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32Array.beq arg 3");

    err = clEnqueueNDRangeKernel(
        g_numleanopencl_ctx.queue,
        kernel,
        1,
        NULL,
        &global,
        &local,
        0,
        NULL,
        NULL);
    if (err != CL_SUCCESS) lean_internal_panic("clEnqueueNDRangeKernel failed for Float32Array.beq");

    cl_uint result = 0;
    err = clEnqueueReadBuffer(
        g_numleanopencl_ctx.queue,
        out,
        CL_TRUE,
        0,
        sizeof(cl_uint),
        &result,
        0,
        NULL,
        NULL);
    clReleaseMemObject(out);
    lean_dec(a_obj);
    lean_dec(b_obj);

    if (err != CL_SUCCESS) lean_internal_panic("clEnqueueReadBuffer failed for Float32Array.beq");
    return result != 0;
}

lean_obj_res numlean_opencl_float32array_add(lean_obj_arg a_obj, lean_obj_arg b_obj) {
    if (!lean_is_external(a_obj) || !lean_is_external(b_obj)) {
        lean_object* a_data = numlean_opencl_float32array_data(a_obj);
        lean_object* b_data = numlean_opencl_float32array_data(b_obj);

        size_t a_size = lean_array_size(a_data);
        size_t b_size = lean_array_size(b_data);
        size_t stop = a_size < b_size ? a_size : b_size;

        for (size_t i = 0; i < stop; i++) {
            float a = lean_unbox_float32(lean_array_get_core(a_data, i));
            float b = lean_unbox_float32(lean_array_get_core(b_data, i));
            a_data = lean_array_uset(a_data, i, lean_box_float32(a + b));
        }

        lean_dec(b_data);
        lean_dec(a_obj);
        lean_dec(b_obj);
        return numlean_opencl_float32array_mk(a_data);
    }

    NumLeanOpenCLFloat32Array* b = (NumLeanOpenCLFloat32Array*)lean_get_external_data(b_obj);
    NumLeanOpenCLFloat32Array* a = (NumLeanOpenCLFloat32Array*)lean_get_external_data(a_obj);
    size_t stop = a->size < b->size ? a->size : b->size;

    if (!lean_is_exclusive(a_obj)) {
        NumLeanOpenCLFloat32Array* cloned = clone_external_array(a);
        lean_dec(a_obj);
        a_obj = box_record(cloned);
        a = cloned;
    }

    if (stop > 0) {
        cl_kernel kernel = get_add_kernel();
        cl_ulong n_arg = (cl_ulong)stop;
        size_t local = 256;
        size_t global = ((stop + local - 1) / local) * local;

        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32Array.add arg 0");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32Array.add arg 1");
        err = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_arg);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32Array.add arg 2");

        err = clEnqueueNDRangeKernel(
            g_numleanopencl_ctx.queue,
            kernel,
            1,
            NULL,
            &global,
            &local,
            0,
            NULL,
            NULL);
        if (err != CL_SUCCESS) lean_internal_panic("clEnqueueNDRangeKernel failed for Float32Array.add");
    }

    lean_dec(b_obj);
    return a_obj;
}

lean_obj_res numlean_opencl_float32arrayopencl_mk(lean_obj_arg data) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_mk(data);
}

lean_obj_res numlean_opencl_float32arrayopencl_of_array(lean_obj_arg data) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_mk(data);
}

lean_obj_res numlean_opencl_float32arrayopencl_data(b_lean_obj_arg xs) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_data(xs);
}

lean_obj_res numlean_opencl_float32arrayopencl_to_array(b_lean_obj_arg xs) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_to_array(xs);
}

lean_obj_res numlean_opencl_float32arrayopencl_empty_with_capacity(b_lean_obj_arg capacity) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_empty_with_capacity(capacity);
}

lean_obj_res numlean_opencl_float32arrayopencl_push(lean_obj_arg xs, float x) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_push(xs, x);
}

lean_obj_res numlean_opencl_float32arrayopencl_size(b_lean_obj_arg xs) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_size(xs);
}

size_t numlean_opencl_float32arrayopencl_usize(b_lean_obj_arg xs) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_usize(xs);
}

float numlean_opencl_float32arrayopencl_get(b_lean_obj_arg xs, b_lean_obj_arg i) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_get(xs, i);
}

float numlean_opencl_float32arrayopencl_get_bang(b_lean_obj_arg xs, b_lean_obj_arg i) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_get_bang(xs, i);
}

uint8_t numlean_opencl_float32arrayopencl_beq(lean_obj_arg a, lean_obj_arg b) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_beq(a, b);
}

lean_obj_res numlean_opencl_float32arrayopencl_add(lean_obj_arg a, lean_obj_arg b) {
    ensure_opencl_initialized();
    return numlean_opencl_float32array_add(a, b);
}

lean_obj_res numlean_opencl_float32arrayopencl_sum(lean_obj_arg obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(obj)) {
        lean_object* data = numlean_opencl_float32array_data(obj);
        size_t n = lean_array_size(data);
        float acc = 0.0f;
        for (size_t i = 0; i < n; i++) {
            acc += lean_unbox_float32(lean_array_get_core(data, i));
        }
        lean_dec(data);
        lean_dec(obj);

        lean_object* out = lean_alloc_array(0, 1);
        out = lean_array_push(out, lean_box_float32(acc));
        return numlean_opencl_float32array_mk(out);
    }

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    NumLeanOpenCLFloat32Array* out = alloc_record(1, 1);

    if (xs->size == 0) {
        float zero = 0.0f;
        cl_int err = clEnqueueWriteBuffer(
            g_numleanopencl_ctx.queue,
            out->buffer,
            CL_FALSE,
            0,
            sizeof(float),
            &zero,
            0,
            NULL,
            NULL);
        lean_dec(obj);
        if (err != CL_SUCCESS) {
            float32array_finalize(out);
            lean_internal_panic("clEnqueueWriteBuffer failed for Float32ArrayOpenCL.sum empty case");
        }
        return box_record(out);
    }

    cl_kernel kernel = get_sum_kernel();
    cl_ulong n_arg = (cl_ulong)xs->size;
    size_t global = 1;
    size_t local = 1;

    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &xs->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32ArrayOpenCL.sum arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &out->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32ArrayOpenCL.sum arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_arg);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for Float32ArrayOpenCL.sum arg 2");

    err = clEnqueueNDRangeKernel(
        g_numleanopencl_ctx.queue,
        kernel,
        1,
        NULL,
        &global,
        &local,
        0,
        NULL,
        NULL);
    lean_dec(obj);
    if (err != CL_SUCCESS) {
        float32array_finalize(out);
        lean_internal_panic("clEnqueueNDRangeKernel failed for Float32ArrayOpenCL.sum");
    }

    return box_record(out);
}

lean_obj_res numlean_opencl_float32opencl_of_float32(float x) {
    ensure_opencl_initialized();
    lean_object* obj = numlean_opencl_float32array_empty_with_capacity(lean_unsigned_to_nat(1));
    return numlean_opencl_float32array_push(obj, x);
}

lean_obj_res numlean_opencl_float32arrayopencl_copy(lean_obj_arg obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(obj)) {
        lean_object* data = numlean_opencl_float32array_data(obj);
        lean_dec(obj);
        return numlean_opencl_float32array_mk(data);
    }

    NumLeanOpenCLFloat32Array* xs = (NumLeanOpenCLFloat32Array*)lean_get_external_data(obj);
    NumLeanOpenCLFloat32Array* copied = clone_external_array(xs);
    lean_dec(obj);
    return box_record(copied);
}

lean_obj_res numlean_opencl_float32arrayopencl_scal(float alpha, lean_obj_arg obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(obj)) {
        lean_object* data = numlean_opencl_float32array_data(obj);
        size_t n = lean_array_size(data);
        for (size_t i = 0; i < n; i++) {
            float v = lean_unbox_float32(lean_array_get_core(data, i));
            data = lean_array_uset(data, i, lean_box_float32(alpha * v));
        }
        lean_dec(obj);
        return numlean_opencl_float32array_mk(data);
    }

    NumLeanOpenCLFloat32Array* xs = exclusive_or_clone_array(&obj);
    if (xs->size > 0) {
        cl_kernel kernel = get_blas1_kernel("numlean_opencl_blas1_scal", &g_scal_kernel);
        cl_ulong n_arg = (cl_ulong)xs->size;
        size_t local = 256;
        size_t global = ((xs->size + local - 1) / local) * local;
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &xs->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for scal arg 0");
        err = clSetKernelArg(kernel, 1, sizeof(float), &alpha);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for scal arg 1");
        err = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_arg);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for scal arg 2");
        cl_event event = NULL;
        err = clEnqueueNDRangeKernel(g_numleanopencl_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, &event);
        if (err != CL_SUCCESS) lean_internal_panic("clEnqueueNDRangeKernel failed for scal");
        numlean_opencl_profile_record("kernel/scal", xs->size, 0, event);
    }
    return obj;
}

lean_obj_res numlean_opencl_float32arrayopencl_map_unsafe(lean_obj_arg obj, b_lean_obj_arg expr_obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(obj)) {
        lean_object* data = numlean_opencl_float32array_data(obj);
        lean_dec(obj);
        obj = numlean_opencl_float32array_mk(data);
    }

    NumLeanOpenCLFloat32Array* xs = exclusive_or_clone_array(&obj);

    if (xs->size > 0) {
        const char* expr = lean_string_cstr(expr_obj);
        cl_program program = NULL;
        cl_kernel kernel = compile_map_unsafe_kernel(expr, &program);

        cl_ulong n_arg = (cl_ulong)xs->size;
        size_t local = 256;
        size_t global = ((xs->size + local - 1) / local) * local;
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &xs->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for mapUnsafe arg 0");
        err = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &n_arg);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for mapUnsafe arg 1");
        cl_event event = NULL;
        err = clEnqueueNDRangeKernel(g_numleanopencl_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, &event);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        if (err != CL_SUCCESS) {
            if (event) clReleaseEvent(event);
            lean_internal_panic("clEnqueueNDRangeKernel failed for mapUnsafe");
        }
        numlean_opencl_profile_record("kernel/mapUnsafe", xs->size, 0, event);
    }

    return obj;
}

lean_obj_res numlean_opencl_float32arrayopencl_map_in_context_unsafe(lean_obj_arg obj, b_lean_obj_arg ctx_obj, b_lean_obj_arg expr_obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(obj)) {
        lean_object* data = numlean_opencl_float32array_data(obj);
        lean_dec(obj);
        obj = numlean_opencl_float32array_mk(data);
    }

    size_t ctx_count = lean_array_size(ctx_obj);
    lean_object** ctx_objects = NULL;
    if (ctx_count > 0) {
        ctx_objects = (lean_object**)malloc(ctx_count * sizeof(lean_object*));
        if (!ctx_objects) lean_internal_panic_out_of_memory();
    }

    for (size_t i = 0; i < ctx_count; i++) {
        lean_object* ctx = lean_array_get_core(ctx_obj, i);
        lean_inc(ctx);
        if (!lean_is_external(ctx)) {
            lean_object* data = numlean_opencl_float32array_data(ctx);
            lean_dec(ctx);
            ctx = numlean_opencl_float32array_mk(data);
        }
        ctx_objects[i] = ctx;
    }

    NumLeanOpenCLFloat32Array* xs = exclusive_or_clone_array(&obj);

    if (xs->size > 0) {
        const char* expr = lean_string_cstr(expr_obj);
        cl_program program = NULL;
        cl_kernel kernel = compile_map_in_context_unsafe_kernel(expr, ctx_count, &program);

        cl_ulong n_arg = (cl_ulong)xs->size;
        size_t local = 256;
        size_t global = ((xs->size + local - 1) / local) * local;
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &xs->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for mapInContextUnsafe arg xs");
        err = clSetKernelArg(kernel, 1, sizeof(cl_ulong), &n_arg);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for mapInContextUnsafe arg n");

        unsigned arg = 2;
        for (size_t i = 0; i < ctx_count; i++) {
            NumLeanOpenCLFloat32Array* ctx = (NumLeanOpenCLFloat32Array*)lean_get_external_data(ctx_objects[i]);
            cl_ulong ctx_size = (cl_ulong)ctx->size;
            err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &ctx->buffer);
            if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for mapInContextUnsafe context buffer");
            err = clSetKernelArg(kernel, arg++, sizeof(cl_ulong), &ctx_size);
            if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for mapInContextUnsafe context size");
        }

        cl_event event = NULL;
        err = clEnqueueNDRangeKernel(g_numleanopencl_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, &event);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        if (err != CL_SUCCESS) {
            if (event) clReleaseEvent(event);
            lean_internal_panic("clEnqueueNDRangeKernel failed for mapInContextUnsafe");
        }
        numlean_opencl_profile_record("kernel/mapInContextUnsafe", xs->size, 0, event);
    }

    for (size_t i = 0; i < ctx_count; i++) {
        lean_dec(ctx_objects[i]);
    }
    free(ctx_objects);
    return obj;
}

lean_obj_res numlean_opencl_float32arrayopencl_axpy(float alpha, lean_obj_arg x_obj, lean_obj_arg y_obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(x_obj) || !lean_is_external(y_obj)) {
        lean_object* x_data = numlean_opencl_float32array_data(x_obj);
        lean_object* y_data = numlean_opencl_float32array_data(y_obj);
        size_t stop = min_size(lean_array_size(x_data), lean_array_size(y_data));
        for (size_t i = 0; i < stop; i++) {
            float x = lean_unbox_float32(lean_array_get_core(x_data, i));
            float y = lean_unbox_float32(lean_array_get_core(y_data, i));
            y_data = lean_array_uset(y_data, i, lean_box_float32(y + alpha * x));
        }
        lean_dec(x_data);
        lean_dec(x_obj);
        lean_dec(y_obj);
        return numlean_opencl_float32array_mk(y_data);
    }

    NumLeanOpenCLFloat32Array* x = (NumLeanOpenCLFloat32Array*)lean_get_external_data(x_obj);
    NumLeanOpenCLFloat32Array* y = exclusive_or_clone_array(&y_obj);
    size_t stop = min_size(x->size, y->size);
    if (stop > 0) {
        cl_kernel kernel = get_blas1_kernel("numlean_opencl_blas1_axpy", &g_axpy_kernel);
        cl_ulong n_arg = (cl_ulong)stop;
        size_t local = 256;
        size_t global = ((stop + local - 1) / local) * local;
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for axpy arg 0");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &y->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for axpy arg 1");
        err = clSetKernelArg(kernel, 2, sizeof(float), &alpha);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for axpy arg 2");
        err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n_arg);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for axpy arg 3");
        cl_event event = NULL;
        err = clEnqueueNDRangeKernel(g_numleanopencl_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, &event);
        if (err != CL_SUCCESS) lean_internal_panic("clEnqueueNDRangeKernel failed for axpy");
        numlean_opencl_profile_record("kernel/axpy", stop, 0, event);
    }
    lean_dec(x_obj);
    return y_obj;
}

lean_obj_res numlean_opencl_float32arrayopencl_swap(lean_obj_arg x_obj, lean_obj_arg y_obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(x_obj) || !lean_is_external(y_obj)) {
        lean_object* x_data = numlean_opencl_float32array_data(x_obj);
        lean_object* y_data = numlean_opencl_float32array_data(y_obj);
        size_t stop = min_size(lean_array_size(x_data), lean_array_size(y_data));
        for (size_t i = 0; i < stop; i++) {
            float x = lean_unbox_float32(lean_array_get_core(x_data, i));
            float y = lean_unbox_float32(lean_array_get_core(y_data, i));
            x_data = lean_array_uset(x_data, i, lean_box_float32(y));
            y_data = lean_array_uset(y_data, i, lean_box_float32(x));
        }
        lean_dec(x_obj);
        lean_dec(y_obj);
        return mk_pair(numlean_opencl_float32array_mk(x_data), numlean_opencl_float32array_mk(y_data));
    }

    NumLeanOpenCLFloat32Array* x = exclusive_or_clone_array(&x_obj);
    NumLeanOpenCLFloat32Array* y = exclusive_or_clone_array(&y_obj);
    size_t stop = min_size(x->size, y->size);
    if (stop > 0) {
        cl_kernel kernel = get_blas1_kernel("numlean_opencl_blas1_swap", &g_swap_kernel);
        cl_ulong n_arg = (cl_ulong)stop;
        size_t local = 256;
        size_t global = ((stop + local - 1) / local) * local;
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for swap arg 0");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &y->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for swap arg 1");
        err = clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_arg);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for swap arg 2");
        err = clEnqueueNDRangeKernel(g_numleanopencl_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) lean_internal_panic("clEnqueueNDRangeKernel failed for swap");
    }
    return mk_pair(x_obj, y_obj);
}

lean_obj_res numlean_opencl_float32arrayopencl_rot(float c, float s, lean_obj_arg x_obj, lean_obj_arg y_obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(x_obj) || !lean_is_external(y_obj)) {
        lean_object* x_data = numlean_opencl_float32array_data(x_obj);
        lean_object* y_data = numlean_opencl_float32array_data(y_obj);
        size_t stop = min_size(lean_array_size(x_data), lean_array_size(y_data));
        for (size_t i = 0; i < stop; i++) {
            float x = lean_unbox_float32(lean_array_get_core(x_data, i));
            float y = lean_unbox_float32(lean_array_get_core(y_data, i));
            x_data = lean_array_uset(x_data, i, lean_box_float32(c * x + s * y));
            y_data = lean_array_uset(y_data, i, lean_box_float32(c * y - s * x));
        }
        lean_dec(x_obj);
        lean_dec(y_obj);
        return mk_pair(numlean_opencl_float32array_mk(x_data), numlean_opencl_float32array_mk(y_data));
    }

    NumLeanOpenCLFloat32Array* x = exclusive_or_clone_array(&x_obj);
    NumLeanOpenCLFloat32Array* y = exclusive_or_clone_array(&y_obj);
    size_t stop = min_size(x->size, y->size);
    if (stop > 0) {
        cl_kernel kernel = get_blas1_kernel("numlean_opencl_blas1_rot", &g_rot_kernel);
        cl_ulong n_arg = (cl_ulong)stop;
        size_t local = 256;
        size_t global = ((stop + local - 1) / local) * local;
        cl_int err;
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for rot arg 0");
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &y->buffer);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for rot arg 1");
        err = clSetKernelArg(kernel, 2, sizeof(float), &c);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for rot arg 2");
        err = clSetKernelArg(kernel, 3, sizeof(float), &s);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for rot arg 3");
        err = clSetKernelArg(kernel, 4, sizeof(cl_ulong), &n_arg);
        if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for rot arg 4");
        err = clEnqueueNDRangeKernel(g_numleanopencl_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) lean_internal_panic("clEnqueueNDRangeKernel failed for rot");
    }
    return mk_pair(x_obj, y_obj);
}

lean_obj_res numlean_opencl_float32arrayopencl_dot(lean_obj_arg x_obj, lean_obj_arg y_obj) {
    ensure_opencl_initialized();
    if (!lean_is_external(x_obj) || !lean_is_external(y_obj)) {
        lean_object* x_data = numlean_opencl_float32array_data(x_obj);
        lean_object* y_data = numlean_opencl_float32array_data(y_obj);
        size_t stop = min_size(lean_array_size(x_data), lean_array_size(y_data));
        float acc = 0.0f;
        for (size_t i = 0; i < stop; i++) {
            acc += lean_unbox_float32(lean_array_get_core(x_data, i)) * lean_unbox_float32(lean_array_get_core(y_data, i));
        }
        lean_dec(x_data);
        lean_dec(y_data);
        lean_dec(x_obj);
        lean_dec(y_obj);
        return scalar_array_from_float(acc);
    }

    NumLeanOpenCLFloat32Array* x = (NumLeanOpenCLFloat32Array*)lean_get_external_data(x_obj);
    NumLeanOpenCLFloat32Array* y = (NumLeanOpenCLFloat32Array*)lean_get_external_data(y_obj);
    NumLeanOpenCLFloat32Array* out = alloc_record(1, 1);
    cl_kernel kernel = get_blas1_kernel("numlean_opencl_blas1_dot", &g_dot_kernel);
    cl_ulong n_arg = (cl_ulong)min_size(x->size, y->size);
    size_t global = 1;
    size_t local = 1;
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &x->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for dot arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &y->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for dot arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &out->buffer);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for dot arg 2");
    err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &n_arg);
    if (err != CL_SUCCESS) lean_internal_panic("clSetKernelArg failed for dot arg 3");
    err = clEnqueueNDRangeKernel(g_numleanopencl_ctx.queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    lean_dec(x_obj);
    lean_dec(y_obj);
    if (err != CL_SUCCESS) {
        float32array_finalize(out);
        lean_internal_panic("clEnqueueNDRangeKernel failed for dot");
    }
    return box_record(out);
}

lean_obj_res numlean_opencl_float32arrayopencl_asum(lean_obj_arg obj) {
    ensure_opencl_initialized();
    cl_kernel kernel = get_blas1_kernel("numlean_opencl_blas1_asum", &g_asum_kernel);
    return scalar_array_from_kernel_unary(obj, kernel);
}

lean_obj_res numlean_opencl_float32arrayopencl_nrm2(lean_obj_arg obj) {
    ensure_opencl_initialized();
    cl_kernel kernel = get_blas1_kernel("numlean_opencl_blas1_nrm2", &g_nrm2_kernel);
    return scalar_array_from_kernel_unary(obj, kernel);
}
