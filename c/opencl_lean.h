#ifndef NUMLEANOPENCL_OPENCL_LEAN_H
#define NUMLEANOPENCL_OPENCL_LEAN_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>
#include <lean/lean.h>

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    char platform_name[256];
    char device_name[256];
    char driver_version[256];
    int initialized;
} NumLeanOpenCLContext;

extern NumLeanOpenCLContext g_numleanopencl_ctx;

typedef struct {
    cl_mem buffer;
    size_t size;
    size_t capacity;
} NumLeanOpenCLFloat32Array;

lean_obj_res numlean_opencl_init(lean_obj_arg world);
lean_obj_res numlean_opencl_float32opencl_of_float32(float x);

lean_obj_res numlean_opencl_float32array_mk(lean_obj_arg data);
lean_obj_res numlean_opencl_float32array_data(b_lean_obj_arg xs);
lean_obj_res numlean_opencl_float32array_to_array(b_lean_obj_arg xs);
lean_obj_res numlean_opencl_float32array_empty_with_capacity(b_lean_obj_arg capacity);
lean_obj_res numlean_opencl_float32array_push(lean_obj_arg xs, float x);
lean_obj_res numlean_opencl_float32array_size(b_lean_obj_arg xs);
size_t numlean_opencl_float32array_usize(b_lean_obj_arg xs);
float numlean_opencl_float32array_get(b_lean_obj_arg xs, b_lean_obj_arg i);
float numlean_opencl_float32array_get_bang(b_lean_obj_arg xs, b_lean_obj_arg i);
uint8_t numlean_opencl_float32array_beq(lean_obj_arg a, lean_obj_arg b);
lean_obj_res numlean_opencl_float32array_add(lean_obj_arg a, lean_obj_arg b);

lean_obj_res numlean_opencl_float32arrayopencl_mk(lean_obj_arg data);
lean_obj_res numlean_opencl_float32arrayopencl_data(b_lean_obj_arg xs);
lean_obj_res numlean_opencl_float32arrayopencl_to_array(b_lean_obj_arg xs);
lean_obj_res numlean_opencl_float32arrayopencl_empty_with_capacity(b_lean_obj_arg capacity);
lean_obj_res numlean_opencl_float32arrayopencl_push(lean_obj_arg xs, float x);
lean_obj_res numlean_opencl_float32arrayopencl_size(b_lean_obj_arg xs);
size_t numlean_opencl_float32arrayopencl_usize(b_lean_obj_arg xs);
float numlean_opencl_float32arrayopencl_get(b_lean_obj_arg xs, b_lean_obj_arg i);
float numlean_opencl_float32arrayopencl_get_bang(b_lean_obj_arg xs, b_lean_obj_arg i);
uint8_t numlean_opencl_float32arrayopencl_beq(lean_obj_arg a, lean_obj_arg b);
lean_obj_res numlean_opencl_float32arrayopencl_add(lean_obj_arg a, lean_obj_arg b);
lean_obj_res numlean_opencl_float32arrayopencl_sum(lean_obj_arg xs);
lean_obj_res numlean_opencl_float32arrayopencl_copy(lean_obj_arg x);
lean_obj_res numlean_opencl_float32arrayopencl_scal(float alpha, lean_obj_arg x);
lean_obj_res numlean_opencl_float32arrayopencl_map_unsafe(lean_obj_arg x, b_lean_obj_arg expr);
lean_obj_res numlean_opencl_float32arrayopencl_map_in_context_unsafe(lean_obj_arg x, b_lean_obj_arg ctx, b_lean_obj_arg expr);
lean_obj_res numlean_opencl_float32arrayopencl_axpy(float alpha, lean_obj_arg x, lean_obj_arg y);
lean_obj_res numlean_opencl_float32arrayopencl_swap(lean_obj_arg x, lean_obj_arg y);
lean_obj_res numlean_opencl_float32arrayopencl_rot(float c, float s, lean_obj_arg x, lean_obj_arg y);
lean_obj_res numlean_opencl_float32arrayopencl_dot(lean_obj_arg x, lean_obj_arg y);
lean_obj_res numlean_opencl_float32arrayopencl_asum(lean_obj_arg x);
lean_obj_res numlean_opencl_float32arrayopencl_nrm2(lean_obj_arg x);

#endif
