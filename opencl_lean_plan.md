# Lean 4 OpenCL FloatArray Bindings — Comprehensive Design Plan

## Overview

A pure (non-IO) `OpenCL.FloatArray` type for Lean 4 with one global OpenCL
context, one in-order command queue, copy-on-write semantics, and runtime-built
map kernels. The key invariant is: **all GPU operations are enqueued on a single
in-order command queue, and OpenCL memory objects are only released through their
OpenCL reference-counted lifetime**, making pure reordering by Lean's compiler
safe under the same single-queue discipline as the CUDA design.

`OpenCL.FloatArray` is intentionally a **32-bit floating-point device array**.
Lean's `FloatArray` stores `Float` values as host `double`s, but the target
machine currently has Intel CPU/GPU OpenCL devices where double support is absent,
incomplete, or not worth relying on. `OpenCL.FloatArray.ofFloatArray` therefore
silently downcasts each Lean `Float` to `float` on transfer to the device, and
`toFloatArray` widens the stored `float` values back to Lean `Float` on read-back.

OpenCL differs from CUDA in two important ways:

- There is no `cudaFreeAsync` equivalent in core OpenCL.
- `cl_mem` objects are reference-counted by the OpenCL runtime; releasing the
  host reference with `clReleaseMemObject` does not invalidate commands already
  queued against that memory object.

The load-bearing design rule is therefore: **never expose raw host-managed GPU
pointers; only store `cl_mem` handles inside Lean external objects, enqueue all
commands on the global in-order queue, and let the OpenCL runtime defer object
destruction until queued users are complete.**

---

## 1. Repository Structure

```
OpenCLLean/
├── lakefile.lean
├── OpenCLLean.lean                 -- root module
├── OpenCL/
│   ├── Init.lean                   -- initOpenCLContext
│   ├── FloatArray.lean             -- Lean-side type and API
│   ├── Expr.lean                   -- expression AST for .map
│   └── Internal/
│       └── Unsafe.lean             -- @[extern] declarations
└── c/
    ├── opencl_lean.h
    ├── opencl_lean_context.c       -- platform/device/context/queue
    ├── opencl_lean_array.c         -- alloc / free / COW
    ├── opencl_lean_ops.c           -- add, scale, zip, reduce
    ├── opencl_lean_jit.c           -- clBuildProgram + kernel cache
    ├── opencl_lean_tracy.c         -- optional Tracy integration
    └── CMakeLists.txt
```

---

## 2. The Global Context

### 2.1 What it holds

```c
// opencl_lean_context.c
typedef struct {
    cl_platform_id    platform;
    cl_device_id      device;
    cl_context        context;
    cl_command_queue  queue;       // THE single in-order queue; all ops go here
    char              device_name[256];
    char              driver_version[256];
    int               initialized;
} OpenCLLeanCtx;

static OpenCLLeanCtx g_ctx = {0};
```

### 2.2 Initialization

```c
lean_obj_res lean_opencl_init(lean_obj_arg world) {
    if (g_ctx.initialized) return lean_io_result_mk_ok(lean_box(0));

    cl_int err;

    // Minimal first version: pick the first GPU device, fallback to CPU.
    g_ctx.platform = choose_platform();
    g_ctx.device   = choose_device(g_ctx.platform);

    g_ctx.context = clCreateContext(
        NULL, 1, &g_ctx.device, NULL, NULL, &err);
    CHECK_CL(err);

    // Use an in-order queue. Add CL_QUEUE_PROFILING_ENABLE when Tracy is on.
    cl_queue_properties props[] = {
#ifdef TRACY_ENABLE
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
#endif
        0
    };

    g_ctx.queue = clCreateCommandQueueWithProperties(
        g_ctx.context, g_ctx.device, props, &err);
    CHECK_CL(err);

    clGetDeviceInfo(g_ctx.device, CL_DEVICE_NAME,
                    sizeof(g_ctx.device_name), g_ctx.device_name, NULL);
    clGetDeviceInfo(g_ctx.device, CL_DRIVER_VERSION,
                    sizeof(g_ctx.driver_version), g_ctx.driver_version, NULL);

    g_ctx.initialized = 1;
    return lean_io_result_mk_ok(lean_box(0));
}
```

```lean
-- OpenCL/Init.lean
@[extern "lean_opencl_init"]
opaque initOpenCLContext : IO Unit

-- Call once at program start; safe to call multiple times.
```

### 2.3 Thread safety

Lean's task system may call IO actions on different OS threads. The global OpenCL
objects (`cl_context`, `cl_command_queue`, `cl_mem`, `cl_kernel`, `cl_program`) are
OpenCL runtime handles. Calls that enqueue work on the same command queue are
thread-safe according to OpenCL's object model, but concurrent calls can still
interleave at the host API level.

The correctness invariant does not depend on host call order beyond this: once a
command is successfully enqueued, the in-order command queue serializes it with all
other commands in enqueue order. For simplicity and predictable behavior, the first
implementation should use a coarse `g_queue_mutex` around every command enqueue and
kernel-argument setup. This avoids races caused by sharing cached `cl_kernel`
objects whose arguments are mutable state.

When an `OpenCL.FloatArray` value crosses a thread boundary via `Task`, Lean sets
its RC to a special shared tombstone value. `lean_is_exclusive` returns false for
such objects, so COW always fires. No aliased in-place writes can occur across
threads.

The other shared mutable structure is the program/kernel cache; protect it with a
separate `g_cache_mutex`.

---

## 3. The FloatArray Object

### 3.1 C layout

```c
// opencl_lean.h
typedef struct {
    cl_mem  buffer;      // OpenCL memory object; owned reference
    size_t  n_elems;
} OpenCLFloatArray;
```

The Lean external object owns one reference to `buffer`. Kernels and queued copy
commands use the OpenCL handle, not a raw pointer.

The buffer element type is always OpenCL `float`, not `double`. This is a semantic
choice, not an implementation accident: the current Intel OpenCL environment should
be treated as `float32`-only for this project. The public name remains
`OpenCL.FloatArray` for Lean ergonomics, but its numeric behavior is closer to a
`Float32Array`.

### 3.2 Finalizer

```c
void lean_opencl_float_array_finalize(void* ptr) {
    OpenCLFloatArray* arr = (OpenCLFloatArray*)ptr;
    if (arr->buffer) {
        // OpenCL reference-counted release. Commands already enqueued against
        // this cl_mem keep it alive until the runtime is done with it.
        clReleaseMemObject(arr->buffer);
    }
    free(arr);
}

static lean_external_class* g_opencl_array_class = NULL;

void lean_opencl_array_class_init(void) {
    if (!g_opencl_array_class) {
        g_opencl_array_class = lean_register_external_class(
            lean_opencl_float_array_finalize, /* foreach= */ NULL);
    }
}
```

### 3.3 Why `clReleaseMemObject` is sufficient

Consider:

```lean
let b := a.map (.mul .x (.lit 2.0))
-- map enqueues a kernel that reads a.buffer and writes b.buffer
-- a's RC later drops to 0 on CPU
-- finalizer calls clReleaseMemObject(a.buffer)
```

OpenCL commands queued against a memory object keep the object valid for the
duration of those commands. The finalizer releases Lean's host-side reference, but
the runtime must not destroy the actual memory until pending users are complete.

This is the OpenCL replacement for the CUDA design's `cudaFreeAsync` invariant.

### 3.4 Lean-side opaque type

```lean
-- OpenCL/Internal/Unsafe.lean
private opaque OpenCLFloatArrayPointed : PointedType := ⟨Unit, ()⟩

-- OpenCL/FloatArray.lean
def OpenCL.FloatArray := OpenCLFloatArrayPointed.type
instance : Inhabited OpenCL.FloatArray := ⟨OpenCLFloatArrayPointed.val⟩
```

---

## 4. Copy-on-Write

### 4.1 The rule

```
lean_is_exclusive(obj) == true  →  RC == 1  →  safe to mutate in place
lean_is_exclusive(obj) == false →  RC  > 1  →  must copy first
```

### 4.2 COW helper in C

```c
cl_mem lean_opencl_cow(lean_obj_arg obj, lean_obj_res* new_obj_out) {
    OpenCLFloatArray* arr = lean_get_external_data(obj);

    if (lean_is_exclusive(obj)) {
        *new_obj_out = obj;
        return arr->buffer;
    }

    cl_int err;
    cl_mem new_buf = clCreateBuffer(
        g_ctx.context,
        CL_MEM_READ_WRITE,
        arr->n_elems * sizeof(float),
        NULL,
        &err);
    CHECK_CL(err);

    pthread_mutex_lock(&g_queue_mutex);
    err = clEnqueueCopyBuffer(
        g_ctx.queue,
        arr->buffer,
        new_buf,
        0,
        0,
        arr->n_elems * sizeof(float),
        0,
        NULL,
        NULL);
    pthread_mutex_unlock(&g_queue_mutex);
    CHECK_CL(err);

    OpenCLFloatArray* new_arr = malloc(sizeof(OpenCLFloatArray));
    new_arr->buffer  = new_buf;
    new_arr->n_elems = arr->n_elems;
    *new_obj_out = lean_alloc_external(g_opencl_array_class, new_arr);

    lean_dec(obj);
    return new_buf;
}
```

### 4.3 In-place vs. out-of-place ops

For binary ops like `add`, reuse the first argument's buffer only when Lean says it
is exclusive. The second argument is read-only and does not need COW.

```c
lean_obj_res lean_opencl_add(lean_obj_arg a_obj, lean_obj_arg b_obj) {
    OpenCLFloatArray* a = lean_get_external_data(a_obj);
    OpenCLFloatArray* b = lean_get_external_data(b_obj);
    assert(a->n_elems == b->n_elems);

    lean_obj_res result_obj;
    cl_mem dst = lean_opencl_cow(a_obj, &result_obj);
    launch_add_kernel(dst, b->buffer, a->n_elems);
    lean_dec(b_obj);
    return result_obj;
}
```

---

## 5. Basic Operations

### 5.1 Allocation

```lean
@[extern "lean_opencl_alloc"]
opaque OpenCL.FloatArray.alloc (n : USize) : OpenCL.FloatArray

@[extern "lean_opencl_fill"]
opaque OpenCL.FloatArray.fill (arr : OpenCL.FloatArray) (v : Float) : OpenCL.FloatArray

@[extern "lean_opencl_of_float_array"]
opaque OpenCL.FloatArray.ofFloatArray (src : FloatArray) : OpenCL.FloatArray
-- Converts host Float/double values to device float32, then enqueues H2D write.
```

### 5.2 Arithmetic

```lean
@[extern "lean_opencl_add"]   opaque OpenCL.FloatArray.add   : OpenCL.FloatArray → OpenCL.FloatArray → OpenCL.FloatArray
@[extern "lean_opencl_sub"]   opaque OpenCL.FloatArray.sub   : OpenCL.FloatArray → OpenCL.FloatArray → OpenCL.FloatArray
@[extern "lean_opencl_mul"]   opaque OpenCL.FloatArray.mul   : OpenCL.FloatArray → OpenCL.FloatArray → OpenCL.FloatArray
@[extern "lean_opencl_scale"] opaque OpenCL.FloatArray.scale : OpenCL.FloatArray → Float → OpenCL.FloatArray
@[extern "lean_opencl_zip"]   opaque OpenCL.FloatArray.zip   : OpenCL.FloatArray → OpenCL.FloatArray → OpenCL.Expr → OpenCL.FloatArray

instance : Add OpenCL.FloatArray := ⟨OpenCL.FloatArray.add⟩
instance : Mul OpenCL.FloatArray := ⟨OpenCL.FloatArray.mul⟩
instance : Sub OpenCL.FloatArray := ⟨OpenCL.FloatArray.sub⟩
```

### 5.3 Synchronising read-back

```lean
-- Blocks until all prior queue work completes, then copies D2H.
@[extern "lean_opencl_to_float_array"]
opaque OpenCL.FloatArray.toFloatArray : OpenCL.FloatArray → IO FloatArray

@[extern "lean_opencl_size"]
opaque OpenCL.FloatArray.size : OpenCL.FloatArray → USize
-- Safe to call without queue sync; reads only the CPU-side n_elems field.
```

```c
lean_obj_res lean_opencl_to_float_array(lean_obj_arg arr_obj, lean_obj_arg world) {
    OpenCLFloatArray* arr = lean_get_external_data(arr_obj);

    pthread_mutex_lock(&g_queue_mutex);
    cl_int err = clFinish(g_ctx.queue);
    pthread_mutex_unlock(&g_queue_mutex);
    CHECK_CL_IO(err);

    OPENCL_TRACY_COLLECT();

    float* tmp = malloc(arr->n_elems * sizeof(float));
    err = clEnqueueReadBuffer(
        g_ctx.queue,
        arr->buffer,
        CL_TRUE,
        0,
        arr->n_elems * sizeof(float),
        tmp,
        0,
        NULL,
        NULL);
    CHECK_CL_IO(err);

    lean_obj_res result = lean_alloc_sarray(sizeof(double), arr->n_elems, arr->n_elems);
    for (size_t i = 0; i < arr->n_elems; i++) {
        ((double*)lean_sarray_cptr(result))[i] = (double)tmp[i];
    }

    free(tmp);
    lean_dec(arr_obj);
    return lean_io_result_mk_ok(result);
}
```

Lean's `FloatArray` stores `Float`, which is represented as C `double`, while
`OpenCL.FloatArray` stores device `float`. The FFI must therefore downcast
`double → float` in `ofFloatArray` and widen `float → double` in `toFloatArray`.
This conversion is intentionally silent to keep the API lightweight, but tests and
documentation must treat OpenCL results as float32 results.

---

## 6. Runtime-Built Map Kernel

### 6.1 Expression language (Lean side)

```lean
-- OpenCL/Expr.lean
inductive OpenCL.Expr
  | x
  | i
  | lit  (v : Float)
  | add  (a b : OpenCL.Expr)
  | sub  (a b : OpenCL.Expr)
  | mul  (a b : OpenCL.Expr)
  | div  (a b : OpenCL.Expr)
  | neg  (e : OpenCL.Expr)
  | sin  (e : OpenCL.Expr)
  | cos  (e : OpenCL.Expr)
  | exp  (e : OpenCL.Expr)
  | log  (e : OpenCL.Expr)
  | sqrt (e : OpenCL.Expr)
  | pow  (base exp : OpenCL.Expr)
  | fma  (a b c : OpenCL.Expr)

def OpenCL.Expr.toSource : OpenCL.Expr → String
  | .x          => "x"
  | .i          => "(float)gid"
  | .lit v      => s!"{v}f"
  | .add a b    => s!"({a.toSource} + {b.toSource})"
  | .sub a b    => s!"({a.toSource} - {b.toSource})"
  | .mul a b    => s!"({a.toSource} * {b.toSource})"
  | .div a b    => s!"({a.toSource} / {b.toSource})"
  | .neg e      => s!"(-{e.toSource})"
  | .sin e      => s!"sin({e.toSource})"
  | .cos e      => s!"cos({e.toSource})"
  | .exp e      => s!"exp({e.toSource})"
  | .log e      => s!"log({e.toSource})"
  | .sqrt e     => s!"sqrt({e.toSource})"
  | .pow a b    => s!"pow({a.toSource}, {b.toSource})"
  | .fma a b c  => s!"fma({a.toSource}, {b.toSource}, {c.toSource})"
```

### 6.2 map and zip

```lean
@[extern "lean_opencl_map_expr"]
opaque OpenCL.FloatArray.map (arr : OpenCL.FloatArray) (expr : OpenCL.Expr) : OpenCL.FloatArray

@[extern "lean_opencl_map_str"]
opaque OpenCL.FloatArray.mapStr (arr : OpenCL.FloatArray) (src : String) : OpenCL.FloatArray
```

### 6.3 OpenCL kernel template

```c
static const char* MAP_TEMPLATE =
"__kernel void opencl_lean_map(__global const float* in,\n"
"                              __global float* out,\n"
"                              ulong n) {\n"
"    size_t gid = get_global_id(0);\n"
"    if (gid >= n) return;\n"
"    float x = in[gid];\n"
"    out[gid] = %s;\n"
"}\n";
```

### 6.4 Program/kernel cache

OpenCL kernels have mutable argument slots. Do **not** share one `cl_kernel` object
across concurrent launches without a queue mutex around `clSetKernelArg` and
`clEnqueueNDRangeKernel`. The simpler first design uses one cached `cl_kernel` per
expression and holds `g_queue_mutex` while setting arguments and enqueueing.

The cache itself needs `g_cache_mutex` because Lean's RC does not protect C-side
global state.

```c
typedef struct KernelCacheEntry {
    char       key[256];       // SHA-256(expr + device + driver + options)
    cl_program program;
    cl_kernel  kernel;
    struct KernelCacheEntry* next;
} KernelCacheEntry;

static KernelCacheEntry* g_kernel_cache = NULL;
static pthread_mutex_t   g_cache_mutex  = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t   g_queue_mutex  = PTHREAD_MUTEX_INITIALIZER;

cl_kernel kernel_cache_get(const char* expr_src) {
    char key[256];
    make_opencl_kernel_key(expr_src, key, sizeof(key));

    pthread_mutex_lock(&g_cache_mutex);

    for (KernelCacheEntry* e = g_kernel_cache; e; e = e->next) {
        if (strcmp(e->key, key) == 0) {
            cl_kernel kernel = e->kernel;
            pthread_mutex_unlock(&g_cache_mutex);
            return kernel;
        }
    }

    cl_kernel kernel = build_program_and_kernel(expr_src, key);
    pthread_mutex_unlock(&g_cache_mutex);
    return kernel;
}
```

### 6.5 Build path

```c
cl_kernel build_program_and_kernel(const char* expr_src, const char* key) {
    char src[4096];
    snprintf(src, sizeof(src), MAP_TEMPLATE, expr_src);

    cl_int err;
    const char* sources[] = { src };
    size_t lengths[] = { strlen(src) };

    cl_program program = clCreateProgramWithSource(
        g_ctx.context, 1, sources, lengths, &err);
    CHECK_CL(err);

    const char* opts = "-cl-std=CL1.2";
    err = clBuildProgram(program, 1, &g_ctx.device, opts, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, g_ctx.device, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        char* log = malloc(log_size + 1);
        clGetProgramBuildInfo(program, g_ctx.device, CL_PROGRAM_BUILD_LOG,
                              log_size, log, NULL);
        log[log_size] = 0;
        lean_internal_panic(log);
    }

    cl_kernel kernel = clCreateKernel(program, "opencl_lean_map", &err);
    CHECK_CL(err);
    cache_insert(key, program, kernel);
    return kernel;
}
```

### 6.6 Disk cache

For CUDA, the natural disk artifact is PTX. For OpenCL, portability is weaker:
`clGetProgramInfo(..., CL_PROGRAM_BINARIES, ...)` returns vendor-specific device
binaries that are only valid for the same device, driver, and build options.

Use a conservative key:

```
SHA-256(expr_source)
+ platform vendor
+ device name
+ device version
+ driver version
+ build options
```

Cache path:

```
~/.cache/opencl_lean/<key>.bin
```

If binary loading fails, delete the file and rebuild from source. Source builds are
the correctness baseline; binary caching is an optimization only.

---

## 7. Kernel Launch Helpers

```c
#define LOCAL_SIZE 256

static size_t round_up(size_t n, size_t group) {
    return ((n + group - 1) / group) * group;
}

void launch_map_kernel(cl_kernel kernel, cl_mem dst, cl_mem src, size_t n) {
    cl_ulong n_arg = (cl_ulong)n;
    size_t local  = LOCAL_SIZE;
    size_t global = round_up(n, local);

    pthread_mutex_lock(&g_queue_mutex);

    CHECK_CL(clSetKernelArg(kernel, 0, sizeof(cl_mem), &src));
    CHECK_CL(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dst));
    CHECK_CL(clSetKernelArg(kernel, 2, sizeof(cl_ulong), &n_arg));

    cl_event evt = NULL;
    cl_int err = clEnqueueNDRangeKernel(
        g_ctx.queue,
        kernel,
        1,
        NULL,
        &global,
        &local,
        0,
        NULL,
        OPENCL_TRACY_EVENT_PTR(&evt));

    OPENCL_TRACY_SET_EVENT(evt);
    pthread_mutex_unlock(&g_queue_mutex);
    CHECK_CL(err);
}
```

For portability, query `CL_DEVICE_MAX_WORK_GROUP_SIZE` during initialization and
clamp `LOCAL_SIZE` if needed.

---

## 8. Error Handling

OpenCL returns `cl_int` status codes from nearly every API. Do not ignore them.

```c
static const char* cl_error_string(cl_int err) {
    switch (err) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    default: return "unknown OpenCL error";
    }
}

#define CHECK_CL(err) \
    do { \
        if ((err) != CL_SUCCESS) lean_internal_panic(cl_error_string(err)); \
    } while (0)
```

For FFI functions returning `IO`, prefer `lean_io_result_mk_error` over panic.
For pure functions, panic is acceptable for unrecoverable OpenCL runtime errors,
but keep build logs and device information in the message.

---

## 9. Tracy GPU Profiling

OpenCL profiling uses `cl_event` timestamps. The queue must be created with
`CL_QUEUE_PROFILING_ENABLE`.

### 9.1 Tracy context

```c
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>
#include <tracy/TracyOpenCL.hpp>

static TracyCLCtx g_tracy_cl_ctx;

void opencl_tracy_init(void) {
    g_tracy_cl_ctx = TracyCLContext(g_ctx.context, g_ctx.device);
}

#define OPENCL_TRACY_ZONE(name)       TracyCLZone(g_tracy_cl_ctx, name)
#define OPENCL_TRACY_SET_EVENT(evt)   TracyCLZoneSetEvent(evt)
#define OPENCL_TRACY_COLLECT()        TracyCLCollect(g_tracy_cl_ctx)
#define OPENCL_TRACY_EVENT_PTR(p)     (p)
#else
#define OPENCL_TRACY_ZONE(name)
#define OPENCL_TRACY_SET_EVENT(evt)
#define OPENCL_TRACY_COLLECT()
#define OPENCL_TRACY_EVENT_PTR(p)     NULL
#endif
```

### 9.2 Shared collect point

The natural collect point is `toFloatArray`, because it already calls `clFinish`.
For programs that run long GPU-only loops, expose an explicit collect function:

```lean
@[extern "lean_opencl_tracy_collect"]
opaque OpenCL.tracyCollect : IO Unit
```

### 9.3 CPU-side zones

Use Tracy CPU zones around expensive FFI paths:

```c
lean_obj_res lean_opencl_map_expr(...) {
    ZoneScopedN("opencl_map_expr");
    {
        ZoneScopedN("opencl_kernel_cache_lookup");
        kernel = kernel_cache_get(expr_src);
    }
    launch_map_kernel(kernel, dst, src, n);
    return result;
}
```

---

## 10. Safety Invariants — Summary

These must hold at all times:

| # | Invariant | Mechanism |
|---|-----------|-----------|
| 1 | No GPU memory is destroyed while queued commands may use it | Store only `cl_mem`; release with `clReleaseMemObject`; OpenCL runtime owns queued uses |
| 2 | COW copies complete before a later kernel writes | Copy and kernel are enqueued on the same in-order queue |
| 3 | `toFloatArray` never reads stale data | `clFinish` before blocking D2H read |
| 4 | Two writable `FloatArray` values never alias | COW creates a fresh buffer when `lean_is_exclusive` is false |
| 5 | H2D writes complete before kernels read | All writes and kernels use the same in-order queue |
| 6 | Runtime build errors surface cleanly | Check `clBuildProgram`; retrieve `CL_PROGRAM_BUILD_LOG` |
| 7 | Kernel cache has no data races | `g_cache_mutex` wraps lookup and build |
| 8 | Kernel argument mutation has no data races | `g_queue_mutex` wraps `clSetKernelArg` plus enqueue |
| 9 | No aliased in-place writes across threads | Lean shared tombstone makes `lean_is_exclusive` false after crossing tasks |
| 10 | Disk cache does not load incompatible binaries | Key includes expression, device, driver, and build options; fallback to source build |
| 11 | Numeric storage is consistently float32 | `ofFloatArray` downcasts Lean `Float`/double to OpenCL `float`; kernels use only `float`; `toFloatArray` widens on read-back |

---

## 11. What Can Still Go Wrong

### 11.1 Out-of-order queues

Do not use `CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE` in the first version. It would
require explicit event dependencies between every copy, kernel, and read. That is
possible but not worth the complexity while preserving a pure Lean API.

### 11.2 Shared `cl_kernel` arguments

`cl_kernel` argument slots are mutable. If two host threads set arguments on the
same cached kernel at the same time, launches can receive mixed arguments. The
minimal fix is `g_queue_mutex` around argument setup and enqueue. A more scalable
future design can clone kernels with `clCloneKernel` when available or cache a small
kernel pool per expression.

### 11.3 OpenCL implementation variance

OpenCL compiler behavior varies heavily by vendor. Keep generated kernels simple,
target `-cl-std=CL1.2` initially, and build a conformance test suite that compares
against Lean/CPU results with tolerances.

### 11.4 Binary cache invalidation

OpenCL binaries are not portable. Always include device and driver information in
the cache key. If binary creation or loading fails, rebuild from source and replace
the cache entry.

### 11.5 Floating-point semantics

Build options such as `-cl-fast-relaxed-math` change NaN, signed zero, denormal,
and rounding behavior. Make fast math an explicit option rather than the default.

Even without fast math, `OpenCL.FloatArray` is not double precision. The API accepts
Lean `Float` values, but stores and computes them as OpenCL `float`. This is
important on the current Intel CPU/GPU OpenCL environment, where double support is
not available or is problematic enough that the first design should not depend on
`cl_khr_fp64`. CPU reference tests should round expected values to float32 before
comparison, or use tolerances appropriate for single precision.

### 11.6 Host pointer lifetime for `ofFloatArray`

If using non-blocking `clEnqueueWriteBuffer`, the host staging memory must stay
alive until the write completes. The simplest safe first implementation copies
Lean's `FloatArray` into a temporary `float*` staging allocation, enqueues the
write, and releases the staging memory via an event callback after completion.

Alternatively, use `CL_TRUE` blocking writes in phase 1, then optimize later.

### 11.7 Device selection

Picking the first GPU is convenient but not always correct. Eventually expose:

```lean
structure OpenCL.DeviceInfo where
  platformName : String
  deviceName   : String
  deviceType   : String

opaque OpenCL.availableDevices : IO (Array OpenCL.DeviceInfo)
opaque OpenCL.initDevice : Nat → IO Unit
```

Keep the pure `FloatArray` API single-device even after device selection is added.

---

## 12. Build System

```cmake
# c/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(OpenCLLean LANGUAGES C CXX)

find_package(OpenCL REQUIRED)

option(TRACY_ENABLE "Enable Tracy profiling" OFF)

add_library(opencl_lean SHARED
    opencl_lean_context.c
    opencl_lean_array.c
    opencl_lean_ops.c
    opencl_lean_jit.c
    opencl_lean_tracy.c)

target_link_libraries(opencl_lean OpenCL::OpenCL)
target_compile_features(opencl_lean PRIVATE c_std_11 cxx_std_17)

if(TRACY_ENABLE)
    find_path(TRACY_INCLUDE_DIR tracy/Tracy.hpp HINTS $ENV{TRACY_DIR}/public)
    target_include_directories(opencl_lean PRIVATE ${TRACY_INCLUDE_DIR})
    target_sources(opencl_lean PRIVATE ${TRACY_INCLUDE_DIR}/TracyClient.cpp)
    target_compile_definitions(opencl_lean PUBLIC TRACY_ENABLE TRACY_ENABLE_OPENCL)
endif()
```

```lean
-- lakefile.lean
require mathlib from git "https://github.com/leanprover-community/mathlib4"

lean_lib OpenCLLean

target_c_lib opencl_lean where
  srcDir := "c"
  build := "cmake"
```

---

## 13. Phased Implementation Plan

### Phase 1 — Scaffolding

- `initOpenCLContext` with platform/device selection and one in-order queue
- Opaque `OpenCL.FloatArray` type with `clReleaseMemObject` finalizer
- `alloc`, `fill`, `ofFloatArray`, `toFloatArray`, `size`
- Explicit `double → float → double` conversion tests documenting silent downcast
- Start with blocking H2D writes to avoid staging lifetime complexity
- Verify round-trip tests and finalizer behavior under repeated allocation

### Phase 2 — Basic ops

- Prebuilt source kernels for `add`, `sub`, `mul`, `scale`
- COW helper using `lean_is_exclusive`
- Tests for shared arrays, aliased inputs, and task boundary sharing
- Compare GPU results against CPU `FloatArray` with tolerances

### Phase 3 — Runtime map

- `OpenCL.Expr` AST and source generator
- `mapStr` for debugging generated expressions
- `clCreateProgramWithSource` / `clBuildProgram` path
- Build-log reporting on compile errors
- In-process kernel cache protected by `g_cache_mutex`

### Phase 4 — Concurrency hardening

- `g_queue_mutex` around all `clSetKernelArg` plus enqueue sequences
- Stress tests with many Lean `Task`s sharing arrays and calling `map`
- Confirm no mixed kernel arguments under ThreadSanitizer where possible
- Consider `clCloneKernel` or per-thread kernel instances only if contention is measurable

### Phase 5 — Disk cache

- Persist OpenCL program binaries to `~/.cache/opencl_lean/`
- Key by expression, platform, device, driver, and build options
- Fallback to source build on binary load failure
- Benchmark cold build vs. warm binary load

### Phase 6 — Tracy profiling

- Add `TRACY_ENABLE` CMake option
- Create command queue with `CL_QUEUE_PROFILING_ENABLE` when Tracy is enabled
- Add `TracyCLZone` around kernel enqueues and copy operations
- Add `OPENCL_TRACY_COLLECT()` in `toFloatArray` and explicit `tracyCollect`
- Add CPU zones for COW, source generation, cache lookup, and program build

### Phase 7 — Hardening

- Device capability checks: max work group size, OpenCL version, fp32 support
- Do not require `cl_khr_fp64`; keep generated kernels float32-only
- Random expression tests against CPU reference implementation
- Allocation pressure tests for `CL_OUT_OF_RESOURCES`
- Fast-math option tests documenting numerical differences
- Benchmark end-to-end throughput and queue gaps in Tracy
