# Lean 4 CUDA FloatArray Bindings — Comprehensive Design Plan

## Overview

A pure (non-IO) `Cuda.FloatArray` type for Lean 4 with stream-ordered memory management,
copy-on-write semantics, and JIT-compiled map kernels via NVRTC. The key invariant is:
**all GPU operations and all frees are enqueued on a single global stream**, making
pure reordering by Lean's compiler safe.

---

## 1. Repository Structure

```
CudaLean/
├── lakefile.lean
├── CudaLean.lean                  -- root module
├── Cuda/
│   ├── Init.lean                  -- initCudaContext
│   ├── FloatArray.lean            -- Lean-side type and API
│   ├── Expr.lean                  -- expression AST for .map
│   └── Internal/
│       └── Unsafe.lean            -- @[extern] declarations
└── c/
    ├── cuda_lean.h
    ├── cuda_lean_context.cu       -- global context / stream
    ├── cuda_lean_array.cu         -- alloc / free / COW
    ├── cuda_lean_ops.cu           -- add, scale, zip, reduce
    ├── cuda_lean_jit.cu           -- NVRTC + kernel cache
    └── CMakeLists.txt
```

---

## 2. The Global Context

### 2.1 What it holds

```c
// cuda_lean_context.cu
typedef struct {
    CUdevice   device;
    CUcontext  ctx;          // driver API context
    cudaStream_t stream;     // THE single stream; all ops go here
    bool       initialized;
} CudaLeanCtx;

static CudaLeanCtx g_ctx = {0};
```

### 2.2 Initialization

```c
lean_obj_res lean_cuda_init(lean_obj_arg world) {
    if (g_ctx.initialized) return lean_io_result_mk_ok(lean_box(0));

    cudaSetDevice(0);
    // Use runtime API — avoids per-thread context headaches
    cudaStreamCreateWithFlags(&g_ctx.stream, cudaStreamNonBlocking);
    g_ctx.initialized = true;
    return lean_io_result_mk_ok(lean_box(0));
}
```

```lean
-- Cuda/Init.lean
@[extern "lean_cuda_init"]
opaque initCudaContext : IO Unit

-- Call once at program start; safe to call multiple times
```

### 2.3 Thread safety

Lean's task system may call IO actions on different OS threads. Since we use the
**runtime API** (not driver API), CUDA uses a per-process default context — no
`cuCtxSetCurrent` needed. The stream itself is thread-safe for concurrent
`cudaMemcpyAsync` / kernel launches; the driver queues them correctly.

When a `Cuda.FloatArray` value crosses a thread boundary via `Task`, Lean sets its
RC to a special **shared tombstone** value. `lean_is_exclusive` returns false for
any such object, so COW always fires — no aliased in-place writes can occur across
threads. This means the RC model is sound for concurrent Lean code without any
additional locking on the COW path.

The one piece that is **not** protected by Lean's RC is the **kernel cache** — see
section 6.4.

---

## 3. The FloatArray Object

### 3.1 C layout

```c
// cuda_lean.h
typedef struct {
    float*   device_ptr;   // device memory; owned by this object
    size_t   n_elems;
} CudaFloatArray;

// Lean external object: Lean manages the header (RC, tag, etc.)
// We only define the finalizer and the data pointer.
```

### 3.2 Finalizer — the critical piece

```c
// NEVER cudaFree directly. Always go through the stream.
void lean_cuda_float_array_finalize(void* ptr) {
    CudaFloatArray* arr = (CudaFloatArray*)ptr;
    if (arr->device_ptr && g_ctx.initialized) {
        cudaFreeAsync(arr->device_ptr, g_ctx.stream);
        // Memory is released only after all prior stream ops complete.
        // This is what makes pure reordering safe.
    }
    free(arr);
}

static lean_external_class* g_cuda_array_class = nullptr;

void lean_cuda_array_class_init() {
    if (!g_cuda_array_class)
        g_cuda_array_class = lean_register_external_class(
            lean_cuda_float_array_finalize, /* foreach= */ nullptr);
}
```

### 3.3 Why `cudaFreeAsync` is the load-bearing invariant

Consider:
```lean
let b := a.map "x * 2"   -- kernel K1 enqueued, b wraps new buffer
-- a's RC later drops to 0 on CPU
-- finalizer calls cudaFreeAsync(a.ptr) → enqueued AFTER K1
-- K1 can safely read a.ptr
```

Without `cudaFreeAsync`, the finalizer would race with K1 on the GPU.

### 3.4 Lean-side opaque type

```lean
-- Cuda/Internal/Unsafe.lean
private opaque CudaFloatArrayPointed : PointedType := ⟨Unit, ()⟩

-- Cuda/FloatArray.lean
def Cuda.FloatArray := CudaFloatArrayPointed.type
instance : Inhabited Cuda.FloatArray := ⟨CudaFloatArrayPointed.val⟩
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
// Returns a device pointer that is safe to write to.
// If exclusive: returns arr->device_ptr (in-place).
// If shared:    enqueues an async copy and returns the new pointer.
// new_obj_out:  the lean object to return to Lean (may be same or new).
float* lean_cuda_cow(lean_obj_arg obj, lean_obj_res* new_obj_out) {
    CudaFloatArray* arr = lean_get_external_data(obj);

    if (lean_is_exclusive(obj)) {
        *new_obj_out = obj;
        return arr->device_ptr;
    }

    // Allocate new buffer
    float* new_ptr;
    cudaMallocAsync(&new_ptr, arr->n_elems * sizeof(float), g_ctx.stream);
    cudaMemcpyAsync(new_ptr, arr->device_ptr,
                    arr->n_elems * sizeof(float),
                    cudaMemcpyDeviceToDevice, g_ctx.stream);

    // Build new Lean object
    CudaFloatArray* new_arr = malloc(sizeof(CudaFloatArray));
    new_arr->device_ptr = new_ptr;
    new_arr->n_elems    = arr->n_elems;
    *new_obj_out = lean_alloc_external(g_cuda_array_class, new_arr);

    lean_dec(obj);   // release our reference to the old object
    return new_ptr;
}
```

### 4.3 In-place vs. out-of-place ops

For binary ops like `add`, both inputs might be shared:

```c
// a + b  →  result may reuse a's buffer if exclusive
lean_obj_res lean_cuda_add(lean_obj_arg a_obj, lean_obj_arg b_obj) {
    CudaFloatArray* a = lean_get_external_data(a_obj);
    CudaFloatArray* b = lean_get_external_data(b_obj);
    assert(a->n_elems == b->n_elems);

    lean_obj_res result_obj;
    float* dst = lean_cuda_cow(a_obj, &result_obj);  // COW on a
    // b is read-only; no COW needed
    launch_add_kernel(dst, b->device_ptr, a->n_elems);
    lean_dec(b_obj);
    return result_obj;
}
```

---

## 5. Basic Operations

### 5.1 Allocation

```lean
@[extern "lean_cuda_alloc"]
opaque Cuda.FloatArray.alloc (n : USize) : Cuda.FloatArray
-- Allocates uninitialised device memory. Call .fill or .ofFloatArray next.

@[extern "lean_cuda_fill"]
opaque Cuda.FloatArray.fill (arr : Cuda.FloatArray) (v : Float) : Cuda.FloatArray

@[extern "lean_cuda_of_float_array"]
opaque Cuda.FloatArray.ofFloatArray (src : FloatArray) : Cuda.FloatArray
-- Async H2D copy; returns immediately
```

### 5.2 Arithmetic

```lean
@[extern "lean_cuda_add"]   opaque Cuda.FloatArray.add   : Cuda.FloatArray → Cuda.FloatArray → Cuda.FloatArray
@[extern "lean_cuda_sub"]   opaque Cuda.FloatArray.sub   : Cuda.FloatArray → Cuda.FloatArray → Cuda.FloatArray
@[extern "lean_cuda_mul"]   opaque Cuda.FloatArray.mul   : Cuda.FloatArray → Cuda.FloatArray → Cuda.FloatArray
@[extern "lean_cuda_scale"] opaque Cuda.FloatArray.scale : Cuda.FloatArray → Float → Cuda.FloatArray
@[extern "lean_cuda_zip"]   opaque Cuda.FloatArray.zip   : Cuda.FloatArray → Cuda.FloatArray → String → Cuda.FloatArray
-- zip applies a JIT-compiled binary expression

instance : Add Cuda.FloatArray := ⟨Cuda.FloatArray.add⟩
instance : Mul Cuda.FloatArray := ⟨Cuda.FloatArray.mul⟩
instance : Sub Cuda.FloatArray := ⟨Cuda.FloatArray.sub⟩
```

### 5.3 Synchronising read-back

```lean
-- Blocks until the stream drains, then copies D2H
@[extern "lean_cuda_to_float_array"]
opaque Cuda.FloatArray.toFloatArray : Cuda.FloatArray → FloatArray

@[extern "lean_cuda_size"]
opaque Cuda.FloatArray.size : Cuda.FloatArray → USize
-- Safe to call without sync; purely reads the CPU-side n_elems field
```

```c
lean_obj_res lean_cuda_to_float_array(lean_obj_arg arr_obj, lean_obj_arg world) {
    CudaFloatArray* arr = lean_get_external_data(arr_obj);
    cudaStreamSynchronize(g_ctx.stream);   // drain everything before reading

    lean_obj_res result = lean_alloc_sarray(sizeof(double), arr->n_elems, arr->n_elems);
    // Note: Lean's FloatArray is double internally
    float* tmp = malloc(arr->n_elems * sizeof(float));
    cudaMemcpy(tmp, arr->device_ptr, arr->n_elems * sizeof(float), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < arr->n_elems; i++)
        ((double*)lean_sarray_cptr(result))[i] = (double)tmp[i];
    free(tmp);
    lean_dec(arr_obj);
    return lean_io_result_mk_ok(result);
}
```

---

## 6. JIT Map Kernel

### 6.1 Expression language (Lean side)

```lean
-- Cuda/Expr.lean
inductive Cuda.Expr
  | x                              -- input element
  | i                              -- element index (USize)
  | lit  (v : Float)
  | add  (a b : Cuda.Expr)
  | sub  (a b : Cuda.Expr)
  | mul  (a b : Cuda.Expr)
  | div  (a b : Cuda.Expr)
  | neg  (e : Cuda.Expr)
  | sin  (e : Cuda.Expr)
  | cos  (e : Cuda.Expr)
  | exp  (e : Cuda.Expr)
  | log  (e : Cuda.Expr)
  | sqrt (e : Cuda.Expr)
  | pow  (base exp : Cuda.Expr)
  | fma  (a b c : Cuda.Expr)      -- fused multiply-add

def Cuda.Expr.toSource : Cuda.Expr → String
  | .x        => "x"
  | .i        => "(float)idx"
  | .lit v    => s!"{v}f"
  | .add a b  => s!"({a.toSource} + {b.toSource})"
  | .mul a b  => s!"({a.toSource} * {b.toSource})"
  | .sin e    => s!"sinf({e.toSource})"
  | .exp e    => s!"expf({e.toSource})"
  | .fma a b c => s!"fmaf({a.toSource}, {b.toSource}, {c.toSource})"
  -- etc.
```

### 6.2 map and zip

```lean
-- Accepts a Cuda.Expr, serialises to string, passes to C
@[extern "lean_cuda_map_expr"]
opaque Cuda.FloatArray.map (arr : Cuda.FloatArray) (expr : Cuda.Expr) : Cuda.FloatArray

-- Also accept raw string for power users / debugging
@[extern "lean_cuda_map_str"]
opaque Cuda.FloatArray.mapStr (arr : Cuda.FloatArray) (src : String) : Cuda.FloatArray
```

### 6.3 NVRTC kernel template

```c
// cuda_lean_jit.cu
static const char* MAP_TEMPLATE = R"(
extern "C" __global__ void cuda_lean_map(
    const float* __restrict__ in,
    float*       __restrict__ out,
    unsigned int n)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = in[idx];
    out[idx] = %s;   // expression injected here
}
)";
```

### 6.4 Kernel cache

The kernel cache is the **only piece of shared mutable state not protected by
Lean's RC**. When a `Cuda.FloatArray` crosses a thread boundary its RC is set to
the shared tombstone value, forcing COW — but the kernel cache is a plain C
data structure that Lean knows nothing about. Two threads calling `.map` with the
same novel expression simultaneously would both find a cache miss and both invoke
NVRTC, causing a double-init race on the cache entry.

The fix is a single coarse mutex around the entire lookup+compile path. This is
cheap in practice: the lock is only ever contended on a cold compile (rare), and
never held during kernel *execution*.

```c
// Key: SHA-256(expr_source) + "_sm" + sm_version
// Value: CUfunction (loaded into the global CUmodule)

typedef struct KernelCacheEntry {
    char       key[128];   // hex SHA-256 + arch suffix
    CUmodule   mod;
    CUfunction fn;
    struct KernelCacheEntry* next;
} KernelCacheEntry;

static KernelCacheEntry*  g_kernel_cache  = nullptr;
static pthread_mutex_t    g_cache_mutex   = PTHREAD_MUTEX_INITIALIZER;
// In production: use a hash map; a linked list is fine for <100 kernels

CUfunction kernel_cache_get(const char* expr_src) {
    char sha[65];   sha256_hex(expr_src, sha);
    char key[128];  snprintf(key, sizeof(key), "%s_sm%d", sha, g_ctx.sm_version);

    pthread_mutex_lock(&g_cache_mutex);

    // 1. Check in-process cache
    for (KernelCacheEntry* e = g_kernel_cache; e; e = e->next) {
        if (strcmp(e->key, key) == 0) {
            CUfunction fn = e->fn;
            pthread_mutex_unlock(&g_cache_mutex);
            return fn;
        }
    }

    // 2. Check disk cache  ~/.cache/cuda_lean/<key>.ptx
    char path[512];
    snprintf(path, sizeof(path), "%s/.cache/cuda_lean/%s.ptx", getenv("HOME"), key);
    CUfunction fn = nullptr;
    if (file_exists(path)) {
        char* ptx = read_file(path);
        CUmodule mod;
        cuModuleLoadData(&mod, ptx);
        cuModuleGetFunction(&fn, mod, "cuda_lean_map");
        cache_insert(key, mod, fn);
        free(ptx);
    } else {
        // 3. Compile with NVRTC (slow path, mutex held — acceptable, rare)
        fn = nvrtc_compile_and_cache(expr_src, key, path);
    }

    pthread_mutex_unlock(&g_cache_mutex);
    return fn;
}
```

### 6.5 NVRTC compile path

```c
CUfunction nvrtc_compile_and_cache(const char* expr, const char* key, const char* ptx_path) {
    char src[4096];
    snprintf(src, sizeof(src), MAP_TEMPLATE, expr);

    nvrtcProgram prog;
    nvrtcCreateProgram(&prog, src, "map.cu", 0, nullptr, nullptr);

    const char* opts[] = {"--gpu-architecture=compute_75", "--use_fast_math"};
    nvrtcResult r = nvrtcCompileProgram(prog, 2, opts);
    if (r != NVRTC_SUCCESS) {
        size_t logSize;  nvrtcGetProgramLogSize(prog, &logSize);
        char* log = malloc(logSize);
        nvrtcGetProgramLog(prog, log);
        lean_internal_panic(log);   // surface compile error to Lean
    }

    size_t ptxSize;  nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    // Persist to disk
    mkdir_p(dirname(ptx_path));
    write_file(ptx_path, ptx);

    CUmodule mod;  CUfunction fn;
    cuModuleLoadData(&mod, ptx);
    cuModuleGetFunction(&fn, mod, "cuda_lean_map");
    cache_insert(key, mod, fn);
    free(ptx);
    return fn;
}
```

---

## 7. Kernel Launch Helpers

```c
#define BLOCK_SIZE 256

void launch_map_kernel(CUfunction fn, float* dst, const float* src, size_t n) {
    unsigned int grid = ((unsigned int)n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = { &src, &dst, &n };
    cuLaunchKernel(fn, grid, 1, 1, BLOCK_SIZE, 1, 1,
                   0, g_ctx.stream, args, nullptr);
    // Non-blocking. Returns immediately.
}
```

---

## 8. Safety Invariants — Summary

These must hold at all times:

| # | Invariant | Mechanism |
|---|-----------|-----------|
| 1 | No GPU memory is freed while a kernel might read it | `cudaFreeAsync` enqueues free on stream |
| 2 | COW copies complete before the new kernel writes | Copy and kernel on same stream; GPU serialises |
| 3 | `toFloatArray` never reads stale data | `cudaStreamSynchronize` before D2H copy |
| 4 | Two `FloatArray` values never alias | COW always produces a fresh buffer when RC > 1 |
| 5 | H2D copies in `ofFloatArray` complete before kernels read | All on same stream |
| 6 | NVRTC compile errors surface cleanly | Check return code; retrieve log; panic with message |
| 7 | Kernel cache is consistent across process runs | PTX keyed by SHA-256 of source + SM version |
| 8 | No aliased in-place writes across threads | Lean sets RC to shared tombstone on thread boundary; `lean_is_exclusive` returns false; COW always fires |
| 9 | Kernel cache has no data races under concurrent `Task` use | `g_cache_mutex` wraps all lookup + compile operations |

---

## 9. What Can Still Go Wrong

### 9.1 Multi-GPU / multi-stream (don't do it yet)
The entire design assumes one device, one stream. Adding a second stream requires
explicit `cudaEventRecord` / `cudaStreamWaitEvent` fencing between streams, which
is incompatible with the pure RC model. Mark the API `@[deprecated]` or document
clearly if you ever expose multi-stream.

### 9.2 `cudaMallocAsync` pool exhaustion
`cudaMallocAsync` uses a memory pool. Under heavy allocation pressure the pool can
fail. Add a fallback to `cudaMalloc` and log a warning.

### 9.3 Lean compiler inlining pure functions
If Lean decides to inline or CSE two calls to a pure `map`, it should still be
correct (idempotent result), but double-check that `lean_is_exclusive` is not
called on a partially-constructed object. The safe rule: only call
`lean_is_exclusive` at the start of each FFI function, on the obj passed in.

### 9.4 `--use_fast_math` semantics
NVRTC's `--use_fast_math` enables `-ffast-math` equivalent, which breaks
IEEE754 compliance (no signed zero, no NaN propagation, etc.). Expose this as a
compile option rather than always-on if users need exact semantics.

### 9.5 PTX forward compatibility
PTX is forward-compatible within a major architecture but not backward-compatible.
The SM version is already encoded in the cache key (section 6.4), so stale cached
PTX from a different GPU will never be loaded — it simply won't be found and will
recompile.

---

## 10. Tracy GPU Profiling

Tracy supports GPU zones for both CUDA and OpenCL via timestamp queries that are
correlated with the CPU timeline. The key idea: GPU work runs asynchronously, but
Tracy timestamps GPU events using device-side timers and then correlates them back
to the CPU clock using a calibration offset computed at startup.

### 10.1 How Tracy GPU contexts work

Tracy provides two GPU backends relevant here:

| Backend | API | Tracy context type |
|---------|-----|--------------------|
| CUDA | `cudaEvent_t` timestamp queries | `TracyGpuContext` (via `tracy/TracyOpenGL.hpp` or direct CUDA support) |
| OpenCL | `cl_event` profiling timestamps | `TracyCLContext` (via `tracy/TracyCL.hpp`) |

Both follow the same pattern:
1. **Calibration** at startup: measure the offset between GPU clock and CPU clock
2. **Zone begin**: record a GPU timestamp via an event, send to Tracy with a source location
3. **Zone end**: record another GPU timestamp
4. **Collect**: periodically call a collect function that reads completed event timestamps and uploads them to Tracy

### 10.2 CUDA Tracy integration

```c
// cuda_lean_tracy.h
#ifdef TRACY_ENABLE
#include <tracy/Tracy.hpp>

// One global GPU context for Tracy
static TracyGpuContext g_tracy_gpu_ctx;

void cuda_tracy_init() {
    TracyGpuContext  // initialises against the current CUDA context
}

// Wrap every kernel launch with GPU zones
#define CUDA_TRACY_ZONE(name)                          \
    TracyGpuZone(name)                                 \
    // Tracy inserts cudaEvent_t record before and after

// Call periodically (e.g. after toFloatArray's stream sync)
#define CUDA_TRACY_COLLECT()  TracyGpuCollect

#else
#define CUDA_TRACY_ZONE(name)
#define CUDA_TRACY_COLLECT()
#endif
```

Wrap kernel launches:

```c
void launch_map_kernel(CUfunction fn, float* dst, const float* src,
                       size_t n, const char* zone_name) {
    CUDA_TRACY_ZONE(zone_name);   // GPU zone open
    unsigned int grid = ((unsigned int)n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    void* args[] = { &src, &dst, &n };
    cuLaunchKernel(fn, grid, 1, 1, BLOCK_SIZE, 1, 1,
                   0, g_ctx.stream, args, nullptr);
    // GPU zone close happens at scope exit (RAII)
}
```

Tracy GPU zones appear in the profiler as coloured bars on a **GPU timeline lane**,
correlated with the CPU lane — you see exactly how long each kernel took on the GPU
and how it aligns with the CPU-side launch call.

### 10.3 OpenCL Tracy integration

OpenCL uses `cl_event` profiling, which requires `CL_QUEUE_PROFILING_ENABLE` on
the command queue:

```c
// In your OpenCL context init:
cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0
};
cl_command_queue queue = clCreateCommandQueueWithProperties(
    ctx, device, props, &err);

// Tracy OpenCL context
TracyCLCtx g_tracy_cl_ctx;

void opencl_tracy_init(cl_context ctx, cl_device_id device) {
#ifdef TRACY_ENABLE
    g_tracy_cl_ctx = TracyCLContext(ctx, device);
#endif
}
```

Wrap kernel enqueues:

```c
cl_event event;
{
    TracyCLZone(g_tracy_cl_ctx, "map_kernel");
    clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,
                           &global_size, &local_size,
                           0, nullptr, &event);
    TracyCLZoneSetEvent(event);   // Tracy will read this event's timestamps
}

// In your collect call (after clFinish or periodically):
TracyCLCollect(g_tracy_cl_ctx);
```

### 10.4 Shared collect point

Both backends need periodic collection. The natural place is inside
`toFloatArray` / `toHostArray` since those already synchronise the stream:

```c
lean_obj_res lean_cuda_to_float_array(lean_obj_arg arr_obj, lean_obj_arg world) {
    CudaFloatArray* arr = lean_get_external_data(arr_obj);
    cudaStreamSynchronize(g_ctx.stream);  // drain stream

    CUDA_TRACY_COLLECT();   // all events now complete; safe to read timestamps
    // ... D2H copy ...
}
```

For programs that never call `toFloatArray`, add an explicit:

```lean
@[extern "lean_cuda_tracy_collect"]
opaque Cuda.tracyCollect : IO Unit
-- Call periodically in long-running GPU loops to keep Tracy fed
```

### 10.5 CPU-side zones from Lean

For CPU-side profiling (JIT compile time, COW copies, etc.) use Tracy's C API
directly in the FFI functions:

```c
lean_obj_res lean_cuda_map_expr(...) {
    ZoneScopedN("cuda_map_expr");          // CPU zone for the whole FFI call
    {
        ZoneScopedN("kernel_cache_lookup");
        fn = kernel_cache_get(expr_src);   // shows JIT vs cache hit
    }
    launch_map_kernel(fn, dst, src, n, expr_src);  // GPU zone inside
    return result;
}
```

You can also expose zone markers to Lean directly:

```lean
@[extern "lean_tracy_zone_begin"]
opaque Tracy.zoneBegin (name : String) : IO Unit

@[extern "lean_tracy_zone_end"]
opaque Tracy.zoneEnd : IO Unit
```

### 10.6 Build system integration

Tracy is header-only with a single `TracyClient.cpp` to compile in:

```cmake
option(TRACY_ENABLE "Enable Tracy profiling" OFF)

if(TRACY_ENABLE)
    find_path(TRACY_INCLUDE_DIR tracy/Tracy.hpp
              HINTS $ENV{TRACY_DIR}/public)
    target_include_directories(cuda_lean PRIVATE ${TRACY_INCLUDE_DIR})
    target_sources(cuda_lean PRIVATE ${TRACY_INCLUDE_DIR}/TracyClient.cpp)
    target_compile_definitions(cuda_lean PUBLIC TRACY_ENABLE)
    # For OpenCL zones:
    target_compile_definitions(cuda_lean PUBLIC TRACY_ENABLE_OPENCL)
endif()
```

Build with profiling: `cmake -DTRACY_ENABLE=ON ..`
Build without: `cmake ..` — all `CUDA_TRACY_ZONE` macros expand to nothing, zero overhead.

### 10.7 What you see in the Tracy profiler

```
CPU  ───[lean_cuda_map_expr]──────────────────────────────────────────
          [cache_lookup]  [kernel_launch_call]
GPU  ────────────────────────────[map_kernel "x*2" 1M elems 0.24ms]──
```

- GPU kernel durations with exact device timing
- CPU/GPU overlap and gaps visible at a glance
- JIT compile spikes on first run vs. cache hits
- COW copy durations (D2D memcpy shows as a GPU zone too)
- `toFloatArray` D2H transfer duration

---

## 11. Build System

Update `CMakeLists.txt` to add Tracy and OpenCL alongside CUDA:

```cmake
# c/CMakeLists.txt
cmake_minimum_required(VERSION 3.18)
project(CudaLean LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCL)

option(TRACY_ENABLE "Enable Tracy profiling" OFF)

add_library(cuda_lean SHARED
    cuda_lean_context.cu
    cuda_lean_array.cu
    cuda_lean_ops.cu
    cuda_lean_jit.cu
    cuda_lean_tracy.cu)

target_link_libraries(cuda_lean
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nvrtc)

if(OpenCL_FOUND)
    target_link_libraries(cuda_lean OpenCL::OpenCL)
    target_compile_definitions(cuda_lean PRIVATE CUDA_LEAN_OPENCL)
endif()

if(TRACY_ENABLE)
    find_path(TRACY_INCLUDE_DIR tracy/Tracy.hpp HINTS $ENV{TRACY_DIR}/public)
    target_include_directories(cuda_lean PRIVATE ${TRACY_INCLUDE_DIR})
    target_sources(cuda_lean PRIVATE ${TRACY_INCLUDE_DIR}/TracyClient.cpp)
    target_compile_definitions(cuda_lean PUBLIC TRACY_ENABLE TRACY_ENABLE_OPENCL)
endif()

target_compile_options(cuda_lean PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
```

```lean
-- lakefile.lean
require mathlib from git "https://github.com/leanprover-community/mathlib4"

lean_lib CudaLean

target_c_lib cuda_lean where
  srcDir := "c"
  build := "cmake"
```

---

## 12. Phased Implementation Plan

### Phase 1 — Scaffolding (no GPU ops yet)
- `initCudaContext` and global stream
- Opaque `Cuda.FloatArray` type with finalizer using `cudaFreeAsync`
- `alloc`, `fill`, `ofFloatArray`, `toFloatArray`
- Verify RC/finalizer behaviour with a simple round-trip test

### Phase 2 — Basic ops
- `add`, `sub`, `mul`, `scale` with COW
- Write tests that stress COW: shared arrays, aliased inputs
- Verify with CUDA-memcheck / compute-sanitizer that no races occur

### Phase 3 — JIT map
- `Cuda.Expr` AST and `toSource` codegen
- NVRTC compile path with error reporting
- In-process kernel cache with `g_cache_mutex`
- `mapStr` for debugging

### Phase 4 — Disk cache + warmup
- PTX persistence to `~/.cache/cuda_lean/`
- SM-version keying
- Warmup helper: `Cuda.FloatArray.warmup : List Cuda.Expr → IO Unit`

### Phase 5 — Tracy profiling
- Add `TRACY_ENABLE` CMake option
- CUDA GPU context + `CUDA_TRACY_ZONE` around all kernel launches
- OpenCL GPU context + `TracyCLZone` around all enqueues
- `CUDA_TRACY_COLLECT()` in `toFloatArray` and explicit `tracyCollect`
- CPU zones for JIT compile, COW, cache lookup
- Verify GPU timeline shows correct durations in Tracy UI

### Phase 6 — Hardening
- `compute-sanitizer` clean run on full test suite
- Stress test: 10k random pure expression DAGs, check RC accounting
- Benchmark: verify JIT latency is zero on warm runs
- Tracy profile of benchmark: confirm GPU utilisation, no unexpected gaps
