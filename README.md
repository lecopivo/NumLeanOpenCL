# NumLeanOpenCL

NumLeanOpenCL is an experimental Lean 4 package for running `Float32` array computations on OpenCL devices while keeping a Lean-facing API and specification model.

The project currently focuses on an OpenCL-backed `OpenCLFloat32Array`, a small kernel-expression DSL, automatic compilation of selected Lean expressions to OpenCL kernels, runtime tests, and profiling tools for understanding queue scheduling and kernel execution.

## Requirements

- Lean 4 via `elan`, using the version in `lean-toolchain`.
- Lake.
- OpenCL headers and an OpenCL runtime/ICD.
- `libOpenCL.so` available at the path configured in `lakefile.lean`.

On Linux, this usually means installing your GPU vendor OpenCL runtime or a CPU runtime such as POCL, plus OpenCL development headers.

## Build And Test

Build the default executable:

```bash
lake build
```

Run the runtime tests:

```bash
lake test
```

The current `mapFinIdx` implementation has a known `sorry` warning because `@[implemented_by]` does not currently work with the required `autoParam` setup. The tests are expected to pass despite that warning.

## Basic Usage

Initialize OpenCL once before using the runtime-backed operations:

```lean
import NumLeanOpenCL

open NumLean

def main : IO Unit := do
  NumLean.initOpenCLContext
  let xs := OpenCLFloat32Array.ofArray #[Float32.ofBits 0x3f800000, Float32.ofBits 0x40000000]
  let ys := xs.scal (Float32.ofBits 0x40000000)
  IO.println ys.toArray
```

Prefer `OpenCLFloat32Array.ofArray` for input creation. It performs one bulk upload from a Lean `Array Float32` into an OpenCL buffer. Do not build large OpenCL arrays with repeated `push`; that creates many tiny operations and makes profiling/scheduling data misleading.

## Profiling API

The profiling API is in `NumLeanOpenCL/OpenCL/Basic.lean`.

```lean
NumLean.startOpenCLProfiling : IO Unit
NumLean.stopOpenCLProfiling : IO Unit
NumLean.clearOpenCLProfile : IO Unit
NumLean.markOpenCLProfile : @& String → IO Unit
NumLean.dumpOpenCLProfile : IO String
```

Typical usage:

```lean
NumLean.clearOpenCLProfile
NumLean.markOpenCLProfile "profile/start"

let xs := OpenCLFloat32Array.ofArray hostInput
NumLean.markOpenCLProfile s!"after/upload/{xs.size}"

let ys := xs.scal alpha
NumLean.markOpenCLProfile s!"after/scal/{ys.size}"

let out := ys.toArray
NumLean.markOpenCLProfile s!"after/readback/{out.size}"

IO.println (← NumLean.dumpOpenCLProfile)
```

`clearOpenCLProfile` clears old records and starts a new capture. `markOpenCLProfile` inserts a marker into the OpenCL command queue. Because Lean expressions can be lazy or hoisted if they are pure, marker labels should depend on values produced by the stage being marked, such as `xs.size` or `out.size`. This forces the relevant computation before the marker is enqueued.

`dumpOpenCLProfile` prints a CSV-like table. Device rows come from OpenCL event profiling. Host rows use a monotonic CPU clock and currently cover kernel compilation/build work such as:

- `compile/blas1`
- `compile/mapUnsafe`
- `compile/mapInContextUnsafe`

The output columns are:

```text
idx,kind,label,work_items,bytes,queued_us,submit_us,start_us,end_us,queue_to_submit_us,submit_to_start_us,duration_us,host_start_us,host_end_us,host_duration_us
```

For `kind = device`, the timing columns are OpenCL event profiling timestamps relative to the first event. For `kind = host`, the host timing columns describe CPU-side work such as compilation, and the device timing columns are filled to make the row visible in the same timeline.

## Profiling Executable

Build the profiling workload:

```bash
lake build opencl_profile
```

Run it with runtime arguments:

```bash
./.lake/build/bin/opencl_profile 65536 3
```

Arguments:

- first argument: array size, default `65536`
- second argument: input salt, default `0`

The runtime arguments are intentional: they prevent the workload from being a closed compile-time constant and make the profiling data easier to trust.

Example workflow:

```bash
./.lake/build/bin/opencl_profile 6553600 2 > profile.txt
```

## Profile Viewer Website

The static viewer is in:

```text
profile-viewer/index.html
```

Open it in a browser and paste the output of `dumpOpenCLProfile`, or load a saved profile text file.

Viewer features:

- sortable event table
- horizontal scheduling timeline
- time axis above the timeline
- drag-to-select zoom on a subinterval
- double-click or `Reset Zoom` to zoom out
- filtering by `kind` (`all`, `device`, `host`)
- label filtering, for example `kernel`, `compile`, `write/`, or `read/`
- idle-gap compression to make dense kernel clusters visible
- separate colors for queue delay, device wait, execution, host compilation, and transfers

Useful ways to inspect a profile:

- Use `Kind = device` to focus on OpenCL queue behavior.
- Use `Label Filter = kernel` to focus on kernels.
- Use `Label Filter = write/` or `read/` to inspect upload/readback.
- Enable `Compress idle gaps` when long host-side gaps make short kernel execution bars hard to see.
- Drag around a dense cluster to zoom into scheduler behavior.

## Interpreting Profile Data

For device rows:

- `queued_us`: when the command was queued by the host.
- `submit_us`: when the command was submitted to the device.
- `start_us`: when the device started executing the command.
- `end_us`: when the device finished the command.
- `queue_to_submit_us`: host/driver queue delay.
- `submit_to_start_us`: device-side wait, often because earlier commands are ahead in an in-order queue.
- `duration_us`: actual command execution time.

Large `submit_to_start_us` values for later kernels are normal on an in-order queue if many commands were queued quickly. The kernels may all be submitted near the same moment but execute sequentially.

Uploads and readbacks can dominate total time for large arrays. Host-side construction or conversion of Lean arrays can also create gaps that are not OpenCL device execution; use markers and host compile rows to distinguish those effects.

## Development Notes

- Keep OpenCL input construction bulk-oriented with `ofArray`.
- Use named `mapFinIdx` definitions when elaboration of inline lambdas becomes fragile.
- Avoid profiling closed, pure workloads if you want to reason about runtime ordering; depend on runtime arguments and force stage results in marker labels.
- The profiling viewer is intentionally static and has no build step.
