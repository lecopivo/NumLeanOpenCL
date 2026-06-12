namespace NumLean

/--
Initialise the process-wide OpenCL context and command queue.

The C implementation currently selects the first available GPU device and falls
back to a CPU device if no GPU is available. It is safe to call this multiple
times; only the first call performs OpenCL initialization.
-/
@[extern "numlean_opencl_init"]
opaque initOpenCLContext : IO Unit

/--
Enable collection of OpenCL profiling records.

When profiling is enabled, instrumented OpenCL operations append records to the
process-wide profile buffer. Device operations are recorded from OpenCL events;
host-side work such as kernel compilation is recorded with a monotonic host
clock. Use `clearOpenCLProfile` before a benchmark to start with an empty buffer.
-/
@[extern "numlean_opencl_profile_start"]
opaque startOpenCLProfiling : IO Unit

/--
Disable collection of new OpenCL profiling records.

Previously collected records remain available until `clearOpenCLProfile` is
called. `dumpOpenCLProfile` may still be used after profiling is stopped.
-/
@[extern "numlean_opencl_profile_stop"]
opaque stopOpenCLProfiling : IO Unit

/--
Clear the process-wide OpenCL profile buffer and start a new capture.

This releases stored OpenCL events, resets profiling counters, establishes a new
host-clock origin for host-side records, and enables profiling. It is the usual
entry point before running a workload that should be profiled.
-/
@[extern "numlean_opencl_profile_clear"]
opaque clearOpenCLProfile : IO Unit

/--
Insert a named marker into the OpenCL profile stream.

The marker is enqueued on the OpenCL command queue, so in an in-order queue it can
be used to bracket and force scheduling of preceding commands. To avoid Lean
laziness/hoisting when profiling pure expressions, make marker labels depend on
values produced by the stage being marked, for example `s!"after/{xs.size}"`.
-/
@[extern "numlean_opencl_profile_mark"]
opaque markOpenCLProfile : @& String → IO Unit

/--
Return the collected OpenCL profile as CSV-like text.

The output contains a short human-readable header followed by rows with columns:
`idx`, `kind`, `label`, `work_items`, `bytes`, device event times, and host times.
Rows with `kind = device` come from OpenCL event profiling. Rows with
`kind = host` record CPU-side work such as kernel compilation. The resulting text
can be pasted into `profile-viewer/index.html` for sorting and timeline viewing.
-/
@[extern "numlean_opencl_profile_dump"]
opaque dumpOpenCLProfile : IO String


open Function in
class HasOpenCLType (HostType : Type u) (DeviceType : outParam (Type v)) where
  toDevice : HostType → DeviceType
  toHost : DeviceType → HostType

  left_inv : LeftInverse toHost toDevice
  right_inv : RightInverse toHost toDevice

structure OpenCL (HostType : Type u) {DeviceType : Type v} [HasOpenCLType HostType DeviceType] where
  value : DeviceType

namespace OpenCL
variable {HostType : Type u} {DeviceType : Type v} [HasOpenCLType HostType DeviceType]

def toHost (x : OpenCL HostType) : HostType := HasOpenCLType.toHost x.value

def fromHost (x : HostType) : OpenCL HostType := ⟨HasOpenCLType.toDevice x⟩

-- @[simp]
-- theorem toHost_fromHost ...

-- @[simp]
-- theorem fromHost_toHost ...

end OpenCL

end NumLean
