namespace NumLean

/--
Initialise the process-wide OpenCL context and command queue.

The C implementation currently selects the first available GPU device and falls
back to a CPU device if no GPU is available. It is safe to call this multiple
times; only the first call performs OpenCL initialization.
-/
@[extern "numlean_opencl_init"]
opaque initOpenCLContext : IO Unit


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
