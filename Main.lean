import NumLeanOpenCL

def main : IO Unit := do
  NumLean.initOpenCLContext
  let xs := NumLean.OpenCLFloat32Array.emptyWithCapacity 16
  let xs := xs.push (Float32.ofBits 0x3f800000)
  let xs := xs.push (Float32.ofBits 0x40000000)
  let ys := NumLean.OpenCLFloat32Array.emptyWithCapacity 16
  let ys := ys.push (Float32.ofBits 0x40400000)
  let zs := xs + ys
  let s := zs.sum
  IO.println s!"OpenCL OpenCLFloat32Array size: {xs.size}"
  IO.println s!"OpenCL OpenCLFloat32Array[0]: {xs[0]}"
  IO.println s!"OpenCL OpenCLFloat32Array add size: {zs.size}"
  IO.println s!"OpenCL OpenCLFloat32Array add[0]: {zs.get! 0}"
  IO.println s!"OpenCL OpenCLFloat32Array add[1]: {zs.get! 1}"
  IO.println s!"OpenCL OpenCLFloat32Array beq self: {zs == zs}"
  IO.println s!"OpenCL OpenCLFloat32 sum: {s.get}"
  IO.println s!"Hello, {hello}!"
