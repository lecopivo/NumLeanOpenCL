import NumLean.Interfaces.BlasOps.Basic
import NumLeanOpenCL.Data.Float32Array.Basic

namespace NumLean

instance : BLASOps OpenCLFloat32Array OpenCLFloat32 where
  axpby := sorry
  scal := sorry

end NumLean
