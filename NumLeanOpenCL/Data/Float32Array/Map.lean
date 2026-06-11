import NumLeanOpenCL.KernelExpr
import NumLeanOpenCL.Data.Float32Array

namespace NumLean

namespace OpenCLFloat32Array

unsafe def mapImpl  (xs : OpenCLFloat32Array) (f : Nat → Float32 → Float32)
    {kernel : KernelExpr Float32 .real}
    (_hkernel : ∀ x i, HasKernelExpr x i [] (f i x) kernel) :
    OpenCLFloat32Array :=
  let kernelCode := Float32KernelExpr.toOpenCL kernel
  mapUnsafe xs kernelCode


-- @[implemented_by mapImpl]
def mapFinIdx (xs : OpenCLFloat32Array) (f : (i : Nat) → Float32 → i < xs.size → Float32)
    {kernel : KernelExpr Float32 .real} {ctx : List OpenCLFloat32Array}
    (_hkernel : ∀ x i h, HasKernelExpr x i ctx (f i x h) kernel := by apply_rulesets [compile_kernel_expr]) :
    OpenCLFloat32Array :=
  let kernelCode := Float32KernelExpr.toOpenCL kernel
  unsafe mapInContextUnsafe xs ctx.toArray kernelCode


-- this is the reference implementation and should be the definition of mapFinIdx
-- todo: once implemented_by is fixed we can prove this theorem by rfl
theorem map_eq_data_map (xs : OpenCLFloat32Array) (f : (i : Nat) → Float32 → i < xs.size → Float32)
    {kernel : KernelExpr Float32 .real} {ctx : List OpenCLFloat32Array}
    (_hkernel : ∀ x i h, HasKernelExpr x i ctx (f i x h) kernel) :
  (xs.mapFinIdx f) = ⟨xs.data.mapFinIdx f⟩ := sorry

@[simp, grind =]
theorem size_mapIdx (xs : OpenCLFloat32Array) (f : (i : Nat) → Float32 → i < xs.size → Float32)
    {kernel : KernelExpr Float32 .real} {ctx : List OpenCLFloat32Array}
    (_hkernel : ∀ x i h, HasKernelExpr x i ctx (f i x h) kernel) :
  (xs.mapFinIdx f).size = xs.size := by rw[map_eq_data_map]; simp [OpenCLFloat32Array.size]

@[simp]
theorem getElem_map (xs : OpenCLFloat32Array) (f : (i : Nat) → Float32 → i < xs.size → Float32)
    {kernel : KernelExpr Float32 .real} {ctx : List OpenCLFloat32Array}
    (_hkernel : ∀ x i h, HasKernelExpr x i ctx (f i x h) kernel)
    (i : Nat) (hi : i < xs.size) :
    (xs.mapFinIdx f)[i]'(by grind) = f i xs[i] hi := by
  simp [map_eq_data_map];
  simp only [getElem, get, toArray]
  have hi' : i < xs.data.size := hi
  simp [getElem!_pos, hi']
