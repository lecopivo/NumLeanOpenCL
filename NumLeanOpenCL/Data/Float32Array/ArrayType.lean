import NumLean.Interfaces.ArrayType.Basic
import NumLeanOpenCL.Data.Float32Array.Basic

namespace NumLean

instance : ArrayType OpenCLFloat32Array OpenCLFloat32 where
  toArray x := x.toArray.map .ofFloat32
  fromArray x := .ofArray (x.map (·.get))

  left_inv := sorry
  right_inv := sorry
  size := sorry
  size_spec := sorry
  emptyWithCapacity := sorry
  emptyWithCapacity_spec := sorry
  uget := sorry
  uget_spec := sorry
  get := sorry
  get_spec := sorry
  uset := sorry
  uset_spec := sorry
  set := sorry
  set_spec := sorry
  pop := sorry
  pop_spec := sorry
  replicate := sorry
  replicate_spec := sorry
  swap := sorry
  swap_spec := sorry
  push := sorry
  push_spec := sorry
  append := sorry
  append_spec := sorry
  copySlice := sorry
  copySlice_spec := sorry
  extractSlice := sorry
  extractSlice_spec := sorry

end NumLean
