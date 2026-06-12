namespace NumLean

/--
OpenCL-backed float32 device array.

The field is the specification model. The runtime representation is provided by
the externs below and stores a `cl_mem` buffer plus CPU-side metadata.
-/
structure OpenCLFloat32Array where
  data : Array Float32

attribute [extern "numlean_opencl_float32arrayopencl_mk"] OpenCLFloat32Array.mk
attribute [extern "numlean_opencl_float32arrayopencl_data"] OpenCLFloat32Array.data

namespace OpenCLFloat32Array

attribute [ext] OpenCLFloat32Array

/-- Allocate an OpenCL float32 buffer with capacity for at least `c` elements. -/
@[extern "numlean_opencl_float32arrayopencl_empty_with_capacity"]
def emptyWithCapacity (c : @& Nat) : OpenCLFloat32Array :=
  { data := Array.mkEmpty c }

def empty : OpenCLFloat32Array :=
  emptyWithCapacity 0

@[extern "numlean_opencl_float32arrayopencl_of_array"]
def ofArray (xs : Array Float32) : OpenCLFloat32Array :=
  ⟨xs⟩

instance : Inhabited OpenCLFloat32Array where
  default := empty

instance : EmptyCollection OpenCLFloat32Array where
  emptyCollection := OpenCLFloat32Array.empty

@[extern "numlean_opencl_float32arrayopencl_push"]
def push : OpenCLFloat32Array -> Float32 -> OpenCLFloat32Array
  | ⟨xs⟩, x => ⟨xs.push x⟩

@[extern "numlean_opencl_float32arrayopencl_size", tagged_return]
def size : (@& OpenCLFloat32Array) -> Nat
  | ⟨xs⟩ => xs.size

@[extern "numlean_opencl_float32arrayopencl_usize"]
def usize (xs : @& OpenCLFloat32Array) : USize :=
  xs.size.toUSize

@[inline]
def isEmpty (xs : OpenCLFloat32Array) : Bool :=
  xs.size == 0

/-- Read the OpenCL buffer back as a host `Array Float32`. -/
@[extern "numlean_opencl_float32arrayopencl_to_array"]
def toArray (xs : @& OpenCLFloat32Array) : Array Float32 := xs.data

@[extern "numlean_opencl_float32arrayopencl_get"]
def get : (xs : @& OpenCLFloat32Array) -> (i : @& Nat) -> (h : i < xs.size := by get_elem_tactic) -> Float32
  | xs, i, _ => xs.toArray[i]!

@[extern "numlean_opencl_float32arrayopencl_get_bang"]
def get! : (xs : @& OpenCLFloat32Array) -> (i : @& Nat) -> Float32
  | xs, i => xs.toArray[i]!

instance : GetElem OpenCLFloat32Array Nat Float32 (fun xs i => i < xs.size) where
  getElem xs i h := xs.get i h

@[extern "numlean_opencl_float32arrayopencl_beq"]
def beq : OpenCLFloat32Array -> OpenCLFloat32Array -> Bool
  | ⟨a⟩, ⟨b⟩ => a == b

instance : BEq OpenCLFloat32Array where
  beq := OpenCLFloat32Array.beq

private partial def addSpecLoop (a b : Array Float32) (i stop : Nat) : Array Float32 :=
  if i < stop then
    addSpecLoop (a.set! i (a[i]! + b[i]!)) b (i + 1) stop
  else
    a

/--
Add `b` into `a` and return an array with `a.size` elements.

Only indices `< min a.size b.size` are modified. The extern implementation mutates
the first OpenCL buffer when possible and returns the first argument.
-/
@[extern "numlean_opencl_float32arrayopencl_add"]
def add : OpenCLFloat32Array -> OpenCLFloat32Array -> OpenCLFloat32Array
  | ⟨a⟩, ⟨b⟩ => ⟨addSpecLoop a b 0 (Nat.min a.size b.size)⟩

instance : Add OpenCLFloat32Array where
  add := OpenCLFloat32Array.add

@[extern "numlean_opencl_float32arrayopencl_copy"]
def copy : OpenCLFloat32Array -> OpenCLFloat32Array
  | ⟨x⟩ => ⟨x⟩

@[extern "numlean_opencl_float32arrayopencl_scal"]
def scal : Float32 -> OpenCLFloat32Array -> OpenCLFloat32Array
  | alpha, ⟨x⟩ => ⟨x.map (alpha * ·)⟩

/--
Unsafe elementwise map using a raw OpenCL expression string.

The string is inserted into a kernel as the expression assigned back to the input
buffer. The variables available to the expression are `x : float` and
`gid : size_t`. The extern implementation mutates the array in-place when it is
exclusive and clones first when it is shared.

This is unsafe because malformed or malicious strings are passed to the OpenCL
compiler without validation.
-/
@[extern "numlean_opencl_float32arrayopencl_map_unsafe"]
unsafe def mapUnsafe : OpenCLFloat32Array -> String -> OpenCLFloat32Array
  | xs, _ => xs

/--
Unsafe elementwise map with additional OpenCL array context.

The string is inserted into a kernel as the expression assigned back to the input
buffer. The variables available to the expression are:

- `x : float`, the current value of the mutated array
- `gid : size_t`, the element index
- `ctx0`, `ctx1`, ... as `__global const float*` context arrays
- `ctx0_size`, `ctx1_size`, ... as `ulong` context sizes

The extern implementation mutates the first array in-place when exclusive and
clones first when shared. This is unsafe because the expression string is passed
to the OpenCL compiler without validation.
-/
@[extern "numlean_opencl_float32arrayopencl_map_in_context_unsafe"]
unsafe def mapInContextUnsafe : OpenCLFloat32Array -> Array OpenCLFloat32Array -> String -> OpenCLFloat32Array
  | xs, _, _ => xs

private partial def axpySpecLoop (alpha : Float32) (x y : Array Float32) (i stop : Nat) : Array Float32 :=
  if i < stop then
    axpySpecLoop alpha x (y.set! i (y[i]! + alpha * x[i]!)) (i + 1) stop
  else
    y

@[extern "numlean_opencl_float32arrayopencl_axpy"]
def axpy : Float32 -> OpenCLFloat32Array -> OpenCLFloat32Array -> OpenCLFloat32Array
  | alpha, ⟨x⟩, ⟨y⟩ => ⟨axpySpecLoop alpha x y 0 (Nat.min x.size y.size)⟩

@[extern "numlean_opencl_float32arrayopencl_swap"]
def swap : OpenCLFloat32Array -> OpenCLFloat32Array -> OpenCLFloat32Array × OpenCLFloat32Array
  | ⟨x⟩, ⟨y⟩ => (⟨y⟩, ⟨x⟩)

private partial def rotSpecLoop (c s : Float32) (x y : Array Float32) (i stop : Nat) : Array Float32 × Array Float32 :=
  if i < stop then
    let xi := x[i]!
    let yi := y[i]!
    rotSpecLoop c s (x.set! i (c * xi + s * yi)) (y.set! i (c * yi - s * xi)) (i + 1) stop
  else
    (x, y)

@[extern "numlean_opencl_float32arrayopencl_rot"]
def rot : Float32 -> Float32 -> OpenCLFloat32Array -> OpenCLFloat32Array -> OpenCLFloat32Array × OpenCLFloat32Array
  | c, s, ⟨x⟩, ⟨y⟩ =>
    let (x', y') := rotSpecLoop c s x y 0 (Nat.min x.size y.size)
    (⟨x'⟩, ⟨y'⟩)

end OpenCLFloat32Array

/-- GPU-resident scalar float32, represented as a size-one OpenCL array. -/
structure OpenCLFloat32 where
  data : OpenCLFloat32Array
  h_size : data.size = 1

namespace OpenCLFloat32

private def ofFloat32Spec (x : Float32) : OpenCLFloat32Array :=
  OpenCLFloat32Array.ofArray #[x]

axiom ofFloat32Spec_size (x : Float32) : (ofFloat32Spec x).size = 1

/-- Create a GPU-resident scalar as a size-one OpenCL array. -/
@[extern "numlean_opencl_float32opencl_of_float32"]
def ofFloat32 (x : Float32) : OpenCLFloat32 :=
  ⟨ofFloat32Spec x, ofFloat32Spec_size x⟩

end OpenCLFloat32

namespace OpenCLFloat32Array

private def sumSpec (xs : OpenCLFloat32Array) : OpenCLFloat32Array :=
  ⟨#[xs.toArray.foldl (· + ·) (0 : Float32)]⟩

axiom sumSpec_size (xs : OpenCLFloat32Array) : (sumSpec xs).size = 1

/--
Sum the array on the GPU and return a GPU scalar.

The extern implementation only enqueues work and returns a `OpenCLFloat32`; it
does not block until the scalar is read with `OpenCLFloat32.get`/`.data`.
-/
@[extern "numlean_opencl_float32arrayopencl_sum"]
def sum (xs : OpenCLFloat32Array) : OpenCLFloat32 :=
  ⟨sumSpec xs, sumSpec_size xs⟩

private partial def dotSpecLoop (x y : Array Float32) (i stop : Nat) (acc : Float32) : Float32 :=
  if i < stop then
    dotSpecLoop x y (i + 1) stop (acc + x[i]! * y[i]!)
  else
    acc

@[extern "numlean_opencl_float32arrayopencl_dot"]
def dot (x y : OpenCLFloat32Array) : OpenCLFloat32 :=
  OpenCLFloat32.ofFloat32 <| dotSpecLoop x.toArray y.toArray 0 (Nat.min x.size y.size) (0 : Float32)

@[extern "numlean_opencl_float32arrayopencl_asum"]
def asum (x : OpenCLFloat32Array) : OpenCLFloat32 :=
  OpenCLFloat32.ofFloat32 <| x.toArray.foldl (fun acc v => acc + if v < (0 : Float32) then -v else v) (0 : Float32)

@[extern "numlean_opencl_float32arrayopencl_nrm2"]
def nrm2 (x : OpenCLFloat32Array) : OpenCLFloat32 :=
  OpenCLFloat32.ofFloat32 <| Float.toFloat32 <| Float.sqrt <| x.toArray.foldl (fun acc v => acc + v.toFloat * v.toFloat) 0.0

end OpenCLFloat32Array

namespace OpenCLFloat32

/-- Read a GPU-resident scalar back to the host. This is the synchronization point. -/
@[inline]
def get (x : @& OpenCLFloat32) : Float32 :=
  OpenCLFloat32Array.get! x.data 0

end OpenCLFloat32

end NumLean
