import NumLeanOpenCL
import NumLeanOpenCL.Data.Float32Array.Map

open NumLean

def one : Float32 := 1
def two : Float32 := 2
def half : Float32 := 0.5

def mkHostInput (n salt : Nat) : Array Float32 :=
  let rec loop (i : Nat) (xs : Array Float32) :=
    if i < n then
      let v := (((i + salt) % 17) + 1) • (1 : Float32)
      loop (i + 1) (xs.push v)
    else
      xs
  loop 0 (Array.mkEmpty n)

def indexMap (xs : OpenCLFloat32Array) : OpenCLFloat32Array :=
  xs.mapFinIdx fun i x _ => x * (i • (1 : Float32)) + x

partial def chainScal (iters : Nat) (x : OpenCLFloat32Array) : OpenCLFloat32Array :=
  if iters == 0 then x else chainScal (iters - 1) (x.scal half)

partial def chainAxpy (iters : Nat) (x y : OpenCLFloat32Array) : OpenCLFloat32Array :=
  if iters == 0 then y else chainAxpy (iters - 1) x (OpenCLFloat32Array.axpy one x y)

def main (args : List String) : IO Unit := do
  let n := args[0]?.bind String.toNat? |>.getD 65536
  let salt := args[1]?.bind String.toNat? |>.getD 0
  NumLean.initOpenCLContext

  let baseHost := mkHostInput n salt
  let auxHost := mkHostInput n (salt + 7)

  NumLean.clearOpenCLProfile
  NumLean.markOpenCLProfile "profile/start"

  let base := OpenCLFloat32Array.ofArray baseHost
  NumLean.markOpenCLProfile s!"after/base-upload/{base.size}"
  let aux := OpenCLFloat32Array.ofArray auxHost
  NumLean.markOpenCLProfile s!"after/aux-upload/{aux.size}"

  let x := base.copy
  let x := chainScal 8 x
  NumLean.markOpenCLProfile s!"after/scal-chain/{x.size}"
  let x := indexMap x
  NumLean.markOpenCLProfile s!"after/mapFinIdx/{x.size}"
  let x := unsafe x.mapUnsafe "x * 1.0001f + (float)(gid & 7)"
  NumLean.markOpenCLProfile s!"after/mapUnsafe/{x.size}"
  let y := chainAxpy 8 x aux
  NumLean.markOpenCLProfile s!"after/axpy-chain/{y.size}"
  let y := unsafe y.mapInContextUnsafe #[x] "x + (gid < ctx0_size ? 0.25f * ctx0[gid] : 0.0f)"
  NumLean.markOpenCLProfile s!"after/context-map/{y.size}"

  let data := y.toArray
  NumLean.markOpenCLProfile s!"after/readback/{data.size}"

  IO.println s!"profile workload: n={n}, salt={salt}, output_size={data.size}"
  IO.println s!"sample: y[0]={data[0]!}, y[n/2]={data[n / 2]!}, y[n-1]={data[n - 1]!}"
  IO.println (← NumLean.dumpOpenCLProfile)
