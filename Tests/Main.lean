import NumLeanOpenCL
import NumLeanOpenCL.Data.Float32Array.Map

open NumLean

def zero : Float32 := 0
def one : Float32 := 1
def two : Float32 := 2
def three : Float32 := 3
def four : Float32 := 4
def five : Float32 := 5
def six : Float32 := 6
def ten : Float32 := 10
def twenty : Float32 := 20
def negOne : Float32 := -1
def negTwo : Float32 := -2
def negThree : Float32 := -3
def twelve : Float32 := 12
def fourteen : Float32 := 14
def twentyFour : Float32 := 24

def mkOpenCL (xs : Array Float32) : OpenCLFloat32Array :=
  OpenCLFloat32Array.ofArray xs

def fail (msg : String) : IO α :=
  throw <| IO.userError msg

def assertEq (label : String) (actual expected : UInt32) : IO Unit :=
  unless actual == expected do
    fail s!"{label}: got {actual}, expected {expected}"

def assertF32Bits (label : String) (actual expected : Float32) : IO Unit :=
  assertEq label actual.toBits expected.toBits

def assertArrayBits (label : String) (actual : OpenCLFloat32Array) (expected : Array Float32) : IO Unit := do
  let data := actual.toArray
  unless data.size == expected.size do
    fail s!"{label}: size got {data.size}, expected {expected.size}"
  for h : i in [0:data.size] do
    assertF32Bits s!"{label}[{i}]" data[i] expected[i]!

def assertScalarBits (label : String) (actual : OpenCLFloat32) (expected : Float32) : IO Unit :=
  assertF32Bits label actual.get expected

def assertNear (label : String) (actual expected tol : Float) : IO Unit := do
  let d := actual - expected
  let abs := if d < 0.0 then -d else d
  unless abs <= tol do
    fail s!"{label}: got {actual}, expected {expected}, tolerance {tol}"

def testBlas1 : IO Unit := do
  let x := mkOpenCL #[one, two, negThree]
  assertArrayBits "copy" x.copy #[one, two, negThree]
  assertArrayBits "scal" (x.copy.scal two) #[two, four, (-6 : Float32)]
  assertArrayBits "mapUnsafe" (unsafe x.copy.mapUnsafe "x * 2.0f + (float)gid") #[two, five, (-4 : Float32)]
  assertArrayBits "mapInContextUnsafe"
    (unsafe (mkOpenCL #[one, two, three]).mapInContextUnsafe #[mkOpenCL #[ten, twenty]] "x + (gid < ctx0_size ? ctx0[gid] : 0.0f)")
    #[(11 : Float32), (22 : Float32), three]

  let ax := mkOpenCL #[one, two, three]
  let ay := mkOpenCL #[ten, twenty]
  assertArrayBits "axpy capped" (OpenCLFloat32Array.axpy two ax ay) #[twelve, twentyFour]

  let addA := mkOpenCL #[one, two, three]
  let addB := mkOpenCL #[ten]
  assertArrayBits "add capped" (addA + addB) #[(11 : Float32), two, three]

  assertScalarBits "sum" (mkOpenCL #[one, two, three] |>.sum) six
  assertScalarBits "dot capped" (OpenCLFloat32Array.dot (mkOpenCL #[one, two, three]) (mkOpenCL #[four, five])) fourteen
  assertScalarBits "asum" (OpenCLFloat32Array.asum (mkOpenCL #[negOne, two, negThree])) six
  assertNear "nrm2" (OpenCLFloat32Array.nrm2 (mkOpenCL #[three, four])).get.toFloat 5.0 0.0001

  let (sx, sy) := OpenCLFloat32Array.swap (mkOpenCL #[one, two]) (mkOpenCL #[three, four, five])
  assertArrayBits "swap x" sx #[three, four]
  assertArrayBits "swap y" sy #[one, two, five]

  let (rx, ry) := OpenCLFloat32Array.rot zero one (mkOpenCL #[one, two]) (mkOpenCL #[three, four])
  assertArrayBits "rot x" rx #[three, four]
  assertArrayBits "rot y" ry #[negOne, negTwo]

  unless (mkOpenCL #[one, two]) == (mkOpenCL #[one, two]) do
    fail "beq equal failed"
  if (mkOpenCL #[one, two]) == (mkOpenCL #[one, three]) then
    fail "beq unequal failed"

def testKernelDSL : IO Unit := do
  let expr : Float32KernelExpr := .x * .lit two + .gid
  let input := #[one, two, negThree]
  let expected := expr.mapArraySpec #[] input
  assertArrayBits "dsl mapExpr" (mkOpenCL input |>.mapExpr expr) expected

  let ctxExpr : Float32KernelExpr :=
    .x + .ifInBounds 0 .gid (.ctx 0 .gid) (.lit zero)
  let ctx := #[#[ten, twenty]]
  let input := #[one, two, three]
  let expected := ctxExpr.mapArraySpec ctx input
  assertArrayBits "dsl mapInContextExpr" (mkOpenCL input |>.mapInContextExpr #[mkOpenCL #[ten, twenty]] ctxExpr) expected

  let source := ctxExpr.toOpenCL
  unless source.contains "ctx0" && source.contains "ctx0_size" do
    fail s!"dsl source missing context references: {source}"

def map1 (xs : OpenCLFloat32Array) : OpenCLFloat32Array :=
  xs.mapFinIdx fun i x _ => x + i • (1 : Float32)

def map2 (xs : OpenCLFloat32Array) : OpenCLFloat32Array :=
  xs.mapFinIdx fun i x _ => x - i • (1 : Float32)

def map3 (xs : OpenCLFloat32Array) : OpenCLFloat32Array :=
  xs.mapFinIdx fun i x _ => x * (i • (1 : Float32)) + x

def map4 (xs : OpenCLFloat32Array) : OpenCLFloat32Array :=
  xs.mapFinIdx fun i x _ =>
    let leftIdx := if i < 1 then xs.size - 1 else i - 1
    let rightIdx := if i + 1 < xs.size then i + 1 else 0
    xs.toArray[leftIdx]?.getD (0 : Float32) - (2 • (1 : Float32)) * x + xs.toArray[rightIdx]?.getD (0 : Float32)

def testMapFinIdx : IO Unit := do
  let xs := mkOpenCL #[one, two, three]
  assertArrayBits "mapFinIdx index"
    (map1 xs)
    #[one, three, five]

  assertArrayBits "mapFinIdx subtraction"
    (map2 xs)
    #[one, one, one]

  assertArrayBits "mapFinIdx fused multiply-add shape"
    (map3 xs)
    #[one, four, (9 : Float32)]

  let ys := mkOpenCL #[one, two, four, ten]
  assertArrayBits "mapFinIdx laplacian wrap"
    (map4 ys)
    #[ten, one, four, (-15 : Float32)]

partial def iterateScal (n : Nat) (x : OpenCLFloat32Array) : OpenCLFloat32Array :=
  if n == 0 then x else iterateScal (n - 1) (x.scal two)

partial def iterateAxpy (n : Nat) (x y : OpenCLFloat32Array) : OpenCLFloat32Array :=
  if n == 0 then y else iterateAxpy (n - 1) x (OpenCLFloat32Array.axpy one x y)

def testExecutionModel : IO Unit := do
  let base := mkOpenCL #[one, two, three]
  let alias := base
  let scaled := base.scal two
  assertArrayBits "cow alias unchanged after scal" alias #[one, two, three]
  assertArrayBits "cow scaled" scaled #[two, four, six]

  let copied := alias.copy
  let copiedScaled := copied.scal ten
  assertArrayBits "copy source unchanged" alias #[one, two, three]
  assertArrayBits "copy target independent" copiedScaled #[ten, twenty, (30 : Float32)]

  let ordered := iterateAxpy 64 (mkOpenCL #[one]) (mkOpenCL #[zero])
  assertArrayBits "queue ordered axpy chain" ordered #[(64 : Float32)]

  let pow := iterateScal 10 (mkOpenCL #[one])
  assertArrayBits "queue ordered scal chain" pow #[(1024 : Float32)]

  let shared := mkOpenCL #[one, two]
  let first := shared + mkOpenCL #[one, one]
  let second := shared + mkOpenCL #[two, two]
  assertArrayBits "shared original remains" shared #[one, two]
  assertArrayBits "shared add first" first #[two, three]
  assertArrayBits "shared add second" second #[three, four]

  let mapShared := mkOpenCL #[one, two]
  let mapAlias := mapShared
  let mapped := unsafe mapShared.mapUnsafe "x + 1.0f"
  assertArrayBits "mapUnsafe cow alias unchanged" mapAlias #[one, two]
  assertArrayBits "mapUnsafe mapped" mapped #[two, three]

  let contextShared := mkOpenCL #[one, two]
  let contextAlias := contextShared
  let contextMapped := unsafe contextShared.mapInContextUnsafe #[mkOpenCL #[three, four]] "x + ctx0[gid]"
  assertArrayBits "mapInContextUnsafe cow alias unchanged" contextAlias #[one, two]
  assertArrayBits "mapInContextUnsafe mapped" contextMapped #[four, six]

def main : IO Unit := do
  NumLean.initOpenCLContext
  testBlas1
  testKernelDSL
  testMapFinIdx
  testExecutionModel
  IO.println "NumLeanOpenCL runtime tests passed"
