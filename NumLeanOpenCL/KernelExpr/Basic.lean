import NumLeanOpenCL.Data.Float32Array
import NumLean.Algebra.Float

namespace NumLean

inductive KernelType where
  | real
  | nat
  | int
deriving Repr, BEq

abbrev KernelType.denote (R : Type) : KernelType -> Type
  | .real => R
  | .nat => Nat
  | .int => Int

-- instance {R} [RLikeOps R] : RLikeOps (KernelType.denote R .real) :=

/-- Well-typed scalar/index expression syntax for kernels. -/
inductive KernelExpr (R : Type) : KernelType -> Type where
  | x : KernelExpr R .real
  | gid : KernelExpr R .nat
  | lit (v : R) : KernelExpr R .real
  | natLit (v : Nat) : KernelExpr R .nat
  | intLit (v : Int) : KernelExpr R .int
  | ctx (arrayIndex : Nat) (elemIndex : KernelExpr R .nat) : KernelExpr R .real
  | ctxSize (arrayIndex : Nat) : KernelExpr R .nat
  | cast (t : KernelType) {s : KernelType} (a : KernelExpr R s) : KernelExpr R t
  | add {t : KernelType} (a b : KernelExpr R t) : KernelExpr R t
  | sub {t : KernelType} (a b : KernelExpr R t) : KernelExpr R t
  | mul {t : KernelType} (a b : KernelExpr R t) : KernelExpr R t
  | div {t : KernelType} (a b : KernelExpr R t) : KernelExpr R t
  | neg {t : KernelType} (a : KernelExpr R t) : KernelExpr R t
  | abs {t : KernelType} (a : KernelExpr R t) : KernelExpr R t
  | sqrt (a : KernelExpr R .real) : KernelExpr R .real
  | sin (a : KernelExpr R .real) : KernelExpr R .real
  | cos (a : KernelExpr R .real) : KernelExpr R .real
  | exp (a : KernelExpr R .real) : KernelExpr R .real
  | log (a : KernelExpr R .real) : KernelExpr R .real
  | fma (a b c : KernelExpr R .real) : KernelExpr R .real
  | min {t : KernelType} (a b : KernelExpr R t) : KernelExpr R t
  | max {t : KernelType} (a b : KernelExpr R t) : KernelExpr R t
  | ifLt {c t : KernelType} (a b : KernelExpr R c) (thenExpr elseExpr : KernelExpr R t) : KernelExpr R t
  | ifInBounds {t : KernelType} (arrayIndex : Nat) (elemIndex : KernelExpr R .nat) (thenExpr elseExpr : KernelExpr R t) : KernelExpr R t
deriving Repr

namespace KernelExpr

instance : Add (KernelExpr R t) where add := .add
instance : Sub (KernelExpr R t) where sub := .sub
instance : Mul (KernelExpr R t) where mul := .mul
instance : Div (KernelExpr R t) where div := .div
instance : Neg (KernelExpr R t) where neg := .neg

def WellFormed (ctxSizes : Array Nat) : KernelExpr R t -> Prop
  | .x | .gid | .lit _ | .natLit _ | .intLit _ => True
  | .ctx arrayIndex elemIndex => arrayIndex < ctxSizes.size ∧ WellFormed ctxSizes elemIndex
  | .ctxSize arrayIndex => arrayIndex < ctxSizes.size
  | .cast _ a | .neg a | .abs a |
      .sqrt a | .sin a | .cos a | .exp a | .log a => WellFormed ctxSizes a
  | .add a b | .sub a b | .mul a b | .div a b | .min a b | .max a b =>
      WellFormed ctxSizes a ∧ WellFormed ctxSizes b
  | .fma a b c => WellFormed ctxSizes a ∧ WellFormed ctxSizes b ∧ WellFormed ctxSizes c
  | .ifLt a b thenExpr elseExpr =>
      WellFormed ctxSizes a ∧ WellFormed ctxSizes b ∧ WellFormed ctxSizes thenExpr ∧ WellFormed ctxSizes elseExpr
  | .ifInBounds arrayIndex elemIndex thenExpr elseExpr =>
      arrayIndex < ctxSizes.size ∧ WellFormed ctxSizes elemIndex ∧ WellFormed ctxSizes thenExpr ∧ WellFormed ctxSizes elseExpr

mutual
  def evalReal {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
      (ctx : Array (Array R)) (xv : R) (gidv : Nat) : KernelExpr R .real -> R
    | .x => xv
    | .lit v => v
    | .ctx arrayIndex elemIndex =>
        let i := evalNat ctx xv gidv elemIndex
        match ctx[arrayIndex]? with
        | some arr => arr[i]?.getD 0
        | none => 0
    | .cast _ a => evalCastReal ctx xv gidv a
    | .add a b => evalReal ctx xv gidv a + evalReal ctx xv gidv b
    | .sub a b => evalReal ctx xv gidv a - evalReal ctx xv gidv b
    | .mul a b => evalReal ctx xv gidv a * evalReal ctx xv gidv b
    | .div a b => evalReal ctx xv gidv a / evalReal ctx xv gidv b
    | .neg a => -evalReal ctx xv gidv a
    | .abs a => let v := evalReal ctx xv gidv a; if v < 0 then -v else v
    | .sqrt a => ROps.sqrt (evalReal ctx xv gidv a)
    | .sin a => RCOps.sin (evalReal ctx xv gidv a)
    | .cos a => RCOps.cos (evalReal ctx xv gidv a)
    | .exp a => RCOps.exp (evalReal ctx xv gidv a)
    | .log a => ROps.log (evalReal ctx xv gidv a)
    | .fma a b c => evalReal ctx xv gidv a * evalReal ctx xv gidv b + evalReal ctx xv gidv c
    | .min a b => let av := evalReal ctx xv gidv a; let bv := evalReal ctx xv gidv b; if av < bv then av else bv
    | .max a b => let av := evalReal ctx xv gidv a; let bv := evalReal ctx xv gidv b; if av < bv then bv else av
    | .ifLt a b thenExpr elseExpr =>
        if evalLt ctx xv gidv a b then evalReal ctx xv gidv thenExpr else evalReal ctx xv gidv elseExpr
    | .ifInBounds arrayIndex elemIndex thenExpr elseExpr =>
        let i := evalNat ctx xv gidv elemIndex
        let inBounds := match ctx[arrayIndex]? with | some arr => decide (i < arr.size) | none => false
        if inBounds then evalReal ctx xv gidv thenExpr else evalReal ctx xv gidv elseExpr
  termination_by e => sizeOf e

  def evalNat {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
      (ctx : Array (Array R)) (xv : R) (gidv : Nat) : KernelExpr R .nat -> Nat
    | .gid => gidv
    | .natLit v => v
    | .ctxSize arrayIndex => ctx.map Array.size |>.getD arrayIndex 0
    | .cast _ a => evalCastNat ctx xv gidv a
    | .add a b => evalNat ctx xv gidv a + evalNat ctx xv gidv b
    | .sub a b => evalNat ctx xv gidv a - evalNat ctx xv gidv b
    | .mul a b => evalNat ctx xv gidv a * evalNat ctx xv gidv b
    | .div a b => evalNat ctx xv gidv a / evalNat ctx xv gidv b
    | .neg a => evalNat ctx xv gidv a
    | .abs a => evalNat ctx xv gidv a
    | .min a b => Nat.min (evalNat ctx xv gidv a) (evalNat ctx xv gidv b)
    | .max a b => Nat.max (evalNat ctx xv gidv a) (evalNat ctx xv gidv b)
    | .ifLt a b thenExpr elseExpr =>
        if evalLt ctx xv gidv a b then evalNat ctx xv gidv thenExpr else evalNat ctx xv gidv elseExpr
    | .ifInBounds arrayIndex elemIndex thenExpr elseExpr =>
        let i := evalNat ctx xv gidv elemIndex
        let inBounds := match ctx[arrayIndex]? with | some arr => decide (i < arr.size) | none => false
        if inBounds then evalNat ctx xv gidv thenExpr else evalNat ctx xv gidv elseExpr
  termination_by e => sizeOf e

  def evalInt {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
      (ctx : Array (Array R)) (xv : R) (gidv : Nat) : KernelExpr R .int -> Int
    | .intLit v => v
    | .cast _ a => evalCastInt ctx xv gidv a
    | .add a b => evalInt ctx xv gidv a + evalInt ctx xv gidv b
    | .sub a b => evalInt ctx xv gidv a - evalInt ctx xv gidv b
    | .mul a b => evalInt ctx xv gidv a * evalInt ctx xv gidv b
    | .div a b => evalInt ctx xv gidv a / evalInt ctx xv gidv b
    | .neg a => -evalInt ctx xv gidv a
    | .abs a => let v := evalInt ctx xv gidv a; if v < 0 then -v else v
    | .min a b => let av := evalInt ctx xv gidv a; let bv := evalInt ctx xv gidv b; if av < bv then av else bv
    | .max a b => let av := evalInt ctx xv gidv a; let bv := evalInt ctx xv gidv b; if av < bv then bv else av
    | .ifLt a b thenExpr elseExpr =>
        if evalLt ctx xv gidv a b then evalInt ctx xv gidv thenExpr else evalInt ctx xv gidv elseExpr
    | .ifInBounds arrayIndex elemIndex thenExpr elseExpr =>
        let i := evalNat ctx xv gidv elemIndex
        let inBounds := match ctx[arrayIndex]? with | some arr => decide (i < arr.size) | none => false
        if inBounds then evalInt ctx xv gidv thenExpr else evalInt ctx xv gidv elseExpr
  termination_by e => sizeOf e

  def evalCastReal {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
      (ctx : Array (Array R)) (xv : R) (gidv : Nat) : {s : KernelType} -> KernelExpr R s -> R
    | .real, a => evalReal ctx xv gidv a
    | .nat, a => evalNat ctx xv gidv a • (1 : R)
    | .int, a => evalInt ctx xv gidv a • (1 : R)
  termination_by _ a => sizeOf a + 1

  def evalCastNat {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
      (ctx : Array (Array R)) (xv : R) (gidv : Nat) : {s : KernelType} -> KernelExpr R s -> Nat
    | .real, _ => 0
    | .nat, a => evalNat ctx xv gidv a
    | .int, a => (evalInt ctx xv gidv a).toNat
  termination_by _ a => sizeOf a + 1

  def evalCastInt {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
      (ctx : Array (Array R)) (xv : R) (gidv : Nat) : {s : KernelType} -> KernelExpr R s -> Int
    | .real, _ => 0
    | .nat, a => Int.ofNat (evalNat ctx xv gidv a)
    | .int, a => evalInt ctx xv gidv a
  termination_by _ a => sizeOf a + 1

  def evalLt {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
      (ctx : Array (Array R)) (xv : R) (gidv : Nat) : {t : KernelType} -> KernelExpr R t -> KernelExpr R t -> Bool
    | .real, a, b => decide (evalReal ctx xv gidv a < evalReal ctx xv gidv b)
    | .nat, a, b => decide (evalNat ctx xv gidv a < evalNat ctx xv gidv b)
    | .int, a, b => decide (evalInt ctx xv gidv a < evalInt ctx xv gidv b)
  termination_by _ a b => sizeOf a + sizeOf b + 1
end

def eval {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
    (ctx : Array (Array R)) (xv : R) (gidv : Nat) {t} (expr : KernelExpr R t) : (KernelType.denote R t) :=
  match t with
  | .real => evalReal ctx xv gidv expr
  | .int => evalInt ctx xv gidv expr
  | .nat => evalNat ctx xv gidv expr

def mapArraySpec {R : Type} [ROps R] [DecidableRel (· < · : R -> R -> Prop)]
    (expr : KernelExpr R .real) (ctx : Array (Array R)) (xs : Array R) : Array R :=
  xs.mapIdx fun gid x => evalReal ctx x gid expr

end KernelExpr

abbrev Float32KernelExpr := KernelExpr Float32 .real
abbrev KernelNatExpr := KernelExpr Float32 .nat
abbrev KernelIntExpr := KernelExpr Float32 .int

namespace Float32KernelExpr

def gid : Float32KernelExpr := KernelExpr.cast .real KernelExpr.gid
def ctxSize (arrayIndex : Nat) : Float32KernelExpr := KernelExpr.cast .real (KernelExpr.ctxSize arrayIndex)
def cast (t : KernelType) {s : KernelType} (a : KernelExpr Float32 s) : KernelExpr Float32 t := KernelExpr.cast t a
def abs (a : Float32KernelExpr) : Float32KernelExpr := KernelExpr.abs a
def sqrt (a : Float32KernelExpr) : Float32KernelExpr := KernelExpr.sqrt a
def sin (a : Float32KernelExpr) : Float32KernelExpr := KernelExpr.sin a
def cos (a : Float32KernelExpr) : Float32KernelExpr := KernelExpr.cos a
def exp (a : Float32KernelExpr) : Float32KernelExpr := KernelExpr.exp a
def log (a : Float32KernelExpr) : Float32KernelExpr := KernelExpr.log a
def fma (a b c : Float32KernelExpr) : Float32KernelExpr := KernelExpr.fma a b c
def min (a b : Float32KernelExpr) : Float32KernelExpr := KernelExpr.min a b
def max (a b : Float32KernelExpr) : Float32KernelExpr := KernelExpr.max a b
def ifLt {t : KernelType} (a b : KernelExpr Float32 t) (thenExpr elseExpr : Float32KernelExpr) : Float32KernelExpr :=
  KernelExpr.ifLt a b thenExpr elseExpr
def ifInBounds (arrayIndex : Nat) (elemIndex : KernelNatExpr) (thenExpr elseExpr : Float32KernelExpr) : Float32KernelExpr :=
  KernelExpr.ifInBounds arrayIndex elemIndex thenExpr elseExpr

private def floatSource (v : Float32) : String := s!"{v.toFloat}f"

mutual
  def toOpenCL : Float32KernelExpr -> String
    | .x => "x"
    | .lit v => floatSource v
    | .ctx arrayIndex elemIndex => s!"ctx{arrayIndex}[{toOpenCLNat elemIndex}]"
    | .cast .real a => toOpenCLCast .real a
    | .add a b => s!"({toOpenCL a} + {toOpenCL b})"
    | .sub a b => s!"({toOpenCL a} - {toOpenCL b})"
    | .mul a b => s!"({toOpenCL a} * {toOpenCL b})"
    | .div a b => s!"({toOpenCL a} / {toOpenCL b})"
    | .neg a => s!"(-{toOpenCL a})"
    | .abs a => s!"fabs({toOpenCL a})"
    | .sqrt a => s!"sqrt({toOpenCL a})"
    | .sin a => s!"sin({toOpenCL a})"
    | .cos a => s!"cos({toOpenCL a})"
    | .exp a => s!"exp({toOpenCL a})"
    | .log a => s!"log({toOpenCL a})"
    | .fma a b c => s!"fma({toOpenCL a}, {toOpenCL b}, {toOpenCL c})"
    | .min a b => s!"fmin({toOpenCL a}, {toOpenCL b})"
    | .max a b => s!"fmax({toOpenCL a}, {toOpenCL b})"
    | .ifLt a b thenExpr elseExpr => s!"(({toOpenCLAny a} < {toOpenCLAny b}) ? {toOpenCL thenExpr} : {toOpenCL elseExpr})"
    | KernelExpr.ifInBounds arrayIndex elemIndex thenExpr elseExpr =>
        s!"(({toOpenCLNat elemIndex} < ctx{arrayIndex}_size) ? {toOpenCL thenExpr} : {toOpenCL elseExpr})"
  termination_by e => sizeOf e

  def toOpenCLNat : KernelNatExpr -> String
    | .gid => "gid"
    | .natLit v => s!"((ulong){v})"
    | .ctxSize arrayIndex => s!"ctx{arrayIndex}_size"
    | .cast .nat a => toOpenCLCast .nat a
    | .add a b => s!"({toOpenCLNat a} + {toOpenCLNat b})"
    | .sub a b => s!"({toOpenCLNat a} - {toOpenCLNat b})"
    | .mul a b => s!"({toOpenCLNat a} * {toOpenCLNat b})"
    | .div a b => s!"({toOpenCLNat a} / {toOpenCLNat b})"
    | .neg a => toOpenCLNat a
    | .abs a => toOpenCLNat a
    | .min a b => s!"min({toOpenCLNat a}, {toOpenCLNat b})"
    | .max a b => s!"max({toOpenCLNat a}, {toOpenCLNat b})"
    | .ifLt a b thenExpr elseExpr => s!"(({toOpenCLAny a} < {toOpenCLAny b}) ? {toOpenCLNat thenExpr} : {toOpenCLNat elseExpr})"
    | KernelExpr.ifInBounds arrayIndex elemIndex thenExpr elseExpr =>
        s!"(({toOpenCLNat elemIndex} < ctx{arrayIndex}_size) ? {toOpenCLNat thenExpr} : {toOpenCLNat elseExpr})"
  termination_by e => sizeOf e

  def toOpenCLInt : KernelIntExpr -> String
    | .intLit v => s!"((long){v})"
    | .cast .int a => toOpenCLCast .int a
    | .add a b => s!"({toOpenCLInt a} + {toOpenCLInt b})"
    | .sub a b => s!"({toOpenCLInt a} - {toOpenCLInt b})"
    | .mul a b => s!"({toOpenCLInt a} * {toOpenCLInt b})"
    | .div a b => s!"({toOpenCLInt a} / {toOpenCLInt b})"
    | .neg a => s!"(-{toOpenCLInt a})"
    | .abs a => s!"labs({toOpenCLInt a})"
    | .min a b => s!"min({toOpenCLInt a}, {toOpenCLInt b})"
    | .max a b => s!"max({toOpenCLInt a}, {toOpenCLInt b})"
    | .ifLt a b thenExpr elseExpr => s!"(({toOpenCLAny a} < {toOpenCLAny b}) ? {toOpenCLInt thenExpr} : {toOpenCLInt elseExpr})"
    | KernelExpr.ifInBounds arrayIndex elemIndex thenExpr elseExpr =>
        s!"(({toOpenCLNat elemIndex} < ctx{arrayIndex}_size) ? {toOpenCLInt thenExpr} : {toOpenCLInt elseExpr})"
  termination_by e => sizeOf e

  def toOpenCLAny : {t : KernelType} -> KernelExpr Float32 t -> String
    | .real, e => toOpenCL e
    | .nat, e => toOpenCLNat e
    | .int, e => toOpenCLInt e
  termination_by _ e => sizeOf e + 1

  def toOpenCLCast (t : KernelType) : {s : KernelType} -> KernelExpr Float32 s -> String
    | .real, a => toOpenCL a
    | .nat, a => match t with
      | .real => s!"((float){toOpenCLNat a})"
      | .nat => toOpenCLNat a
      | .int => s!"((long){toOpenCLNat a})"
    | .int, a => match t with
      | .real => s!"((float){toOpenCLInt a})"
      | .nat => s!"((ulong){toOpenCLInt a})"
      | .int => toOpenCLInt a
  termination_by _ a => sizeOf a + 1
end

def eval (ctx : Array (Array Float32)) (x : Float32) (gid : Nat) (expr : Float32KernelExpr) : Float32 :=
  KernelExpr.eval ctx x gid expr

def mapArraySpec (expr : Float32KernelExpr) (ctx : Array (Array Float32)) (xs : Array Float32) : Array Float32 :=
  KernelExpr.mapArraySpec expr ctx xs

end Float32KernelExpr

namespace OpenCLFloat32Array

@[inline]
def mapExpr (xs : OpenCLFloat32Array) (expr : Float32KernelExpr) : OpenCLFloat32Array :=
  unsafe xs.mapUnsafe expr.toOpenCL

@[inline]
def mapInContextExpr (xs : OpenCLFloat32Array) (ctx : Array OpenCLFloat32Array) (expr : Float32KernelExpr) : OpenCLFloat32Array :=
  unsafe xs.mapInContextUnsafe ctx expr.toOpenCL

end OpenCLFloat32Array

end NumLean
