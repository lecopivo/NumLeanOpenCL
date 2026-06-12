import NumLeanOpenCL.KernelExpr.Basic
import NumLeanOpenCL.KernelExpr.Register
import NumLeanOpenCL.Tactic.ApplyRuleSets
import NumLeanOpenCL.Tactic.ApplyRuleSets.RuleProc

namespace NumLean

structure HasKernelExpr (x : Float32) (i : Nat) (ctx : List OpenCLFloat32Array) {t}
    (e : KernelType.denote Float32 t) (kernel : KernelExpr Float32 t) : Prop where
  denote : kernel.eval (ctx.map (·.data)).toArray x i = e

structure HasContextIndex (ctx : List OpenCLFloat32Array) (xs : OpenCLFloat32Array) (i : Nat) : Prop where
  in_context : ctx[i]? = some xs

namespace HasKernelExpr

variable {x : Float32} {i : Nat}

theorem expr_congr {x : Float32} {i : Nat} {ctx : List OpenCLFloat32Array} {t}
    {e : KernelType.denote Float32 t} {kernel kernel' : KernelExpr Float32 t}
    (h : HasKernelExpr x i ctx e kernel) (h' : kernel = kernel') :
    HasKernelExpr x i ctx e kernel' := by
  subst h'
  exact h

theorem expr_ctx_congr {x : Float32} {i : Nat} {ctx ctx' : List OpenCLFloat32Array} {t}
    {e : KernelType.denote Float32 t} {kernel kernel' : KernelExpr Float32 t}
    (h : HasKernelExpr x i ctx e kernel) (hctx : ctx = ctx') (h' : kernel = kernel') :
    HasKernelExpr x i ctx' e kernel' := by
  subst hctx
  subst h'
  exact h

open Lean Meta in
@[compile_kernel_expr high]
ruleproc collect_context , (x : Float32) (i : Nat) (ctx : List OpenCLFloat32Array) {t}
    (e : KernelType.denote Float32 t) (kernel : KernelExpr Float32 t) :
    HasKernelExpr x i ctx e kernel :=
  fun argOrigin goal => do
    let ctx ← instantiateMVars ctx
    let e ← instantiateMVars e

    if ctx.isMVar && ¬e.isMVar then
      let fvars := (← (e.collectFVars.run {})).2.fvarIds
      let arrayTy := mkConst ``OpenCLFloat32Array

      let mut arrayFVars := #[]
      for id in fvars do
        let ty ← id.getType
        if ← isDefEq ty arrayTy then
          arrayFVars := arrayFVars.push id

      let mut xs ← mkAppOptM ``List.nil #[arrayTy]
      for id in arrayFVars.reverse do
        xs ← mkAppM ``List.cons #[.fvar id, xs]

      ctx.mvarId!.assign xs
      return ← NumLeanOpenCL.Tactic.ApplyRuleSets.applyRuleSets argOrigin goal

    return none

open Lean Meta in
@[compile_kernel_expr]
ruleproc has_context_index , (ctx : List OpenCLFloat32Array) (xs : OpenCLFloat32Array) (i : Nat) : HasContextIndex ctx xs i :=
  fun _ goal => do
    let ctx ← instantiateMVars ctx
    let xs ← instantiateMVars xs
    let i ← instantiateMVars i

    if ¬ctx.isMVar && xs.isFVar then
      let rec getContextFVars (ctx : Expr) (acc : Array FVarId) : Option (Array FVarId) := do
        match ctx with
        | mkApp3 (.const ``List.cons _) _ (.fvar id) ctx' =>
            getContextFVars ctx' (acc.push id)
        | Expr.app (Expr.const ``List.nil _) _ => some acc
        | _ => none
      let some ctxFVars := getContextFVars ctx #[] | return none
      let .fvar id := xs | return none
      let some n := ctxFVars.idxOf? id | return none
      if i.isMVar then
        i.mvarId!.assign (mkNatLit n)
      let prf ← mkFreshExprMVar goal
      let fields ← prf.mvarId!.constructor
      match fields with
      | [field] =>
        try field.refl catch _ => return none
        return prf
      | _ => return none
    return none


@[compile_kernel_expr]
theorem x_rule : HasKernelExpr x i ctx x .x := by
  constructor;
  simp [KernelExpr.eval, KernelExpr.evalReal]

@[compile_kernel_expr]
theorem i_rule : HasKernelExpr x i ctx i .gid := by
  constructor;
  simp [KernelExpr.eval, KernelExpr.evalNat]

theorem lit_rule_proof (v : Float32) : HasKernelExpr x i ctx v (.lit v) := by
  constructor
  simp [KernelExpr.eval, KernelExpr.evalReal]

theorem nat_lit_rule_proof (v : Nat) : HasKernelExpr x i ctx v (.natLit v) := by
  constructor
  simp [KernelExpr.eval, KernelExpr.evalNat]

theorem int_lit_rule_proof (v : Int) : HasKernelExpr x i ctx v (.intLit v) := by
  constructor
  simp [KernelExpr.eval, KernelExpr.evalInt]

open Lean Meta in
@[compile_kernel_expr low]
ruleproc lit_rule , (x : Float32) (i : Nat) (ctx : List OpenCLFloat32Array)
    (v : Float32) (kernel : KernelExpr Float32 .real) : HasKernelExpr x i ctx v kernel :=
  fun _ _ => do
    let v ← instantiateMVars v
    let kernel ← instantiateMVars kernel
    if !kernel.isMVar then
      return none
    try
      discard <| Meta.evalExpr Float32 (mkConst ``Float32) v
    catch _ =>
      return none
    kernel.mvarId!.assign (← mkAppOptM ``KernelExpr.lit #[some (mkConst ``Float32), some v])
    return some (← mkAppOptM ``HasKernelExpr.lit_rule_proof #[some x, some i, some ctx, some v])

open Lean Meta in
@[compile_kernel_expr low]
ruleproc nat_lit_rule , (x : Float32) (i : Nat) (ctx : List OpenCLFloat32Array)
    (v : Nat) (kernel : KernelExpr Float32 .nat) : HasKernelExpr x i ctx v kernel :=
  fun _ _ => do
    let v ← instantiateMVars v
    let kernel ← instantiateMVars kernel
    if !kernel.isMVar then
      return none
    try
      discard <| Meta.evalExpr Nat (mkConst ``Nat) v
    catch _ =>
      return none
    kernel.mvarId!.assign (← mkAppOptM ``KernelExpr.natLit #[some (mkConst ``Float32), some v])
    return some (← mkAppOptM ``HasKernelExpr.nat_lit_rule_proof #[some x, some i, some ctx, some v])

open Lean Meta in
@[compile_kernel_expr low]
ruleproc int_lit_rule , (x : Float32) (i : Nat) (ctx : List OpenCLFloat32Array)
    (v : Int) (kernel : KernelExpr Float32 .int) : HasKernelExpr x i ctx v kernel :=
  fun _ _ => do
    let v ← instantiateMVars v
    let kernel ← instantiateMVars kernel
    if !kernel.isMVar then
      return none
    try
      discard <| Meta.evalExpr Int (mkConst ``Int) v
    catch _ =>
      return none
    kernel.mvarId!.assign (← mkAppOptM ``KernelExpr.intLit #[some (mkConst ``Float32), some v])
    return some (← mkAppOptM ``HasKernelExpr.int_lit_rule_proof #[some x, some i, some ctx, some v])

@[compile_kernel_expr]
theorem add_real {a b : Float32} {ka kb : KernelExpr Float32 .real}
    (ha : HasKernelExpr x i ctx a ka) (hb : HasKernelExpr x i ctx b kb) :
    HasKernelExpr x i ctx (a + b) (ka + kb) := by
  constructor
  have hka : KernelExpr.evalReal (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  have hkb : KernelExpr.evalReal (ctx.map (·.data)).toArray x i kb = b := by simpa [KernelExpr.eval] using hb.denote
  change KernelExpr.evalReal (ctx.map (·.data)).toArray x i (KernelExpr.add ka kb) = a + b
  simp [KernelExpr.evalReal, hka, hkb]

@[compile_kernel_expr]
theorem sub_real {a b : Float32} {ka kb : KernelExpr Float32 .real}
    (ha : HasKernelExpr x i ctx a ka) (hb : HasKernelExpr x i ctx b kb) :
    HasKernelExpr x i ctx (a - b) (ka - kb) := by
  constructor
  have hka : KernelExpr.evalReal (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  have hkb : KernelExpr.evalReal (ctx.map (·.data)).toArray x i kb = b := by simpa [KernelExpr.eval] using hb.denote
  change KernelExpr.evalReal (ctx.map (·.data)).toArray x i (KernelExpr.sub ka kb) = a - b
  simp [KernelExpr.evalReal, hka, hkb]

@[compile_kernel_expr]
theorem mul_real {a b : Float32} {ka kb : KernelExpr Float32 .real}
    (ha : HasKernelExpr x i ctx a ka) (hb : HasKernelExpr x i ctx b kb) :
    HasKernelExpr x i ctx (a * b) (ka * kb) := by
  constructor
  have hka : KernelExpr.evalReal (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  have hkb : KernelExpr.evalReal (ctx.map (·.data)).toArray x i kb = b := by simpa [KernelExpr.eval] using hb.denote
  change KernelExpr.evalReal (ctx.map (·.data)).toArray x i (KernelExpr.mul ka kb) = a * b
  simp [KernelExpr.evalReal, hka, hkb]

@[compile_kernel_expr]
theorem div_real {a b : Float32} {ka kb : KernelExpr Float32 .real}
    (ha : HasKernelExpr x i ctx a ka) (hb : HasKernelExpr x i ctx b kb) :
    HasKernelExpr x i ctx (a / b) (ka / kb) := by
  constructor
  have hka : KernelExpr.evalReal (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  have hkb : KernelExpr.evalReal (ctx.map (·.data)).toArray x i kb = b := by simpa [KernelExpr.eval] using hb.denote
  change KernelExpr.evalReal (ctx.map (·.data)).toArray x i (KernelExpr.div ka kb) = a / b
  simp [KernelExpr.evalReal, hka, hkb]

@[compile_kernel_expr]
theorem neg_real {a : Float32} {ka : KernelExpr Float32 .real}
    (ha : HasKernelExpr x i ctx a ka) :
    HasKernelExpr x i ctx (-a) (-ka) := by
  constructor
  have hka : KernelExpr.evalReal (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  change KernelExpr.evalReal (ctx.map (·.data)).toArray x i (KernelExpr.neg ka) = -a
  simp [KernelExpr.evalReal, hka]

@[compile_kernel_expr]
theorem add_nat {a b : Nat} {ka kb : KernelExpr Float32 .nat}
    (ha : HasKernelExpr x i ctx a ka) (hb : HasKernelExpr x i ctx b kb) :
    HasKernelExpr x i ctx (a + b) (ka + kb) := by
  constructor
  have hka : KernelExpr.evalNat (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  have hkb : KernelExpr.evalNat (ctx.map (·.data)).toArray x i kb = b := by simpa [KernelExpr.eval] using hb.denote
  change KernelExpr.evalNat (ctx.map (·.data)).toArray x i (KernelExpr.add ka kb) = a + b
  simp [KernelExpr.evalNat, hka, hkb]

@[compile_kernel_expr]
theorem sub_nat {a b : Nat} {ka kb : KernelExpr Float32 .nat}
    (ha : HasKernelExpr x i ctx a ka) (hb : HasKernelExpr x i ctx b kb) :
    HasKernelExpr x i ctx (a - b) (ka - kb) := by
  constructor
  have hka : KernelExpr.evalNat (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  have hkb : KernelExpr.evalNat (ctx.map (·.data)).toArray x i kb = b := by simpa [KernelExpr.eval] using hb.denote
  change KernelExpr.evalNat (ctx.map (·.data)).toArray x i (KernelExpr.sub ka kb) = a - b
  simp [KernelExpr.evalNat, hka, hkb]

@[compile_kernel_expr]
theorem mul_nat {a b : Nat} {ka kb : KernelExpr Float32 .nat}
    (ha : HasKernelExpr x i ctx a ka) (hb : HasKernelExpr x i ctx b kb) :
    HasKernelExpr x i ctx (a * b) (ka * kb) := by
  constructor
  have hka : KernelExpr.evalNat (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  have hkb : KernelExpr.evalNat (ctx.map (·.data)).toArray x i kb = b := by simpa [KernelExpr.eval] using hb.denote
  change KernelExpr.evalNat (ctx.map (·.data)).toArray x i (KernelExpr.mul ka kb) = a * b
  simp [KernelExpr.evalNat, hka, hkb]

@[compile_kernel_expr]
theorem cast_real_of_nat {a : Nat} {ka : KernelExpr Float32 .nat}
    (ha : HasKernelExpr x i ctx a ka) :
    HasKernelExpr x i ctx (a • (1 : Float32)) (KernelExpr.cast .real ka) := by
  constructor
  have hka : KernelExpr.evalNat (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  simp [KernelExpr.eval, KernelExpr.evalReal, KernelExpr.evalCastReal, hka]

@[compile_kernel_expr]
theorem cast_real_of_int {a : Int} {ka : KernelExpr Float32 .int}
    (ha : HasKernelExpr x i ctx a ka) :
    HasKernelExpr x i ctx (a • (1 : Float32)) (KernelExpr.cast .real ka) := by
  constructor
  have hka : KernelExpr.evalInt (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  simp [KernelExpr.eval, KernelExpr.evalReal, KernelExpr.evalCastReal, hka]

@[compile_kernel_expr]
theorem if_lt_nat {a b c d : Nat} {ka kb kc kd : KernelExpr Float32 .nat}
    (ha : HasKernelExpr x i ctx a ka) (hb : HasKernelExpr x i ctx b kb)
    (hc : HasKernelExpr x i ctx c kc) (hd : HasKernelExpr x i ctx d kd) :
    HasKernelExpr x i ctx (if a < b then c else d) (KernelExpr.ifLt ka kb kc kd) := by
  constructor
  have hka : KernelExpr.evalNat (ctx.map (·.data)).toArray x i ka = a := by simpa [KernelExpr.eval] using ha.denote
  have hkb : KernelExpr.evalNat (ctx.map (·.data)).toArray x i kb = b := by simpa [KernelExpr.eval] using hb.denote
  have hkc : KernelExpr.evalNat (ctx.map (·.data)).toArray x i kc = c := by simpa [KernelExpr.eval] using hc.denote
  have hkd : KernelExpr.evalNat (ctx.map (·.data)).toArray x i kd = d := by simpa [KernelExpr.eval] using hd.denote
  change KernelExpr.evalNat (ctx.map (·.data)).toArray x i (KernelExpr.ifLt ka kb kc kd) = if a < b then c else d
  simp [KernelExpr.evalNat, KernelExpr.evalLt, hka, hkb, hkc, hkd]

@[compile_kernel_expr]
theorem ctx_get_elem (xs : OpenCLFloat32Array) (idx : Nat)
    {eidx} (hidx : HasKernelExpr x i ctx idx eidx)
    {ixs} (hxs : HasContextIndex ctx xs ixs) :
    HasKernelExpr x i ctx (xs.toArray[idx]?.getD 0) (.ctx ixs eidx) := by
  constructor
  have hh := hidx.denote
  simp [KernelExpr.eval] at hh
  simp [KernelExpr.eval, KernelExpr.evalReal, hxs.in_context, hh, OpenCLFloat32Array.toArray]

@[compile_kernel_expr]
theorem ctx_get_elem_bounded (xs : OpenCLFloat32Array) (idx : Nat) (h : idx < xs.size)
    {eidx} (hidx : HasKernelExpr x i ctx idx eidx)
    {ixs} (hxs : HasContextIndex ctx xs ixs) :
    HasKernelExpr x i ctx xs[idx] (.ctx ixs eidx) := by
  constructor
  have hh := hidx.denote
  simp [KernelExpr.eval] at hh
  cases xs with
  | mk data =>
    simp [KernelExpr.eval, KernelExpr.evalReal, hxs.in_context, hh,
      OpenCLFloat32Array.size] at h ⊢
    simp [h]
    change data[idx] = OpenCLFloat32Array.get { data := data } idx h
    simp [OpenCLFloat32Array.get, OpenCLFloat32Array.toArray]
    simpa using (getElem!_pos data idx h).symm

@[compile_kernel_expr]
theorem ctx_size (xs : OpenCLFloat32Array)
    {ixs} (hxs : HasContextIndex ctx xs ixs) :
    HasKernelExpr x i ctx xs.size (.ctxSize ixs) := by
  constructor;
  simp [KernelExpr.eval, KernelExpr.evalNat, hxs.in_context]
  rfl

section Examples

private def one : Float32 := 1
private def two : Float32 := 2
private def three : Float32 := 3

variable (x : Float32) (i : Nat) (xs ys : OpenCLFloat32Array)

example : HasKernelExpr x i [] x KernelExpr.x := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] one (KernelExpr.lit one) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] i KernelExpr.gid := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] (i • (1 : Float32)) (KernelExpr.cast .real KernelExpr.gid) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] (x + one) (KernelExpr.x + KernelExpr.lit one) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] (x - one) (KernelExpr.x - KernelExpr.lit one) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] (x * two) (KernelExpr.x * KernelExpr.lit two) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] (x / two) (KernelExpr.x / KernelExpr.lit two) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] (-x) (-KernelExpr.x) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] ((x + one) * (-x - three))
    ((KernelExpr.x + KernelExpr.lit one) * (-KernelExpr.x - KernelExpr.lit three)) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] (i + 7) (KernelExpr.gid + KernelExpr.natLit 7) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasKernelExpr x i [] (i * 7) (KernelExpr.gid * KernelExpr.natLit 7) := by
  apply HasKernelExpr.expr_congr
  apply_rulesets [compile_kernel_expr]
  rfl

example : HasContextIndex [xs, ys] xs 0 := by
  apply_rulesets [compile_kernel_expr]

example : HasContextIndex [xs, ys] ys 1 := by
  apply_rulesets [compile_kernel_expr]

example : HasKernelExpr x i [xs] (xs.toArray[i]?.getD 0) (.ctx 0 KernelExpr.gid) := by
  apply HasKernelExpr.expr_ctx_congr
  apply_rulesets [compile_kernel_expr]
  rfl; rfl

example : HasKernelExpr x i [ys] (ys.size) (.ctxSize 0) := by
  apply HasKernelExpr.expr_ctx_congr
  apply_rulesets [compile_kernel_expr]
  rfl; rfl

example : HasKernelExpr x i [xs] (x + xs.toArray[i]?.getD 0)
    (KernelExpr.x + .ctx 0 KernelExpr.gid) := by
  apply HasKernelExpr.expr_ctx_congr
  apply_rulesets [compile_kernel_expr]
  rfl; rfl

example : HasKernelExpr x i [xs, ys] (xs.toArray[i]?.getD 0 + ys.size • (1 : Float32))
    (.ctx 0 KernelExpr.gid + KernelExpr.cast .real (.ctxSize 1)) := by
  apply HasKernelExpr.expr_ctx_congr
  apply_rulesets [compile_kernel_expr]
  rfl; rfl

example (h : i + 10 < xs.size) : HasKernelExpr x i [xs] (x + xs[i + 1])
    (KernelExpr.x + .ctx 0 (.add KernelExpr.gid (.natLit 1))) := by
  apply HasKernelExpr.expr_ctx_congr
  apply_rulesets [compile_kernel_expr]
  rfl; rfl


example : HasKernelExpr x i [] (x + i • (1 : Float32)) (t:=.real)
    (KernelExpr.x + KernelExpr.cast KernelType.real KernelExpr.gid) := by
  apply HasKernelExpr.expr_ctx_congr
  apply_rulesets [compile_kernel_expr]
  rfl; rfl


end Examples
