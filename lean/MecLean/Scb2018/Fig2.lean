import MecLean.Scb2018.MISs

set_option linter.unusedSectionVars false

/-!
# Lee–Bareinboim (2018), Fig. 2: the worked example of `POMISs`

Source: the paper and its unofficial AI-corrected version (see `README.md`): Sec. 3
(Fig. 2, `fig:mlct-ib`, and the text after Prop. 5) and App. C (the worked example).  The graph
is read from Fig. 2; it agrees with every MUCT and IB the
text states for `G`, `G_{\overline X}`, `G_{\overline Z}` and `G_{\overline W}`.

Vertices `S = 0, T = 1, W = 2, X = 3, Y = 4, Z = 5`.  `G`: `S → W`, `W → Y`, `T → X`, `T → Y`,
`Z → X`, `X → Y`, `W ↔ X`, `Z ↔ Y`.

Every check is a kernel computation (`decide +kernel`).  `pomis_iff` uses `thm9`, whose "if"
direction rests on `ScbMec.pomis_diag`; `ib_fixed_points` checks the graphical condition of
Thm. 6 on all 64 subsets, independently of the algorithm.

The statement that the run examines only four intervened graphs concerns the execution, not
the output.  `no_recursion` checks the data that decide the calls of `subPOMISs` (by
`subPOMISs_eq`): after pivot `X` the list `π'` is empty, pivot `Z` is pruned, and after pivot
`W` the list `π'` is empty, so no recursive call occurs.  `recursion_alt` checks the remark that
with `W` before `Z` a recursive call follows pivot `W`.
-/

namespace Mec.Scb2018.Fig2

open Mec ScbMec Alg

abbrev S : Fin 6 := 0
abbrev T : Fin 6 := 1
abbrev W : Fin 6 := 2
abbrev X : Fin 6 := 3
abbrev Y : Fin 6 := 4
abbrev Z : Fin 6 := 5

/-- `fig:mlct-example-full`: the causal diagram `G`. -/
def G : ADMG 6 where
  dir i j := decide ((i = S ∧ j = W) ∨ (i = W ∧ j = Y) ∨ (i = T ∧ j = X) ∨ (i = T ∧ j = Y) ∨
    (i = Z ∧ j = X) ∨ (i = X ∧ j = Y))
  bi i j := decide ((i = W ∧ j = X) ∨ (i = X ∧ j = W) ∨ (i = Z ∧ j = Y) ∨ (i = Y ∧ j = Z))

theorem G_wf : G.WF := by decide +kernel

/-- Fig. 2(a): the MUCT is `{W, X, Y, Z}` and the IB is `{S, T}`. -/
theorem muct_G : muct G Y ∅ = {W, X, Y, Z} ∧ ib G Y ∅ = {S, T} := by decide +kernel

/-- Fig. 2(b): `MUCT(G_{\overline X}, Y) = {Y}` and `IB(G_{\overline X}, Y) = {T, W, X}`. -/
theorem muct_GX : muct G Y {X} = {Y} ∧ ib G Y {X} = {T, W, X} := by decide +kernel

/-- Fig. 2(c): `G_{\overline Z}` has the same MUCT and IB as `G_{\overline X}`. -/
theorem muct_GZ : muct G Y {Z} = {Y} ∧ ib G Y {Z} = {T, W, X} := by decide +kernel

/-- Fig. 2(d): `MUCT(G_{\overline W}, Y) = {X, Y, Z}` and `IB(G_{\overline W}, Y) = {T, W}`. -/
theorem muct_GW : muct G Y {W} = {X, Y, Z} ∧ ib G Y {W} = {T, W} := by decide +kernel

/-- `pa(Y)_G = {T, W, X}` is a POMIS: `MUCT(G_{\overline{pa(Y)}}, Y) = {Y}`. -/
theorem muct_paY : muct G Y {T, W, X} = {Y} ∧ ib G Y {T, W, X} = {T, W, X} := by
  decide +kernel

/-- Line 2 of `POMISs`: `H = G_{\overline{\{S,T\}}}[{W,X,Y,Z} ∪ {S,T}]` is `G`. -/
theorem H_eq : compress G (muct G Y ∅) (ib G Y ∅) = G := by decide +kernel

/-- The reverse topological order `(Y, X, Z, T, W, S)` of App. C. -/
def ord : List (Fin 6) := [Y, X, Z, T, W, S]

theorem ord_revTopo :
    RevTopo (compress G (muct G Y ∅) (ib G Y ∅)) (muct G Y ∅ ∪ ib G Y ∅) ord := by
  decide +kernel

/-- Line 3: the candidate list is `(Y, X, Z, T, W, S) ∩ {W, X, Z} = (X, Z, W)`. -/
theorem candidates : ord.filter (fun v => decide (v ∈ muct G Y ∅ ∧ v ≠ Y)) = [X, Z, W] := by
  decide +kernel

/-- The output of `POMISs`: `{S, T}`, then `{T, W, X}` (pivot `X`); the pivot `Z` is pruned
because its border meets `{X}`; then `{T, W}` (pivot `W`). -/
theorem output : POMISs G Y ord = [{S, T}, {T, W, X}, {T, W}] := by decide +kernel

/-- App. C, order `(X, Z, W)`: after pivot `X`, `π' = (Z, W) ∩ MUCT(H_{\overline X}, Y)` is
empty; pivot `Z` is pruned since its border meets `O' = {X}`; after pivot `W`, `π'` is the empty
suffix.  So `subPOMISs` makes no recursive call, and `POMISs` computes the MUCT and IB of `G`,
`G_{\overline X}`, `G_{\overline Z}` and `G_{\overline W}` only (`H = G` by `H_eq`). -/
theorem no_recursion :
    [Z, W].filter (fun v => decide (v ∈ muct G Y {X})) = [] ∧
      ib G Y {Z} ∩ ({X} : Finset (Fin 6)) ≠ ∅ := by
  decide +kernel

/-- App. C, order `(X, W, Z)`: pivot `W` is not pruned (`O' = {X}`) and leaves `π' = (Z)`, so a
recursive call follows on `G_{\overline{\{T,W\}}}[\{X,Y,Z\} ∪ \{T,W\}]`, which represents
`G_{\overline{\{W,Z\}}}` after pivot `Z` (Prop. 8). -/
theorem recursion_alt :
    ib G Y {W} ∩ ({X} : Finset (Fin 6)) = ∅ ∧
      [Z].filter (fun v => decide (v ∈ muct G Y {W})) = [Z] := by
  decide +kernel

/-- The set of POMISs is `{{S, T}, {T, W}, {T, W, X}}` (by Thm. 9 and `output`). -/
theorem pomis_iff (P : Finset (Fin 6)) :
    POMIS (ConfDiag G) Y P ↔ P ∈ ({{S, T}, {T, W}, {T, W, X}} : Finset (Finset (Fin 6))) := by
  rw [← thm9_of_revTopo_compress G_wf ord_revTopo P, output]
  revert P
  decide +kernel

/-- The graphical condition of Thm. 6 over all subsets: `IB(G_{\overline P}, Y) = P` exactly for
`{S, T}`, `{T, W}` and `{T, W, X}`. -/
theorem ib_fixed_points : ∀ P : Finset (Fin 6), (Y ∉ P ∧ ib G Y P = P) ↔
    P ∈ ({{S, T}, {T, W}, {T, W, X}} : Finset (Finset (Fin 6))) := by
  decide +kernel

/-- Another reverse topological order, with `W` before `Z`, gives the same output. -/
theorem output_alt : POMISs G Y [Y, X, W, Z, T, S] = [{S, T}, {T, W, X}, {T, W}] := by
  decide +kernel

/-- Algorithm 4 on `G`: `ord` is a reverse topological order of `G[An(Y)_G]`, and `MISs` lists
18 sets, each once. -/
theorem miss_G : RevTopo (restrict G (anc G Y)) (anc G Y) ord ∧ (MISs G Y ord).length = 18 ∧
    (MISs G Y ord).Nodup := by
  decide +kernel

end Mec.Scb2018.Fig2
