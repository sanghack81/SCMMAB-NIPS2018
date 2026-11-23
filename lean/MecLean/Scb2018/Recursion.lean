import MecLean.Scb2018.Thm6

set_option linter.unusedSectionVars false

/-!
# Structural Causal Bandits (Lee–Bareinboim, NeurIPS 2018): territories under further interventions

Source: the paper and its unofficial AI-corrected version (see `README.md`): Prop. 7
(`lem:ignorability`) and Prop. 8 (`lem:recursion`) of Sec. 4; Prop. 12 (`prop:dontmisspomis`)
and the proofs of App. C.

## Territories after intervention

`DiagMUCT G y W T` (`MecLean.Common.DiagTerritory`) is `MUCT(G_{\overline W}, Y)`: in
`G_{\overline W}` the vertices of `W` have no parent and no bidirected edge, so no closure step
from `Y` reaches them, and the library leaves them out of `diagH G y W`.  `diagBorder G T` is
then `IB(G_{\overline W}, Y) = Pa(T)_{G_{\overline W}} \ T`, since no edge into `T` is cut.
`muct_cut`, `ib_cut`, `diagMUCT_cut_iff` and `diagBorder_cut` prove this for the mutilated
graph `cut G W` read with no further intervention.

* `cut G X` is `G_{\overline X}`, `restrict G S` is `G[S]` and `compress G T X` is
  `G_{\overline X}[T ∪ X]`.  All graphs keep the vertex type `Fin n`: a vertex outside `S` stays
  in `G[S]` without edges.  Bidirected edges stay symmetric (`wf_cut`, `wf_restrict`).
* `muct G y W` computes `MUCT(G_{\overline W}, Y)` by a closure iteration: it is the `reachSet`
  of `y` under one directed or bidirected edge whose target lies in `diagH G y W`.
  `diagMUCT_muct` and `eq_muct`: it is the unique `DiagMUCT`.  `ib G y W` is
  `diagBorder G (muct G y W)`.
* `prop7`, `prop7_cut`: Prop. 7.  `prop8`, `prop8_cut`: Prop. 8; `prop8_muct`, `prop8_general`:
  the same for every `W'`.  `prop12_diag`, `prop12`, `prop12_cut`: Prop. 12.  The `_cut` forms
  state the results on the mutilated graphs `cut G W` themselves; the others use the library's
  reading of `DiagMUCT G y W` as `MUCT(G_{\overline W}, Y)`.

## Modelling choices

* The MUCT algorithm of App. B (Alg. 3) alternates c-components and descendants inside
  `G[An(Y)_G]`; `muct` adds both kinds of neighbours in every round.  Both compute the least set
  containing `Y` and closed under these edges.
* Prop. 8 holds for every `W'`, not only `W' ⊆ T \ {Y}` (`prop8_muct`, `prop8_general`).
* Prop. 12 needs only `W' ⊆ W`; the paper assumes `W' ⊂ W`.

## Steps added to the paper's proofs

* Prop. 7: the proof in App. C assumes that `T` is the MUCT of `G_{\overline X}`.  This needs the
  fact that every vertex with a directed path to `Y` avoiding `X = Pa(T) \ T` lies in `T`
  (`diagH_border_subset`).
* Prop. 8: the proof compares the edges of the two graphs inside `T ∪ X` only.  It also needs that
  the MUCT of `G_{\overline{W ∪ W'}}` lies in `T` (`muct_union_subset`, from the closure of `T`),
  and that the two graphs differ in the bidirected edges at `X` and outside `T ∪ X`, which the
  closure from `Y` never uses (`reach_terr_compress`, `reach_terr_of_compress`).
-/

namespace Mec.Scb2018

open Mec ScbMec

variable {n : ℕ}

namespace Alg

/-! ## Graph operations -/

/-- `G_{\overline X}`: the directed edges into `X` and the bidirected edges with an endpoint in
`X` are removed. -/
def cut (G : ADMG n) (X : Finset (Fin n)) : ADMG n where
  dir i j := G.dir i j && decide (j ∉ X)
  bi i j := G.bi i j && (decide (i ∉ X) && decide (j ∉ X))

/-- `G[S]` on the same vertex type: the edges with an endpoint outside `S` are removed. -/
def restrict (G : ADMG n) (S : Finset (Fin n)) : ADMG n where
  dir i j := G.dir i j && (decide (i ∈ S) && decide (j ∈ S))
  bi i j := G.bi i j && (decide (i ∈ S) && decide (j ∈ S))

/-- `G_{\overline X}[T ∪ X]`, the graph passed to the recursive calls of `POMISs`. -/
def compress (G : ADMG n) (T X : Finset (Fin n)) : ADMG n := restrict (cut G X) (T ∪ X)

section Ops

variable {G : ADMG n}

theorem cut_dir {X : Finset (Fin n)} {i j : Fin n} :
    (cut G X).dir i j = true ↔ G.dir i j = true ∧ j ∉ X := by
  simp [cut]

theorem cut_bi {X : Finset (Fin n)} {i j : Fin n} :
    (cut G X).bi i j = true ↔ G.bi i j = true ∧ i ∉ X ∧ j ∉ X := by
  simp [cut]

theorem restrict_dir {S : Finset (Fin n)} {i j : Fin n} :
    (restrict G S).dir i j = true ↔ G.dir i j = true ∧ i ∈ S ∧ j ∈ S := by
  simp [restrict]

theorem restrict_bi {S : Finset (Fin n)} {i j : Fin n} :
    (restrict G S).bi i j = true ↔ G.bi i j = true ∧ i ∈ S ∧ j ∈ S := by
  simp [restrict]

theorem compress_dir {T X : Finset (Fin n)} {i j : Fin n} :
    (compress G T X).dir i j = true ↔ G.dir i j = true ∧ j ∉ X ∧ i ∈ T ∪ X ∧ j ∈ T ∪ X := by
  simp only [compress, restrict_dir, cut_dir, and_assoc]

theorem compress_bi {T X : Finset (Fin n)} {i j : Fin n} :
    (compress G T X).bi i j = true ↔
      G.bi i j = true ∧ i ∉ X ∧ j ∉ X ∧ i ∈ T ∪ X ∧ j ∈ T ∪ X := by
  simp only [compress, restrict_bi, cut_bi, and_assoc]

/-- A reach transfers along a map of its steps. -/
theorem reach_of_step {R R' : Fin n → Fin n → Bool} {D D' : Finset (Fin n)}
    (h : ∀ a b, R a b = true → b ∉ D → R' a b = true ∧ b ∉ D') {u v : Fin n}
    (hr : Reach R D u v) : Reach R' D' u v := by
  induction hr with
  | refl => exact Reach.refl _
  | @tail b c _ hbc ih => exact Reach.tail ih (h b c hbc.1 hbc.2).1 (h b c hbc.1 hbc.2).2

/-- A directed path of `G_{\overline C}` avoiding `D` is a directed path of `G` avoiding
`C ∪ D`. -/
theorem cut_ancAvoid {C D : Finset (Fin n)} {u v : Fin n} :
    (cut G C).AncAvoid D u v ↔ G.AncAvoid (C ∪ D) u v := by
  constructor
  · intro h
    refine reach_of_step (R := (cut G C).dir) (R' := G.dir) (D := D) (D' := C ∪ D)
      (fun a b hab hb => ⟨(cut_dir.mp hab).1, fun hbCD => ?_⟩) h
    rcases Finset.mem_union.mp hbCD with h' | h'
    · exact (cut_dir.mp hab).2 h'
    · exact hb h'
  · intro h
    exact reach_of_step (R := G.dir) (R' := (cut G C).dir) (D := C ∪ D) (D' := D)
      (fun a b hab hb => ⟨cut_dir.mpr ⟨hab, fun h' => hb (Finset.mem_union_left _ h')⟩,
        fun h' => hb (Finset.mem_union_right _ h')⟩) h

theorem wf_cut (hG : G.WF) (X : Finset (Fin n)) : (cut G X).WF := by
  refine ⟨fun i => ?_, fun i => ?_, fun i j => ?_, fun i j hij hji => ?_⟩
  · simp [cut, hG.dir_irrefl i]
  · simp [cut, hG.bi_irrefl i]
  · show (G.bi i j && (decide (i ∉ X) && decide (j ∉ X))) =
      (G.bi j i && (decide (j ∉ X) && decide (i ∉ X)))
    rw [hG.bi_symm i j, Bool.and_comm (decide (i ∉ X))]
  · exact hG.acyclic (cut_dir.mp hij).1
      (Reach.mono_rel (R := (cut G X).dir) (R' := G.dir)
        (fun a b _ hab => (cut_dir.mp hab).1) hji)

theorem wf_restrict (hG : G.WF) (S : Finset (Fin n)) : (restrict G S).WF := by
  refine ⟨fun i => ?_, fun i => ?_, fun i j => ?_, fun i j hij hji => ?_⟩
  · simp [restrict, hG.dir_irrefl i]
  · simp [restrict, hG.bi_irrefl i]
  · show (G.bi i j && (decide (i ∈ S) && decide (j ∈ S))) =
      (G.bi j i && (decide (j ∈ S) && decide (i ∈ S)))
    rw [hG.bi_symm i j, Bool.and_comm (decide (i ∈ S))]
  · exact hG.acyclic (restrict_dir.mp hij).1
      (Reach.mono_rel (R := (restrict G S).dir) (R' := G.dir)
        (fun a b _ hab => (restrict_dir.mp hab).1) hji)

theorem wf_compress (hG : G.WF) (T X : Finset (Fin n)) : (compress G T X).WF :=
  wf_restrict (wf_cut hG X) (T ∪ X)

end Ops

/-! ## The minimal territory as a closure -/

/-- One directed or bidirected edge: the step of the territory closure. -/
def terrRel (G : ADMG n) (t w : Fin n) : Bool := G.dir t w || G.bi t w

/-- `MUCT(G_{\overline W}, Y)`, computed by a closure iteration: `reachSet` iterates, `n` times
from `{y}`, the step that adds every vertex of `diagH G y W` reached from the current set by one
directed or bidirected edge. -/
def muct (G : ADMG n) (y : Fin n) (W : Finset (Fin n)) : Finset (Fin n) :=
  reachSet (terrRel G) (Finset.univ \ diagH G y W) y

/-- `IB(G_{\overline W}, Y) = Pa(T)_G \ T` for `T = muct G y W`. -/
def ib (G : ADMG n) (y : Fin n) (W : Finset (Fin n)) : Finset (Fin n) :=
  diagBorder G (muct G y W)

section Closure

variable {G : ADMG n} {y : Fin n} {W T S : Finset (Fin n)}

theorem terrRel_iff {t w : Fin n} :
    terrRel G t w = true ↔ G.dir t w = true ∨ G.bi t w = true := by
  simp [terrRel]

theorem mem_muct {v : Fin n} :
    v ∈ muct G y W ↔ Reach (terrRel G) (Finset.univ \ diagH G y W) y v := by
  unfold muct
  exact reach_iff_mem_reachSet.symm

theorem y_mem_diagH (hy : y ∉ W) : y ∈ diagH G y W := mem_diagH.mpr ⟨hy, Reach.refl y⟩

theorem muct_subset_diagH (hy : y ∉ W) : muct G y W ⊆ diagH G y W := by
  intro v hv
  rw [mem_muct] at hv
  by_cases hyv : y = v
  · subst hyv
    exact y_mem_diagH hy
  · simpa using Reach.last_not_mem hv hyv

theorem diagUCT_muct (hy : y ∉ W) : DiagUCT G y W (muct G y W) := by
  refine ⟨muct_subset_diagH hy, mem_muct.mpr (Reach.refl y), fun t ht w hw hd => ?_,
    fun t ht w hw hb => ?_⟩
  · exact mem_muct.mpr (Reach.tail (mem_muct.mp ht) (terrRel_iff.mpr (Or.inl hd))
      (by simpa using hw))
  · exact mem_muct.mpr (Reach.tail (mem_muct.mp ht) (terrRel_iff.mpr (Or.inr hb))
      (by simpa using hw))

/-- The closure lies in every UC-territory. -/
theorem muct_subset_of_diagUCT (hT : DiagUCT G y W T) : muct G y W ⊆ T := by
  intro v hv
  rw [mem_muct] at hv
  induction hv with
  | refl => exact hT.2.1
  | @tail b c _ hbc ih =>
    have hcH : c ∈ diagH G y W := by simpa using hbc.2
    rcases terrRel_iff.mp hbc.1 with h | h
    · exact hT.2.2.1 b ih c hcH h
    · exact hT.2.2.2 b ih c hcH h

theorem diagMUCT_muct (hy : y ∉ W) : DiagMUCT G y W (muct G y W) :=
  ⟨diagUCT_muct hy, fun _ hT' hne hU =>
    hne (Finset.Subset.antisymm (Finset.mem_powerset.mp hT') (muct_subset_of_diagUCT hU))⟩

theorem y_not_mem_of_diagMUCT (hT : DiagMUCT G y W T) : y ∉ W :=
  (mem_diagH.mp (hT.1.1 hT.1.2.1)).1

/-- The minimal UC-territory is unique: it is the closure. -/
theorem eq_muct (hT : DiagMUCT G y W T) : T = muct G y W := by
  by_contra hne
  exact hT.2 (muct G y W) (Finset.mem_powerset.mpr (muct_subset_of_diagUCT hT.1))
    (Ne.symm hne) (diagUCT_muct (y_not_mem_of_diagMUCT hT))

theorem diagMUCT_iff (hy : y ∉ W) : DiagMUCT G y W T ↔ T = muct G y W := by
  refine ⟨eq_muct, fun h => ?_⟩
  rw [h]
  exact diagMUCT_muct hy

theorem diagH_anti {W₁ W₂ : Finset (Fin n)} (h : W₁ ⊆ W₂) : diagH G y W₂ ⊆ diagH G y W₁ := by
  intro v hv
  obtain ⟨hv1, hv2⟩ := mem_diagH.mp hv
  exact mem_diagH.mpr ⟨fun h' => hv1 (h h'), Reach.mono_avoid h hv2⟩

/-- Territories shrink when the intervention set grows. -/
theorem muct_anti {W₁ W₂ : Finset (Fin n)} (h : W₁ ⊆ W₂) : muct G y W₂ ⊆ muct G y W₁ := by
  intro v hv
  rw [mem_muct] at hv ⊢
  refine Reach.mono_avoid (fun u hu => ?_) hv
  simp only [Finset.mem_sdiff, Finset.mem_univ, true_and] at hu ⊢
  exact fun h' => hu (diagH_anti h h')

/-- A vertex of a UC-territory has a directed path to `y` inside the territory. -/
theorem reach_in_terr (hT : DiagUCT G y W T) {t : Fin n} (ht : t ∈ T) :
    Reach G.dir (Finset.univ \ T) t y := by
  have key : ∀ a, G.AncAvoid W a y → a ∈ T → Reach G.dir (Finset.univ \ T) a y := by
    intro a h
    unfold ADMG.AncAvoid Reach at h
    induction h using Relation.ReflTransGen.head_induction_on with
    | refl => exact fun _ => Reach.refl _
    | @head a c hac hcy ih =>
      intro ha
      have hcH : c ∈ diagH G y W := mem_diagH.mpr ⟨hac.2, hcy⟩
      have hcT : c ∈ T := hT.2.2.1 a ha c hcH hac.1
      exact Reach.head hac.1 (by simpa using hcT) (ih hcT)
  exact key t (mem_diagH.mp (hT.1 ht)).2 ht

/-- Every vertex with a directed path to `y` avoiding the border of a UC-territory `T` (and
possibly more) lies in `T`. -/
theorem diagH_border_subset (hT : DiagUCT G y W T) (Z : Finset (Fin n)) :
    diagH G y (diagBorder G T ∪ Z) ⊆ T := by
  intro v hv
  obtain ⟨hv1, hv2⟩ := mem_diagH.mp hv
  exact diag_mem_of_avoid_border hT (fun h => hv1 (Finset.mem_union_left _ h))
    (Reach.mono_avoid Finset.subset_union_left hv2)

/-- A UC-territory lies in `diagH` of every intervention set disjoint from it. -/
theorem subset_diagH_of_disjoint (hT : DiagUCT G y W T) {V : Finset (Fin n)}
    (hV : Disjoint V T) : T ⊆ diagH G y V := by
  intro t ht
  refine mem_diagH.mpr ⟨fun h => Finset.disjoint_left.mp hV h ht, ?_⟩
  refine Reach.mono_avoid (fun v hv => ?_) (reach_in_terr hT ht)
  simp only [Finset.mem_sdiff, Finset.mem_univ, true_and]
  exact fun h => Finset.disjoint_left.mp hV hv h

/-- A minimal territory for `W` is the minimal territory for every intervention set `V`
disjoint from it in whose `diagH` it stays closed. -/
theorem diagMUCT_of_closed (hT : DiagMUCT G y W T) {V : Finset (Fin n)} (hV : Disjoint V T)
    (hdir : ∀ t ∈ T, ∀ w ∈ diagH G y V, G.dir t w = true → w ∈ T)
    (hbi : ∀ t ∈ T, ∀ w ∈ diagH G y V, G.bi t w = true → w ∈ T) : DiagMUCT G y V T := by
  have hsub := subset_diagH_of_disjoint hT.1 hV
  refine ⟨⟨hsub, hT.1.2.1, hdir, hbi⟩, fun T' hT' hne hU => ?_⟩
  have hT'T := Finset.mem_powerset.mp hT'
  refine hT.2 T' hT' hne ⟨hT'T.trans hT.1.1, hU.2.1, fun t ht w hw hd => ?_,
    fun t ht w hw hb => ?_⟩
  · exact hU.2.2.1 t ht w (hsub (hT.1.2.2.1 t (hT'T ht) w hw hd)) hd
  · exact hU.2.2.2 t ht w (hsub (hT.1.2.2.2 t (hT'T ht) w hw hb)) hb

/-- Prop. 7, territory form: intervening on the border and outside the territory keeps it. -/
theorem prop7_diag (hT : DiagMUCT G y W T) {Z : Finset (Fin n)} (hZ : Disjoint Z T) :
    DiagMUCT G y (diagBorder G T ∪ Z) T := by
  have hV : Disjoint (diagBorder G T ∪ Z) T := by
    refine Finset.disjoint_left.mpr (fun v hv hvT => ?_)
    rcases Finset.mem_union.mp hv with h | h
    · exact (mem_diagBorder.mp h).1 hvT
    · exact Finset.disjoint_left.mp hZ h hvT
  have hD := diagH_border_subset hT.1 Z
  exact diagMUCT_of_closed hT hV (fun _ _ w hw _ => hD hw) (fun _ _ w hw _ => hD hw)

/-- Additional interventions outside a minimal territory keep it (App. C, proof of Thm. 9,
second closure fact). -/
theorem diagMUCT_union (hT : DiagMUCT G y W T) {Z : Finset (Fin n)} (hZ : Disjoint Z T) :
    DiagMUCT G y (W ∪ Z) T := by
  have hV : Disjoint (W ∪ Z) T := by
    refine Finset.disjoint_left.mpr (fun v hv hvT => ?_)
    rcases Finset.mem_union.mp hv with h | h
    · exact (mem_diagH.mp (hT.1.1 hvT)).1 h
    · exact Finset.disjoint_left.mp hZ h hvT
  have hsub : diagH G y (W ∪ Z) ⊆ diagH G y W := diagH_anti Finset.subset_union_left
  exact diagMUCT_of_closed hT hV (fun t ht w hw hd => hT.1.2.2.1 t ht w (hsub hw) hd)
    (fun t ht w hw hb => hT.1.2.2.2 t ht w (hsub hw) hb)

/-- Every `IB(G_{\overline S}, Y)` with `y ∉ S` is its own interventional border. -/
theorem ib_self (hy : y ∉ S) :
    DiagMUCT G y (ib G y S) (muct G y S) ∧ diagBorder G (muct G y S) = ib G y S := by
  have h := prop7_diag (diagMUCT_muct (G := G) hy) (Finset.disjoint_empty_left _)
  rw [Finset.union_empty] at h
  exact ⟨h, rfl⟩

/-- The graphical condition of Thm. 6 in computable form. -/
theorem border_iff (X : Finset (Fin n)) :
    (∃ T, DiagMUCT G y X T ∧ diagBorder G T = X) ↔ y ∉ X ∧ ib G y X = X := by
  constructor
  · rintro ⟨T, hT, hB⟩
    refine ⟨y_not_mem_of_diagMUCT hT, ?_⟩
    unfold ib
    rw [← eq_muct hT]
    exact hB
  · rintro ⟨hy, hB⟩
    exact ⟨muct G y X, diagMUCT_muct hy, hB⟩

end Closure

/-! ## The mutilated graph read without further intervention -/

section Cut

variable {G : ADMG n} {y : Fin n} {W T : Finset (Fin n)}

/-- `MUCT(G_{\overline W}, Y)` of the mutilated graph `cut G W` itself equals `muct G y W`: in
`cut G W` the vertices of `W` have no incoming edge, so the closure from `y` never reaches
them. -/
theorem muct_cut (hy : y ∉ W) : muct (cut G W) y ∅ = muct G y W := by
  have hD : ∀ v, v ∉ W → (v ∈ diagH (cut G W) y ∅ ↔ v ∈ diagH G y W) := by
    intro v hv
    rw [mem_diagH, mem_diagH, cut_ancAvoid, Finset.union_empty]
    exact ⟨fun h => ⟨hv, h.2⟩, fun h => ⟨Finset.notMem_empty v, h.2⟩⟩
  have k1 : ∀ v, Reach (terrRel (cut G W)) (Finset.univ \ diagH (cut G W) y ∅) y v →
      v ∉ W ∧ Reach (terrRel G) (Finset.univ \ diagH G y W) y v := by
    intro v h
    induction h with
    | refl => exact ⟨hy, Reach.refl _⟩
    | @tail b c _ hbc ih =>
      have hcW : c ∉ W := by
        rcases terrRel_iff.mp hbc.1 with h' | h'
        · exact (cut_dir.mp h').2
        · exact (cut_bi.mp h').2.2
      have hcD : c ∈ diagH (cut G W) y ∅ := by simpa using hbc.2
      refine ⟨hcW, Reach.tail ih.2 ?_ (by simpa using (hD c hcW).mp hcD)⟩
      rcases terrRel_iff.mp hbc.1 with h' | h'
      · exact terrRel_iff.mpr (Or.inl (cut_dir.mp h').1)
      · exact terrRel_iff.mpr (Or.inr (cut_bi.mp h').1)
  have k2 : ∀ v, Reach (terrRel G) (Finset.univ \ diagH G y W) y v →
      v ∉ W ∧ Reach (terrRel (cut G W)) (Finset.univ \ diagH (cut G W) y ∅) y v := by
    intro v h
    induction h with
    | refl => exact ⟨hy, Reach.refl _⟩
    | @tail b c _ hbc ih =>
      have hcH : c ∈ diagH G y W := by simpa using hbc.2
      have hcW : c ∉ W := (mem_diagH.mp hcH).1
      refine ⟨hcW, Reach.tail ih.2 ?_ (by simpa using (hD c hcW).mpr hcH)⟩
      rcases terrRel_iff.mp hbc.1 with h' | h'
      · exact terrRel_iff.mpr (Or.inl (cut_dir.mpr ⟨h', hcW⟩))
      · exact terrRel_iff.mpr (Or.inr (cut_bi.mpr ⟨h', ih.1, hcW⟩))
  ext v
  rw [mem_muct, mem_muct]
  exact ⟨fun h => (k1 v h).2, fun h => (k2 v h).2⟩

/-- The border of a territory disjoint from `W` is the same in `G_{\overline W}` and `G`. -/
theorem diagBorder_cut (hT : Disjoint W T) : diagBorder (cut G W) T = diagBorder G T := by
  ext v
  rw [mem_diagBorder, mem_diagBorder]
  constructor
  · rintro ⟨hv, t, ht, h⟩
    exact ⟨hv, t, ht, (cut_dir.mp h).1⟩
  · rintro ⟨hv, t, ht, h⟩
    exact ⟨hv, t, ht, cut_dir.mpr ⟨h, fun h' => Finset.disjoint_left.mp hT h' ht⟩⟩

/-- `IB(G_{\overline W}, Y)` of the mutilated graph `cut G W` itself equals `ib G y W`. -/
theorem ib_cut (hy : y ∉ W) : ib (cut G W) y ∅ = ib G y W := by
  unfold ib
  rw [muct_cut hy]
  refine diagBorder_cut (Finset.disjoint_left.mpr (fun v hvW hvT => ?_))
  exact (mem_diagH.mp (muct_subset_diagH hy hvT)).1 hvW

/-- `DiagMUCT G y W` is the minimal territory of the mutilated graph `G_{\overline W}` with no
further intervention. -/
theorem diagMUCT_cut_iff (hy : y ∉ W) : DiagMUCT (cut G W) y ∅ T ↔ DiagMUCT G y W T := by
  rw [diagMUCT_iff (Finset.notMem_empty y), diagMUCT_iff hy, muct_cut hy]

end Cut

/-! ## The compressed graph `G_{\overline X}[T ∪ X]` -/

section Compress

variable {G : ADMG n} {y : Fin n} {W T W' : Finset (Fin n)}

theorem not_mem_diagBorder_of_mem {v : Fin n} (hv : v ∈ T) : v ∉ diagBorder G T :=
  fun h => (mem_diagBorder.mp h).1 hv

/-- Inside `T`, the directed edges of `compress G T (diagBorder G T)` are those of `G`. -/
theorem compress_dir_iff {a b : Fin n} (ha : a ∈ T) (hb : b ∈ T) :
    (compress G T (diagBorder G T)).dir a b = true ↔ G.dir a b = true := by
  rw [compress_dir]
  exact ⟨fun h => h.1, fun h => ⟨h, not_mem_diagBorder_of_mem hb, Finset.mem_union_left _ ha,
    Finset.mem_union_left _ hb⟩⟩

/-- Inside `T`, the bidirected edges of `compress G T (diagBorder G T)` are those of `G`. -/
theorem compress_bi_iff {a b : Fin n} (ha : a ∈ T) (hb : b ∈ T) :
    (compress G T (diagBorder G T)).bi a b = true ↔ G.bi a b = true := by
  rw [compress_bi]
  exact ⟨fun h => h.1, fun h => ⟨h, not_mem_diagBorder_of_mem ha, not_mem_diagBorder_of_mem hb,
    Finset.mem_union_left _ ha, Finset.mem_union_left _ hb⟩⟩

theorem mem_of_compress_dir {a b : Fin n}
    (h : (compress G T (diagBorder G T)).dir a b = true) : b ∈ T := by
  obtain ⟨-, hbX, -, hb⟩ := compress_dir.mp h
  rcases Finset.mem_union.mp hb with hb | hb
  · exact hb
  · exact absurd hb hbX

theorem mem_of_compress_bi {a b : Fin n}
    (h : (compress G T (diagBorder G T)).bi a b = true) : b ∈ T := by
  obtain ⟨-, -, hbX, -, hb⟩ := compress_bi.mp h
  rcases Finset.mem_union.mp hb with hb | hb
  · exact hb
  · exact absurd hb hbX

/-- From a vertex of `T`, the directed paths to `y` of `H_{\overline{W'}}` and of
`G_{\overline{W ∪ W'}}` agree. -/
theorem compress_ancAvoid_iff (hT : DiagUCT G y W T) {a : Fin n} (ha : a ∈ T) :
    (compress G T (diagBorder G T)).AncAvoid W' a y ↔ G.AncAvoid (W ∪ W') a y := by
  constructor
  · intro h
    revert ha
    unfold ADMG.AncAvoid Reach at h
    induction h using Relation.ReflTransGen.head_induction_on with
    | refl => exact fun _ => Reach.refl _
    | @head a c hac _ ih =>
      intro ha
      have hcT : c ∈ T := mem_of_compress_dir hac.1
      have hcW : c ∉ W := (mem_diagH.mp (hT.1 hcT)).1
      refine Reach.head ((compress_dir_iff ha hcT).mp hac.1) (fun h' => ?_) (ih hcT)
      rcases Finset.mem_union.mp h' with h' | h'
      · exact hcW h'
      · exact hac.2 h'
  · intro h
    revert ha
    unfold ADMG.AncAvoid Reach at h
    induction h using Relation.ReflTransGen.head_induction_on with
    | refl => exact fun _ => Reach.refl _
    | @head a c hac hcy ih =>
      intro ha
      have hcW : c ∉ W := fun h' => hac.2 (Finset.mem_union_left _ h')
      have hcH : c ∈ diagH G y W :=
        mem_diagH.mpr ⟨hcW, Reach.mono_avoid Finset.subset_union_left hcy⟩
      have hcT : c ∈ T := hT.2.2.1 a ha c hcH hac.1
      exact Reach.head ((compress_dir_iff ha hcT).mpr hac.1)
        (fun h' => hac.2 (Finset.mem_union_right _ h')) (ih hcT)

theorem mem_diagH_compress_iff (hT : DiagUCT G y W T) {a : Fin n} (ha : a ∈ T) :
    a ∈ diagH (compress G T (diagBorder G T)) y W' ↔ a ∈ diagH G y (W ∪ W') := by
  have haW : a ∉ W := (mem_diagH.mp (hT.1 ha)).1
  rw [mem_diagH, mem_diagH, compress_ancAvoid_iff hT ha]
  constructor
  · rintro ⟨h1, h2⟩
    refine ⟨fun h => ?_, h2⟩
    rcases Finset.mem_union.mp h with h | h
    · exact haW h
    · exact h1 h
  · rintro ⟨h1, h2⟩
    exact ⟨fun h => h1 (Finset.mem_union_right _ h), h2⟩

theorem reach_terr_compress (hT : DiagUCT G y W T) {v : Fin n}
    (h : Reach (terrRel G) (Finset.univ \ diagH G y (W ∪ W')) y v) :
    v ∈ T ∧ Reach (terrRel (compress G T (diagBorder G T)))
      (Finset.univ \ diagH (compress G T (diagBorder G T)) y W') y v := by
  induction h with
  | refl => exact ⟨hT.2.1, Reach.refl _⟩
  | @tail b c _ hbc ih =>
    obtain ⟨hbT, hr⟩ := ih
    have hcD : c ∈ diagH G y (W ∪ W') := by simpa using hbc.2
    have hcH : c ∈ diagH G y W := diagH_anti Finset.subset_union_left hcD
    have hcT : c ∈ T := by
      rcases terrRel_iff.mp hbc.1 with h' | h'
      · exact hT.2.2.1 b hbT c hcH h'
      · exact hT.2.2.2 b hbT c hcH h'
    refine ⟨hcT, Reach.tail hr ?_ ?_⟩
    · rcases terrRel_iff.mp hbc.1 with h' | h'
      · exact terrRel_iff.mpr (Or.inl ((compress_dir_iff hbT hcT).mpr h'))
      · exact terrRel_iff.mpr (Or.inr ((compress_bi_iff hbT hcT).mpr h'))
    · simpa using (mem_diagH_compress_iff hT hcT).mpr hcD

theorem reach_terr_of_compress (hT : DiagUCT G y W T) {v : Fin n}
    (h : Reach (terrRel (compress G T (diagBorder G T)))
      (Finset.univ \ diagH (compress G T (diagBorder G T)) y W') y v) :
    v ∈ T ∧ Reach (terrRel G) (Finset.univ \ diagH G y (W ∪ W')) y v := by
  induction h with
  | refl => exact ⟨hT.2.1, Reach.refl _⟩
  | @tail b c _ hbc ih =>
    obtain ⟨hbT, hr⟩ := ih
    have hcD : c ∈ diagH (compress G T (diagBorder G T)) y W' := by simpa using hbc.2
    rcases terrRel_iff.mp hbc.1 with h' | h'
    · have hcT : c ∈ T := mem_of_compress_dir h'
      exact ⟨hcT, Reach.tail hr (terrRel_iff.mpr (Or.inl ((compress_dir_iff hbT hcT).mp h')))
        (by simpa using (mem_diagH_compress_iff hT hcT).mp hcD)⟩
    · have hcT : c ∈ T := mem_of_compress_bi h'
      exact ⟨hcT, Reach.tail hr (terrRel_iff.mpr (Or.inr ((compress_bi_iff hbT hcT).mp h')))
        (by simpa using (mem_diagH_compress_iff hT hcT).mp hcD)⟩

theorem muct_compress_eq (hT : DiagUCT G y W T) :
    muct (compress G T (diagBorder G T)) y W' = muct G y (W ∪ W') := by
  ext v
  rw [mem_muct, mem_muct]
  exact ⟨fun h => (reach_terr_of_compress hT h).2, fun h => (reach_terr_compress hT h).2⟩

theorem muct_union_subset (hT : DiagUCT G y W T) : muct G y (W ∪ W') ⊆ T :=
  fun _ hv => (reach_terr_compress hT (mem_muct.mp hv)).1

/-- The border of a subset of `T` is the same in `compress G T (diagBorder G T)` and `G`. -/
theorem diagBorder_compress {T' : Finset (Fin n)} (hT' : T' ⊆ T) :
    diagBorder (compress G T (diagBorder G T)) T' = diagBorder G T' := by
  ext v
  rw [mem_diagBorder, mem_diagBorder]
  constructor
  · rintro ⟨hvT', t, ht, h⟩
    exact ⟨hvT', t, ht, (compress_dir.mp h).1⟩
  · rintro ⟨hvT', t, ht, h⟩
    refine ⟨hvT', t, ht, compress_dir.mpr ⟨h, not_mem_diagBorder_of_mem (hT' ht), ?_,
      Finset.mem_union_left _ (hT' ht)⟩⟩
    by_cases hvT : v ∈ T
    · exact Finset.mem_union_left _ hvT
    · exact Finset.mem_union_right _ (mem_diagBorder.mpr ⟨hvT, t, hT' ht, h⟩)

end Compress

end Alg

open Alg

/-! ## Props. 7, 8 and 12 -/

section Props

variable {G : ADMG n} {y : Fin n} {W T W' : Finset (Fin n)}

/-- **Prop. 7** (`lem:ignorability`).  Let `T = MUCT(G_{\overline W}, Y)` and
`X = IB(G_{\overline W}, Y) = diagBorder G T`.  For every `Z` disjoint from `T`,
`MUCT(G_{\overline{X ∪ Z}}, Y) = T` and `IB(G_{\overline{X ∪ Z}}, Y) = X`. -/
theorem prop7 (hT : DiagMUCT G y W T) {Z : Finset (Fin n)} (hZ : Disjoint Z T) :
    DiagMUCT G y (diagBorder G T ∪ Z) T ∧ muct G y (diagBorder G T ∪ Z) = T ∧
      ib G y (diagBorder G T ∪ Z) = diagBorder G T := by
  have h := prop7_diag hT hZ
  have e : muct G y (diagBorder G T ∪ Z) = T := (eq_muct h).symm
  refine ⟨h, e, ?_⟩
  unfold ib
  rw [e]

/-- **Prop. 7** on the mutilated graph itself: `MUCT(G_{\overline{X ∪ Z}}, Y) = T` and
`IB(G_{\overline{X ∪ Z}}, Y) = X`, computed in `cut G (X ∪ Z)` with no further intervention. -/
theorem prop7_cut (hT : DiagMUCT G y W T) {Z : Finset (Fin n)} (hZ : Disjoint Z T) :
    muct (cut G (diagBorder G T ∪ Z)) y ∅ = T ∧
      ib (cut G (diagBorder G T ∪ Z)) y ∅ = diagBorder G T := by
  have h := prop7 hT hZ
  have hy : y ∉ diagBorder G T ∪ Z := y_not_mem_of_diagMUCT h.1
  rw [muct_cut hy, ib_cut hy]
  exact h.2

/-- **Prop. 8** (`lem:recursion`), computable form, for every further intervention `W'`: with
`T = MUCT(G_{\overline W}, Y)` and `H = G_{\overline X}[T ∪ X]` for `X = diagBorder G T`,
`H_{\overline{W'}}` and `G_{\overline{W ∪ W'}}` have the same MUCT and the same IB. -/
theorem prop8_muct (hT : DiagMUCT G y W T) :
    muct (compress G T (diagBorder G T)) y W' = muct G y (W ∪ W') ∧
      ib (compress G T (diagBorder G T)) y W' = ib G y (W ∪ W') := by
  have e := muct_compress_eq (W' := W') hT.1
  refine ⟨e, ?_⟩
  unfold ib
  rw [e]
  exact diagBorder_compress (muct_union_subset hT.1)

/-- **Prop. 8**, territory form, for every `W'` with `y ∉ W'`. -/
theorem prop8_general (hT : DiagMUCT G y W T) (hy : y ∉ W') {T' : Finset (Fin n)} :
    (DiagMUCT (compress G T (diagBorder G T)) y W' T' ↔ DiagMUCT G y (W ∪ W') T') ∧
      (DiagMUCT G y (W ∪ W') T' →
        diagBorder (compress G T (diagBorder G T)) T' = diagBorder G T') := by
  have hyW : y ∉ W := y_not_mem_of_diagMUCT hT
  have hyU : y ∉ W ∪ W' := by
    rw [Finset.mem_union]
    rintro (h | h)
    · exact hyW h
    · exact hy h
  rw [diagMUCT_iff hy, diagMUCT_iff hyU, muct_compress_eq hT.1]
  refine ⟨Iff.rfl, fun h => ?_⟩
  rw [h]
  exact diagBorder_compress (muct_union_subset hT.1)

/-- **Prop. 8** (`lem:recursion`).  Let `T` and `X` be the MUCT and IB of `G_{\overline W}` and
`H = G_{\overline X}[T ∪ X]`.  For every `W' ⊆ T \ {Y}`, `H_{\overline{W'}}` and
`G_{\overline{W ∪ W'}}` have the same MUCT and IB. -/
theorem prop8 (hT : DiagMUCT G y W T) (hW' : W' ⊆ T \ {y}) {T' : Finset (Fin n)} :
    (DiagMUCT (compress G T (diagBorder G T)) y W' T' ↔ DiagMUCT G y (W ∪ W') T') ∧
      (DiagMUCT G y (W ∪ W') T' →
        diagBorder (compress G T (diagBorder G T)) T' = diagBorder G T') :=
  prop8_general hT (fun h => (Finset.mem_sdiff.mp (hW' h)).2 (Finset.mem_singleton_self y))

/-- **Prop. 8** on the mutilated graphs themselves: for `W' ⊆ T \ {y}`, the graphs
`H_{\overline{W'}} = cut H W'` and `G_{\overline{W ∪ W'}} = cut G (W ∪ W')`, read with no
further intervention, have the same MUCT and IB. -/
theorem prop8_cut (hT : DiagMUCT G y W T) (hW' : W' ⊆ T \ {y}) :
    muct (cut (compress G T (diagBorder G T)) W') y ∅ = muct (cut G (W ∪ W')) y ∅ ∧
      ib (cut (compress G T (diagBorder G T)) W') y ∅ = ib (cut G (W ∪ W')) y ∅ := by
  have hy : y ∉ W' := fun h => (Finset.mem_sdiff.mp (hW' h)).2 (Finset.mem_singleton_self y)
  have hyU : y ∉ W ∪ W' := by
    rw [Finset.mem_union]
    rintro (h | h)
    · exact y_not_mem_of_diagMUCT hT h
    · exact hy h
  rw [muct_cut hy, muct_cut hyU, ib_cut hy, ib_cut hyU]
  exact prop8_muct hT

/-- **Prop. 12** (`prop:dontmisspomis`), graphical form: if `W` is its own interventional
border and `W' ⊆ W`, then `W \ W' ⊂ MUCT(G_{\overline{W'}}, Y) ∪ IB(G_{\overline{W'}}, Y)`. -/
theorem prop12_diag {TW T' : Finset (Fin n)} (hW : DiagMUCT G y W TW)
    (hB : diagBorder G TW = W) (hW' : W' ⊆ W) (hT' : DiagMUCT G y W' T') :
    W \ W' ⊂ T' ∪ diagBorder G T' := by
  have hyW : y ∉ W := y_not_mem_of_diagMUCT hW
  have hsub : TW ⊆ T' := by
    rw [eq_muct hW, eq_muct hT']
    exact muct_anti hW'
  have hincl : W \ W' ⊆ T' ∪ diagBorder G T' := by
    intro w hw
    have hwB : w ∈ diagBorder G TW := by
      rw [hB]
      exact (Finset.mem_sdiff.mp hw).1
    obtain ⟨-, t, ht, hwt⟩ := mem_diagBorder.mp hwB
    by_cases hwT : w ∈ T'
    · exact Finset.mem_union_left _ hwT
    · exact Finset.mem_union_right _ (mem_diagBorder.mpr ⟨hwT, t, hsub ht, hwt⟩)
  rw [Finset.ssubset_iff_of_subset hincl]
  exact ⟨y, Finset.mem_union_left _ hT'.1.2.1, fun h => hyW (Finset.mem_sdiff.mp h).1⟩

/-- **Prop. 12** (`prop:dontmisspomis`).  If `W` is a POMIS and `W' ⊆ W`, then
`W \ W' ⊂ MUCT(G_{\overline{W'}}, Y) ∪ IB(G_{\overline{W'}}, Y)`.  Uses only the "only if" part
of Thm. 6. -/
theorem prop12 (hG : G.WF) {T' : Finset (Fin n)} (hP : POMIS (ConfDiag G) y W)
    (hW' : W' ⊆ W) (hT' : DiagMUCT G y W' T') : W \ W' ⊂ T' ∪ diagBorder G T' := by
  obtain ⟨TW, hTW, hB⟩ := thm6_onlyif hG hP
  exact prop12_diag hTW hB hW' hT'

/-- **Prop. 12** on the mutilated graph itself:
`W \ W' ⊂ MUCT(G_{\overline{W'}}, Y) ∪ IB(G_{\overline{W'}}, Y)`, computed in `cut G W'`. -/
theorem prop12_cut (hG : G.WF) (hP : POMIS (ConfDiag G) y W) (hW' : W' ⊆ W) :
    W \ W' ⊂ muct (cut G W') y ∅ ∪ ib (cut G W') y ∅ := by
  have hy : y ∉ W' := fun h => hP.1.1 (hW' h)
  rw [muct_cut hy, ib_cut hy]
  exact prop12 hG hP hW' (diagMUCT_muct hy)

end Props

end Mec.Scb2018
