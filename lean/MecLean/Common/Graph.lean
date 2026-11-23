import Mathlib

set_option linter.unusedSectionVars false

/-!
# Shared graph vocabulary

Vertices are `Fin n`.  All graph data are Boolean matrices, so every finite example can be
checked by kernel reduction (`decide`).

* `Reach R D a b`: `b` is reached from `a` by `R`-steps whose *targets* avoid `D`; the start may
  lie in `D`.  With `D = X` this is a path from `a` whose later vertices avoid `X`, e.g. a path in
  `G_{\overline X}` or a path that is proper with respect to `X`.
* `ADMG`: directed (`dir i j` is `i → j`) and bidirected (`bi`) edges.  Causal diagrams may carry
  both `i → j` and `i ↔ j`; a MAG is a simple ADMG (`MecLean.Common.MAG`).
* `PMG`: partial mixed graphs with endpoint marks.  `m i j` is the mark at `j` on the edge between
  `i` and `j` (`none` when they are not adjacent).
-/

open Finset

namespace Mec

/-- Endpoint marks of partial mixed graphs. -/
inductive Mark
  | tail
  | arrow
  | circle
  deriving DecidableEq, Repr, Inhabited

variable {n : ℕ}

/-! ## Reachability and its decision procedure -/

/-- `Reach R D a b`: `b` is reached from `a` by `R`-steps whose targets avoid `D`. -/
def Reach (R : Fin n → Fin n → Bool) (D : Finset (Fin n)) : Fin n → Fin n → Prop :=
  Relation.ReflTransGen (fun x y => R x y = true ∧ y ∉ D)

section ReachDecide

variable (R : Fin n → Fin n → Bool) (D : Finset (Fin n))

/-- One round of extension by an `R`-step into a vertex outside `D`. -/
def stepSet (W : Finset (Fin n)) : Finset (Fin n) :=
  W ∪ Finset.univ.filter (fun c => c ∉ D ∧ ∃ b ∈ W, R b c = true)

/-- `n` rounds of extension from `{x}`. -/
def reachSet (x : Fin n) : Finset (Fin n) := (stepSet R D)^[n] {x}

variable {R D}

theorem subset_stepSet (W : Finset (Fin n)) : W ⊆ stepSet R D W := Finset.subset_union_left

theorem iterate_mono_le (W : Finset (Fin n)) {a b : ℕ} (hab : a ≤ b) :
    (stepSet R D)^[a] W ⊆ (stepSet R D)^[b] W := by
  induction hab with
  | refl => exact subset_rfl
  | step _ ih =>
    rw [Function.iterate_succ_apply']
    exact ih.trans (subset_stepSet _)

theorem iterate_stable (W : Finset (Fin n)) {k : ℕ}
    (hk : (stepSet R D)^[k + 1] W = (stepSet R D)^[k] W) :
    ∀ j, k ≤ j → (stepSet R D)^[j] W = (stepSet R D)^[k] W := by
  have hk' : stepSet R D ((stepSet R D)^[k] W) = (stepSet R D)^[k] W := by
    rw [Function.iterate_succ_apply'] at hk
    exact hk
  intro j hj
  induction hj with
  | refl => rfl
  | step _ ih => rw [Function.iterate_succ_apply', ih, hk']

theorem card_iterate (W : Finset (Fin n)) (hW : W.Nonempty) :
    ∀ m : ℕ, (∀ k < m, (stepSet R D)^[k + 1] W ≠ (stepSet R D)^[k] W) →
      m + 1 ≤ ((stepSet R D)^[m] W).card := by
  intro m
  induction m with
  | zero =>
    intro _
    simpa using Finset.card_pos.mpr hW
  | succ m ih =>
    intro h
    have h1 := ih (fun k hk => h k (Nat.lt_succ_of_lt hk))
    have hne := h m (Nat.lt_succ_self m)
    have hsub : (stepSet R D)^[m] W ⊆ (stepSet R D)^[m + 1] W :=
      iterate_mono_le W (Nat.le_succ m)
    have hss : (stepSet R D)^[m] W ⊂ (stepSet R D)^[m + 1] W :=
      Finset.ssubset_iff_subset_ne.mpr ⟨hsub, fun he => hne he.symm⟩
    have := Finset.card_lt_card hss
    omega

theorem stepSet_reachSet (x : Fin n) : stepSet R D (reachSet R D x) = reachSet R D x := by
  obtain ⟨k, hk, hkk⟩ : ∃ k < n, (stepSet R D)^[k + 1] {x} = (stepSet R D)^[k] {x} := by
    by_contra hcon
    push_neg at hcon
    have h := card_iterate (R := R) (D := D) {x} (Finset.singleton_nonempty x) n hcon
    have hle : ((stepSet R D)^[n] {x}).card ≤ n := by
      simpa using Finset.card_le_univ ((stepSet R D)^[n] {x})
    omega
  have h1 := iterate_stable {x} hkk (n + 1) (by omega)
  have h2 := iterate_stable {x} hkk n (by omega)
  have h3 : (stepSet R D)^[n + 1] {x} = (stepSet R D)^[n] {x} := h1.trans h2.symm
  rw [Function.iterate_succ_apply'] at h3
  exact h3

theorem mem_reachSet_of_reach {x y : Fin n} (h : Reach R D x y) : y ∈ reachSet R D x := by
  induction h with
  | refl =>
    exact iterate_mono_le {x} (Nat.zero_le n) (by simp)
  | @tail b c _ hbc ih =>
    rw [← stepSet_reachSet (R := R) (D := D) x]
    unfold stepSet
    exact Finset.mem_union_right _ (Finset.mem_filter.mpr ⟨Finset.mem_univ _, hbc.2, b, ih, hbc.1⟩)

theorem reach_of_mem_iterate {x : Fin n} :
    ∀ (m : ℕ) {y : Fin n}, y ∈ (stepSet R D)^[m] {x} → Reach R D x y := by
  intro m
  induction m with
  | zero =>
    intro y hy
    simp only [Function.iterate_zero, id_eq, Finset.mem_singleton] at hy
    subst hy
    exact Relation.ReflTransGen.refl
  | succ m ih =>
    intro y hy
    rw [Function.iterate_succ_apply'] at hy
    unfold stepSet at hy
    rcases Finset.mem_union.mp hy with hy | hy
    · exact ih hy
    · obtain ⟨_, hyD, b, hb, hby⟩ := Finset.mem_filter.mp hy
      exact Relation.ReflTransGen.tail (ih hb) ⟨hby, hyD⟩

theorem reach_iff_mem_reachSet {x y : Fin n} : Reach R D x y ↔ y ∈ reachSet R D x :=
  ⟨mem_reachSet_of_reach, reach_of_mem_iterate n⟩

instance decReach (x y : Fin n) : Decidable (Reach R D x y) :=
  decidable_of_iff _ reach_iff_mem_reachSet.symm

end ReachDecide

namespace Reach

variable {R R' : Fin n → Fin n → Bool} {D D' : Finset (Fin n)}

theorem refl (a : Fin n) : Reach R D a a := Relation.ReflTransGen.refl

theorem tail {a b c : Fin n} (h : Reach R D a b) (hbc : R b c = true) (hc : c ∉ D) :
    Reach R D a c := Relation.ReflTransGen.tail h ⟨hbc, hc⟩

theorem head {a b c : Fin n} (hab : R a b = true) (hb : b ∉ D) (h : Reach R D b c) :
    Reach R D a c := Relation.ReflTransGen.head ⟨hab, hb⟩ h

theorem single {a b : Fin n} (hab : R a b = true) (hb : b ∉ D) : Reach R D a b :=
  Relation.ReflTransGen.single ⟨hab, hb⟩

theorem trans {a b c : Fin n} (h₁ : Reach R D a b) (h₂ : Reach R D b c) : Reach R D a c :=
  Relation.ReflTransGen.trans h₁ h₂

/-- Avoiding more vertices is harder. -/
theorem mono_avoid (hD : D' ⊆ D) {a b : Fin n} (h : Reach R D a b) : Reach R D' a b := by
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hbc ih => exact Relation.ReflTransGen.tail ih ⟨hbc.1, fun h' => hbc.2 (hD h')⟩

/-- Stepwise transfer to another relation. -/
theorem mono_rel (hR : ∀ x y, y ∉ D → R x y = true → R' x y = true) {a b : Fin n}
    (h : Reach R D a b) : Reach R' D a b := by
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | @tail x y _ hxy ih => exact Relation.ReflTransGen.tail ih ⟨hR x y hxy.2 hxy.1, hxy.2⟩

/-- The last vertex of a nontrivial reach avoids `D`. -/
theorem last_not_mem {a b : Fin n} (h : Reach R D a b) (hne : a ≠ b) : b ∉ D := by
  rcases Relation.ReflTransGen.cases_tail h with h' | ⟨c, _, hcb⟩
  · exact absurd h'.symm hne
  · exact hcb.2

/-- Choose the last vertex of `S` on a reach that avoids `T`: from there the reach avoids
`T ∪ S`. -/
theorem last_mem {S T : Finset (Fin n)} {s k : Fin n} (hs : s ∈ S) (hsT : s ∉ T)
    (h : Reach R T s k) : ∃ x ∈ S, x ∉ T ∧ Reach R (T ∪ S) x k := by
  induction h with
  | refl => exact ⟨s, hs, hsT, Relation.ReflTransGen.refl⟩
  | @tail b c _ hbc ih =>
    obtain ⟨x, hxS, hxT, hx⟩ := ih
    by_cases hcS : c ∈ S
    · exact ⟨c, hcS, hbc.2, Relation.ReflTransGen.refl⟩
    · exact ⟨x, hxS, hxT, Relation.ReflTransGen.tail hx ⟨hbc.1, by simp [hbc.2, hcS]⟩⟩

/-- A reach avoiding `D` either avoids `D ∪ S` or has a last vertex in `S`, from which it avoids
`D ∪ S`. -/
theorem split_last {R : Fin n → Fin n → Bool} {D : Finset (Fin n)} (S : Finset (Fin n))
    {a b : Fin n} (h : Reach R D a b) :
    Reach R (D ∪ S) a b ∨ ∃ c ∈ S, c ∉ D ∧ Reach R (D ∪ S) c b := by
  induction h with
  | refl => exact Or.inl (Reach.refl a)
  | @tail b' c _ hbc ih =>
    by_cases hcS : c ∈ S
    · exact Or.inr ⟨c, hcS, hbc.2, Reach.refl c⟩
    · have hc : c ∉ D ∪ S := by simp [hbc.2, hcS]
      rcases ih with ih | ⟨c', hc'S, hc'D, ih⟩
      · exact Or.inl (Reach.tail ih hbc.1 hc)
      · exact Or.inr ⟨c', hc'S, hc'D, Reach.tail ih hbc.1 hc⟩

end Reach

/-! ## Acyclic directed mixed graphs -/

/-- Directed (`dir i j`: `i → j`) and bidirected (`bi i j`: `i ↔ j`) edges. -/
structure ADMG (n : ℕ) where
  dir : Fin n → Fin n → Bool
  bi : Fin n → Fin n → Bool

namespace ADMG

variable (G : ADMG n)

theorem ext_iff' {G H : ADMG n} : G = H ↔ G.dir = H.dir ∧ G.bi = H.bi := by
  cases G; cases H; simp

instance : DecidableEq (ADMG n) := fun G H => decidable_of_iff _ ext_iff'.symm

/-- Adjacency. -/
def adj (i j : Fin n) : Bool := G.dir i j || G.dir j i || G.bi i j

/-- Reflexive ancestry. -/
def Anc (a b : Fin n) : Prop := Reach G.dir ∅ a b

/-- A directed path from `a` to `b` whose vertices after `a` avoid `D`; e.g. `a ∈ An(b)` in
`G_{\overline D}` when `a ∈ D`, and `a ∈ An(b)` in `G \ D` when `a ∉ D`. -/
def AncAvoid (D : Finset (Fin n)) (a b : Fin n) : Prop := Reach G.dir D a b

instance (a b : Fin n) : Decidable (G.Anc a b) := by unfold Anc; infer_instance
instance (D : Finset (Fin n)) (a b : Fin n) : Decidable (G.AncAvoid D a b) := by
  unfold AncAvoid; infer_instance

/-- Well-formedness: irreflexive edges, symmetric bidirected edges, no directed cycle. -/
def WF : Prop :=
  (∀ i, G.dir i i = false) ∧ (∀ i, G.bi i i = false) ∧ (∀ i j, G.bi i j = G.bi j i) ∧
    (∀ i j, G.dir i j = true → ¬ G.Anc j i)

instance : Decidable G.WF := by unfold WF; infer_instance

/-- At most one edge between two vertices. -/
def Simple : Prop := ∀ i j, G.dir i j = true → G.dir j i = false ∧ G.bi i j = false

instance : Decidable G.Simple := by unfold Simple; infer_instance

/-- No bidirected edge between ancestrally related vertices. -/
def Ancestral : Prop := ∀ i j, G.bi i j = true → ¬ G.Anc i j

instance : Decidable G.Ancestral := by unfold Ancestral; infer_instance

/-- Mark at `j` on the edge between `i` and `j`, for a simple ADMG. -/
def markAt (i j : Fin n) : Option Mark :=
  if G.dir i j || G.bi i j then some .arrow else if G.dir j i then some .tail else none

theorem anc_of_avoid {D : Finset (Fin n)} {a b : Fin n} (h : G.AncAvoid D a b) : G.Anc a b :=
  Reach.mono_avoid (Finset.empty_subset D) h

theorem adj_comm (i j : Fin n) (hbi : G.bi i j = G.bi j i) : G.adj i j = G.adj j i := by
  unfold adj
  rw [hbi]
  cases G.dir i j <;> cases G.dir j i <;> cases G.bi j i <;> rfl

end ADMG

/-! ## Partial mixed graphs -/

/-- `m i j` is the mark at `j` on the edge between `i` and `j`, or `none` if non-adjacent. -/
structure PMG (n : ℕ) where
  m : Fin n → Fin n → Option Mark

namespace PMG

variable (P : PMG n)

theorem ext_iff' {P Q : PMG n} : P = Q ↔ P.m = Q.m := by
  cases P; cases Q; simp

instance : DecidableEq (PMG n) := fun P Q => decidable_of_iff _ ext_iff'.symm

def adj (i j : Fin n) : Bool := (P.m i j).isSome

/-- An edge that is possibly directed from `i` to `j`: no arrowhead at `i`, no tail at `j`.
These are the edges `i → j`, `i o→ j`, `i o–o j` (and `i –o j`).  `i` is a possible parent of
`j` exactly when `possDir i j`. -/
def possDir (i j : Fin n) : Bool :=
  decide ((P.m i j).isSome = true ∧ P.m j i ≠ some .arrow ∧ P.m i j ≠ some .tail)

/-- A definite directed edge `i → j`. -/
def dirEdge (i j : Fin n) : Bool := decide (P.m j i = some .tail ∧ P.m i j = some .arrow)

/-- A circle edge `i o–o j`. -/
def circEdge (i j : Fin n) : Bool := decide (P.m i j = some .circle ∧ P.m j i = some .circle)

/-- Possibly directed path from `a` to `b` whose vertices after `a` avoid `D`. -/
def PossAnc (D : Finset (Fin n)) (a b : Fin n) : Prop := Reach P.possDir D a b

instance (D : Finset (Fin n)) (a b : Fin n) : Decidable (P.PossAnc D a b) := by
  unfold PossAnc; infer_instance

/-- Well-formedness: irreflexive and symmetric adjacency. -/
def WF : Prop := (∀ i, P.m i i = none) ∧ ∀ i j, (P.m i j).isSome = (P.m j i).isSome

instance : Decidable P.WF := by unfold WF; infer_instance

end PMG

/-- The mark matrix of a simple ADMG. -/
def ADMG.toPMG (G : ADMG n) : PMG n := ⟨G.markAt⟩

end Mec
