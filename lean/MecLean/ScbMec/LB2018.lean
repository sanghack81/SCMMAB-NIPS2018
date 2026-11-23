import MecLean.ScbMec.MIS
import MecLean.Common.DiagTerritory

set_option linter.unusedSectionVars false

/-!
# Lee–Bareinboim (2018), Thm. 6, "if": a set that is its own interventional border is a POMIS

For a causal diagram `G`: if `T` is the minimal UC-territory after intervention on `X` and
`Pa(T) \ T = X`, then some SCM conforming to `G` makes `X` strictly better than every other
MIS.  The paper's proof (its Props. 4 and 5) sketches a colour-coded witness; this file gives a
complete proof with its own witness.

## Graph facts

* `mem_of_avoid`: every vertex outside `X` with a directed path to `y` avoiding `X` lies in `T`.
* `reachT_of_avoid`: such a vertex reaches `y` by a directed path inside `T`.
* `exists_mem_not_mem`: an MIS `W ≠ X` disjoint from `T` misses some vertex of `X`.
* `exists_split`: for `W` meeting `T`, call `v ∈ T` *live* if no directed path from `v` to `y`
  inside `T` meets `W`.  The live vertices contain `y` and are closed under directed edges
  inside `T`; by minimality of `T` they are not closed under bidirected edges, so some `a ↔ b`
  inside `T` has `a` live and `b` not live.

## The witness

Every root of `Mech.ofADMG` is a fair coin, and every vertex takes values in
`Roots n → ZMod K`, one coordinate per root.  Vertex `v` records the bits of its own roots at
their coordinates.  A vertex of `T` adds, at each coordinate `l`, the coordinate `l` of its
parents in `T`, and of a parent `p ∉ T` only when `l` is the private root of `p`.  At a fixed
intervention the coordinate `l` of `y` is then `C + [coin l] · cnt`, where `cnt` counts the
directed paths that carry the record of `l` to `y` and avoid the intervened set (`resp_coord`).

The reward is `1` exactly when each coordinate of `y` takes one of the two values it can take
under `do(X = 0)`, so the arm `X = 0` has mean `1`.  For an MIS `W ≠ X` some coordinate has
`0 < cnt_W ≠ cnt_X` (`key_coord`): a private coordinate of a vertex of `X \ W` when `W` misses
`T`, and the coordinate of the coin of `a ↔ b` from `exists_split` otherwise.  Flipping that
coin moves the coordinate by `cnt_W`, so at most one of the two positions is a value of
`do(X = 0)`, and each arm of `W` has mean at most `1/2` (`E_le_half`).  `K` exceeds twice every
path count, so these differences do not vanish modulo `K`.
-/

open Finset

namespace Mec.ScbMec

open Mec

variable {n : ℕ}

namespace LB18

/-! ## Graph facts -/

section Graph

variable {G : ADMG n} {y : Fin n} {X T : Finset (Fin n)}

theorem mem_diagH {v : Fin n} : v ∈ diagH G y X ↔ v ∉ X ∧ G.AncAvoid X v y := by
  unfold diagH
  rw [Finset.mem_filter]
  simp only [Finset.mem_univ, true_and]

theorem mem_diagBorder {v : Fin n} :
    v ∈ diagBorder G T ↔ v ∉ T ∧ ∃ t ∈ T, G.dir v t = true := by
  unfold diagBorder
  rw [Finset.mem_filter]
  simp only [Finset.mem_univ, true_and]

theorem not_mem_of_mem_border (hB : diagBorder G T = X) {x : Fin n} (hx : x ∈ X) : x ∉ T := by
  rw [← hB] at hx
  exact (mem_diagBorder.mp hx).1

theorem avoid_of_mem (hT : DiagMUCT G y X T) {t : Fin n} (ht : t ∈ T) : G.AncAvoid X t y :=
  (mem_diagH.mp (hT.1.1 ht)).2

/-- A vertex outside `X` with a directed path to `y` avoiding `X` lies in `T`: where such a path
enters `T`, the vertex before it would be a parent of `T` outside `X`. -/
theorem mem_of_avoid (hT : DiagMUCT G y X T) (hB : diagBorder G T = X) {v : Fin n}
    (hvX : v ∉ X) (h : G.AncAvoid X v y) : v ∈ T := by
  unfold ADMG.AncAvoid Reach at h
  revert hvX
  induction h using Relation.ReflTransGen.head_induction_on with
  | refl => exact fun _ => hT.1.2.1
  | @head a c hac _ ih =>
    intro haX
    have hcT : c ∈ T := ih hac.2
    by_contra haT
    exact haX (by rw [← hB]; exact mem_diagBorder.mpr ⟨haT, c, hcT, hac.1⟩)

/-- Such a vertex reaches `y` by a directed path inside `T`. -/
theorem reachT_of_avoid (hT : DiagMUCT G y X T) (hB : diagBorder G T = X) {a : Fin n}
    (h : G.AncAvoid X a y) : Reach G.dir Tᶜ a y := by
  unfold ADMG.AncAvoid Reach at h
  induction h using Relation.ReflTransGen.head_induction_on with
  | refl => exact Reach.refl _
  | @head a c hac hcy ih =>
    have hcT : c ∈ T := mem_of_avoid hT hB hac.2 hcy
    exact Reach.head hac.1 (fun h => (Finset.mem_compl.mp h) hcT) ih

/-- Two vertices that reach each other are equal. -/
theorem eq_of_reach_reach (hG : G.WF) {D D' : Finset (Fin n)} {a b : Fin n}
    (h1 : Reach G.dir D a b) (h2 : Reach G.dir D' b a) : a = b := by
  rcases Relation.ReflTransGen.cases_tail h1 with h | ⟨c, hac, hcb⟩
  · exact h.symm
  · exfalso
    have hac' : Reach G.dir D a c := hac
    exact hG.acyclic hcb.1 (Reach.trans (Reach.mono_avoid (Finset.empty_subset D') h2)
      (Reach.mono_avoid (Finset.empty_subset D) hac'))

/-- A directed path inside `T` from a vertex of `T` ends in `T`. -/
theorem mem_of_reachT {v w : Fin n} (h : Reach G.dir Tᶜ v w) (hv : v ∈ T) : w ∈ T := by
  rcases Relation.ReflTransGen.cases_tail h with h' | ⟨c, -, hcw⟩
  · rw [h']
    exact hv
  · by_contra hw
    exact hcw.2 (Finset.mem_compl.mpr hw)

theorem compl_union_sub {S : Finset (Fin n)} (hS : ∀ s ∈ S, s ∉ T) : Tᶜ ∪ S ⊆ Tᶜ := by
  intro v hv
  rcases Finset.mem_union.mp hv with h | h
  · exact h
  · exact Finset.mem_compl.mpr (hS v h)

/-- An MIS `W ≠ X` disjoint from `T` misses some vertex of `X`: otherwise a vertex of `W \ X`
would reach `y` avoiding `W`, and the path would enter `T` from a vertex of `X ⊆ W`. -/
theorem exists_mem_not_mem (hT : DiagMUCT G y X T) (hB : diagBorder G T = X)
    {W : Finset (Fin n)} (hW : AncCrit G y W) (hWT : ∀ w ∈ W, w ∉ T) (hne : W ≠ X) :
    ∃ x ∈ X, x ∉ W := by
  by_contra hcon
  have hXW : ∀ x ∈ X, x ∈ W := fun x hx => by
    by_contra h
    exact hcon ⟨x, hx, h⟩
  obtain ⟨w, hwW, hwX⟩ : ∃ w ∈ W, w ∉ X := by
    by_contra h
    apply hne
    have hWX : W ⊆ X := fun v hv => by
      by_contra hvX
      exact h ⟨v, hv, hvX⟩
    exact Finset.Subset.antisymm hWX (fun v hv => hXW v hv)
  have key : ∀ v, G.AncAvoid W v y → v ∉ T → v ∉ X → False := by
    intro v h
    unfold ADMG.AncAvoid Reach at h
    induction h using Relation.ReflTransGen.head_induction_on with
    | refl => exact fun hyT _ => hyT hT.1.2.1
    | @head a c hac _ ih =>
      intro haT haX
      have hcX : c ∉ X := fun h => hac.2 (hXW c h)
      by_cases hcT : c ∈ T
      · exact haX (by rw [← hB]; exact mem_diagBorder.mpr ⟨haT, c, hcT, hac.1⟩)
      · exact ih hcT hcX
  exact key w (hW w hwW) (hWT w hwW) hwX

/-- If `W` meets `T`, some bidirected edge `a ↔ b` inside `T` joins a live vertex `a` (no
directed path from `a` to `y` inside `T` meets `W`) to a vertex `b` that is not live.
Otherwise the live vertices would form a smaller UC-territory. -/
theorem exists_split (hG : G.WF) (hT : DiagMUCT G y X T) (hB : diagBorder G T = X)
    {W : Finset (Fin n)} (hyW : y ∉ W) {s₀ : Fin n} (hs₀W : s₀ ∈ W) (hs₀T : s₀ ∈ T) :
    ∃ a ∈ T, ∃ b ∈ T, G.bi a b = true ∧
      (∀ s ∈ W, ¬ (Reach G.dir Tᶜ a s ∧ Reach G.dir Tᶜ s y)) ∧
      ∃ s ∈ W, Reach G.dir Tᶜ b s ∧ Reach G.dir Tᶜ s y := by
  by_contra hcon
  obtain ⟨Lv, hmem⟩ : ∃ Lv : Finset (Fin n), ∀ v, v ∈ Lv ↔
      v ∈ T ∧ ∀ s ∈ W, ¬ (Reach G.dir Tᶜ v s ∧ Reach G.dir Tᶜ s y) :=
    ⟨T.filter (fun v => ∀ s ∈ W, ¬ (Reach G.dir Tᶜ v s ∧ Reach G.dir Tᶜ s y)),
      fun v => Finset.mem_filter⟩
  have hsub : Lv ⊆ T := fun v hv => ((hmem v).mp hv).1
  have hH : ∀ w ∈ diagH G y X, w ∈ T := fun w hw =>
    mem_of_avoid hT hB (mem_diagH.mp hw).1 (mem_diagH.mp hw).2
  have hUCT : DiagUCT G y X Lv := by
    refine ⟨fun v hv => hT.1.1 (hsub hv), (hmem y).mpr ⟨hT.1.2.1, fun s hsW hs => ?_⟩,
      fun t ht w hw htw => ?_, fun t ht w hw htw => ?_⟩
    · have hys : y = s := eq_of_reach_reach hG hs.1 hs.2
      rw [← hys] at hsW
      exact hyW hsW
    · have hwT := hH w hw
      refine (hmem w).mpr ⟨hwT, fun s hsW hs => ((hmem t).mp ht).2 s hsW ⟨?_, hs.2⟩⟩
      exact Reach.head htw (fun h => (Finset.mem_compl.mp h) hwT) hs.1
    · have hwT := hH w hw
      exact (hmem w).mpr ⟨hwT, fun s hsW hs =>
        hcon ⟨t, hsub ht, w, hwT, htw, ((hmem t).mp ht).2, s, hsW, hs⟩⟩
  have hne : Lv ≠ T := by
    intro h
    have hs₀ : s₀ ∈ Lv := by
      rw [h]
      exact hs₀T
    exact ((hmem s₀).mp hs₀).2 s₀ hs₀W
      ⟨Reach.refl s₀, reachT_of_avoid hT hB (avoid_of_mem hT hs₀T)⟩
  exact hT.2 Lv (Finset.mem_powerset.mpr hsub) hne hUCT

end Graph

/-- A directed path inside `T` from a live vertex `a` to a vertex `w` that reaches `y` inside
`T` avoids `W`. -/
theorem reach_avoid_of_live {G : ADMG n} {T W : Finset (Fin n)} {a z : Fin n}
    (hlive : ∀ s ∈ W, ¬ (Reach G.dir Tᶜ a s ∧ Reach G.dir Tᶜ s z)) {w : Fin n}
    (h : Reach G.dir Tᶜ a w) (hwz : Reach G.dir Tᶜ w z) : Reach G.dir (Tᶜ ∪ W) a w := by
  revert hwz
  induction h with
  | refl => exact fun _ => Reach.refl _
  | @tail b c hab hbc ih =>
    intro hcz
    have hbz : Reach G.dir Tᶜ b z := Reach.head hbc.1 hbc.2 hcz
    refine Reach.tail (ih hbz) hbc.1 ?_
    rw [Finset.mem_union, not_or]
    exact ⟨hbc.2, fun hcW => hlive c hcW ⟨Reach.tail hab hbc.1 hbc.2, hcz⟩⟩

theorem mem_T_of_inr_mem {G : ADMG n} {T : Finset (Fin n)} {a b w : Fin n} (haT : a ∈ T)
    (hbT : b ∈ T) (h : (Sum.inr (a, b) : Roots n) ∈ rootScope G w) : w ∈ T := by
  rcases mem_rootScope.mp h with h | ⟨a', b', h, -, hw⟩
  · exact absurd h (by simp)
  · simp only [Sum.inr.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    rcases hw with rfl | rfl
    · exact haT
    · exact hbT

/-! ## Path counts -/

section Count

variable {G : ADMG n}

/-- The parents `p` of `v` whose coordinate `l` a vertex `v ∈ T` reads: every parent in `T`, and
a parent outside `T` only at its private coordinate. -/
def par (G : ADMG n) (T : Finset (Fin n)) (l : Roots n) (v : Fin n) : Finset (Fin n) :=
  Finset.univ.filter (fun p => G.dir p v = true ∧ (p ∈ T ∨ l = Sum.inl p))

theorem mem_par {T : Finset (Fin n)} {l : Roots n} {v p : Fin n} :
    p ∈ par G T l v ↔ G.dir p v = true ∧ (p ∈ T ∨ l = Sum.inl p) := by
  unfold par
  rw [Finset.mem_filter]
  simp only [Finset.mem_univ, true_and]

/-- One step of the path count `cnt`. -/
def cntF (G : ADMG n) (T S : Finset (Fin n)) (l : Roots n) (v : Fin n) (z : Fin n → ℕ) : ℕ :=
  if v ∈ S then 0 else
    (if l ∈ rootScope G v then 1 else 0) + if v ∈ T then ∑ p ∈ par G T l v, z p else 0

/-- `cnt G T S l v`: the number of directed paths that carry the record of the root `l` to `v`
in the witness under an intervention on `S`.  A path starts at a vertex whose roots contain
`l`, continues inside `T` through vertices that read coordinate `l`, and avoids `S`. -/
def cnt (G : ADMG n) (T S : Finset (Fin n)) (l : Roots n) : Fin n → ℕ :=
  sol (X := fun _ => ℕ) (cntF G T S l)

theorem cnt_spec (hG : G.WF) (T S : Finset (Fin n)) (l : Roots n) (v : Fin n) :
    cnt G T S l v = cntF G T S l v (cnt G T S l) := by
  unfold cnt
  refine sol_spec (X := fun _ => ℕ) (topRank G) (fun i => ?_) v
  intro z z' h
  have hs : ∑ p ∈ par G T l i, z p = ∑ p ∈ par G T l i, z' p :=
    Finset.sum_congr rfl (fun p hp => h p (topRank_lt hG (mem_par.mp hp).1))
  unfold cntF
  rw [hs]

theorem cnt_of_mem (hG : G.WF) {T S : Finset (Fin n)} {l : Roots n} {v : Fin n} (hv : v ∈ S) :
    cnt G T S l v = 0 := by
  rw [cnt_spec hG T S l v]
  unfold cntF
  rw [ite_eq_left hv]

theorem cnt_of_not_mem (hG : G.WF) {T S : Finset (Fin n)} {l : Roots n} {v : Fin n}
    (hv : v ∉ S) : cnt G T S l v = (if l ∈ rootScope G v then 1 else 0) +
      if v ∈ T then ∑ p ∈ par G T l v, cnt G T S l p else 0 := by
  rw [cnt_spec hG T S l v]
  unfold cntF
  rw [ite_eq_right hv]

theorem one_le_cnt (hG : G.WF) {T S : Finset (Fin n)} {l : Roots n} {v : Fin n} (hv : v ∉ S)
    (hl : l ∈ rootScope G v) : 1 ≤ cnt G T S l v := by
  rw [cnt_of_not_mem hG hv, ite_eq_left hl]
  omega

/-- Blocking more vertices leaves fewer paths: the count under `S` is at most the count under
`S'` when every vertex of `S' \ S` neither records `l` nor lies in `T`. -/
theorem cnt_le (hG : G.WF) {T S S' : Finset (Fin n)} {l : Roots n}
    (h : ∀ w ∈ S', w ∉ S → l ∉ rootScope G w ∧ w ∉ T) (v : Fin n) :
    cnt G T S l v ≤ cnt G T S' l v := by
  refine rank_induction (topRank G) (P := fun v => cnt G T S l v ≤ cnt G T S' l v)
    (fun v ih => ?_) v
  by_cases hvS : v ∈ S
  · rw [cnt_of_mem hG hvS]
    exact Nat.zero_le _
  · by_cases hvS' : v ∈ S'
    · obtain ⟨h1, h2⟩ := h v hvS' hvS
      rw [cnt_of_not_mem hG hvS, ite_eq_right h1, ite_eq_right h2]
      omega
    · rw [cnt_of_not_mem hG hvS, cnt_of_not_mem hG hvS']
      have hs : (if v ∈ T then ∑ p ∈ par G T l v, cnt G T S l p else 0) ≤
          (if v ∈ T then ∑ p ∈ par G T l v, cnt G T S' l p else 0) := by
        by_cases hvT : v ∈ T
        · rw [ite_eq_left hvT, ite_eq_left hvT]
          exact Finset.sum_le_sum (fun p hp => ih p (topRank_lt hG (mem_par.mp hp).1))
        · simp [hvT]
      omega

theorem cnt_le_empty (hG : G.WF) (T S : Finset (Fin n)) (l : Roots n) (v : Fin n) :
    cnt G T S l v ≤ cnt G T ∅ l v :=
  cnt_le hG (fun w hw _ => absurd hw (Finset.notMem_empty w)) v

/-- The count grows along an edge into a vertex of `T` outside `S` that reads coordinate `l`. -/
theorem cnt_le_child (hG : G.WF) {T S : Finset (Fin n)} {l : Roots n} {v w : Fin n}
    (hvw : G.dir v w = true) (hwS : w ∉ S) (hwT : w ∈ T) (hv : v ∈ T ∨ l = Sum.inl v) :
    cnt G T S l v ≤ cnt G T S l w := by
  rw [cnt_of_not_mem hG hwS, ite_eq_left hwT]
  have h1 : cnt G T S l v ≤ ∑ p ∈ par G T l w, cnt G T S l p :=
    Finset.single_le_sum (fun p _ => Nat.zero_le _) (mem_par.mpr ⟨hvw, hv⟩)
  omega

/-- The count grows along a directed path inside `T` avoiding `S`. -/
theorem cnt_le_reach (hG : G.WF) {T S : Finset (Fin n)} {l : Roots n} {v w : Fin n}
    (h : Reach G.dir (Tᶜ ∪ S) v w) (hv : v ∈ T) : cnt G T S l v ≤ cnt G T S l w := by
  have key : w ∈ T ∧ cnt G T S l v ≤ cnt G T S l w := by
    induction h with
    | refl => exact ⟨hv, le_rfl⟩
    | @tail b c _ hbc ih =>
      have hcT : c ∈ T := by
        by_contra hc
        exact hbc.2 (Finset.mem_union_left _ (Finset.mem_compl.mpr hc))
      have hcS : c ∉ S := fun hc => hbc.2 (Finset.mem_union_right _ hc)
      exact ⟨hcT, le_trans ih.2 (cnt_le_child hG hbc.1 hcS hcT (Or.inl ih.1))⟩
  exact key.2

/-- A strict gap between two counts passes to a child in `T`. -/
theorem cnt_lt_child (hG : G.WF) {T S₁ S₂ : Finset (Fin n)} {l : Roots n}
    (hle : ∀ v, cnt G T S₁ l v ≤ cnt G T S₂ l v) {v w : Fin n}
    (hvw : G.dir v w = true) (hvT : v ∈ T) (hwT : w ∈ T) (hwS₂ : w ∉ S₂)
    (hlt : cnt G T S₁ l v < cnt G T S₂ l v) : cnt G T S₁ l w < cnt G T S₂ l w := by
  by_cases hwS₁ : w ∈ S₁
  · rw [cnt_of_mem hG hwS₁]
    have h1 := cnt_le_child hG (S := S₂) (l := l) hvw hwS₂ hwT (Or.inl hvT)
    omega
  · rw [cnt_of_not_mem hG hwS₁, cnt_of_not_mem hG hwS₂, ite_eq_left hwT, ite_eq_left hwT]
    have hs : ∑ p ∈ par G T l w, cnt G T S₁ l p < ∑ p ∈ par G T l w, cnt G T S₂ l p :=
      Finset.sum_lt_sum (fun p _ => hle p) ⟨v, mem_par.mpr ⟨hvw, Or.inl hvT⟩, hlt⟩
    omega

/-- A strict gap between two counts passes along a directed path inside `T`. -/
theorem cnt_lt_reach (hG : G.WF) {T S₁ S₂ : Finset (Fin n)} {l : Roots n}
    (hle : ∀ v, cnt G T S₁ l v ≤ cnt G T S₂ l v) (hS₂ : ∀ u ∈ T, u ∉ S₂) {v w : Fin n}
    (h : Reach G.dir Tᶜ v w) (hv : v ∈ T) (hlt : cnt G T S₁ l v < cnt G T S₂ l v) :
    cnt G T S₁ l w < cnt G T S₂ l w := by
  have key : w ∈ T ∧ cnt G T S₁ l w < cnt G T S₂ l w := by
    induction h with
    | refl => exact ⟨hv, hlt⟩
    | @tail b c _ hbc ih =>
      have hcT : c ∈ T := by
        by_contra hc
        exact hbc.2 (Finset.mem_compl.mpr hc)
      exact ⟨hcT, cnt_lt_child hG hle hbc.1 ih.1 hcT (hS₂ c hcT) ih.2⟩
  exact key.2

/-- Under an intervention on `X ∋ x₀`, no path carries the private record of `x₀`. -/
theorem cnt_inl_zero (hG : G.WF) {T X : Finset (Fin n)} {x₀ : Fin n} (hx₀ : x₀ ∈ X)
    (v : Fin n) : cnt G T X (Sum.inl x₀) v = 0 := by
  refine rank_induction (topRank G) (P := fun v => cnt G T X (Sum.inl x₀) v = 0)
    (fun v ih => ?_) v
  by_cases hvX : v ∈ X
  · exact cnt_of_mem hG hvX
  · have h1 : (Sum.inl x₀ : Roots n) ∉ rootScope G v := by
      rw [mem_rootScope]
      rintro (h | ⟨a, b, h, -⟩)
      · exact hvX (by rw [← Sum.inl_injective h]; exact hx₀)
      · exact absurd h (by simp)
    have h2 : ∑ p ∈ par G T (Sum.inl x₀) v, cnt G T X (Sum.inl x₀) p = 0 :=
      Finset.sum_eq_zero (fun p hp => ih p (topRank_lt hG (mem_par.mp hp).1))
    rw [cnt_of_not_mem hG hvX, ite_eq_right h1, h2]
    simp

end Count

/-! ## The coordinate that separates an MIS from `X` -/

section Key

variable {G : ADMG n} {y : Fin n} {X T : Finset (Fin n)}

/-- For every MIS `W ≠ X`, some coordinate is carried to `y` by a positive number of paths
avoiding `W`, and by a different number of paths avoiding `X`. -/
theorem key_coord (hG : G.WF) (hT : DiagMUCT G y X T) (hB : diagBorder G T = X)
    {W : Finset (Fin n)} (hyW : y ∉ W) (hW : AncCrit G y W) (hne : W ≠ X) :
    ∃ l, 0 < cnt G T W l y ∧ cnt G T W l y ≠ cnt G T X l y := by
  have hXT : ∀ x ∈ X, x ∉ T := fun x hx => not_mem_of_mem_border hB hx
  by_cases hWT : ∃ s ∈ W, s ∈ T
  · obtain ⟨s₀, hs₀W, hs₀T⟩ := hWT
    obtain ⟨a, haT, b, hbT, hab, hlive, s, hsW, hbs, hsy⟩ :=
      exists_split hG hT hB hyW hs₀W hs₀T
    refine ⟨Sum.inr (a, b), ?_, ?_⟩
    · have hay : Reach G.dir Tᶜ a y := reachT_of_avoid hT hB (avoid_of_mem hT haT)
      have haW : a ∉ W := fun h => hlive a h ⟨Reach.refl a, hay⟩
      have h1 : 1 ≤ cnt G T W (Sum.inr (a, b)) a :=
        one_le_cnt hG haW (mem_rootScope.mpr (Or.inr ⟨a, b, rfl, hab, Or.inl rfl⟩))
      have h2 : cnt G T W (Sum.inr (a, b)) a ≤ cnt G T W (Sum.inr (a, b)) y :=
        cnt_le_reach hG (reach_avoid_of_live hlive hay (Reach.refl y)) haT
      omega
    · have hle : ∀ v, cnt G T W (Sum.inr (a, b)) v ≤ cnt G T X (Sum.inr (a, b)) v :=
        cnt_le hG (fun w hwX _ =>
          ⟨fun hl => hXT w hwX (mem_T_of_inr_mem haT hbT hl), hXT w hwX⟩)
      have hsT : s ∈ T := mem_of_reachT hbs hbT
      have hbX : b ∉ X := fun h => hXT b h hbT
      have h1 : 1 ≤ cnt G T X (Sum.inr (a, b)) b :=
        one_le_cnt hG hbX (mem_rootScope.mpr (Or.inr ⟨a, b, rfl, hab, Or.inr rfl⟩))
      have h2 : cnt G T X (Sum.inr (a, b)) b ≤ cnt G T X (Sum.inr (a, b)) s :=
        cnt_le_reach hG (Reach.mono_avoid (compl_union_sub hXT) hbs) hbT
      have h3 : cnt G T W (Sum.inr (a, b)) s = 0 := cnt_of_mem hG hsW
      have h4 : cnt G T W (Sum.inr (a, b)) s < cnt G T X (Sum.inr (a, b)) s := by omega
      exact Nat.ne_of_lt (cnt_lt_reach hG hle (fun u hu hX => hXT u hX hu) hsy hsT h4)
  · have hWT' : ∀ w ∈ W, w ∉ T := fun w hw hwT => hWT ⟨w, hw, hwT⟩
    obtain ⟨x₀, hx₀X, hx₀W⟩ := exists_mem_not_mem hT hB hW hWT' hne
    have hx₀B : x₀ ∈ diagBorder G T := by
      rw [hB]
      exact hx₀X
    obtain ⟨-, t, htT, hx₀t⟩ := mem_diagBorder.mp hx₀B
    have h1 : 1 ≤ cnt G T W (Sum.inl x₀) x₀ :=
      one_le_cnt hG hx₀W (mem_rootScope.mpr (Or.inl rfl))
    have h2 : cnt G T W (Sum.inl x₀) x₀ ≤ cnt G T W (Sum.inl x₀) t :=
      cnt_le_child hG hx₀t (fun h => hWT' t h htT) htT (Or.inr rfl)
    have h3 : cnt G T W (Sum.inl x₀) t ≤ cnt G T W (Sum.inl x₀) y :=
      cnt_le_reach hG (Reach.mono_avoid (compl_union_sub hWT')
        (reachT_of_avoid hT hB (avoid_of_mem hT htT))) htT
    refine ⟨Sum.inl x₀, by omega, ?_⟩
    rw [cnt_inl_zero hG hx₀X y]
    omega

end Key

/-! ## The witness model -/

/-- The equation of vertex `v`: the bits of its own roots at their coordinates, plus, for
`v ∈ T`, the coordinates of its parents that it reads (`par`). -/
def cf (G : ADMG n) (T : Finset (Fin n)) (K : ℕ) (v : Fin n) (z : Fin n → Roots n → ZMod K)
    (u : Roots n → Bool) : Roots n → ZMod K := fun l =>
  (if l ∈ rootScope G v ∧ u l = true then (1 : ZMod K) else 0) +
    if v ∈ T then ∑ p ∈ par G T l v, z p l else 0

theorem cf_pa (G : ADMG n) (T : Finset (Fin n)) (K : ℕ) (i : Fin n)
    (x x' : Fin n → Roots n → ZMod K) (u : Roots n → Bool)
    (h : ∀ j, G.dir j i = true → x j = x' j) : cf G T K i x u = cf G T K i x' u := by
  funext l
  have hs : ∑ p ∈ par G T l i, x p l = ∑ p ∈ par G T l i, x' p l :=
    Finset.sum_congr rfl (fun p hp => congrFun (h p (mem_par.mp hp).1) l)
  simp only [cf, hs]

theorem cf_ex (G : ADMG n) (T : Finset (Fin n)) (K : ℕ) (i : Fin n)
    (x : Fin n → Roots n → ZMod K) (u u' : Roots n → Bool)
    (h : ∀ l ∈ rootScope G i, u l = u' l) : cf G T K i x u = cf G T K i x u' := by
  funext l
  by_cases hl : l ∈ rootScope G i
  · simp only [cf, h l hl]
  · simp [cf, hl]

/-- The mechanisms of the witness; its causal diagram is `G`. -/
def mech (G : ADMG n) (hG : G.WF) (T : Finset (Fin n)) (K : ℕ) :
    Mech n (fun _ => Roots n → ZMod K) (Roots n) (fun _ => Bool) :=
  Mech.ofADMG G hG (cf G T K) (cf_pa G T K) (cf_ex G T K)

/-- The witness SCM: every root is a fair coin. -/
noncomputable def wscm (G : ADMG n) (hG : G.WF) (T : Finset (Fin n)) (K : ℕ) :
    SCM n (fun _ => Roots n → ZMod K) (Roots n) (fun _ => Bool) where
  toMech := mech G hG T K
  p _ _ := 1 / 2
  p_nonneg _ _ := by norm_num
  p_sum _ := by norm_num [Fintype.sum_bool]

/-- The witness model. -/
noncomputable def wmodel (G : ADMG n) (hG : G.WF) (T : Finset (Fin n)) (K : ℕ) [NeZero K] :
    Model n where
  X := fun _ => Roots n → ZMod K
  L := Roots n
  U := fun _ => Bool
  scm := wscm G hG T K

theorem wconf {G : ADMG n} (hG : G.WF) (T : Finset (Fin n)) (K : ℕ) [NeZero K] :
    ConfDiag G (wmodel G hG T K) := by
  show (mech G hG T K).diag = G
  exact Mech.ofADMG_diag G hG _ _ _

/-- The arm of `X` used in the witness: every coordinate `0`. -/
def zarm (K : ℕ) : Fin n → Roots n → ZMod K := fun _ _ => 0

theorem resp_out {G : ADMG n} (hG : G.WF) (T : Finset (Fin n)) (K : ℕ) {S : Finset (Fin n)}
    {s : Fin n → Roots n → ZMod K} {u : Roots n → Bool} {v : Fin n} (hv : v ∉ S)
    (l : Roots n) :
    (mech G hG T K).resp S s u v l =
      (if l ∈ rootScope G v ∧ u l = true then (1 : ZMod K) else 0) +
        if v ∈ T then ∑ p ∈ par G T l v, (mech G hG T K).resp S s u p l else 0 := by
  rw [Mech.resp_of_not_mem hv]
  rfl

/-- At a fixed intervention, coordinate `l` of every vertex depends on the coins only through
the coin `l`, which adds the path count `cnt`. -/
theorem resp_coord {G : ADMG n} (hG : G.WF) (T : Finset (Fin n)) (K : ℕ) (S : Finset (Fin n))
    (s : Fin n → Roots n → ZMod K) (u : Roots n → Bool) (l : Roots n) (v : Fin n) :
    (mech G hG T K).resp S s u v l = (mech G hG T K).resp S s (fun _ => false) v l +
      if u l = true then ((cnt G T S l v : ℕ) : ZMod K) else 0 := by
  refine rank_induction (topRank G) (P := fun v => (mech G hG T K).resp S s u v l =
      (mech G hG T K).resp S s (fun _ => false) v l +
        if u l = true then ((cnt G T S l v : ℕ) : ZMod K) else 0) (fun v ih => ?_) v
  by_cases hvS : v ∈ S
  · rw [Mech.resp_of_mem hvS, Mech.resp_of_mem hvS, cnt_of_mem hG hvS]
    simp
  · rw [resp_out hG T K hvS, resp_out hG T K hvS, cnt_of_not_mem hG hvS]
    have hs : ∑ p ∈ par G T l v, (mech G hG T K).resp S s u p l =
        ∑ p ∈ par G T l v, (mech G hG T K).resp S s (fun _ => false) p l +
          ∑ p ∈ par G T l v, (if u l = true then ((cnt G T S l p : ℕ) : ZMod K) else 0) := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl (fun p hp => ih p (topRank_lt hG (mem_par.mp hp).1))
    rw [hs]
    by_cases hl : l ∈ rootScope G v <;> by_cases hvT : v ∈ T <;>
      rcases Bool.eq_false_or_eq_true (u l) with hu | hu <;> simp [hl, hvT, hu]
    ring

/-- The value at `y` agrees, at every coordinate, with a value that `y` takes under
`do(X = 0)`. -/
def Good (G : ADMG n) (hG : G.WF) (T X : Finset (Fin n)) (K : ℕ) (y : Fin n)
    (v : Roots n → ZMod K) : Prop :=
  ∀ l, v l = (mech G hG T K).resp X (zarm K) (fun _ => false) y l ∨
    v l = (mech G hG T K).resp X (zarm K) (fun _ => false) y l + ((cnt G T X l y : ℕ) : ZMod K)

open Classical in
/-- The witness reward: `1` on the values of `Good`, `0` elsewhere. -/
noncomputable def reward (G : ADMG n) (hG : G.WF) (T X : Finset (Fin n)) (K : ℕ) (y : Fin n)
    (v : Roots n → ZMod K) : ℝ :=
  if Good G hG T X K y v then 1 else 0

theorem reward_pair {G : ADMG n} (hG : G.WF) (T X : Finset (Fin n)) (K : ℕ) (y : Fin n)
    {a b : Roots n → ZMod K} (h : ¬ (Good G hG T X K y a ∧ Good G hG T X K y b)) :
    reward G hG T X K y a + reward G hG T X K y b ≤ 1 := by
  have hle : ∀ v, reward G hG T X K y v ≤ 1 := fun v => by
    unfold reward
    split_ifs <;> norm_num
  have hz : ∀ v, ¬ Good G hG T X K y v → reward G hG T X K y v = 0 := fun v hv => by
    unfold reward
    rw [ite_eq_right hv]
  by_cases ha : Good G hG T X K y a
  · have hb : ¬ Good G hG T X K y b := fun hb => h ⟨ha, hb⟩
    rw [hz b hb]
    linarith [hle a]
  · rw [hz a ha]
    linarith [hle b]

/-! ## Averaging over the coins -/

theorem E_eq_one {X : Fin n → Type} {L : Type} {U : L → Type} [Fintype L] [DecidableEq L]
    [∀ l, Fintype (U l)] (M : SCM n X L U) {g : (∀ l, U l) → ℝ} (hg : ∀ u, g u = 1) :
    M.E g = 1 := by
  unfold SCM.E
  calc ∑ u, M.prob u * g u = ∑ u, M.prob u :=
        Finset.sum_congr rfl (fun u _ => by rw [hg u, mul_one])
    _ = 1 := M.sum_prob

/-- With fair coins, a function that is `1` at no pair of values differing in one coin has mean
at most `1/2`. -/
theorem E_le_half {X : Fin n → Type} {L : Type} [Fintype L] [DecidableEq L]
    (M : SCM n X L (fun _ => Bool)) (hp : ∀ l a, M.p l a = 1 / 2) (l₀ : L)
    {g : (L → Bool) → ℝ} (hg : ∀ u, g u + g (Function.update u l₀ (!u l₀)) ≤ 1) :
    M.E g ≤ 1 / 2 := by
  have hinv : Function.Involutive (fun u : L → Bool => Function.update u l₀ (!u l₀)) := by
    intro u
    funext l
    by_cases hl : l = l₀
    · subst hl
      simp
    · simp [Function.update_of_ne hl]
  have hprob : ∀ u : L → Bool, M.prob (Function.update u l₀ (!u l₀)) = M.prob u := by
    intro u
    unfold SCM.prob
    simp only [hp]
  have hswap : ∑ u : L → Bool, M.prob u * g u = ∑ u : L → Bool,
      M.prob (Function.update u l₀ (!u l₀)) * g (Function.update u l₀ (!u l₀)) :=
    (Equiv.sum_comp (Function.Involutive.toPerm _ hinv) (fun u => M.prob u * g u)).symm
  have h2 : 2 * ∑ u : L → Bool, M.prob u * g u =
      ∑ u : L → Bool, M.prob u * (g u + g (Function.update u l₀ (!u l₀))) := by
    calc 2 * ∑ u : L → Bool, M.prob u * g u
        = ∑ u : L → Bool, M.prob u * g u + ∑ u : L → Bool, M.prob u * g u := two_mul _
      _ = ∑ u : L → Bool, M.prob u * g u + ∑ u : L → Bool,
            M.prob (Function.update u l₀ (!u l₀)) * g (Function.update u l₀ (!u l₀)) := by
          rw [← hswap]
      _ = ∑ u : L → Bool, M.prob u * (g u + g (Function.update u l₀ (!u l₀))) := by
          rw [← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl (fun u _ => ?_)
          rw [hprob]
          ring
  have h3 : ∑ u : L → Bool, M.prob u * (g u + g (Function.update u l₀ (!u l₀))) ≤ 1 :=
    calc ∑ u : L → Bool, M.prob u * (g u + g (Function.update u l₀ (!u l₀)))
        ≤ ∑ u : L → Bool, M.prob u * 1 :=
          Finset.sum_le_sum (fun u _ => mul_le_mul_of_nonneg_left (hg u) (M.prob_nonneg u))
      _ = 1 := by
          simp only [mul_one]
          exact M.sum_prob
  have hE : M.E g = ∑ u : L → Bool, M.prob u * g u := rfl
  rw [hE]
  linarith

theorem best_le (m : Model n) {y : Fin n} {r : m.X y → ℝ} {S : Finset (Fin n)} {c : ℝ}
    (h : ∀ x, m.mu y r S x ≤ c) : m.best y r S ≤ c := by
  obtain ⟨x, hx⟩ := m.exists_best y r S
  rw [← hx]
  exact h x

/-- Two values that differ by `d` at one coordinate cannot both be among `B` and `B + m`
modulo `K` when `0 < d`, `d ≠ m` and `d + m < K`. -/
theorem zmod_aux {K : ℕ} {d m : ℕ} (hd : 0 < d) (hdm : d ≠ m) (hK : d + m < K)
    {B C : ZMod K} (h1 : C = B ∨ C = B + m) (h2 : C + d = B ∨ C + d = B + m) : False := by
  have hd0 : (d : ZMod K) ≠ 0 := by
    intro h
    rw [ZMod.natCast_eq_zero_iff] at h
    exact absurd (Nat.le_of_dvd hd h) (by omega)
  have hdm' : (d : ZMod K) ≠ (m : ZMod K) := by
    intro h
    rw [ZMod.natCast_eq_natCast_iff', Nat.mod_eq_of_lt (by omega : d < K),
      Nat.mod_eq_of_lt (by omega : m < K)] at h
    exact hdm h
  have hdm'' : (d : ZMod K) + (m : ZMod K) ≠ 0 := by
    intro h
    rw [← Nat.cast_add, ZMod.natCast_eq_zero_iff] at h
    exact absurd (Nat.le_of_dvd (by omega) h) (by omega)
  rcases h1 with h1 | h1 <;> rcases h2 with h2 | h2
  · exact hd0 (by linear_combination h2 - h1)
  · exact hdm' (by linear_combination h2 - h1)
  · exact hdm'' (by linear_combination h2 - h1)
  · exact hd0 (by linear_combination h2 - h1)

theorem pair_aux {K : ℕ} {d m : ℕ} (hd : 0 < d) (hdm : d ≠ m) (hK : d + m < K)
    {B C a b : ZMod K} (c : Bool) (ha : a = C + if c = true then (d : ZMod K) else 0)
    (hb : b = C + if (!c) = true then (d : ZMod K) else 0)
    (h1 : a = B ∨ a = B + m) (h2 : b = B ∨ b = B + m) : False := by
  cases c
  · simp at ha hb
    rw [ha] at h1
    rw [hb] at h2
    exact zmod_aux hd hdm hK h1 h2
  · simp at ha hb
    rw [ha] at h1
    rw [hb] at h2
    exact zmod_aux hd hdm hK h2 h1

theorem mu_X {G : ADMG n} (hG : G.WF) (T X : Finset (Fin n)) (K : ℕ) [NeZero K] (y : Fin n) :
    (wmodel G hG T K).mu y (reward G hG T X K y) X (zarm K) = 1 := by
  show (wscm G hG T K).E
    (fun u => reward G hG T X K y ((mech G hG T K).resp X (zarm K) u y)) = 1
  refine E_eq_one (wscm G hG T K) (fun u => ?_)
  have hg : Good G hG T X K y ((mech G hG T K).resp X (zarm K) u y) := by
    intro l
    rw [resp_coord hG T K X (zarm K) u l y]
    rcases Bool.eq_false_or_eq_true (u l) with hu | hu
    · right
      simp [hu]
    · left
      simp [hu]
  show reward G hG T X K y ((mech G hG T K).resp X (zarm K) u y) = 1
  unfold reward
  rw [ite_eq_left hg]

theorem mu_le_half {G : ADMG n} (hG : G.WF) (T X : Finset (Fin n)) (K : ℕ) [NeZero K]
    (y : Fin n) {W : Finset (Fin n)} {l : Roots n} (hd : 0 < cnt G T W l y)
    (hdm : cnt G T W l y ≠ cnt G T X l y) (hK : cnt G T W l y + cnt G T X l y < K)
    (w : Fin n → Roots n → ZMod K) :
    (wmodel G hG T K).mu y (reward G hG T X K y) W w ≤ 1 / 2 := by
  show (wscm G hG T K).E
    (fun u => reward G hG T X K y ((mech G hG T K).resp W w u y)) ≤ 1 / 2
  refine E_le_half (wscm G hG T K) (fun _ _ => rfl) l (fun u => ?_)
  show reward G hG T X K y ((mech G hG T K).resp W w u y) +
    reward G hG T X K y ((mech G hG T K).resp W w (Function.update u l (!u l)) y) ≤ 1
  refine reward_pair hG T X K y (fun h => ?_)
  have e1 := resp_coord hG T K W w u l y
  have e2 := resp_coord hG T K W w (Function.update u l (!u l)) l y
  rw [Function.update_self] at e2
  exact pair_aux hd hdm hK (u l) e1 e2 (h.1 l) (h.2 l)

end LB18

/-- **Lee–Bareinboim (2018), Thm. 6, "if"**. -/
theorem pomis_diag {G : ADMG n} (hG : G.WF) {y : Fin n} {X T : Finset (Fin n)}
    (hT : DiagMUCT G y X T) (hB : diagBorder G T = X) : POMIS (ConfDiag G) y X := by
  have hyX : y ∉ X := fun h => LB18.not_mem_of_mem_border hB h hT.1.2.1
  have hcrit : AncCrit G y X := by
    intro x hx
    have hx' : x ∈ diagBorder G T := by
      rw [hB]
      exact hx
    obtain ⟨-, t, ht, hxt⟩ := LB18.mem_diagBorder.mp hx'
    have htH := LB18.mem_diagH.mp (hT.1.1 ht)
    exact Reach.head hxt htH.1 htH.2
  refine ⟨dmis_diag_of_ancCrit hG hyX hcrit, ?_⟩
  obtain ⟨K, hK0, hK⟩ : ∃ K : ℕ, 0 < K ∧ ∀ l, 2 * LB18.cnt G T ∅ l y < K := by
    refine ⟨2 * (∑ l, LB18.cnt G T ∅ l y) + 1, by omega, fun l => ?_⟩
    have h1 : LB18.cnt G T ∅ l y ≤ ∑ l', LB18.cnt G T ∅ l' y :=
      Finset.single_le_sum (f := fun l' => LB18.cnt G T ∅ l' y) (fun i _ => Nat.zero_le _)
        (Finset.mem_univ l)
    omega
  have : NeZero K := ⟨by omega⟩
  refine ⟨LB18.wmodel G hG T K, LB18.reward G hG T X K y, LB18.wconf hG T K,
    fun W hW hne => ?_⟩
  have hyW : y ∉ W := hW.1
  have hWcrit : AncCrit G y W := (mis_diag_iff hG hyW).mp (dmis_mis hW)
  obtain ⟨l, hl0, hlne⟩ := LB18.key_coord hG hT hB hyW hWcrit hne
  have hsum : LB18.cnt G T W l y + LB18.cnt G T X l y < K := by
    have h1 := LB18.cnt_le_empty hG T W l y
    have h2 := LB18.cnt_le_empty hG T X l y
    have h3 := hK l
    omega
  have hup : (LB18.wmodel G hG T K).best y (LB18.reward G hG T X K y) W ≤ 1 / 2 :=
    LB18.best_le _ (fun w => LB18.mu_le_half hG T X K y hl0 hlne hsum w)
  have hlow : 1 ≤ (LB18.wmodel G hG T K).best y (LB18.reward G hG T X K y) X :=
    (LB18.mu_X hG T X K y).symm.le.trans ((LB18.wmodel G hG T K).mu_le_best y _ X _)
  linarith

end Mec.ScbMec
