import MecLean.Common.PAG

set_option linter.unusedSectionVars false

/-!
# Recursive structural causal models

`Mech n X L U` is the deterministic part of an SCM: observed variables `Fin n` with domains
`X i`, exogenous roots `l : L` with values `U l`.  Vertex `i` lists endogenous arguments `pa i`
and exogenous arguments `ex i`; its equation `f i` may ignore listed arguments (the usual
graphical convention).  A rank function witnesses recursiveness.

* `M.diag`: the causal diagram, `j → i` for `j ∈ pa i` and `i ↔ j` when `i ≠ j` share a root.
* `M.resp S x u`: the potential response to `do(S = x)` at exogenous value `u`; outside `S`
  the values of `x` are ignored.
* `SCM`: a `Mech` with independent finite root distributions; `SCM.prob`, `SCM.P`, `SCM.E`.

The pointwise lemmas below (effectiveness, the agreement lemma, restriction, composition and
parent-fixed responses) hold for exogenous spaces of any type; only `SCM` assumes finiteness.
-/

open Finset

namespace Mec

variable {n : ℕ}

/-- `φ` reads only the coordinates satisfying `D`. -/
def DepOn {X : Fin n → Type*} {α : Sort*} (D : Fin n → Prop) (φ : (∀ j, X j) → α) : Prop :=
  ∀ x x' : (∀ j, X j), (∀ j, D j → x j = x' j) → φ x = φ x'

/-- Strong induction along a rank function. -/
theorem rank_induction (rank : Fin n → ℕ) {P : Fin n → Prop}
    (h : ∀ i, (∀ j, rank j < rank i → P j) → P i) (i : Fin n) : P i := by
  have key : ∀ k, ∀ i, rank i = k → P i := by
    intro k
    induction k using Nat.strong_induction_on with
    | _ k ih => exact fun i hi => h i (fun j hj => ih (rank j) (by omega) j rfl)
  exact key _ i rfl

/-! ## Solutions of recursive systems -/

section Solve

variable {X : Fin n → Type*} [∀ i, Inhabited (X i)]

/-- The solution of a recursive system, by `n` rounds of substitution. -/
def sol (F : ∀ i, (∀ j, X j) → X i) : ∀ i, X i :=
  (fun x i => F i x)^[n] (fun _ => default)

/-- Number of vertices of smaller rank. -/
def height (rank : Fin n → ℕ) (i : Fin n) : ℕ :=
  (Finset.univ.filter (fun j => rank j < rank i)).card

theorem height_lt {rank : Fin n → ℕ} {i j : Fin n} (h : rank j < rank i) :
    height rank j < height rank i := by
  unfold height
  apply Finset.card_lt_card
  refine (Finset.ssubset_iff_of_subset ?_).mpr ⟨j, by simp [h], by simp⟩
  intro k hk
  simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk ⊢
  exact lt_trans hk h

theorem height_lt_n (rank : Fin n → ℕ) (i : Fin n) : height rank i < n := by
  unfold height
  have hss : (Finset.univ.filter (fun j => rank j < rank i)) ⊂ Finset.univ :=
    (Finset.ssubset_iff_of_subset (Finset.filter_subset _ _)).mpr
      ⟨i, Finset.mem_univ _, by simp⟩
  have := Finset.card_lt_card hss
  simpa using this

theorem sol_spec {F : ∀ i, (∀ j, X j) → X i} (rank : Fin n → ℕ)
    (hF : ∀ i, DepOn (fun j => rank j < rank i) (F i)) (i : Fin n) :
    sol F i = F i (sol F) := by
  have key : ∀ k : ℕ, ∀ i : Fin n, height rank i < k →
      ((fun x i => F i x)^[k] (fun _ => default)) i =
        ((fun x i => F i x)^[k + 1] (fun _ => default)) i := by
    intro k
    induction k with
    | zero => intro i hi; exact absurd hi (Nat.not_lt_zero _)
    | succ k ih =>
      intro i hi
      rw [Function.iterate_succ_apply', Function.iterate_succ_apply' (n := k + 1)]
      apply hF i
      intro j hj
      exact ih j (lt_of_lt_of_le (height_lt hj) (Nat.lt_succ_iff.mp hi))
  have h := key n i (height_lt_n rank i)
  rw [Function.iterate_succ_apply'] at h
  exact h

/-- Uniqueness: any solution of a recursive system is `sol`. -/
theorem eq_sol {F : ∀ i, (∀ j, X j) → X i} (rank : Fin n → ℕ)
    (hF : ∀ i, DepOn (fun j => rank j < rank i) (F i)) {z : ∀ j, X j}
    (hz : ∀ i, z i = F i z) : z = sol F := by
  funext i
  refine rank_induction rank (P := fun i => z i = sol F i) (fun i ih => ?_) i
  rw [hz i, sol_spec rank hF i]
  exact hF i _ _ (fun j hj => ih j hj)

end Solve

/-! ## Mechanisms -/

/-- The deterministic part of a recursive SCM. -/
structure Mech (n : ℕ) (X : Fin n → Type) (L : Type) (U : L → Type) where
  pa : Fin n → Finset (Fin n)
  ex : Fin n → Finset L
  rank : Fin n → ℕ
  pa_rank : ∀ {i j : Fin n}, j ∈ pa i → rank j < rank i
  f : ∀ i, (∀ j, X j) → (∀ l, U l) → X i
  f_pa : ∀ i (x x' : ∀ j, X j) (u : ∀ l, U l), (∀ j ∈ pa i, x j = x' j) → f i x u = f i x' u
  f_ex : ∀ i (x : ∀ j, X j) (u u' : ∀ l, U l), (∀ l ∈ ex i, u l = u' l) → f i x u = f i x u'

namespace Mech

variable {X : Fin n → Type} {L : Type} {U : L → Type} (M : Mech n X L U)

/-- The causal diagram: listed endogenous arguments and shared exogenous roots. -/
def diag [DecidableEq L] : ADMG n where
  dir j i := decide (j ∈ M.pa i)
  bi i j := decide (i ≠ j ∧ (M.ex i ∩ M.ex j).Nonempty)

variable [DecidableEq L]

theorem diag_dir {i j : Fin n} : M.diag.dir j i = true ↔ j ∈ M.pa i := by simp [diag]

theorem rank_le_of_ancAvoid {D : Finset (Fin n)} {a b : Fin n} (h : M.diag.AncAvoid D a b) :
    M.rank a ≤ M.rank b := by
  induction h with
  | refl => exact le_rfl
  | tail _ hbc ih => exact le_trans ih (le_of_lt (M.pa_rank ((M.diag_dir).mp hbc.1)))

/-- A nontrivial directed path of the causal diagram strictly increases the rank. -/
theorem rank_lt_of_ancAvoid {D : Finset (Fin n)} {a b : Fin n}
    (h : M.diag.AncAvoid D a b) (hab : a ≠ b) : M.rank a < M.rank b := by
  rcases Relation.ReflTransGen.cases_tail h with h' | ⟨c, hac, hcb⟩
  · exact absurd h'.symm hab
  · exact lt_of_le_of_lt (M.rank_le_of_ancAvoid (D := D) hac)
      (M.pa_rank ((M.diag_dir).mp hcb.1))

theorem diag_wf : M.diag.WF := by
  refine ⟨fun i => ?_, fun i => by simp [diag], fun i j => ?_, fun i j hij hji => ?_⟩
  · simp only [diag, decide_eq_false_iff_not]
    exact fun h => lt_irrefl _ (M.pa_rank h)
  · simp only [diag]
    rw [Finset.inter_comm]
    simp only [ne_comm]
  · have h1 := M.pa_rank ((M.diag_dir).mp hij)
    have h2 := M.rank_le_of_ancAvoid hji
    omega

/-- The induced MAG of the SCM. -/
def mag : ADMG n := M.diag.mag

/-! ### Interventions and potential responses -/

variable [∀ i, Inhabited (X i)]

/-- The system after `do(S = x)` at exogenous value `u`. -/
def sys (S : Finset (Fin n)) (x : ∀ i, X i) (u : ∀ l, U l) : ∀ i, (∀ j, X j) → X i :=
  fun i z => if i ∈ S then x i else M.f i z u

theorem sys_dep (S : Finset (Fin n)) (x : ∀ i, X i) (u : ∀ l, U l) (i : Fin n) :
    DepOn (fun j => M.rank j < M.rank i) (M.sys S x u i) := by
  intro z z' h
  unfold sys
  split_ifs
  · rfl
  · exact M.f_pa i z z' u (fun j hj => h j (M.pa_rank hj))

/-- The potential response to `do(S = x)` at `u`. -/
def resp (S : Finset (Fin n)) (x : ∀ i, X i) (u : ∀ l, U l) : ∀ i, X i := sol (M.sys S x u)

theorem resp_spec (S : Finset (Fin n)) (x : ∀ i, X i) (u : ∀ l, U l) (i : Fin n) :
    M.resp S x u i = if i ∈ S then x i else M.f i (M.resp S x u) u :=
  sol_spec M.rank (M.sys_dep S x u) i

theorem eq_resp (S : Finset (Fin n)) (x : ∀ i, X i) (u : ∀ l, U l) {z : ∀ i, X i}
    (hz : ∀ i, z i = if i ∈ S then x i else M.f i z u) : z = M.resp S x u :=
  eq_sol M.rank (M.sys_dep S x u) hz

variable {M}

/-- Effectiveness. -/
theorem resp_of_mem {S : Finset (Fin n)} {x : ∀ i, X i} {u : ∀ l, U l} {i : Fin n}
    (hi : i ∈ S) : M.resp S x u i = x i := by
  rw [M.resp_spec, if_pos hi]

theorem resp_of_not_mem {S : Finset (Fin n)} {x : ∀ i, X i} {u : ∀ l, U l} {i : Fin n}
    (hi : i ∉ S) : M.resp S x u i = M.f i (M.resp S x u) u := by
  rw [M.resp_spec, if_neg hi]

/-- Values outside the intervened set are irrelevant. -/
theorem resp_congr {S : Finset (Fin n)} {x x' : ∀ i, X i} (u : ∀ l, U l)
    (h : ∀ i ∈ S, x i = x' i) : M.resp S x u = M.resp S x' u := by
  apply M.eq_resp
  intro i
  by_cases hi : i ∈ S
  · rw [if_pos hi, M.resp_of_mem hi, h i hi]
  · rw [if_neg hi, M.resp_of_not_mem hi]

/-- Agreement lemma.  Let `Δ` contain every vertex whose mechanism differs between `do(S₁ = x₁)`
and `do(S₂ = x₂)`.  Then the two responses agree at every vertex not reached from `Δ` by a
directed path whose later vertices avoid the vertices fixed in both. -/
theorem resp_agree {S₁ S₂ : Finset (Fin n)} {x₁ x₂ : ∀ i, X i} (u : ∀ l, U l)
    (Δ : Fin n → Prop)
    (hΔ : ∀ i, ¬ Δ i → (i ∈ S₁ ↔ i ∈ S₂) ∧ (i ∈ S₁ → x₁ i = x₂ i))
    {v : Fin n} (hv : ∀ d, Δ d → ¬ M.diag.AncAvoid (S₁ ∩ S₂) d v) :
    M.resp S₁ x₁ u v = M.resp S₂ x₂ u v := by
  revert hv
  refine rank_induction M.rank (P := fun v => (∀ d, Δ d → ¬ M.diag.AncAvoid (S₁ ∩ S₂) d v) →
    M.resp S₁ x₁ u v = M.resp S₂ x₂ u v) (fun w ih hw => ?_) v
  have hnw : ¬ Δ w := fun h => hw w h (Reach.refl w)
  obtain ⟨hmem, hval⟩ := hΔ w hnw
  by_cases h1 : w ∈ S₁
  · rw [M.resp_of_mem h1, M.resp_of_mem (hmem.mp h1)]
    exact hval h1
  · have h2 : w ∉ S₂ := fun h => h1 (hmem.mpr h)
    rw [M.resp_of_not_mem h1, M.resp_of_not_mem h2]
    apply M.f_pa
    intro j hj
    apply ih j (M.pa_rank hj)
    intro d hd hreach
    exact hw d hd (Reach.tail hreach ((M.diag_dir).mpr hj) (by simp [h1]))

/-- Restriction: interventions in `S \ T` that reach `y` only through `T` can be dropped. -/
theorem resp_restrict {S T : Finset (Fin n)} (hTS : T ⊆ S) (x : ∀ i, X i) (u : ∀ l, U l)
    {y : Fin n} (h : ∀ d ∈ S, d ∉ T → ¬ M.diag.AncAvoid T d y) :
    M.resp S x u y = M.resp T x u y := by
  apply M.resp_agree u (fun d => d ∈ S ∧ d ∉ T)
  · intro i hi
    have hi' : i ∈ S → i ∈ T := fun h => by_contra fun h' => hi ⟨h, h'⟩
    exact ⟨⟨hi', fun h => hTS h⟩, fun _ => rfl⟩
  · intro d hd
    rw [Finset.inter_eq_right.mpr hTS]
    exact h d hd.1 hd.2

/-- Composition: additionally fixing variables at their current values changes nothing. -/
theorem resp_extend {S T : Finset (Fin n)} (hST : S ⊆ T) (x x' : ∀ i, X i) (u : ∀ l, U l)
    (hS : ∀ i ∈ S, x' i = x i) (hT : ∀ i ∈ T, i ∉ S → x' i = M.resp S x u i) :
    M.resp T x' u = M.resp S x u := by
  symm
  apply M.eq_resp
  intro i
  by_cases hiT : i ∈ T
  · rw [if_pos hiT]
    by_cases hiS : i ∈ S
    · rw [M.resp_of_mem hiS, hS i hiS]
    · rw [hT i hiT hiS]
  · rw [if_neg hiT, M.resp_of_not_mem (fun h => hiT (hST h))]

/-- A response whose parents are all intervened evaluates its equation at the intervention. -/
theorem resp_of_pa_subset {S : Finset (Fin n)} {x : ∀ i, X i} {u : ∀ l, U l} {i : Fin n}
    (hi : i ∉ S) (hpa : M.pa i ⊆ S) : M.resp S x u i = M.f i x u := by
  rw [M.resp_of_not_mem hi]
  apply M.f_pa
  intro j hj
  exact M.resp_of_mem (hpa hj)

end Mech

/-! ## Probabilities -/

/-- An SCM: mechanisms with mutually independent finite exogenous roots. -/
structure SCM (n : ℕ) (X : Fin n → Type) (L : Type) (U : L → Type) [Fintype L]
    [∀ l, Fintype (U l)] extends Mech n X L U where
  p : ∀ l, U l → ℝ
  p_nonneg : ∀ l a, 0 ≤ p l a
  p_sum : ∀ l, ∑ a, p l a = 1

namespace SCM

variable {X : Fin n → Type} {L : Type} {U : L → Type} [Fintype L] [DecidableEq L]
  [∀ l, Fintype (U l)] (M : SCM n X L U)

/-- Probability of an exogenous value. -/
noncomputable def prob (u : ∀ l, U l) : ℝ := ∏ l, M.p l (u l)

/-- Probability of an exogenous event. -/
noncomputable def P (E : (∀ l, U l) → Prop) [DecidablePred E] : ℝ :=
  ∑ u, if E u then M.prob u else 0

/-- Expectation of an exogenous function. -/
noncomputable def E (g : (∀ l, U l) → ℝ) : ℝ := ∑ u, M.prob u * g u

theorem prob_nonneg (u : ∀ l, U l) : 0 ≤ M.prob u :=
  Finset.prod_nonneg (fun l _ => M.p_nonneg l (u l))

theorem sum_prob : ∑ u : (∀ l, U l), M.prob u = 1 := by
  unfold prob
  rw [← Fintype.prod_sum (fun l a => M.p l a)]
  simp [M.p_sum]

/-- With a one-point exogenous space the expectation is the value at that point. -/
theorem E_of_subsingleton [hsub : Subsingleton (∀ l, U l)] (g : (∀ l, U l) → ℝ)
    (u₀ : ∀ l, U l) : M.E g = g u₀ := by
  have h1 : ∑ u : (∀ l, U l), M.prob u = M.prob u₀ := Fintype.sum_subsingleton _ u₀
  rw [M.sum_prob] at h1
  unfold E
  rw [Fintype.sum_subsingleton _ u₀, ← h1, one_mul]

theorem P_congr {E₁ E₂ : (∀ l, U l) → Prop} [DecidablePred E₁] [DecidablePred E₂]
    (h : ∀ u, E₁ u ↔ E₂ u) : M.P E₁ = M.P E₂ := by
  unfold P
  refine Finset.sum_congr rfl (fun u _ => ?_)
  by_cases h1 : E₁ u
  · rw [if_pos h1, if_pos ((h u).mp h1)]
  · rw [if_neg h1, if_neg (fun h2 => h1 ((h u).mpr h2))]

theorem E_congr {g₁ g₂ : (∀ l, U l) → ℝ} (h : ∀ u, g₁ u = g₂ u) : M.E g₁ = M.E g₂ := by
  unfold E
  exact Finset.sum_congr rfl (fun u _ => by rw [h u])

theorem P_nonneg (E : (∀ l, U l) → Prop) [DecidablePred E] : 0 ≤ M.P E := by
  unfold P
  exact Finset.sum_nonneg (fun u _ => by split_ifs <;> simp [M.prob_nonneg u])

theorem P_mono {E₁ E₂ : (∀ l, U l) → Prop} [DecidablePred E₁] [DecidablePred E₂]
    (h : ∀ u, E₁ u → E₂ u) : M.P E₁ ≤ M.P E₂ := by
  unfold P
  refine Finset.sum_le_sum (fun u _ => ?_)
  by_cases h1 : E₁ u
  · rw [if_pos h1, if_pos (h u h1)]
  · rw [if_neg h1]
    split_ifs <;> simp [M.prob_nonneg u]

end SCM

/-! ## Compatibility and the shared transfer lemma -/

namespace Mech

variable {X : Fin n → Type} {L : Type} {U : L → Type} [DecidableEq L] (M : Mech n X L U)

/-- The SCM is compatible with the PAG `P` of the class of `M0`: its induced MAG is in the
class. -/
def CompatPAG (M0 : ADMG n) : Prop := MEC M0 M.mag

variable {M}

/-- **Shared transfer lemma.**  In an SCM compatible with a PAG, every directed path of the
causal diagram is a possibly directed path of the PAG through the same vertices. -/
theorem possAnc_of_ancAvoid {P : PMG n} {M0 : ADMG n} (hP : IsPAG P M0) (hc : M.CompatPAG M0)
    {D : Finset (Fin n)} {a b : Fin n} (h : M.diag.AncAvoid D a b) : P.PossAnc D a b :=
  hP.possAnc_of_ancAvoid hc (ADMG.mag_ancAvoid M.diag_wf h)

/-- In an SCM compatible with a PAG, every listed endogenous argument is a possible parent. -/
theorem possDir_of_mem_pa {P : PMG n} {M0 : ADMG n} (hP : IsPAG P M0) (hc : M.CompatPAG M0)
    {i j : Fin n} (h : j ∈ M.pa i) : P.possDir j i = true :=
  hP.possDir_of_dir hc (ADMG.mag_dir_of_dir M.diag_wf ((M.diag_dir).mpr h))

end Mech

/-- The MIS core for mechanisms: the interventions of `S` outside `T ⊆ S` can be dropped when
every vertex of `S` with a diagram path to `y` avoiding `S` lies in `T`.  (On a path from
`S \ T` to `y` avoiding `T`, the last vertex of `S \ T` reaches `y` avoiding `S`.) -/
theorem Mech.resp_core {X : Fin n → Type} {L : Type} {U : L → Type} [DecidableEq L]
    [∀ i, Inhabited (X i)] {M : Mech n X L U} {S T : Finset (Fin n)} {y : Fin n}
    (hTS : T ⊆ S) (hT : ∀ z ∈ S, M.diag.AncAvoid S z y → z ∈ T) (x : ∀ i, X i)
    (u : ∀ l, U l) : M.resp S x u y = M.resp T x u y := by
  apply Mech.resp_restrict hTS x u
  intro d hdS hdT hpath
  obtain ⟨c, hc, hcT, hcpath⟩ :=
    Reach.last_mem (S := S \ T) (Finset.mem_sdiff.mpr ⟨hdS, hdT⟩) hdT hpath
  rw [Finset.union_sdiff_of_subset hTS] at hcpath
  exact hcT (hT c (Finset.mem_sdiff.mp hc).1 hcpath)


end Mec
