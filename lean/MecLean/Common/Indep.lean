import MecLean.Common.Model

set_option linter.unusedSectionVars false

/-!
# Root dependence and independence of disjoint root blocks

* `Mech.resp_roots`: the response of `v` to `do(S = x)` reads only the exogenous roots of the
  non-intervened vertices with a directed path to `v` avoiding `S`.
* `SCM.merge`, `SCM.sum_merge`: averaging over two independent copies of the exogenous roots,
  merged on a set of roots `D` and its complement, is averaging over one copy.  This is how
  independence of disjoint root blocks enters the proofs.
-/

open Finset

namespace Mec

variable {n : ℕ}

namespace Mech

variable {X : Fin n → Type} {L : Type} {U : L → Type} [DecidableEq L] [∀ i, Inhabited (X i)]
  {M : Mech n X L U}

/-- The response of `v` to `do(S = x)` reads only the roots of the vertices outside `S` with a
directed path to `v` whose later vertices avoid `S`. -/
theorem resp_roots {S : Finset (Fin n)} (x : ∀ i, X i) {u u' : ∀ l, U l} {v : Fin n}
    (h : ∀ w, w ∉ S → M.diag.AncAvoid S w v → ∀ l ∈ M.ex w, u l = u' l) :
    M.resp S x u v = M.resp S x u' v := by
  revert h
  refine rank_induction M.rank (P := fun v => (∀ w, w ∉ S → M.diag.AncAvoid S w v →
    ∀ l ∈ M.ex w, u l = u' l) → M.resp S x u v = M.resp S x u' v) (fun v ih h => ?_) v
  by_cases hv : v ∈ S
  · rw [M.resp_of_mem hv, M.resp_of_mem hv]
  · rw [M.resp_of_not_mem hv, M.resp_of_not_mem hv,
      M.f_ex v _ u u' (h v hv (Reach.refl v))]
    apply M.f_pa
    intro j hj
    apply ih j (M.pa_rank hj)
    intro w hw hwj
    exact h w hw (Reach.tail hwj ((M.diag_dir).mpr hj) hv)

end Mech

namespace SCM

variable {X : Fin n → Type} {L : Type} {U : L → Type} [Fintype L] [DecidableEq L]
  [∀ l, Fintype (U l)] (M : SCM n X L U)

/-- The roots in `D` from `a`, the others from `b`. -/
def merge (D : Finset L) (a b : ∀ l, U l) : ∀ l, U l := fun l => if l ∈ D then a l else b l

theorem merge_of_mem {D : Finset L} {a b : ∀ l, U l} {l : L} (h : l ∈ D) :
    merge D a b l = a l := if_pos h

theorem merge_of_not_mem {D : Finset L} {a b : ∀ l, U l} {l : L} (h : l ∉ D) :
    merge D a b l = b l := if_neg h

/-- Exchange the two merged vectors. -/
def mergeSwap (D : Finset L) (ab : (∀ l, U l) × (∀ l, U l)) : (∀ l, U l) × (∀ l, U l) :=
  (merge D ab.1 ab.2, merge D ab.2 ab.1)

theorem mergeSwap_involutive (D : Finset L) : Function.Involutive (mergeSwap (U := U) D) := by
  rintro ⟨a, b⟩
  simp only [mergeSwap, Prod.mk.injEq]
  constructor <;> funext l <;> by_cases hl : l ∈ D <;> simp [merge, hl]

/-- Two independent copies of the roots, merged on `D` and its complement, have the law of one
copy. -/
theorem sum_merge (D : Finset L) (φ : (∀ l, U l) → ℝ) :
    ∑ a : (∀ l, U l), ∑ b : (∀ l, U l), M.prob a * M.prob b * φ (merge D a b) =
      ∑ w : (∀ l, U l), M.prob w * φ w := by
  have hprob : ∀ a b, M.prob (merge D a b) * M.prob (merge D b a) = M.prob a * M.prob b := by
    intro a b
    unfold prob
    rw [← Finset.prod_mul_distrib, ← Finset.prod_mul_distrib]
    refine Finset.prod_congr rfl (fun l _ => ?_)
    by_cases hl : l ∈ D <;> simp [merge, hl, mul_comm]
  have hmerge : ∀ a b : (∀ l, U l), merge D (merge D a b) (merge D b a) = a := by
    intro a b; funext l; by_cases hl : l ∈ D <;> simp [merge, hl]
  calc ∑ a : (∀ l, U l), ∑ b : (∀ l, U l), M.prob a * M.prob b * φ (merge D a b)
      = ∑ ab : (∀ l, U l) × (∀ l, U l), M.prob ab.1 * M.prob ab.2 * φ (merge D ab.1 ab.2) :=
        (Fintype.sum_prod_type' (fun a b => M.prob a * M.prob b * φ (merge D a b))).symm
    _ = ∑ ab : (∀ l, U l) × (∀ l, U l), M.prob (mergeSwap D ab).1 * M.prob (mergeSwap D ab).2 *
          φ (merge D (mergeSwap D ab).1 (mergeSwap D ab).2) :=
        (Equiv.sum_comp (Function.Involutive.toPerm _ (mergeSwap_involutive D))
          (fun ab => M.prob ab.1 * M.prob ab.2 * φ (merge D ab.1 ab.2))).symm
    _ = ∑ ab : (∀ l, U l) × (∀ l, U l), M.prob ab.1 * M.prob ab.2 * φ ab.1 := by
        refine Finset.sum_congr rfl (fun ab _ => ?_)
        simp only [mergeSwap]
        rw [hprob, hmerge]
    _ = ∑ a : (∀ l, U l), ∑ b : (∀ l, U l), M.prob a * M.prob b * φ a :=
        Fintype.sum_prod_type' (fun a b => M.prob a * M.prob b * φ a)
    _ = ∑ w : (∀ l, U l), M.prob w * φ w := by
        refine Finset.sum_congr rfl (fun a _ => ?_)
        rw [← Finset.sum_mul, ← Finset.mul_sum, M.sum_prob, mul_one]

/-- If `g a b` is a function of `a` on the roots in `D` and of `b` on the others, its average
over one copy of the roots is the iterated average over two independent copies. -/
theorem E_split (D : Finset L) (g : (∀ l, U l) → ℝ) (h : (∀ l, U l) → (∀ l, U l) → ℝ)
    (hg : ∀ a b, g (merge D a b) = h a b) :
    M.E g = ∑ a : (∀ l, U l), M.prob a * M.E (h a) := by
  unfold E
  rw [← M.sum_merge D g]
  refine Finset.sum_congr rfl (fun a _ => ?_)
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl (fun b _ => ?_)
  rw [hg a b, mul_assoc]

end SCM

end Mec
