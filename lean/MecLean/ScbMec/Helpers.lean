import MecLean.Common.Model

set_option linter.unusedSectionVars false

/-! # Small facts used by the examples of Fig. 1 and Appendix A -/

open Finset

namespace Mec.ScbMec

open Mec

variable {n : ℕ}

/-- Reward `1` for `true` and `0` for `false`. -/
noncomputable def boolReward (b : Bool) : ℝ := if b then 1 else 0

/-- The expectation of a constant. -/
theorem scmE_const {X : Fin n → Type} {L : Type} {U : L → Type} [Fintype L] [DecidableEq L]
    [∀ l, Fintype (U l)] (M : SCM n X L U) {F : (∀ l, U l) → ℝ} {c : ℝ}
    (hF : ∀ u, F u = c) : M.E F = c := by
  unfold SCM.E
  calc ∑ u, M.prob u * F u = ∑ u, M.prob u * c :=
        Finset.sum_congr rfl (fun u _ => by rw [hF u])
    _ = c := by rw [← Finset.sum_mul, M.sum_prob, one_mul]

/-- The expectation of a function of one root coordinate under the product distribution. -/
theorem scmE_coord {X : Fin n → Type} {L : Type} [Fintype L] [DecidableEq L] {V : Type}
    [Fintype V] (M : SCM n X L (fun _ => V)) (l₀ : L) (g : V → ℝ) {F : (L → V) → ℝ}
    (hF : ∀ u, F u = g (u l₀)) : M.E F = ∑ a, M.p l₀ a * g a := by
  calc M.E F = ∑ u : L → V, ∏ l, (M.p l (u l) * (if l = l₀ then g (u l) else 1)) := by
        unfold SCM.E SCM.prob
        refine Finset.sum_congr rfl (fun u _ => ?_)
        rw [Finset.prod_mul_distrib, Fintype.prod_ite_eq', hF u]
    _ = ∏ l, ∑ a, M.p l a * (if l = l₀ then g a else 1) :=
        (Fintype.prod_sum (fun l a => M.p l a * (if l = l₀ then g a else 1))).symm
    _ = ∑ a, M.p l₀ a * (if l₀ = l₀ then g a else 1) :=
        Fintype.prod_eq_single l₀ (fun l hl => by simp [hl, M.p_sum])
    _ = ∑ a, M.p l₀ a * g a := by simp

/-- An upper bound on every arm bounds the best arm. -/
theorem best_le_of_forall (m : Model n) {y : Fin n} {r : m.X y → ℝ} {S : Finset (Fin n)}
    {c : ℝ} (h : ∀ x, m.mu y r S x ≤ c) : m.best y r S ≤ c := by
  obtain ⟨x, hx⟩ := m.exists_best y r S
  rw [← hx]
  exact h x

end Mec.ScbMec
