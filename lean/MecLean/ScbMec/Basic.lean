import MecLean.Common.Model

set_option linter.unusedSectionVars false

/-!
# Minimal and possibly-optimal intervention sets

## Model classes

A *class* is a predicate on bundled finite SCMs (`Model n`).  "Conforming to" a causal diagram
or a MAG is `ConfDiag` or `ConfMAG`.  Domains are arbitrary finite sets; the reward is `r(Y)`
for an arbitrary `r : Ω_Y → ℝ`.

## Definitions

An MIS loses, for every proper subset, some arm in some SCM of the class; a DMIS does so in a
single SCM; a POMIS is a DMIS that is strictly better than every other DMIS in some SCM.  For a
causal diagram MIS and DMIS coincide (`dmis_of_mis_diag`), so `POMIS (ConfDiag G)` is the POMIS of
Lee–Bareinboim (2018), Def. 2.
-/

open Finset

namespace Mec.ScbMec

open Mec

variable {n : ℕ}

/-! ## Classes of SCMs -/

/-- SCMs whose causal diagram is `G`. -/
def ConfDiag (G : ADMG n) (m : Model n) : Prop := m.scm.toMech.diag = G

/-- SCMs whose induced MAG is `M`. -/
def ConfMAG (M : ADMG n) (m : Model n) : Prop := m.scm.toMech.mag = M

/-! ## Minimal and possibly-optimal intervention sets -/

/-- MIS: every proper subset of `S` differs from `S` at some arm of some SCM of the class. -/
def MIS (C : Model n → Prop) (y : Fin n) (S : Finset (Fin n)) : Prop :=
  y ∉ S ∧ ∀ T, T ⊂ S →
    ∃ (m : Model n) (r : m.X y → ℝ) (x : ∀ i, m.X i), C m ∧ m.mu y r T x ≠ m.mu y r S x

/-- DMIS: one SCM of the class separates every proper subset. -/
def DMIS (C : Model n → Prop) (y : Fin n) (S : Finset (Fin n)) : Prop :=
  y ∉ S ∧ ∃ (m : Model n) (r : m.X y → ℝ), C m ∧
    ∀ T, T ⊂ S → ∃ x : ∀ i, m.X i, m.mu y r T x ≠ m.mu y r S x

/-- POMIS: a DMIS strictly better than every other DMIS in some SCM of the class. -/
def POMIS (C : Model n → Prop) (y : Fin n) (S : Finset (Fin n)) : Prop :=
  DMIS C y S ∧ ∃ (m : Model n) (r : m.X y → ℝ), C m ∧
    ∀ W, DMIS C y W → W ≠ S → m.best y r W < m.best y r S

/-! ## Graphical criteria -/

/-- `S ⊆ An(y)_{M_{\overline S}}` (Lee–Bareinboim, Prop. 1). -/
def AncCrit (M : ADMG n) (y : Fin n) (S : Finset (Fin n)) : Prop := ∀ x ∈ S, M.AncAvoid S x y

instance (M : ADMG n) (y : Fin n) (S : Finset (Fin n)) : Decidable (AncCrit M y S) := by
  unfold AncCrit; infer_instance

/-! ## Elementary facts -/

theorem MIS.mono {C C' : Model n → Prop} (hC : ∀ m, C m → C' m) {y : Fin n}
    {S : Finset (Fin n)} (h : MIS C y S) : MIS C' y S := by
  refine ⟨h.1, fun T hT => ?_⟩
  obtain ⟨m, r, x, hm, hne⟩ := h.2 T hT
  exact ⟨m, r, x, hC m hm, hne⟩

theorem DMIS.mono {C C' : Model n → Prop} (hC : ∀ m, C m → C' m) {y : Fin n}
    {S : Finset (Fin n)} (h : DMIS C y S) : DMIS C' y S := by
  obtain ⟨hy, m, r, hm, hsep⟩ := h
  exact ⟨hy, m, r, hC m hm, hsep⟩

/-- A DMIS is an MIS (any class). -/
theorem dmis_mis {C : Model n → Prop} {y : Fin n} {S : Finset (Fin n)} (h : DMIS C y S) :
    MIS C y S := by
  obtain ⟨hy, m, r, hm, hsep⟩ := h
  refine ⟨hy, fun T hT => ?_⟩
  obtain ⟨x, hx⟩ := hsep T hT
  exact ⟨m, r, x, hm, hx⟩

/-- A model conforming to a causal diagram conforms to the diagram's induced MAG. -/
theorem confMAG_of_confDiag {G : ADMG n} {m : Model n} (h : ConfDiag G m) : ConfMAG G.mag m := by
  unfold ConfMAG Mech.mag
  unfold ConfDiag at h
  rw [h]

end Mec.ScbMec
