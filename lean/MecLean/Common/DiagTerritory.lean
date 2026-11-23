import MecLean.Common.Graph

set_option linter.unusedSectionVars false

/-!
# Territories of causal diagrams (Lee–Bareinboim 2018)

The minimal UC-territory (MUCT) and interventional border (IB) of a causal diagram `G` after
intervention on `X`.  A causal diagram may carry both `a → b` and `a ↔ b`, so these are stated
on the ADMG directly.
-/

open Finset

namespace Mec

variable {n : ℕ}

/-- `An(Y)_{𝒢_{\overline X}} \ X`: the vertices outside `X` with a directed path to `y` whose
later vertices avoid `X`. -/
def diagH (G : ADMG n) (y : Fin n) (X : Finset (Fin n)) : Finset (Fin n) :=
  Finset.univ.filter (fun v => v ∉ X ∧ G.AncAvoid X v y)

/-- A UC-territory of the causal diagram `G` after intervention on `X` (Lee–Bareinboim 2018), as
used in App. l. 76 and l. 134: a set containing `y`, inside `diagH G y X`, closed under directed
and bidirected edges inside `diagH G y X`.  Closure under single edges inside `diagH` is closure
under directed and bidirected paths of `G[diagH]`.  Intervened vertices never enter the
territory (App. l. 134), so edges into `X` play no role. -/
def DiagUCT (G : ADMG n) (y : Fin n) (X T : Finset (Fin n)) : Prop :=
  T ⊆ diagH G y X ∧ y ∈ T ∧
    (∀ t ∈ T, ∀ w ∈ diagH G y X, G.dir t w = true → w ∈ T) ∧
    (∀ t ∈ T, ∀ w ∈ diagH G y X, G.bi t w = true → w ∈ T)

instance (G : ADMG n) (y : Fin n) (X T : Finset (Fin n)) : Decidable (DiagUCT G y X T) := by
  unfold DiagUCT; infer_instance

/-- A minimal UC-territory of the causal diagram `G` after intervention on `X`. -/
def DiagMUCT (G : ADMG n) (y : Fin n) (X T : Finset (Fin n)) : Prop :=
  DiagUCT G y X T ∧ ∀ T' ∈ T.powerset, T' ≠ T → ¬ DiagUCT G y X T'

instance (G : ADMG n) (y : Fin n) (X T : Finset (Fin n)) : Decidable (DiagMUCT G y X T) := by
  unfold DiagMUCT; infer_instance

/-- `Pa(T)_𝒢 \ T`.  For `T` disjoint from `X` it is also `Pa(T)_{𝒢_{\overline X}} \ T`. -/
def diagBorder (G : ADMG n) (T : Finset (Fin n)) : Finset (Fin n) :=
  Finset.univ.filter (fun v => v ∉ T ∧ ∃ t ∈ T, G.dir v t = true)

end Mec
