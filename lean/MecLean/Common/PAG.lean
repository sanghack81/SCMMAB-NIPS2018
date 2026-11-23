import MecLean.Common.MAG

set_option linter.unusedSectionVars false

/-!
# PAGs as summaries of Markov equivalence classes

`IsPAG P M0`: `P` has the skeleton of the MAG `M0`, and a non-circle mark of `P` is a mark shared
by every MAG Markov equivalent to `M0`; every other endpoint is a circle.  This is the semantic
definition of the PAG of `[M0]`; no orientation rule is used, so the definition does not depend
on the completeness of FCI.  A MAG belongs to `[P]` exactly when it is in `MEC M0`.

The transfer lemma `IsPAG.possAnc_of_ancAvoid` says that a directed path of a represented MAG is
possibly directed in `P`.
-/

open Finset

namespace Mec

variable {n : ℕ}

/-- The PAG mark at `j` on the edge between `i` and `j` for the class of `M0`. -/
noncomputable def pagMark (M0 : ADMG n) (i j : Fin n) : Option Mark := by
  classical
  exact if M0.adj i j = false then none
    else if ∀ M, MEC M0 M → M.markAt i j = some .arrow then some .arrow
    else if ∀ M, MEC M0 M → M.markAt i j = some .tail then some .tail
    else some .circle

/-- `P` is the PAG of the Markov equivalence class of the MAG `M0`. -/
def IsPAG (P : PMG n) (M0 : ADMG n) : Prop := M0.IsMAG ∧ ∀ i j, P.m i j = pagMark M0 i j

theorem MEC.refl {M0 : ADMG n} (h : M0.IsMAG) : MEC M0 M0 := ⟨h, fun _ _ _ => Iff.rfl⟩

theorem MEC.symm {M0 M : ADMG n} (h0 : M0.IsMAG) (h : MEC M0 M) : MEC M M0 :=
  ⟨h0, fun S a b => (h.2 S a b).symm⟩

theorem MEC.trans {M0 M M' : ADMG n} (h : MEC M0 M) (h' : MEC M M') : MEC M0 M' :=
  ⟨h'.1, fun S a b => (h'.2 S a b).trans (h.2 S a b)⟩

variable {P : PMG n} {M0 M : ADMG n}

theorem IsPAG.adj (hP : IsPAG P M0) (hM : MEC M0 M) (i j : Fin n) : P.adj i j = M.adj i j := by
  classical
  rw [MEC.adj_eq hP.1 hM]
  unfold PMG.adj
  rw [hP.2 i j]
  unfold pagMark
  cases h : M0.adj i j
  · simp
  · simp only [Bool.true_eq_false, if_false]
    split_ifs <;> rfl

theorem IsPAG.mark (hP : IsPAG P M0) (hM : MEC M0 M) {i j : Fin n} {k : Mark}
    (h : P.m i j = some k) (hk : k ≠ .circle) : M.markAt i j = some k := by
  classical
  rw [hP.2 i j] at h
  unfold pagMark at h
  split_ifs at h with h1 h2 h3
  · cases h
    exact h2 M hM
  · cases h
    exact h3 M hM
  · cases h
    exact absurd rfl hk

/-- A PAG has no edge from a vertex to itself. -/
theorem IsPAG.m_self (hP : IsPAG P M0) (a : Fin n) : P.m a a = none := by
  rw [hP.2 a a]
  unfold pagMark
  simp [hP.1.wf.adj_irrefl a]

/-- No vertex is a possible parent of itself. -/
theorem IsPAG.possDir_self (hP : IsPAG P M0) (a : Fin n) : P.possDir a a = false := by
  simp [PMG.possDir, hP.m_self a]

/-- A directed edge of a represented MAG is a possibly directed edge of the PAG. -/
theorem IsPAG.possDir_of_dir (hP : IsPAG P M0) (hM : MEC M0 M) {i j : Fin n}
    (h : M.dir i j = true) : P.possDir i j = true := by
  obtain ⟨hij, hji⟩ := hM.1.markAt_of_dir h
  have hadj : P.adj i j = true := by
    rw [hP.adj hM]
    simp [ADMG.adj, h]
  unfold PMG.possDir
  rw [decide_eq_true_iff]
  refine ⟨hadj, ?_, ?_⟩
  · intro harr
    have := hP.mark hM harr (by decide)
    rw [hji] at this
    exact absurd this (by decide)
  · intro htail
    have := hP.mark hM htail (by decide)
    rw [hij] at this
    exact absurd this (by decide)

/-- A directed path of a represented MAG is a possibly directed path of the PAG, with the same
vertices. -/
theorem IsPAG.possAnc_of_ancAvoid (hP : IsPAG P M0) (hM : MEC M0 M) {D : Finset (Fin n)}
    {a b : Fin n} (h : M.AncAvoid D a b) : P.PossAnc D a b :=
  Reach.mono_rel (fun _ _ _ hxy => hP.possDir_of_dir hM hxy) h

/-- A PAG all of whose marks are circles, shown by explicit members of the class. -/
theorem isPAG_of_circles (hM0 : M0.IsMAG)
    (hP : ∀ i j, P.m i j = if M0.adj i j = true then some .circle else none)
    (Ms : List (ADMG n)) (hMs : ∀ M ∈ Ms, MEC M0 M)
    (hvar : ∀ i j, M0.adj i j = true →
      (∃ M ∈ Ms, M.markAt i j = some .tail) ∧ (∃ M ∈ Ms, M.markAt i j = some .arrow)) :
    IsPAG P M0 := by
  classical
  refine ⟨hM0, fun i j => ?_⟩
  rw [hP i j]
  unfold pagMark
  cases h : M0.adj i j
  · simp
  · obtain ⟨⟨Mt, hMt, ht⟩, ⟨Ma, hMa, ha⟩⟩ := hvar i j h
    simp only [Bool.true_eq_false, if_false, if_true]
    have hna : ¬ ∀ M, MEC M0 M → M.markAt i j = some .arrow := by
      intro hall
      have := hall Mt (hMs Mt hMt)
      rw [ht] at this
      exact absurd this (by decide)
    have hnt : ¬ ∀ M, MEC M0 M → M.markAt i j = some .tail := by
      intro hall
      have := hall Ma (hMs Ma hMa)
      rw [ha] at this
      exact absurd this (by decide)
    rw [if_neg hna, if_neg hnt]

/-- A PAG checked mark by mark: circles by explicit members of the class with a tail and with
an arrowhead, every other mark by its invariance over the class. -/
theorem isPAG_of_marks (hM0 : M0.IsMAG)
    (hnone : ∀ i j, P.m i j = none ↔ M0.adj i j = false)
    (hcirc : ∀ i j, P.m i j = some .circle →
      (∃ M, MEC M0 M ∧ M.markAt i j = some .tail) ∧ (∃ M, MEC M0 M ∧ M.markAt i j = some .arrow))
    (hfix : ∀ i j k, P.m i j = some k → k ≠ .circle → ∀ M, MEC M0 M → M.markAt i j = some k) :
    IsPAG P M0 := by
  classical
  refine ⟨hM0, fun i j => ?_⟩
  unfold pagMark
  cases hm : P.m i j with
  | none =>
    rw [if_pos ((hnone i j).mp hm)]
  | some k =>
    have hadj : M0.adj i j = true := by
      by_contra h
      simp only [Bool.not_eq_true] at h
      rw [(hnone i j).mpr h] at hm
      exact absurd hm (by simp)
    rw [if_neg (by simp [hadj])]
    cases k with
    | arrow =>
      rw [if_pos (hfix i j .arrow hm (by decide))]
    | tail =>
      have hna : ¬ ∀ M, MEC M0 M → M.markAt i j = some .arrow := by
        intro hall
        have := hall M0 (MEC.refl hM0)
        rw [hfix i j .tail hm (by decide) M0 (MEC.refl hM0)] at this
        exact absurd this (by decide)
      rw [if_neg hna, if_pos (hfix i j .tail hm (by decide))]
    | circle =>
      obtain ⟨⟨Mt, hMt, ht⟩, ⟨Ma, hMa, ha⟩⟩ := hcirc i j hm
      have hna : ¬ ∀ M, MEC M0 M → M.markAt i j = some .arrow := by
        intro hall
        have := hall Mt hMt
        rw [ht] at this
        exact absurd this (by decide)
      have hnt : ¬ ∀ M, MEC M0 M → M.markAt i j = some .tail := by
        intro hall
        have := hall Ma hMa
        rw [ha] at this
        exact absurd this (by decide)
      rw [if_neg hna, if_neg hnt]

end Mec
