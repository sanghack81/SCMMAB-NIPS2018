import MecLean.ScbMec.LB2018
import MecLean.Common.Indep

set_option linter.unusedSectionVars false

/-!
# Structural Causal Bandits (Lee–Bareinboim, NeurIPS 2018): Theorem 6

Source: the paper and its unofficial AI-corrected version (see `README.md`).  Thm. 6: given `⟨G, Y⟩`, a set
`X ⊆ V \ {Y}` is a POMIS iff `IB(G_X̄, Y) = X`.

* `mu_le_best_diagBorder`: for any UC-territory `T` of `G` after intervention on `W` and any SCM
  conforming to `G`, every arm of `W` is matched by the best arm of `Pa(T) \ T`.  The paper derives
  this ("only if" part, and Prop. 10) from do-calculus Rules 2 and 3; the proof here uses only
  composition, exclusion and the independence of disjoint root blocks.  In a causal diagram no visibility hypothesis is needed: a root
  shared between `T` and the rest of `An(Y)_{G_W̄} \ W` is a bidirected edge, which the territory
  absorbs.
* `thm6_onlyif`, `thm6_if`, `thm6`: both directions; "if" is `ScbMec.pomis_diag`.
-/

open Finset

namespace Mec.Scb2018

open Mec ScbMec

variable {n : ℕ}

/-! ## Territories of a causal diagram -/

section Territory

variable {G : ADMG n} {y : Fin n} {W T : Finset (Fin n)}

theorem mem_diagH {v : Fin n} : v ∈ diagH G y W ↔ v ∉ W ∧ G.AncAvoid W v y := by
  unfold diagH
  rw [Finset.mem_filter]
  simp only [Finset.mem_univ, true_and]

theorem mem_diagBorder {v : Fin n} :
    v ∈ diagBorder G T ↔ v ∉ T ∧ ∃ t ∈ T, G.dir v t = true := by
  unfold diagBorder
  rw [Finset.mem_filter]
  simp only [Finset.mem_univ, true_and]

/-- A directed path avoiding `W` that ends in `diagH` stays in `diagH`. -/
theorem reach_inside_diagH {a w : Fin n} (h : G.AncAvoid W a w) (hw : w ∈ diagH G y W) :
    Reach G.dir (Finset.univ \ diagH G y W) a w := by
  have hwy := (mem_diagH.mp hw).2
  unfold ADMG.AncAvoid Reach at h
  induction h using Relation.ReflTransGen.head_induction_on with
  | refl => exact Reach.refl _
  | @head a c hac hcw ih =>
    refine Reach.head hac.1 ?_ ih
    have hcH : c ∈ diagH G y W :=
      mem_diagH.mpr ⟨hac.2, Reach.trans (R := G.dir) (D := W) hcw hwy⟩
    simpa using hcH

theorem not_mem_of_mem_diagUCT (hT : DiagUCT G y W T) {t : Fin n} (ht : t ∈ T) : t ∉ W :=
  (mem_diagH.mp (hT.1 ht)).1

/-- A UC-territory is closed under directed paths that avoid `W` and end in `diagH`. -/
theorem diag_desc_closed (hT : DiagUCT G y W T) {t w : Fin n} (ht : t ∈ T)
    (h : G.AncAvoid W t w) (hw : w ∈ diagH G y W) : w ∈ T := by
  have hr := reach_inside_diagH h hw
  clear h hw
  induction hr with
  | refl => exact ht
  | @tail b c _ hbc ih =>
    have hcH : c ∈ diagH G y W := by simpa using hbc.2
    exact hT.2.2.1 b ih c hcH hbc.1

theorem diag_mem_of_avoid_border (hT : DiagUCT G y W T) {v : Fin n}
    (hv : v ∉ diagBorder G T) (h : G.AncAvoid (diagBorder G T) v y) : v ∈ T := by
  unfold ADMG.AncAvoid Reach at h
  revert hv
  induction h using Relation.ReflTransGen.head_induction_on with
  | refl => exact fun _ => hT.2.1
  | @head a c hac _ ih =>
    intro ha
    by_contra haT
    exact ha (mem_diagBorder.mpr ⟨haT, c, ih hac.2, hac.1⟩)

theorem diag_avoid_border_of_mem (hT : DiagUCT G y W T) {t : Fin n} (ht : t ∈ T) :
    G.AncAvoid (diagBorder G T) t y := by
  have key : ∀ a, G.AncAvoid W a y → a ∈ T → G.AncAvoid (diagBorder G T) a y := by
    intro a h
    unfold ADMG.AncAvoid Reach at h
    induction h using Relation.ReflTransGen.head_induction_on with
    | refl => exact fun _ => Reach.refl _
    | @head a c hac hcy ih =>
      intro ha
      have hcH : c ∈ diagH G y W := mem_diagH.mpr ⟨hac.2, hcy⟩
      have hcT := diag_desc_closed hT ha (Reach.single hac.1 hac.2) hcH
      exact Reach.head hac.1 (fun hc => (mem_diagBorder.mp hc).1 hcT) (ih hcT)
  exact key t (mem_diagH.mp (hT.1 ht)).2 ht

theorem diagBorder_mem_diagH (hT : DiagUCT G y W T) {w : Fin n} (hw : w ∈ diagBorder G T)
    (hwW : w ∉ W) : w ∈ diagH G y W := by
  obtain ⟨-, t, ht, hwt⟩ := mem_diagBorder.mp hw
  obtain ⟨htW, hty⟩ := mem_diagH.mp (hT.1 ht)
  exact mem_diagH.mpr ⟨hwW, Reach.head hwt htW hty⟩

theorem ancCrit_diagBorder (hT : DiagUCT G y W T) : AncCrit G y (diagBorder G T) := by
  intro w hw
  obtain ⟨-, t, ht, hwt⟩ := mem_diagBorder.mp hw
  exact Reach.head hwt (fun h => (mem_diagBorder.mp h).1 ht) (diag_avoid_border_of_mem hT ht)

theorem y_not_mem_diagBorder (hT : DiagUCT G y W T) : y ∉ diagBorder G T :=
  fun h => (mem_diagBorder.mp h).1 hT.2.1

/-- The border of a UC-territory is a DMIS (equivalently an MIS) of `G`. -/
theorem dmis_diagBorder (hG : G.WF) (hT : DiagUCT G y W T) :
    DMIS (ConfDiag G) y (diagBorder G T) :=
  dmis_diag_of_ancCrit hG (y_not_mem_diagBorder hT) (ancCrit_diagBorder hT)

/-- `diagH` itself is a UC-territory when it contains `y`. -/
theorem diagUCT_whole (hy : y ∈ diagH G y W) : DiagUCT G y W (diagH G y W) :=
  ⟨subset_rfl, hy, fun _ _ w hw _ => hw, fun _ _ w hw _ => hw⟩

/-- Every UC-territory of a causal diagram contains a minimal one. -/
theorem exists_diagMUCT (h : DiagUCT G y W T) : ∃ T', T' ⊆ T ∧ DiagMUCT G y W T' := by
  obtain ⟨T', hT'mem, hmin⟩ := Finset.exists_min_image (T.powerset.filter (DiagUCT G y W))
    Finset.card ⟨T, Finset.mem_filter.mpr ⟨Finset.mem_powerset_self T, h⟩⟩
  rw [Finset.mem_filter, Finset.mem_powerset] at hT'mem
  refine ⟨T', hT'mem.1, hT'mem.2, fun T'' hT'' hne hU => ?_⟩
  rw [Finset.mem_powerset] at hT''
  have hlt : T''.card < T'.card :=
    Finset.card_lt_card (Finset.ssubset_iff_subset_ne.mpr ⟨hT'', hne⟩)
  have hle := hmin T''
    (Finset.mem_filter.mpr ⟨Finset.mem_powerset.mpr (hT''.trans hT'mem.1), hU⟩)
  omega

end Territory

/-! ## The inequality and the necessity direction of Thm. 6 -/

section Main

variable {G : ADMG n} {y : Fin n} {W T : Finset (Fin n)}

/-- **Necessity inequality** (Thm. 6 "only if" and Prop. 10 of the paper, without do-calculus).
For a UC-territory `T` of `G` after intervention on `W` and an SCM conforming to `G`, each arm of
`W` is matched by the best arm of `Pa(T) \ T`. -/
theorem mu_le_best_diagBorder (hT : DiagUCT G y W T) (m : Model n) (hm : ConfDiag G m)
    (r : m.X y → ℝ) (x : ∀ i, m.X i) : m.mu y r W x ≤ m.best y r (diagBorder G T) := by
  have hGd : m.scm.toMech.diag = G := hm
  have hB : ∀ v, v ∉ W → ∀ i ∈ diagBorder G T, i ∉ W → m.scm.toMech.diag.AncAvoid W v i →
      v ∈ diagH G y W ∧ v ∉ T := by
    intro v hvW i hiB hiW hvi
    rw [hGd] at hvi
    have hiH := diagBorder_mem_diagH hT hiB hiW
    have hvH : v ∈ diagH G y W :=
      mem_diagH.mpr ⟨hvW, Reach.trans (R := G.dir) (D := W) hvi (mem_diagH.mp hiH).2⟩
    exact ⟨hvH, fun hvT => (mem_diagBorder.mp hiB).1 (diag_desc_closed hT hvT hvi hiH)⟩
  obtain ⟨D, hD⟩ : ∃ D : Finset m.L, ∀ l, l ∈ D ↔ ∃ v, v ∉ W ∧
      (∃ i ∈ diagBorder G T, i ∉ W ∧ m.scm.toMech.diag.AncAvoid W v i) ∧
        l ∈ m.scm.toMech.ex v :=
    ⟨Finset.univ.filter (fun l => ∃ v, v ∉ W ∧
      (∃ i ∈ diagBorder G T, i ∉ W ∧ m.scm.toMech.diag.AncAvoid W v i) ∧
        l ∈ m.scm.toMech.ex v),
      fun l => by rw [Finset.mem_filter]; simp only [Finset.mem_univ, true_and]⟩
  have hYD : ∀ w, w ∉ diagBorder G T →
      m.scm.toMech.diag.AncAvoid (diagBorder G T) w y → ∀ l ∈ m.scm.toMech.ex w, l ∉ D := by
    intro w hwB hwy l hlw hlD
    obtain ⟨v, hvW, ⟨i, hiB, hiW, hvi⟩, hlv⟩ := (hD l).mp hlD
    rw [hGd] at hwy
    have hwT : w ∈ T := diag_mem_of_avoid_border hT hwB hwy
    obtain ⟨hvH, hvT⟩ := hB v hvW i hiB hiW hvi
    have hwv : w ≠ v := fun e => hvT (e ▸ hwT)
    have hbi : m.scm.toMech.diag.bi w v = true := by
      simp only [Mech.diag, decide_eq_true_eq]
      exact ⟨hwv, l, Finset.mem_inter.mpr ⟨hlw, hlv⟩⟩
    rw [hGd] at hbi
    exact hvT (hT.2.2.2 w hwT v hvH hbi)
  obtain ⟨z, hz⟩ : ∃ z : (∀ l, m.U l) → ∀ i, m.X i, ∀ u i,
      z u i = if i ∈ W then x i else m.scm.toMech.resp W x u i :=
    ⟨fun u i => if i ∈ W then x i else m.scm.toMech.resp W x u i, fun _ _ => rfl⟩
  have hcons : ∀ u, m.scm.toMech.resp W x u y =
      m.scm.toMech.resp (diagBorder G T) (z u) u y := by
    intro u
    have h1 : m.scm.toMech.resp (W ∪ diagBorder G T) (z u) u = m.scm.toMech.resp W x u :=
      Mech.resp_extend Finset.subset_union_left x (z u) u
        (fun i hi => by rw [hz, if_pos hi]) (fun i _ hi => by rw [hz, if_neg hi])
    rw [← congrFun h1 y]
    apply Mech.resp_restrict Finset.subset_union_right (z u) u
    intro d hd hdB hdy
    have hdW : d ∈ W := (Finset.mem_union.mp hd).resolve_right hdB
    rw [hGd] at hdy
    exact not_mem_of_mem_diagUCT hT (diag_mem_of_avoid_border hT hdB hdy) hdW
  have hsplit : ∀ a b : (∀ l, m.U l), r (m.scm.toMech.resp W x (SCM.merge D a b) y) =
      r (m.scm.toMech.resp (diagBorder G T) (z a) b y) := by
    intro a b
    rw [hcons]
    congr 1
    have e1 : m.scm.toMech.resp (diagBorder G T) (z (SCM.merge D a b)) (SCM.merge D a b) =
        m.scm.toMech.resp (diagBorder G T) (z a) (SCM.merge D a b) := by
      apply Mech.resp_congr
      intro i hi
      rw [hz, hz]
      by_cases hiW : i ∈ W
      · rw [if_pos hiW, if_pos hiW]
      · rw [if_neg hiW, if_neg hiW]
        apply Mech.resp_roots
        intro v hvW hvi l hl
        exact SCM.merge_of_mem ((hD l).mpr ⟨v, hvW, ⟨i, hi, hiW, hvi⟩, hl⟩)
    rw [e1]
    apply Mech.resp_roots
    intro v hvB hvy l hl
    exact SCM.merge_of_not_mem (hYD v hvB hvy l hl)
  have hE : m.mu y r W x = ∑ a : (∀ l, m.U l), m.scm.prob a *
      m.mu y r (diagBorder G T) (z a) := by
    unfold Model.mu
    exact m.scm.E_split D _ (fun a b => r (m.scm.toMech.resp (diagBorder G T) (z a) b y))
      hsplit
  rw [hE]
  calc ∑ a : (∀ l, m.U l), m.scm.prob a * m.mu y r (diagBorder G T) (z a)
      ≤ ∑ a : (∀ l, m.U l), m.scm.prob a * m.best y r (diagBorder G T) :=
        Finset.sum_le_sum (fun a _ => mul_le_mul_of_nonneg_left
          (m.mu_le_best y r _ (z a)) (m.scm.prob_nonneg a))
    _ = m.best y r (diagBorder G T) := by rw [← Finset.sum_mul, m.scm.sum_prob, one_mul]

theorem best_le_best_diagBorder (hT : DiagUCT G y W T) (m : Model n) (hm : ConfDiag G m)
    (r : m.X y → ℝ) : m.best y r W ≤ m.best y r (diagBorder G T) := by
  obtain ⟨x, hx⟩ := m.exists_best y r W
  rw [← hx]
  exact mu_le_best_diagBorder hT m hm r x

/-- A POMIS of a causal diagram is the border of every UC-territory relative to itself. -/
theorem diagBorder_eq_of_pomis (hG : G.WF) (hP : POMIS (ConfDiag G) y W)
    (hT : DiagUCT G y W T) : diagBorder G T = W := by
  by_contra hne
  obtain ⟨-, m, r, hm, hbest⟩ := hP
  have h1 := hbest _ (dmis_diagBorder hG hT) hne
  have h2 := best_le_best_diagBorder hT m hm r
  linarith

/-- **Thm. 6, "only if".** -/
theorem thm6_onlyif (hG : G.WF) (hP : POMIS (ConfDiag G) y W) :
    ∃ T, DiagMUCT G y W T ∧ diagBorder G T = W := by
  have hyH : y ∈ diagH G y W := mem_diagH.mpr ⟨hP.1.1, Reach.refl y⟩
  obtain ⟨T, -, hT⟩ := exists_diagMUCT (diagUCT_whole hyH)
  exact ⟨T, hT, diagBorder_eq_of_pomis hG hP hT.1⟩

/-- **Thm. 6, "if"** (the paper's Prop. 5 with `W = X`). -/
theorem thm6_if (hG : G.WF) (hT : DiagMUCT G y W T) (hB : diagBorder G T = W) :
    POMIS (ConfDiag G) y W :=
  pomis_diag hG hT hB

/-- **Thm. 6.** -/
theorem thm6 (hG : G.WF) :
    POMIS (ConfDiag G) y W ↔ ∃ T, DiagMUCT G y W T ∧ diagBorder G T = W :=
  ⟨thm6_onlyif hG, fun ⟨_, hT, hB⟩ => thm6_if hG hT hB⟩

end Main

end Mec.Scb2018
