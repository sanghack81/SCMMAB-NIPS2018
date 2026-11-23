import MecLean.Scb2018.Thm6

set_option linter.unusedSectionVars false

/-!
# Structural Causal Bandits (NeurIPS 2018): Props. 2, 4, 5, 10 and Cors. 3, 11

* `prop5`, `prop4`: `IB(G_W̄, Y)` is a POMIS for every `W` (Prop. 5), in particular `IB(G, Y)`
  (Prop. 4).  From Thm. 6 and `diagMUCT_border` (the MUCT after intervening on `W` is also the
  MUCT after intervening on its border; the special case `Z = ∅` of Prop. 7).
* `pa_pomis`: `pa(Y)_G` is always a POMIS (text after Prop. 5).
* `prop2`, `cor3`: if `Y` is not confounded with its ancestors (in particular if `G` is
  Markovian), `pa(Y)_G` is the only POMIS.
* `subsumption` (Prop. 10): for a UC-territory `T` of `G` with border `X` and any `Z`,
  `μ_{z} ≤ max μ_{s}` with `S = (T ∩ Z) ∪ X`.  The paper uses do-calculus Rules 2 and 3; the
  proof here uses composition, exclusion and independence of disjoint root blocks.  It holds for
  every UC-territory, not only the minimal one.
* `cor11` (Cor. 11): no POMIS contains an ancestor of the border that is not in the border.
-/

open Finset

namespace Mec.Scb2018

open Mec ScbMec

variable {n : ℕ}

section

variable {G : ADMG n} {y : Fin n} {W T : Finset (Fin n)}

/-- After intervening on its border, the vertices with a directed path to `y` are exactly the
territory. -/
theorem diagH_border (hT : DiagUCT G y W T) : diagH G y (diagBorder G T) = T := by
  ext v
  rw [mem_diagH]
  constructor
  · rintro ⟨hvB, hvy⟩
    exact diag_mem_of_avoid_border hT hvB hvy
  · intro hv
    exact ⟨fun h => (mem_diagBorder.mp h).1 hv, diag_avoid_border_of_mem hT hv⟩

/-- The MUCT after intervening on `W` is also the MUCT after intervening on its border. -/
theorem diagMUCT_border (hT : DiagMUCT G y W T) : DiagMUCT G y (diagBorder G T) T := by
  have hH := diagH_border hT.1
  refine ⟨⟨fun v hv => by rw [hH]; exact hv, hT.1.2.1,
    fun t _ w hw _ => by rw [hH] at hw; exact hw,
    fun t _ w hw _ => by rw [hH] at hw; exact hw⟩, fun T' hT' hne hU => hT.2 T' hT' hne ?_⟩
  rw [Finset.mem_powerset] at hT'
  refine ⟨hT'.trans hT.1.1, hU.2.1, fun t ht w hw hdir => ?_, fun t ht w hw hbi => ?_⟩
  · have hwT : w ∈ T := hT.1.2.2.1 t (hT' ht) w hw hdir
    exact hU.2.2.1 t ht w (by rw [hH]; exact hwT) hdir
  · have hwT : w ∈ T := hT.1.2.2.2 t (hT' ht) w hw hbi
    exact hU.2.2.2 t ht w (by rw [hH]; exact hwT) hbi

/-- **Prop. 5.** `IB(G_W̄, Y)` is a POMIS. -/
theorem prop5 (hG : G.WF) (hT : DiagMUCT G y W T) : POMIS (ConfDiag G) y (diagBorder G T) :=
  thm6_if hG (diagMUCT_border hT) rfl

/-- **Prop. 4.** `IB(G, Y)` is a POMIS. -/
theorem prop4 (hG : G.WF) (hT : DiagMUCT G y ∅ T) : POMIS (ConfDiag G) y (diagBorder G T) :=
  prop5 hG hT

end

/-! ## Parents of `Y` -/

/-- `pa(Y)_G`. -/
def paSet (G : ADMG n) (y : Fin n) : Finset (Fin n) :=
  Finset.univ.filter (fun v => G.dir v y = true)

section

variable {G : ADMG n} {y : Fin n}

theorem diagBorder_singleton (hG : G.WF) : diagBorder G {y} = paSet G y := by
  ext v
  simp only [diagBorder, paSet, Finset.mem_filter, Finset.mem_univ, true_and,
    Finset.mem_singleton, exists_eq_left]
  constructor
  · rintro ⟨-, h⟩
    exact h
  · intro h
    refine ⟨fun e => ?_, h⟩
    rw [e, hG.dir_irrefl] at h
    exact Bool.false_ne_true h

theorem y_not_mem_paSet (hG : G.WF) : y ∉ paSet G y := by
  simp [paSet, hG.dir_irrefl]

/-- After intervening on `pa(Y)`, only `Y` reaches `Y`. -/
theorem diagH_paSet (hG : G.WF) : diagH G y (paSet G y) = {y} := by
  ext v
  rw [mem_diagH, Finset.mem_singleton]
  constructor
  · rintro ⟨hv, h⟩
    rcases Relation.ReflTransGen.cases_tail h with e | ⟨c, hvc, hcy⟩
    · exact e.symm
    · exfalso
      have hcP : c ∈ paSet G y := by simp [paSet, hcy.1]
      by_cases hvc' : v = c
      · exact hv (hvc' ▸ hcP)
      · exact Reach.last_not_mem hvc hvc' hcP
  · intro hv
    rw [hv]
    exact ⟨y_not_mem_paSet hG, Reach.refl _⟩

/-- `{Y}` is the MUCT after intervening on `W` when `Y` is not confounded with its ancestors
in `diagH`. -/
theorem diagMUCT_singleton (hG : G.WF) {W : Finset (Fin n)} (hyW : y ∉ W)
    (hnc : ∀ v ∈ diagH G y W, G.bi y v = false) : DiagMUCT G y W {y} := by
  refine ⟨⟨?_, Finset.mem_singleton_self y, ?_, ?_⟩, ?_⟩
  · intro v hv
    rw [Finset.mem_singleton] at hv
    rw [hv]
    exact mem_diagH.mpr ⟨hyW, Reach.refl _⟩
  · intro t ht w hw hdir
    rw [Finset.mem_singleton] at ht
    rw [ht] at hdir
    exact absurd (G.anc_of_avoid (mem_diagH.mp hw).2) (hG.acyclic hdir)
  · intro t ht w hw hbi
    rw [Finset.mem_singleton] at ht
    rw [ht, hnc w hw] at hbi
    exact absurd hbi (by simp)
  · intro T' hT' hne hU
    rw [Finset.mem_powerset] at hT'
    exact hne (Finset.Subset.antisymm hT' (Finset.singleton_subset_iff.mpr hU.2.1))

/-- **`pa(Y)_G` is always a POMIS** (text after Prop. 5). -/
theorem pa_pomis (hG : G.WF) : POMIS (ConfDiag G) y (paSet G y) := by
  refine thm6_if hG (diagMUCT_singleton hG (y_not_mem_paSet hG) (fun v hv => ?_))
    (diagBorder_singleton hG)
  rw [diagH_paSet hG, Finset.mem_singleton] at hv
  rw [hv]
  exact hG.bi_irrefl y

/-- **Prop. 2.** If `Y` is not confounded with its ancestors, `pa(Y)_G` is the only POMIS. -/
theorem prop2 (hG : G.WF) (hnc : ∀ v, G.Anc v y → G.bi y v = false) {W : Finset (Fin n)} :
    POMIS (ConfDiag G) y W ↔ W = paSet G y := by
  constructor
  · intro hP
    obtain ⟨T, hT, hB⟩ := thm6_onlyif hG hP
    have h1 := diagMUCT_singleton hG hP.1.1
      (fun v hv => hnc v (G.anc_of_avoid (mem_diagH.mp hv).2))
    have hTy : T = {y} := by
      by_contra hne
      exact hT.2 {y} (Finset.mem_powerset.mpr (Finset.singleton_subset_iff.mpr hT.1.2.1))
        (fun h => hne h.symm) h1.1
    rw [← hB, hTy, diagBorder_singleton hG]
  · rintro rfl
    exact pa_pomis hG

/-- **Cor. 3 (Markovian POMIS).** -/
theorem cor3 (hG : G.WF) (hmark : ∀ i j, G.bi i j = false) {W : Finset (Fin n)} :
    POMIS (ConfDiag G) y W ↔ W = paSet G y :=
  prop2 hG (fun v _ => hmark y v)

end

/-! ## Prop. 10 (subsumption) and Cor. 11 -/

section

variable {G : ADMG n} {y : Fin n} {T : Finset (Fin n)}

theorem diag_mem_of_avoid {W S : Finset (Fin n)} (hT : DiagUCT G y W T)
    (hXS : diagBorder G T ⊆ S) {v : Fin n} (hv : v ∉ diagBorder G T) (h : G.AncAvoid S v y) :
    v ∈ T :=
  diag_mem_of_avoid_border hT hv (Reach.mono_avoid hXS h)

/-- **Prop. 10 (Subsumption).**  For a UC-territory `T` of `G` with border `X` and any `Z`,
every arm of `Z` is matched by the best arm of `(T ∩ Z) ∪ X`. -/
theorem subsumption (hT : DiagUCT G y ∅ T) {Z : Finset (Fin n)} (m : Model n)
    (hm : ConfDiag G m) (r : m.X y → ℝ) (z : ∀ i, m.X i) :
    m.mu y r Z z ≤ m.best y r ((T ∩ Z) ∪ diagBorder G T) := by
  have hGd : m.scm.toMech.diag = G := hm
  have hXS : diagBorder G T ⊆ (T ∩ Z) ∪ diagBorder G T := Finset.subset_union_right
  have hSZ : ∀ i ∈ (T ∩ Z) ∪ diagBorder G T, i ∉ Z → i ∈ diagBorder G T := by
    intro i hi hiZ
    rcases Finset.mem_union.mp hi with h | h
    · exact absurd (Finset.mem_inter.mp h).2 hiZ
    · exact h
  have hB : ∀ v, v ∉ Z → ∀ i ∈ diagBorder G T, i ∉ Z → m.scm.toMech.diag.AncAvoid Z v i →
      v ∈ diagH G y ∅ ∧ v ∉ T := by
    intro v _ i hiB _ hvi
    rw [hGd] at hvi
    have hiH := diagBorder_mem_diagH hT hiB (Finset.notMem_empty i)
    have hvi' : G.AncAvoid ∅ v i := Reach.mono_avoid (Finset.empty_subset Z) hvi
    have hvH : v ∈ diagH G y ∅ := mem_diagH.mpr ⟨Finset.notMem_empty v,
      Reach.trans (R := G.dir) (D := ∅) hvi' (mem_diagH.mp hiH).2⟩
    exact ⟨hvH, fun hvT => (mem_diagBorder.mp hiB).1 (diag_desc_closed hT hvT hvi' hiH)⟩
  obtain ⟨D, hD⟩ : ∃ D : Finset m.L, ∀ l, l ∈ D ↔ ∃ v, v ∉ Z ∧
      (∃ i ∈ diagBorder G T, i ∉ Z ∧ m.scm.toMech.diag.AncAvoid Z v i) ∧
        l ∈ m.scm.toMech.ex v :=
    ⟨Finset.univ.filter (fun l => ∃ v, v ∉ Z ∧
      (∃ i ∈ diagBorder G T, i ∉ Z ∧ m.scm.toMech.diag.AncAvoid Z v i) ∧
        l ∈ m.scm.toMech.ex v),
      fun l => by rw [Finset.mem_filter]; simp only [Finset.mem_univ, true_and]⟩
  have hYD : ∀ w, w ∉ (T ∩ Z) ∪ diagBorder G T →
      m.scm.toMech.diag.AncAvoid ((T ∩ Z) ∪ diagBorder G T) w y →
        ∀ l ∈ m.scm.toMech.ex w, l ∉ D := by
    intro w hwS hwy l hlw hlD
    obtain ⟨v, hvZ, ⟨i, hiB, hiZ, hvi⟩, hlv⟩ := (hD l).mp hlD
    rw [hGd] at hwy
    have hwB : w ∉ diagBorder G T := fun h => hwS (hXS h)
    have hwT : w ∈ T := diag_mem_of_avoid hT hXS hwB hwy
    obtain ⟨hvH, hvT⟩ := hB v hvZ i hiB hiZ hvi
    have hwv : w ≠ v := fun e => hvT (e ▸ hwT)
    have hbi : m.scm.toMech.diag.bi w v = true := by
      simp only [Mech.diag, decide_eq_true_eq]
      exact ⟨hwv, l, Finset.mem_inter.mpr ⟨hlw, hlv⟩⟩
    rw [hGd] at hbi
    exact hvT (hT.2.2.2 w hwT v hvH hbi)
  obtain ⟨z', hz⟩ : ∃ z' : (∀ l, m.U l) → ∀ i, m.X i, ∀ u i,
      z' u i = if i ∈ Z then z i else m.scm.toMech.resp Z z u i :=
    ⟨fun u i => if i ∈ Z then z i else m.scm.toMech.resp Z z u i, fun _ _ => rfl⟩
  have hcons : ∀ u, m.scm.toMech.resp Z z u y =
      m.scm.toMech.resp ((T ∩ Z) ∪ diagBorder G T) (z' u) u y := by
    intro u
    have h1 : m.scm.toMech.resp (Z ∪ ((T ∩ Z) ∪ diagBorder G T)) (z' u) u =
        m.scm.toMech.resp Z z u :=
      Mech.resp_extend Finset.subset_union_left z (z' u) u
        (fun i hi => by rw [hz, if_pos hi]) (fun i _ hi => by rw [hz, if_neg hi])
    rw [← congrFun h1 y]
    apply Mech.resp_restrict Finset.subset_union_right (z' u) u
    intro d hd hdS hdy
    have hdZ : d ∈ Z := by
      rcases Finset.mem_union.mp hd with h | h
      · exact h
      · exact absurd h hdS
    rw [hGd] at hdy
    have hdB : d ∉ diagBorder G T := fun h => hdS (hXS h)
    have hdT : d ∈ T := diag_mem_of_avoid hT hXS hdB hdy
    exact hdS (Finset.mem_union_left _ (Finset.mem_inter.mpr ⟨hdT, hdZ⟩))
  have hsplit : ∀ a b : (∀ l, m.U l), r (m.scm.toMech.resp Z z (SCM.merge D a b) y) =
      r (m.scm.toMech.resp ((T ∩ Z) ∪ diagBorder G T) (z' a) b y) := by
    intro a b
    rw [hcons]
    congr 1
    have e1 : m.scm.toMech.resp ((T ∩ Z) ∪ diagBorder G T) (z' (SCM.merge D a b))
        (SCM.merge D a b) =
        m.scm.toMech.resp ((T ∩ Z) ∪ diagBorder G T) (z' a) (SCM.merge D a b) := by
      apply Mech.resp_congr
      intro i hi
      rw [hz, hz]
      by_cases hiZ : i ∈ Z
      · rw [if_pos hiZ, if_pos hiZ]
      · rw [if_neg hiZ, if_neg hiZ]
        have hiB := hSZ i hi hiZ
        apply Mech.resp_roots
        intro v hvZ hvi l hl
        exact SCM.merge_of_mem ((hD l).mpr ⟨v, hvZ, ⟨i, hiB, hiZ, hvi⟩, hl⟩)
    rw [e1]
    apply Mech.resp_roots
    intro v hvS hvy l hl
    exact SCM.merge_of_not_mem (hYD v hvS hvy l hl)
  have hE : m.mu y r Z z = ∑ a : (∀ l, m.U l), m.scm.prob a *
      m.mu y r ((T ∩ Z) ∪ diagBorder G T) (z' a) := by
    unfold Model.mu
    exact m.scm.E_split D _
      (fun a b => r (m.scm.toMech.resp ((T ∩ Z) ∪ diagBorder G T) (z' a) b y)) hsplit
  rw [hE]
  calc ∑ a : (∀ l, m.U l), m.scm.prob a * m.mu y r ((T ∩ Z) ∪ diagBorder G T) (z' a)
      ≤ ∑ a : (∀ l, m.U l), m.scm.prob a * m.best y r ((T ∩ Z) ∪ diagBorder G T) :=
        Finset.sum_le_sum (fun a _ => mul_le_mul_of_nonneg_left
          (m.mu_le_best y r _ (z' a)) (m.scm.prob_nonneg a))
    _ = m.best y r ((T ∩ Z) ∪ diagBorder G T) := by
        rw [← Finset.sum_mul, m.scm.sum_prob, one_mul]

/-- **Cor. 11.**  No POMIS contains an ancestor of the border `X = IB(G, Y)` outside `X`. -/
theorem cor11 (hG : G.WF) (hT : DiagUCT G y ∅ T) {W : Finset (Fin n)}
    (hP : POMIS (ConfDiag G) y W) {w x : Fin n} (hw : w ∈ W) (hx : x ∈ diagBorder G T)
    (hwx : G.Anc w x) : w ∈ diagBorder G T := by
  by_contra hwB
  have hxH : x ∈ diagH G y ∅ := diagBorder_mem_diagH hT hx (Finset.notMem_empty x)
  have hwT : w ∉ T := fun hwT => (mem_diagBorder.mp hx).1 (diag_desc_closed hT hwT hwx hxH)
  have hwS : w ∉ (T ∩ W) ∪ diagBorder G T := by
    rw [Finset.mem_union, Finset.mem_inter]
    rintro (⟨h, -⟩ | h)
    · exact hwT h
    · exact hwB h
  have hyS : y ∉ (T ∩ W) ∪ diagBorder G T := by
    rw [Finset.mem_union, Finset.mem_inter]
    rintro (⟨-, h⟩ | h)
    · exact hP.1.1 h
    · exact (mem_diagBorder.mp h).1 hT.2.1
  have hyS' : y ∉ ((T ∩ W) ∪ diagBorder G T).filter
      (fun v => G.AncAvoid ((T ∩ W) ∪ diagBorder G T) v y) :=
    fun h => hyS (Finset.filter_subset _ _ h)
  have hD := dmis_diag_of_ancCrit hG hyS' (ancCrit_core G y ((T ∩ W) ∪ diagBorder G T))
  have hne : ((T ∩ W) ∪ diagBorder G T).filter
      (fun v => G.AncAvoid ((T ∩ W) ∪ diagBorder G T) v y) ≠ W := by
    intro h
    exact hwS (Finset.filter_subset _ _ (h ▸ hw))
  obtain ⟨-, m, r, hm, hbest⟩ := hP
  have h1 := hbest _ hD hne
  have h2 : m.best y r W ≤ m.best y r ((T ∩ W) ∪ diagBorder G T) := by
    obtain ⟨z, hz⟩ := m.exists_best y r W
    rw [← hz]
    exact subsumption hT m hm r z
  have hGd : m.scm.toMech.diag = G := hm
  have h3 : m.best y r ((T ∩ W) ∪ diagBorder G T) = m.best y r
      (((T ∩ W) ∪ diagBorder G T).filter
        (fun v => G.AncAvoid ((T ∩ W) ∪ diagBorder G T) v y)) := by
    apply m.best_congr
    intro x
    refine m.mu_eq_of_core r (Finset.filter_subset _ _) (fun v hv hpath => ?_) x
    rw [hGd] at hpath
    exact Finset.mem_filter.mpr ⟨hv, hpath⟩
  linarith

end

end Mec.Scb2018
