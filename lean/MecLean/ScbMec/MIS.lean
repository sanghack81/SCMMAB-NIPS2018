import MecLean.ScbMec.Basic

set_option linter.unusedSectionVars false

/-!
# MIS and DMIS in causal diagrams

Lee–Bareinboim (2018), Prop. 1, through a single Boolean witness model that separates every
proper subset of a set satisfying the ancestral criterion.
-/

open Finset


namespace Mec.ScbMec

open Mec

variable {n : ℕ}

/-! ## Pointwise exclusion and the MIS core -/

/-- A directed path from `z` avoiding `S.erase z` after `z` avoids `S`: it cannot return to
`z`. -/
theorem ancAvoid_of_erase {G : ADMG n} (hG : G.WF) {S : Finset (Fin n)} {z y : Fin n}
    (h : G.AncAvoid (S.erase z) z y) : G.AncAvoid S z y := by
  unfold ADMG.AncAvoid at h ⊢
  induction h with
  | refl => exact Reach.refl z
  | @tail b c _ hbc ih =>
    refine Reach.tail ih hbc.1 (fun hcS => ?_)
    by_cases hcz : c = z
    · rw [hcz] at hbc
      exact hG.acyclic hbc.1 (G.anc_of_avoid ih)
    · exact hbc.2 (Finset.mem_erase.mpr ⟨hcz, hcS⟩)

/-- The MIS core: removing from `S` the vertices with no directed path to `y` in
`M_{\overline S}`, for the induced MAG `M` of the model, changes no arm's mean. -/
theorem mu_core (m : Model n) {y : Fin n} (r : m.X y → ℝ) (S : Finset (Fin n))
    (x : ∀ i, m.X i) :
    m.mu y r S x = m.mu y r (S.filter (fun z => m.scm.toMech.mag.AncAvoid S z y)) x := by
  refine m.mu_eq_of_core r (Finset.filter_subset _ _) (fun z hz hpath => ?_) x
  exact Finset.mem_filter.mpr ⟨hz, ADMG.mag_ancAvoid m.scm.toMech.diag_wf hpath⟩

/-- The MIS core satisfies the ancestral criterion in the induced MAG. -/
theorem ancCrit_core (M : ADMG n) (y : Fin n) (S : Finset (Fin n)) :
    AncCrit M y (S.filter (fun z => M.AncAvoid S z y)) := by
  intro z hz
  exact Reach.mono_avoid (Finset.filter_subset _ _) (Finset.mem_filter.mp hz).2

/-- Every element of an MIS has, in some model of the class, a diagram path to `y` avoiding
`S`; otherwise dropping it changes no arm's mean in any model of the class. -/
theorem MIS.exists_ancAvoid {C : Model n → Prop} {y : Fin n} {S : Finset (Fin n)}
    (h : MIS C y S) {z : Fin n} (hz : z ∈ S) :
    ∃ m : Model n, C m ∧ m.scm.toMech.diag.AncAvoid S z y := by
  obtain ⟨m, r, x, hm, hne⟩ := h.2 (S.erase z) (Finset.erase_ssubset hz)
  by_contra hcon
  apply hne
  refine (m.mu_eq_of_core r (Finset.erase_subset z S) (fun w hw hpath => ?_) x).symm
  refine Finset.mem_erase.mpr ⟨fun hwz => ?_, hw⟩
  rw [hwz] at hpath
  exact hcon ⟨m, hm, hpath⟩

/-! ## The Boolean witness -/

/-- The Boolean OR mechanism on `G`: a vertex is `true` iff some parent is. -/
def orF (G : ADMG n) (i : Fin n) (z : Fin n → Bool) (_u : Roots n → Unit) : Bool :=
  decide (∃ j, G.dir j i = true ∧ z j = true)

theorem orF_pa (G : ADMG n) (i : Fin n) (z z' : Fin n → Bool) (u : Roots n → Unit)
    (h : ∀ j, G.dir j i = true → z j = z' j) : orF G i z u = orF G i z' u := by
  unfold orF
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_iff]
  exact exists_congr (fun j => and_congr_right (fun hj => by rw [h j hj]))

/-- The deterministic Boolean OR mechanism whose causal diagram is `G`. -/
def orMech (G : ADMG n) (hG : G.WF) : Mech n (fun _ => Bool) (Roots n) (fun _ => Unit) :=
  Mech.ofADMG G hG (orF G) (orF_pa G) (fun _ _ _ _ _ => rfl)

theorem orMech_diag (G : ADMG n) (hG : G.WF) : (orMech G hG).diag = G := by
  unfold orMech
  exact Mech.ofADMG_diag G hG _ _ _

theorem orMech_f (G : ADMG n) (hG : G.WF) (i : Fin n) (z : Fin n → Bool)
    (u : Roots n → Unit) :
    (orMech G hG).f i z u = decide (∃ j, G.dir j i = true ∧ z j = true) := rfl

/-- If an intervention sets every vertex of `T` to `false`, every vertex is `false`. -/
theorem orMech_resp_false (G : ADMG n) (hG : G.WF) {T : Finset (Fin n)} {x : Fin n → Bool}
    (hx : ∀ i ∈ T, x i = false) (u : Roots n → Unit) (v : Fin n) :
    (orMech G hG).resp T x u v = false := by
  have h : (fun _ : Fin n => false) = (orMech G hG).resp T x u := by
    apply Mech.eq_resp
    intro i
    by_cases hi : i ∈ T
    · rw [if_pos hi, hx i hi]
    · rw [if_neg hi, orMech_f]
      simp
  rw [← h]

/-- Under `do(S = x)` with `x w = true` for `w ∈ S`, every vertex reached from `w` by a directed
path avoiding `S` is `true`. -/
theorem orMech_resp_true (G : ADMG n) (hG : G.WF) {S : Finset (Fin n)} {x : Fin n → Bool}
    {w : Fin n} (hw : w ∈ S) (hxw : x w = true) (u : Roots n → Unit) {v : Fin n}
    (h : G.AncAvoid S w v) : (orMech G hG).resp S x u v = true := by
  unfold ADMG.AncAvoid at h
  induction h with
  | refl =>
    rw [Mech.resp_of_mem hw]
    exact hxw
  | @tail b c _ hbc ih =>
    rw [Mech.resp_of_not_mem hbc.2, orMech_f]
    exact decide_eq_true ⟨b, hbc.1, ih⟩

/-- One Boolean model on the causal diagram `G` separating every proper subset of `S` whose
missing element has a directed path to `y` in `G_{\overline S}`. -/
theorem or_witness {G : ADMG n} (hG : G.WF) (y : Fin n) (S : Finset (Fin n)) (hy : y ∉ S) :
    ∃ (m : Model n) (r : m.X y → ℝ), ConfDiag G m ∧
      ∀ T ⊆ S, ∀ w ∈ S, w ∉ T → G.AncAvoid S w y → ∃ x, m.mu y r T x ≠ m.mu y r S x := by
  refine ⟨Model.ofDet (orMech G hG), fun b : Bool => if b then 1 else 0, orMech_diag G hG, ?_⟩
  intro T _ w hwS hwT hpath
  have hx : ∀ i ∈ T, decide (i = w) = false := fun i hi =>
    decide_eq_false (fun hiw => hwT (hiw ▸ hi))
  have hxw : decide (w = w) = true := decide_eq_true rfl
  refine ⟨fun i => decide (i = w), ?_⟩
  rw [Model.ofDet_mu, Model.ofDet_mu,
    orMech_resp_false G hG (x := fun i => decide (i = w)) hx,
    orMech_resp_true G hG hwS (x := fun i => decide (i = w)) hxw _ hpath]
  simp

/-! ## Causal diagrams -/

/-- A set satisfying the ancestral criterion in `G` is a DMIS of the class of `G`. -/
theorem dmis_diag_of_ancCrit {G : ADMG n} (hG : G.WF) {y : Fin n} {S : Finset (Fin n)}
    (hy : y ∉ S) (hcrit : AncCrit G y S) : DMIS (ConfDiag G) y S := by
  obtain ⟨m, r, hm, hsep⟩ := or_witness hG y S hy
  refine ⟨hy, m, r, hm, fun T hT => ?_⟩
  obtain ⟨w, hwS, hwT⟩ := Finset.exists_of_ssubset hT
  exact hsep T (Finset.ssubset_def.mp hT).1 w hwS hwT (hcrit w hwS)

/-- Lee–Bareinboim (2018), Prop. 1. -/
theorem mis_diag_iff {G : ADMG n} (hG : G.WF) {y : Fin n} {S : Finset (Fin n)} (hy : y ∉ S) :
    MIS (ConfDiag G) y S ↔ AncCrit G y S := by
  refine ⟨fun h z hz => ?_, fun hcrit => dmis_mis (dmis_diag_of_ancCrit hG hy hcrit)⟩
  obtain ⟨m, hm, hpath⟩ := h.exists_ancAvoid hz
  rw [show m.scm.toMech.diag = G from hm] at hpath
  exact hpath

/-- MIS and DMIS coincide for causal diagrams. -/
theorem dmis_of_mis_diag {G : ADMG n} (hG : G.WF) {y : Fin n} {S : Finset (Fin n)}
    (h : MIS (ConfDiag G) y S) : DMIS (ConfDiag G) y S :=
  dmis_diag_of_ancCrit hG h.1 ((mis_diag_iff hG h.1).mp h)

end Mec.ScbMec
