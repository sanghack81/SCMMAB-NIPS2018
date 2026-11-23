import MecLean.Common.MSep

set_option linter.unusedSectionVars false

/-!
# MAGs, induced MAGs, Markov equivalence classes and PAGs

* `IsMAG M`: a simple, well-formed, ancestral ADMG in which every non-adjacent pair is
  m-separated by some set of other vertices (maximality as in Richardson–Spirtes).
* `G.mag`: the MAG induced by a causal diagram `G` (ancestral projection).  Two vertices are
  adjacent iff no set of other vertices m-separates them in `G`; the edge is `a → b` when `a` is an
  ancestor of `b` and `a ↔ b` when neither is an ancestor of the other.  (The adjacency rule is
  equivalent to the inducing-path rule by Richardson–Spirtes, Thm. 4.2; we take the
  separation form as the definition.)
* `MEC M0 M`: `M` is a MAG Markov equivalent to `M0`.
* `IsPAG P M0`: the marks of `P` are the invariant marks of `MEC M0`; a mark is a circle exactly
  when it varies over the class.
-/

open Finset

namespace Mec

variable {n : ℕ}

namespace ADMG

variable (G : ADMG n)

/-- No set of other vertices m-separates `a` and `b`. -/
def Insep (a b : Fin n) : Prop := ∀ S : Finset (Fin n), a ∉ S → b ∉ S → MConn G S a b

instance (a b : Fin n) : Decidable (G.Insep a b) := by unfold Insep; infer_instance

/-- Maximality: every non-adjacent pair is m-separated by some set of other vertices. -/
def Maximal : Prop :=
  ∀ i j, i ≠ j → G.adj i j = false → ∃ S : Finset (Fin n), i ∉ S ∧ j ∉ S ∧ ¬ MConn G S i j

instance : Decidable G.Maximal := by unfold Maximal; infer_instance

/-- A maximal ancestral graph without undirected edges. -/
def IsMAG : Prop := G.WF ∧ G.Simple ∧ G.Ancestral ∧ G.Maximal

instance : Decidable G.IsMAG := by unfold IsMAG; infer_instance

/-- The MAG induced by the causal diagram `G` (ancestral projection). -/
def mag : ADMG n where
  dir i j := decide (G.Insep i j ∧ G.Anc i j)
  bi i j := decide (G.Insep i j ∧ ¬ G.Anc i j ∧ ¬ G.Anc j i)

variable {G}

theorem WF.dir_irrefl (h : G.WF) (i : Fin n) : G.dir i i = false := h.1 i
theorem WF.bi_irrefl (h : G.WF) (i : Fin n) : G.bi i i = false := h.2.1 i
theorem WF.bi_symm (h : G.WF) (i j : Fin n) : G.bi i j = G.bi j i := h.2.2.1 i j
theorem WF.acyclic (h : G.WF) {i j : Fin n} (hij : G.dir i j = true) : ¬ G.Anc j i :=
  h.2.2.2 i j hij

theorem WF.adj_irrefl (h : G.WF) (i : Fin n) : G.adj i i = false := by
  simp [adj, h.dir_irrefl i, h.bi_irrefl i]

theorem WF.ne_of_adj (h : G.WF) {i j : Fin n} (hij : G.adj i j = true) : i ≠ j := by
  rintro rfl
  rw [h.adj_irrefl] at hij
  exact Bool.false_ne_true hij

theorem WF.adj_comm (h : G.WF) (i j : Fin n) : G.adj i j = G.adj j i :=
  G.adj_comm i j (h.bi_symm i j)

theorem anc_of_dir {i j : Fin n} (h : G.dir i j = true) : G.Anc i j :=
  Reach.single h (Finset.notMem_empty j)

/-- Adjacent vertices are inseparable. -/
theorem insep_of_adj (hG : G.WF) {a b : Fin n} (hadj : G.adj a b = true) : G.Insep a b := by
  intro S ha hb
  have hab := hG.ne_of_adj hadj
  unfold adj at hadj
  simp only [Bool.or_eq_true] at hadj
  rcases hadj with (h | h) | h
  · exact mconn_of_edge G hab ha hb .fwd h
  · exact mconn_of_edge G hab ha hb .bwd h
  · exact mconn_of_edge G hab ha hb .bi h

theorem not_insep_self (a : Fin n) : ¬ G.Insep a a := fun h =>
  not_mconn_self G ∅ a (h ∅ (Finset.notMem_empty a) (Finset.notMem_empty a))

/-- In a maximal graph, inseparability is adjacency. -/
theorem insep_iff_adj (hG : G.WF) (hmax : G.Maximal) {a b : Fin n} :
    G.Insep a b ↔ G.adj a b = true := by
  refine ⟨fun h => ?_, insep_of_adj hG⟩
  by_contra hna
  have hab : a ≠ b := by
    rintro rfl
    exact not_insep_self a h
  obtain ⟨S, ha, hb, hS⟩ := hmax a b hab (by simpa using hna)
  exact hS (h S ha hb)

theorem IsMAG.wf (h : G.IsMAG) : G.WF := h.1
theorem IsMAG.simple (h : G.IsMAG) : G.Simple := h.2.1
theorem IsMAG.ancestral (h : G.IsMAG) : G.Ancestral := h.2.2.1
theorem IsMAG.maximal (h : G.IsMAG) : G.Maximal := h.2.2.2

/-- In a MAG an edge `i → j` excludes the other edge types. -/
theorem IsMAG.dir_excl (h : G.IsMAG) {i j : Fin n} (hij : G.dir i j = true) :
    G.dir j i = false ∧ G.bi i j = false ∧ G.bi j i = false := by
  obtain ⟨h1, h2⟩ := h.simple i j hij
  exact ⟨h1, h2, by rw [← h.wf.bi_symm]; exact h2⟩

theorem IsMAG.markAt_of_dir (h : G.IsMAG) {i j : Fin n} (hij : G.dir i j = true) :
    G.markAt i j = some .arrow ∧ G.markAt j i = some .tail := by
  obtain ⟨h1, h2, h3⟩ := h.dir_excl hij
  constructor
  · simp [markAt, hij]
  · simp [markAt, h1, h3, hij]

theorem markAt_isSome (i j : Fin n) : (G.markAt i j).isSome = G.adj i j := by
  unfold markAt adj
  cases h1 : G.dir i j <;> cases h2 : G.dir j i <;> cases h3 : G.bi i j <;> simp

/-- In a MAG every adjacency is realized by an edge kind whose arrowheads are the marks. -/
theorem IsMAG.exists_kind (hG : G.IsMAG) {a c : Fin n} (h : G.adj a c = true) :
    ∃ e : EK, e.ok G a c = true ∧ e.headFar = decide (G.markAt a c = some .arrow) ∧
      e.headNear = decide (G.markAt c a = some .arrow) := by
  unfold ADMG.adj at h
  simp only [Bool.or_eq_true] at h
  rcases h with (h | h) | h
  · obtain ⟨h1, _, h3⟩ := hG.dir_excl h
    refine ⟨.fwd, h, ?_, ?_⟩
    · simp [EK.headFar, ADMG.markAt, h]
    · simp [EK.headNear, ADMG.markAt, h1, h3, h]
  · obtain ⟨h1, _, h3⟩ := hG.dir_excl h
    refine ⟨.bwd, h, ?_, ?_⟩
    · simp [EK.headFar, ADMG.markAt, h1, h3, h]
    · simp [EK.headNear, ADMG.markAt, h]
  · have h' : G.bi c a = true := by rw [hG.wf.bi_symm]; exact h
    refine ⟨.bi, h, ?_, ?_⟩
    · simp [EK.headFar, ADMG.markAt, h]
    · simp [EK.headNear, ADMG.markAt, h']

/-- The induced MAG keeps every directed edge of the causal diagram. -/
theorem mag_dir_of_dir (hG : G.WF) {i j : Fin n} (h : G.dir i j = true) : G.mag.dir i j = true := by
  have hadj : G.adj i j = true := by simp [adj, h]
  simp only [mag, decide_eq_true_eq]
  exact ⟨insep_of_adj hG hadj, anc_of_dir h⟩

/-- Directed paths of the causal diagram are directed paths of its induced MAG. -/
theorem mag_ancAvoid (hG : G.WF) {D : Finset (Fin n)} {a b : Fin n} (h : G.AncAvoid D a b) :
    G.mag.AncAvoid D a b :=
  Reach.mono_rel (fun _ _ _ hxy => mag_dir_of_dir hG hxy) h

theorem mag_anc (hG : G.WF) {a b : Fin n} (h : G.Anc a b) : G.mag.Anc a b :=
  mag_ancAvoid hG h

/-- A MAG is its own induced MAG. -/
theorem mag_self (h : G.IsMAG) : G.mag = G := by
  have hwf := h.wf
  have hins : ∀ i j, G.Insep i j ↔ G.adj i j = true := fun i j => insep_iff_adj hwf h.maximal
  rw [ext_iff']
  constructor
  · funext i j
    simp only [mag]
    apply Bool.eq_iff_iff.mpr
    rw [decide_eq_true_iff, hins]
    constructor
    · rintro ⟨hadj, hanc⟩
      by_contra hij
      simp only [Bool.not_eq_true] at hij
      unfold adj at hadj
      simp only [hij, Bool.false_or, Bool.or_eq_true] at hadj
      rcases hadj with hji | hbi
      · exact hwf.acyclic hji hanc
      · exact h.ancestral i j hbi hanc
    · intro hij
      exact ⟨by simp [adj, hij], anc_of_dir hij⟩
  · funext i j
    simp only [mag]
    apply Bool.eq_iff_iff.mpr
    rw [decide_eq_true_iff, hins]
    constructor
    · rintro ⟨hadj, hni, hnj⟩
      by_contra hbi
      simp only [Bool.not_eq_true] at hbi
      unfold adj at hadj
      simp only [hbi, Bool.or_false, Bool.or_eq_true] at hadj
      rcases hadj with hij | hji
      · exact hni (anc_of_dir hij)
      · exact hnj (anc_of_dir hji)
    · intro hbi
      refine ⟨by simp [adj, hbi], h.ancestral i j hbi, ?_⟩
      exact h.ancestral j i (by rw [hwf.bi_symm]; exact hbi)

end ADMG

/-! ## Markov equivalence -/

/-- `G` and `H` have the same m-separations. -/
def MarkovEquiv (G H : ADMG n) : Prop := ∀ S a b, MConn G S a b ↔ MConn H S a b

instance (G H : ADMG n) : Decidable (MarkovEquiv G H) := by unfold MarkovEquiv; infer_instance

/-- `M` belongs to the Markov equivalence class of `M0`. -/
def MEC (M0 M : ADMG n) : Prop := M.IsMAG ∧ MarkovEquiv M M0

instance (M0 M : ADMG n) : Decidable (MEC M0 M) := by unfold MEC; infer_instance

theorem MarkovEquiv.insep {G H : ADMG n} (h : MarkovEquiv G H) {a b : Fin n} :
    G.Insep a b ↔ H.Insep a b :=
  ⟨fun hg S ha hb => (h S a b).mp (hg S ha hb), fun hh S ha hb => (h S a b).mpr (hh S ha hb)⟩

/-- Markov-equivalent MAGs have the same adjacencies. -/
theorem MEC.adj_eq {M0 M : ADMG n} (hM0 : M0.IsMAG) (hM : MEC M0 M) (a b : Fin n) :
    M.adj a b = M0.adj a b := by
  have h := hM.2.insep (a := a) (b := b)
  rw [ADMG.insep_iff_adj hM.1.wf hM.1.maximal, ADMG.insep_iff_adj hM0.wf hM0.maximal] at h
  cases h1 : M.adj a b <;> cases h2 : M0.adj a b <;> simp_all

/-- An unshielded collider of a MAG in the class of `M0` is a collider of `M0`.  Stated with
marks: `a *→ c ←* b`, `a, b` non-adjacent. -/
theorem MEC.collider_of_collider {M0 M : ADMG n} (hM0 : M0.IsMAG) (hM : MEC M0 M)
    {a c b : Fin n} (hab : a ≠ b) (hnadj : M.adj a b = false)
    (hac : M.adj a c = true) (hcb : M.adj c b = true)
    (hcolM : M.markAt a c = some .arrow ∧ M.markAt b c = some .arrow) :
    M0.markAt a c = some .arrow ∧ M0.markAt b c = some .arrow := by
  obtain ⟨S, ha, hb, hS⟩ := hM.1.maximal a b hab hnadj
  have hcS : c ∉ S := by
    intro hc
    apply hS
    obtain ⟨e₁, he₁, hf₁, -⟩ := hM.1.exists_kind hac
    obtain ⟨e₂, he₂, -, hn₂⟩ := hM.1.exists_kind hcb
    refine mconn_of_two M (hM.1.wf.ne_of_adj hac) hab (hM.1.wf.ne_of_adj hcb) ha hb e₁ e₂
      he₁ he₂ ?_
    have hcol : (e₁.headFar && e₂.headNear) = true := by
      rw [hf₁, hn₂]
      simp [hcolM.1, hcolM.2]
    rw [if_pos hcol]
    exact decide_eq_true ⟨c, hc, Reach.refl c⟩
  have hS0 : ¬ MConn M0 S a b := fun h => hS ((hM.2 S a b).mpr h)
  by_contra hcol
  apply hS0
  have hac0 : M0.adj a c = true := by rw [← MEC.adj_eq hM0 hM a c]; exact hac
  have hcb0 : M0.adj c b = true := by rw [← MEC.adj_eq hM0 hM c b]; exact hcb
  obtain ⟨e₁, he₁, hf₁, -⟩ := hM0.exists_kind hac0
  obtain ⟨e₂, he₂, -, hn₂⟩ := hM0.exists_kind hcb0
  refine mconn_of_two M0 (hM0.wf.ne_of_adj hac0) hab (hM0.wf.ne_of_adj hcb0) ha hb e₁ e₂
    he₁ he₂ ?_
  have hnc : (e₁.headFar && e₂.headNear) = false := by
    rw [hf₁, hn₂]
    rcases not_and_or.mp hcol with h | h <;> simp [h]
  rw [if_neg (by simp [hnc])]
  simp [hcS]

end Mec
