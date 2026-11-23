import MecLean.Scb2018.Algorithm

set_option linter.unusedSectionVars false

/-!
# Structural Causal Bandits (Lee–Bareinboim, NeurIPS 2018): Algorithm 4 `MISs`

Source: the paper and its unofficial AI-corrected version (see `README.md`): App. A,
Alg. 4 (`alg:enum_miss`); Sec. 2, Def. 1 and Prop. 1 (`prop:minimality`).

* `subMISs G y X W` is `subMISs(G, Y, X, W)`: record `X`; for each `W_i ∈ W`, recurse on
  `H = G_{\overline{W_i}}[An(Y)_{G_{\overline{W_i}}}]` with `X ∪ {W_i}` and `W^{i+1:} ∩ V(H)`.
  Like `subPOMISs`, it is defined through a fuel argument starting from `|W|`.
* `MISs G y ord` is `MISs(G, Y)`, given `ord = reversed(topological-sort(H))` for
  `H = G[An(Y)_G]`.
* `mem_MISs_iff`: if `ord` is a reverse topological order of `G` on `An(Y)_G`, `MISs` lists
  exactly the sets `X ∌ Y` with `X ⊆ An(Y)_{G_{\overline X}}` (`AncCrit`).
* `mem_MISs_iff_mis`: for a well-formed causal diagram these are the MISs (Prop. 1,
  `mis_diag_iff`).  `mem_MISs_iff_of_revTopo_restrict`: the same under the hypothesis of the
  pseudocode, a reverse topological order of `H = G[An(Y)_G]`.

The recursion of the paper is correct: the reverse topological order guarantees that a later
candidate is never a descendant of an earlier one, so adding it cannot cut the directed path of
an earlier vertex.  (With a topological order instead, `A → B → Y` would give `{A, B}`.)

## Modelling choices

* Graphs keep the vertex type `Fin n`, so `V(H) = An(Y)_G`; line 4 intersects the order with
  `An(Y)_G \ {Y}` explicitly, and the recursion intersects the suffix with
  `An(Y)_{G_{\overline{W_i}}}`.
* The output is a list of `Finset`s; the theorems are about membership.  The `O(mn²)` running
  time is not formalized.
-/

namespace Mec.Scb2018

open Mec ScbMec

variable {n : ℕ}

namespace Alg

/-- `An(Y)_G`. -/
def anc (G : ADMG n) (y : Fin n) : Finset (Fin n) := Finset.univ.filter (fun v => G.Anc v y)

theorem mem_anc {G : ADMG n} {y v : Fin n} : v ∈ anc G y ↔ G.Anc v y := by
  unfold anc
  simp

/-- `G_{\overline w}[An(Y)_{G_{\overline w}}]`, the graph `H` of line 9 of `subMISs`. -/
def misGraph (G : ADMG n) (y w : Fin n) : ADMG n := restrict (cut G {w}) (anc (cut G {w}) y)

/-- `subMISs(G, Y, X, W)` with fuel `k`. -/
def subMISsAux (y : Fin n) : ℕ → ADMG n → Finset (Fin n) → List (Fin n) → List (Finset (Fin n))
  | 0, _, X, _ => [X]
  | k + 1, G, X, W => X :: forLoop (fun _ w rest =>
      subMISsAux y k (misGraph G y w) (insert w X)
        (rest.filter (fun v => decide (v ∈ anc (cut G {w}) y)))) ∅ W

/-- `subMISs(G, Y, X, W)` (Alg. 4, lines 6–12). -/
def subMISs (G : ADMG n) (y : Fin n) (X : Finset (Fin n)) (W : List (Fin n)) :
    List (Finset (Fin n)) :=
  subMISsAux y W.length G X W

/-- `MISs(G, Y)` (Alg. 4, lines 1–5); `ord` is `reversed(topological-sort(H))`. -/
def MISs (G : ADMG n) (y : Fin n) (ord : List (Fin n)) : List (Finset (Fin n)) :=
  subMISs (restrict G (anc G y)) y ∅ (ord.filter (fun v => decide (v ∈ anc G y ∧ v ≠ y)))

/-! ## Directed paths in the restricted graphs -/

/-- Restricting to the ancestors of `y` keeps the directed paths to `y`. -/
theorem restrict_anc_ancAvoid {K : ADMG n} {y : Fin n} {D : Finset (Fin n)} {v : Fin n} :
    (restrict K (anc K y)).AncAvoid D v y ↔ K.AncAvoid D v y := by
  constructor
  · intro h
    exact Reach.mono_rel (R := (restrict K (anc K y)).dir) (R' := K.dir)
      (fun a b _ hab => (restrict_dir.mp hab).1) h
  · intro h
    unfold ADMG.AncAvoid Reach at h
    induction h using Relation.ReflTransGen.head_induction_on with
    | refl => exact Reach.refl _
    | @head a c hac hcy ih =>
      have hcA : c ∈ anc K y := mem_anc.mpr (K.anc_of_avoid hcy)
      have haA : a ∈ anc K y := mem_anc.mpr (K.anc_of_avoid (Reach.head hac.1 hac.2 hcy))
      exact Reach.head (restrict_dir.mpr ⟨hac.1, haA, hcA⟩) hac.2 ih

/-- `G` is the graph held by `subMISs` after selecting `X` in `G0`: its directed paths to `y`
avoiding `D` are those of `G0` avoiding `X ∪ D`. -/
def MRep (G0 G : ADMG n) (y : Fin n) (X : Finset (Fin n)) : Prop :=
  ∀ (D : Finset (Fin n)) (v : Fin n), G.AncAvoid D v y ↔ G0.AncAvoid (X ∪ D) v y

theorem mrep_root (G0 : ADMG n) (y : Fin n) : MRep G0 (restrict G0 (anc G0 y)) y ∅ := by
  intro D v
  rw [restrict_anc_ancAvoid, Finset.empty_union]

theorem MRep.step {G0 G : ADMG n} {y : Fin n} {X : Finset (Fin n)} (h : MRep G0 G y X)
    (w : Fin n) : MRep G0 (misGraph G y w) y (insert w X) := by
  intro D v
  unfold misGraph
  rw [restrict_anc_ancAvoid, cut_ancAvoid, h ({w} ∪ D) v]
  have e : X ∪ ({w} ∪ D) = insert w X ∪ D := by
    ext u
    simp only [Finset.mem_union, Finset.mem_insert, Finset.mem_singleton]
    tauto
  rw [e]

theorem MRep.mem_anc_cut {G0 G : ADMG n} {y : Fin n} {X : Finset (Fin n)} (h : MRep G0 G y X)
    (w v : Fin n) : v ∈ anc (cut G {w}) y ↔ G0.AncAvoid (insert w X) v y := by
  rw [mem_anc]
  show (cut G {w}).AncAvoid ∅ v y ↔ _
  rw [cut_ancAvoid, h ({w} ∪ ∅) v]
  have e : X ∪ ({w} ∪ ∅) = insert w X := by
    ext u
    simp only [Finset.mem_union, Finset.mem_insert, Finset.mem_singleton,
      Finset.notMem_empty]
    tauto
  rw [e]

/-- A later candidate cannot cut the directed path of an earlier vertex: the vertices after
`x` on a directed path to `y` precede `x` in a reverse topological order. -/
theorem ancAvoid_union_of_idx {G0 : ADMG n} {y : Fin n} {ord : List (Fin n)}
    (hord : ∀ a ∈ anc G0 y, ∀ b ∈ anc G0 y, G0.dir a b = true → ord.idxOf b < ord.idxOf a)
    {X E : Finset (Fin n)} {x : Fin n} (h : G0.AncAvoid X x y)
    (hE : ∀ v ∈ E, ord.idxOf x ≤ ord.idxOf v) : G0.AncAvoid (X ∪ E) x y := by
  have key : ∀ a, G0.AncAvoid X a y → ord.idxOf a ≤ ord.idxOf x →
      G0.AncAvoid (X ∪ E) a y := by
    intro a ha
    unfold ADMG.AncAvoid Reach at ha
    induction ha using Relation.ReflTransGen.head_induction_on with
    | refl => exact fun _ => Reach.refl _
    | @head a c hac hcy ih =>
      intro hle
      have hcA : c ∈ anc G0 y := mem_anc.mpr (G0.anc_of_avoid hcy)
      have haA : a ∈ anc G0 y := mem_anc.mpr (G0.anc_of_avoid (Reach.head hac.1 hac.2 hcy))
      have hlt := hord a haA c hcA hac.1
      refine Reach.head hac.1 (fun hc => ?_) (ih (by omega))
      rcases Finset.mem_union.mp hc with hc | hc
      · exact hac.2 hc
      · have := hE c hc
        omega
  exact key x h le_rfl

/-! ## Correctness -/

/-- The invariants pass from `subMISs(G, Y, X, W)` to the call for `W_i = w`. -/
theorem child_inv {G0 G : ADMG n} {y : Fin n} {ord : List (Fin n)}
    (hord : ∀ a ∈ anc G0 y, ∀ b ∈ anc G0 y, G0.dir a b = true → ord.idxOf b < ord.idxOf a)
    {X : Finset (Fin n)} {W pre rest : List (Fin n)} {w : Fin n}
    (hR : MRep G0 G y X) (hX : AncCrit G0 y X) (hW : ∀ v ∈ W, v ∉ X ∧ G0.AncAvoid X v y)
    (hs : W.Pairwise (fun a b => ord.idxOf a < ord.idxOf b))
    (hb : ∀ x ∈ X, ∀ v ∈ W, ord.idxOf x < ord.idxOf v) (hsplit : W = pre ++ w :: rest) :
    MRep G0 (misGraph G y w) y (insert w X) ∧ AncCrit G0 y (insert w X) ∧
      (∀ v ∈ rest.filter (fun v => decide (v ∈ anc (cut G {w}) y)),
        v ∉ insert w X ∧ G0.AncAvoid (insert w X) v y) ∧
      (rest.filter (fun v => decide (v ∈ anc (cut G {w}) y))).Pairwise
        (fun a b => ord.idxOf a < ord.idxOf b) ∧
      (∀ x ∈ insert w X, ∀ v ∈ rest.filter (fun v => decide (v ∈ anc (cut G {w}) y)),
        ord.idxOf x < ord.idxOf v) := by
  have hwW : w ∈ W := by rw [hsplit]; simp
  have hs' := hs
  rw [hsplit, List.pairwise_append, List.pairwise_cons] at hs'
  obtain ⟨-, ⟨hwrest, hrest⟩, -⟩ := hs'
  have hwrest' : ∀ v ∈ rest, ord.idxOf w < ord.idxOf v := hwrest
  have hrestW : ∀ v ∈ rest, v ∈ W := fun v hv => by rw [hsplit]; simp [hv]
  have hins : insert w X ⊆ X ∪ {w} := by
    intro u hu
    rw [Finset.mem_insert] at hu
    rw [Finset.mem_union, Finset.mem_singleton]
    tauto
  refine ⟨hR.step w, ?_, ?_, hrest.filter _, ?_⟩
  · intro x hx
    rw [Finset.mem_insert] at hx
    rcases hx with hx | hx
    · rw [hx]
      exact Reach.mono_avoid hins (ancAvoid_union_of_idx hord (hW w hwW).2
        (fun v hv => le_of_eq (congrArg (fun a => ord.idxOf a)
          (Finset.mem_singleton.mp hv).symm)))
    · exact Reach.mono_avoid hins (ancAvoid_union_of_idx hord (hX x hx)
        (fun v hv => by
          rw [Finset.mem_singleton.mp hv]
          exact le_of_lt (hb x hx w hwW)))
  · intro v hv
    simp only [List.mem_filter, decide_eq_true_eq] at hv
    obtain ⟨hvrest, hvA⟩ := hv
    rw [hR.mem_anc_cut] at hvA
    refine ⟨fun h => ?_, hvA⟩
    rw [Finset.mem_insert] at h
    rcases h with h | h
    · have := hwrest' v hvrest
      rw [h] at this
      exact lt_irrefl _ this
    · exact (hW v (hrestW v hvrest)).1 h
  · intro x hx v hv
    have hvrest := (List.mem_filter.mp hv).1
    rw [Finset.mem_insert] at hx
    rcases hx with hx | hx
    · rw [hx]
      exact hwrest' v hvrest
    · exact hb x hx v (hrestW v hvrest)

theorem mem_subMISsAux {G0 : ADMG n} {y : Fin n} {ord : List (Fin n)}
    (hord : ∀ a ∈ anc G0 y, ∀ b ∈ anc G0 y, G0.dir a b = true → ord.idxOf b < ord.idxOf a) :
    ∀ (k : ℕ) (G : ADMG n) (X : Finset (Fin n)) (W : List (Fin n)), W.length ≤ k →
      MRep G0 G y X → AncCrit G0 y X → (∀ v ∈ W, v ∉ X ∧ G0.AncAvoid X v y) →
      W.Pairwise (fun a b => ord.idxOf a < ord.idxOf b) →
      (∀ x ∈ X, ∀ v ∈ W, ord.idxOf x < ord.idxOf v) →
      ∀ Z : Finset (Fin n), Z ∈ subMISsAux y k G X W ↔
        X ⊆ Z ∧ (∀ z ∈ Z, z ∉ X → z ∈ W) ∧ AncCrit G0 y Z := by
  intro k
  induction k with
  | zero =>
    intro G X W hlen _ hX _ _ _ Z
    have hW : W = [] := List.length_eq_zero_iff.mp (Nat.le_zero.mp hlen)
    subst hW
    simp only [subMISsAux, List.mem_singleton]
    constructor
    · intro hZ
      rw [hZ]
      exact ⟨subset_rfl, fun z hz hzX => absurd hz hzX, hX⟩
    · rintro ⟨h1, h2, -⟩
      refine Finset.Subset.antisymm (fun z hz => ?_) h1
      by_contra hzX
      exact absurd (h2 z hz hzX) List.not_mem_nil
  | succ k ih =>
    intro G X W hlen hR hX hW hs hb Z
    simp only [subMISsAux, List.mem_cons]
    rw [mem_forLoop]
    constructor
    · rintro (hZ | ⟨pre, w, rest, hsplit, hZ⟩)
      · rw [hZ]
        exact ⟨subset_rfl, fun z hz hzX => absurd hz hzX, hX⟩
      · obtain ⟨c1, c2, c3, c4, c5⟩ := child_inv hord hR hX hW hs hb hsplit
        have hlen' : (rest.filter (fun v => decide (v ∈ anc (cut G {w}) y))).length ≤ k := by
          have h1 := List.length_filter_le (fun v => decide (v ∈ anc (cut G {w}) y)) rest
          have h2 : W.length = pre.length + (rest.length + 1) := by
            rw [hsplit, List.length_append, List.length_cons]
          omega
        obtain ⟨h1, h2, h3⟩ := (ih _ _ _ hlen' c1 c2 c3 c4 c5 Z).mp hZ
        refine ⟨fun u hu => h1 (Finset.mem_insert_of_mem hu), fun z hz hzX => ?_, h3⟩
        by_cases hzw : z = w
        · rw [hzw, hsplit]
          simp
        · have hz' := h2 z hz (by rw [Finset.mem_insert]; tauto)
          rw [hsplit]
          simp [(List.mem_filter.mp hz').1]
    · rintro ⟨h1, h2, h3⟩
      by_cases hZX : Z = X
      · left
        exact hZX
      · right
        have hss : X ⊂ Z := Finset.ssubset_iff_subset_ne.mpr ⟨h1, Ne.symm hZX⟩
        obtain ⟨z, hzZ, hzX⟩ := Finset.exists_of_ssubset hss
        obtain ⟨pre, w, rest, hsplit, ⟨hwZ, -⟩, hpre⟩ :=
          exists_first (fun v => v ∈ Z ∧ v ∉ X) W ⟨z, h2 z hzZ hzX, hzZ, hzX⟩
        have hpre' : ∀ v ∈ pre, ¬ (v ∈ Z ∧ v ∉ X) := hpre
        refine ⟨pre, w, rest, hsplit, ?_⟩
        obtain ⟨c1, c2, c3, c4, c5⟩ := child_inv hord hR hX hW hs hb hsplit
        have hlen' : (rest.filter (fun v => decide (v ∈ anc (cut G {w}) y))).length ≤ k := by
          have h1 := List.length_filter_le (fun v => decide (v ∈ anc (cut G {w}) y)) rest
          have h2 : W.length = pre.length + (rest.length + 1) := by
            rw [hsplit, List.length_append, List.length_cons]
          omega
        refine (ih _ _ _ hlen' c1 c2 c3 c4 c5 Z).mpr ⟨?_, fun u hu huX => ?_, h3⟩
        · intro u hu
          rw [Finset.mem_insert] at hu
          rcases hu with hu | hu
          · rw [hu]
            exact hwZ
          · exact h1 hu
        · have huX' : u ∉ X := fun h => huX (Finset.mem_insert_of_mem h)
          have huw : u ≠ w := fun h => huX (by rw [h]; exact Finset.mem_insert_self w X)
          have huW := h2 u hu huX'
          rw [hsplit, List.mem_append, List.mem_cons] at huW
          refine List.mem_filter.mpr ⟨?_, decide_eq_true ((hR.mem_anc_cut w u).mpr ?_)⟩
          · rcases huW with h | h | h
            · exact absurd ⟨hu, huX'⟩ (hpre' u h)
            · exact absurd h huw
            · exact h
          · refine Reach.mono_avoid (fun v hv => ?_) (h3 u hu)
            rw [Finset.mem_insert] at hv
            rcases hv with hv | hv
            · rw [hv]
              exact hwZ
            · exact h1 hv

/-- A reverse topological order of `H = G[An(Y)_G]` on its vertices is one of `G` on
`An(Y)_G`. -/
theorem revTopo_of_restrict {G : ADMG n} {y : Fin n} {ord : List (Fin n)}
    (h : RevTopo (restrict G (anc G y)) (anc G y) ord) : RevTopo G (anc G y) ord :=
  ⟨h.1, h.2.1, fun a ha b hb hab => h.2.2 a ha b hb (restrict_dir.mpr ⟨hab, ha, hb⟩)⟩

end Alg

open Alg

/-- **Algorithm 4** (`alg:enum_miss`) lists exactly the sets `X ∌ Y` with
`X ⊆ An(Y)_{G_{\overline X}}`, given a reverse topological order of `G` on `An(Y)_G`. -/
theorem mem_MISs_iff {G : ADMG n} {y : Fin n} {ord : List (Fin n)}
    (hord : RevTopo G (anc G y) ord) (X : Finset (Fin n)) :
    X ∈ MISs G y ord ↔ y ∉ X ∧ AncCrit G y X := by
  have hW0 : ∀ v ∈ ord.filter (fun v => decide (v ∈ anc G y ∧ v ≠ y)),
      v ∉ (∅ : Finset (Fin n)) ∧ G.AncAvoid ∅ v y := by
    intro v hv
    simp only [List.mem_filter, decide_eq_true_eq] at hv
    exact ⟨Finset.notMem_empty v, mem_anc.mp hv.2.1⟩
  have key := mem_subMISsAux hord.2.2 _ _ _ _ le_rfl (mrep_root G y)
    (fun x hx => absurd hx (Finset.notMem_empty x)) hW0 ((pairwise_idxOf ord hord.1).filter _)
    (fun x hx => absurd hx (Finset.notMem_empty x)) X
  unfold MISs subMISs
  rw [key]
  constructor
  · rintro ⟨-, h2, h3⟩
    refine ⟨fun hy => ?_, h3⟩
    have := h2 y hy (Finset.notMem_empty y)
    simp at this
  · rintro ⟨hy, h3⟩
    refine ⟨Finset.empty_subset _, fun z hz _ => ?_, h3⟩
    have hzA : z ∈ anc G y := mem_anc.mpr (G.anc_of_avoid (h3 z hz))
    exact List.mem_filter.mpr ⟨hord.2.1 z hzA, decide_eq_true ⟨hzA, fun h => hy (h ▸ hz)⟩⟩

/-- **Algorithm 4** returns exactly the MISs of a well-formed causal diagram (with Prop. 1). -/
theorem mem_MISs_iff_mis {G : ADMG n} (hG : G.WF) {y : Fin n} {ord : List (Fin n)}
    (hord : RevTopo G (anc G y) ord) (X : Finset (Fin n)) :
    X ∈ MISs G y ord ↔ MIS (ConfDiag G) y X := by
  rw [mem_MISs_iff hord]
  constructor
  · rintro ⟨hy, h⟩
    exact (mis_diag_iff hG hy).mpr h
  · intro h
    exact ⟨h.1, (mis_diag_iff hG h.1).mp h⟩

/-- **Algorithm 4** under the hypothesis of the pseudocode: `ord` is a reverse topological order
of `H = G[An(Y)_G]` on its vertices. -/
theorem mem_MISs_iff_of_revTopo_restrict {G : ADMG n} (hG : G.WF) {y : Fin n}
    {ord : List (Fin n)} (hord : RevTopo (restrict G (anc G y)) (anc G y) ord)
    (X : Finset (Fin n)) : X ∈ MISs G y ord ↔ MIS (ConfDiag G) y X :=
  mem_MISs_iff_mis hG (revTopo_of_restrict hord) X

end Mec.Scb2018
