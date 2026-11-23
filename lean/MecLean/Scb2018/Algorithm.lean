import MecLean.Scb2018.Recursion

set_option linter.unusedSectionVars false

/-!
# Structural Causal Bandits (Lee–Bareinboim, NeurIPS 2018): Algorithm `POMISs` and Thm. 9

Source: the paper and its unofficial AI-corrected version (see `README.md`): Alg. 1
(`alg:pomis-efficient`) and Thm. 9 of Sec. 4; App. C, proof of Thm. 9.

## The algorithm

* `forLoop body O π` is the loop `for π_i ∈ π` of `subPOMISs`: the body receives
  `O ∪ π^{1:i-1}`, `π_i` and `π^{i+1:}`, and the outputs are concatenated.
* `pomisBody` is lines 7–9 for one pivot `q`: `T = MUCT(G_{\overline q}, Y)`,
  `X = IB(G_{\overline q}, Y)`, `π' = π^{i+1:} ∩ T`; if `X ∩ O' = ∅`, record `X` and, if
  `π' ≠ ∅`, recurse on `G_{\overline X}[T ∪ X]`.
* `subPOMISs G y π O` is `subPOMISs(G, Y, π, O)`.  It is defined through `subPOMISsAux` with a
  fuel argument, starting from `|π|`, so that it reduces in the kernel; `subPOMISs_eq` proves
  that `subPOMISs` satisfies the recursion of the pseudocode, with no fuel.
* `POMISs G y ord` is `POMISs(G, Y)`, given the list `ord`, the output of
  `reversed(topological-sort(H))` for `H = G_{\overline X}[T ∪ X]` (lines 2–3).

## Results

* `thm9_graphical`: if `ord` is a reverse topological order of `G` on `MUCT(G, Y)`
  (`RevTopo`), then `X ∈ POMISs G y ord` iff `IB(G_{\overline X}, Y) = X`.
* `thm9`: for a well-formed causal diagram, `X ∈ POMISs G y ord` iff `X` is a POMIS
  (via `thm6`, whose "if" direction is `ScbMec.pomis_diag`).
* `thm9_of_revTopo_compress`: the same under the hypothesis of the pseudocode, a reverse
  topological order of `H = G_{\overline X}[T ∪ X]` on its vertices `T ∪ X`.

## Modelling choices

* Sets are `Finset`s and the output is a list; the theorems are about membership.
* The order is an input with its defining property (`RevTopo`), and only its restriction to
  `MUCT(G, Y)` matters.  Since every graph keeps the vertex type `Fin n`, the vertices of `H` are
  `T ∪ X`, and line 3 intersects the order with `T \ {Y}` explicitly.
* MUCT and IB are computed by the closure `muct` and `ib` of `MecLean.Scb2018.Recursion`, in
  the current graph, with the pivot as intervention set.
* The excluded set `O' = O ∪ π^{1:i-1}` is accumulated by `forLoop`.

## The proof

Soundness: every compressed graph represents the pivots chosen along its recursion path (`Rep`,
from Prop. 8 by `rep_compress` and `Rep.trans`), so every recorded set is some
`IB(G_{\overline S}, Y)`, which is its own border (`ib_self`, from Prop. 7).  Completeness
follows App. C with the invariants `Inv`; `prune_ok` is its argument that the branch of the
earliest target pivot is not pruned.  The text uses without stating that the excluded set lies
in `MUCT(G, Y)` (`Inv.excl`), so that the order compares its members with the vertices of later
territories.  The test `π' ≠ ∅` of line 9 is redundant, since `subPOMISs` on an empty list
returns nothing; it is kept.
-/

namespace Mec.Scb2018

open Mec ScbMec

variable {n : ℕ}

namespace Alg

/-! ## The loop -/

/-- The loop `for π_i ∈ π`: the body receives `O ∪ π^{1:i-1}`, `π_i` and `π^{i+1:}`. -/
def forLoop {α : Type*} (body : Finset (Fin n) → Fin n → List (Fin n) → List α) :
    Finset (Fin n) → List (Fin n) → List α
  | _, [] => []
  | O, w :: rest => body O w rest ++ forLoop body (insert w O) rest

theorem union_toFinset_cons (O : Finset (Fin n)) (w : Fin n) (pre : List (Fin n)) :
    O ∪ (w :: pre).toFinset = insert w O ∪ pre.toFinset := by
  ext v
  simp only [Finset.mem_union, Finset.mem_insert, List.mem_toFinset, List.mem_cons]
  tauto

theorem mem_forLoop {α : Type*} {body : Finset (Fin n) → Fin n → List (Fin n) → List α}
    {x : α} : ∀ (π : List (Fin n)) (O : Finset (Fin n)), x ∈ forLoop body O π ↔
      ∃ pre q post, π = pre ++ q :: post ∧ x ∈ body (O ∪ pre.toFinset) q post := by
  intro π
  induction π with
  | nil =>
    intro O
    simp only [forLoop]
    constructor
    · intro h
      simp at h
    · rintro ⟨pre, q, post, h, -⟩
      cases pre <;> simp at h
  | cons w rest ih =>
    intro O
    simp only [forLoop, List.mem_append]
    rw [ih]
    constructor
    · rintro (h | ⟨pre, q, post, hs, h⟩)
      · exact ⟨[], w, rest, rfl, by simpa using h⟩
      · refine ⟨w :: pre, q, post, by simp [hs], ?_⟩
        rwa [union_toFinset_cons]
    · rintro ⟨pre, q, post, hs, h⟩
      cases pre with
      | nil =>
        left
        simp only [List.nil_append, List.cons.injEq] at hs
        obtain ⟨rfl, rfl⟩ := hs
        simpa using h
      | cons a pre =>
        right
        simp only [List.cons_append, List.cons.injEq] at hs
        obtain ⟨rfl, rfl⟩ := hs
        exact ⟨pre, q, post, rfl, by rwa [union_toFinset_cons] at h⟩

theorem forLoop_congr {α : Type*} {b₁ b₂ : Finset (Fin n) → Fin n → List (Fin n) → List α} :
    ∀ (π : List (Fin n)) (O : Finset (Fin n)),
      (∀ O' q post, post.length < π.length → b₁ O' q post = b₂ O' q post) →
        forLoop b₁ O π = forLoop b₂ O π := by
  intro π
  induction π with
  | nil => intro O _; rfl
  | cons w rest ih =>
    intro O h
    simp only [forLoop]
    rw [h O w rest (by rw [List.length_cons]; omega),
      ih (insert w O) (fun O' q post hl => h O' q post (by rw [List.length_cons]; omega))]

/-- The first element of a list with a property. -/
theorem exists_first {α : Type*} (p : α → Prop) :
    ∀ l : List α, (∃ x ∈ l, p x) →
      ∃ pre x post, l = pre ++ x :: post ∧ p x ∧ ∀ z ∈ pre, ¬ p z
  | [], ⟨x, hx, _⟩ => by simp at hx
  | a :: l, ⟨x, hx, hpx⟩ => by
    by_cases ha : p a
    · exact ⟨[], a, l, rfl, ha, fun z hz => by simp at hz⟩
    · have hxl : x ∈ l := by
        rcases List.mem_cons.mp hx with e | h
        · rw [e] at hpx
          exact absurd hpx ha
        · exact h
      obtain ⟨pre, b, post, h1, h2, h3⟩ := exists_first p l ⟨x, hxl, hpx⟩
      refine ⟨a :: pre, b, post, by simp [h1], h2, fun z hz => ?_⟩
      rcases List.mem_cons.mp hz with e | h
      · rw [e]
        exact ha
      · exact h3 z h

/-! ## Algorithm `POMISs` -/

/-- Lines 8–9 of `subPOMISs`: if `X ∩ O' = ∅`, record `X` and, if `π' ≠ ∅`, recurse on
`G_{\overline X}[T ∪ X]`; `rec` is the recursive call. -/
def pomisStep (rec : ADMG n → List (Fin n) → Finset (Fin n) → List (Finset (Fin n)))
    (G : ADMG n) (T X : Finset (Fin n)) (π' : List (Fin n)) (O' : Finset (Fin n)) :
    List (Finset (Fin n)) :=
  if X ∩ O' = ∅ then X :: (if π' = [] then [] else rec (compress G T X) π' O') else []

/-- Lines 6–9 of `subPOMISs` for the pivot `q = π_i`, with `O' = O ∪ π^{1:i-1}` and
`post = π^{i+1:}`: `T = MUCT(G_{\overline q}, Y)`, `X = IB(G_{\overline q}, Y)` and
`π' = π^{i+1:} ∩ T`. -/
def pomisBody (y : Fin n) (rec : ADMG n → List (Fin n) → Finset (Fin n) → List (Finset (Fin n)))
    (G : ADMG n) (O' : Finset (Fin n)) (q : Fin n) (post : List (Fin n)) :
    List (Finset (Fin n)) :=
  pomisStep rec G (muct G y {q}) (ib G y {q})
    (post.filter (fun v => decide (v ∈ muct G y {q}))) O'

/-- `subPOMISs(G, Y, π, O)` with fuel `k`. -/
def subPOMISsAux (y : Fin n) : ℕ → ADMG n → List (Fin n) → Finset (Fin n) →
    List (Finset (Fin n))
  | 0, _, _, _ => []
  | k + 1, G, π, O => forLoop (pomisBody y (fun G' π' O' => subPOMISsAux y k G' π' O') G) O π

/-- `subPOMISs(G, Y, π, O)` (Alg. 1, lines 4–11). -/
def subPOMISs (G : ADMG n) (y : Fin n) (π : List (Fin n)) (O : Finset (Fin n)) :
    List (Finset (Fin n)) :=
  subPOMISsAux y π.length G π O

/-- `POMISs(G, Y)` (Alg. 1, lines 1–3); `ord` is `reversed(topological-sort(H))`. -/
def POMISs (G : ADMG n) (y : Fin n) (ord : List (Fin n)) : List (Finset (Fin n)) :=
  ib G y ∅ :: subPOMISs (compress G (muct G y ∅) (ib G y ∅)) y
    (ord.filter (fun v => decide (v ∈ muct G y ∅ ∧ v ≠ y))) ∅

/-! ## The fuel is only a termination device -/

theorem pomisBody_congr {y : Fin n}
    {rec₁ rec₂ : ADMG n → List (Fin n) → Finset (Fin n) → List (Finset (Fin n))}
    (G : ADMG n) (O' : Finset (Fin n)) (q : Fin n) (post : List (Fin n))
    (h : ∀ G' π' O'', π'.length ≤ post.length → rec₁ G' π' O'' = rec₂ G' π' O'') :
    pomisBody y rec₁ G O' q post = pomisBody y rec₂ G O' q post := by
  unfold pomisBody pomisStep
  rw [h _ _ _ (List.length_filter_le _ _)]

theorem subPOMISsAux_fuel {y : Fin n} :
    ∀ (k₁ k₂ : ℕ) (G : ADMG n) (π : List (Fin n)) (O : Finset (Fin n)),
      π.length ≤ k₁ → π.length ≤ k₂ → subPOMISsAux y k₁ G π O = subPOMISsAux y k₂ G π O := by
  intro k₁
  induction k₁ with
  | zero =>
    intro k₂ G π O h₁ _
    have hπ : π = [] := List.length_eq_zero_iff.mp (Nat.le_zero.mp h₁)
    subst hπ
    cases k₂ <;> rfl
  | succ k₁ ih =>
    intro k₂ G π O h₁ h₂
    cases π with
    | nil => cases k₂ <;> rfl
    | cons w rest =>
      cases k₂ with
      | zero => simp at h₂
      | succ k₂ =>
        simp only [subPOMISsAux]
        apply forLoop_congr
        intro O' q post hl
        apply pomisBody_congr
        intro G' π' O'' hπ'
        rw [List.length_cons] at h₁ h₂ hl
        exact ih k₂ G' π' O'' (by omega) (by omega)

/-- `subPOMISs` satisfies the recursion of Alg. 1, lines 4–11, without fuel. -/
theorem subPOMISs_eq (G : ADMG n) (y : Fin n) (π : List (Fin n)) (O : Finset (Fin n)) :
    subPOMISs G y π O =
      forLoop (pomisBody y (fun G' π' O' => subPOMISs G' y π' O') G) O π := by
  unfold subPOMISs
  cases π with
  | nil => rfl
  | cons w rest =>
    simp only [List.length_cons, subPOMISsAux]
    apply forLoop_congr
    intro O' q post hl
    apply pomisBody_congr
    intro G' π' O'' hπ'
    rw [List.length_cons] at hl
    exact subPOMISsAux_fuel _ _ G' π' O'' (by omega) le_rfl

/-! ## Representation of the pivots by the compressed graphs -/

/-- `G` represents the intervention `S` on `G0`: every further intervention `W'` gives the MUCT
and IB of `G0` after intervention on `S ∪ W'`. -/
def Rep (G0 G : ADMG n) (y : Fin n) (S : Finset (Fin n)) : Prop :=
  ∀ W' : Finset (Fin n), muct G y W' = muct G0 y (S ∪ W') ∧ ib G y W' = ib G0 y (S ∪ W')

/-- Prop. 8: the compressed graph represents the intervention. -/
theorem rep_compress {G0 : ADMG n} {y : Fin n} {W T : Finset (Fin n)}
    (hT : DiagMUCT G0 y W T) : Rep G0 (compress G0 T (diagBorder G0 T)) y W :=
  fun _ => prop8_muct hT

theorem Rep.trans {G0 G G' : ADMG n} {y : Fin n} {S S₁ : Finset (Fin n)} (h₁ : Rep G0 G y S)
    (h₂ : Rep G G' y S₁) : Rep G0 G' y (S ∪ S₁) := by
  intro W'
  obtain ⟨e1, e2⟩ := h₂ W'
  obtain ⟨e3, e4⟩ := h₁ (S₁ ∪ W')
  rw [e1, e2, e3, e4, Finset.union_assoc]
  exact ⟨rfl, rfl⟩

/-! ## Soundness -/

theorem sound_aux {G0 : ADMG n} {y : Fin n} {X : Finset (Fin n)} :
    ∀ (k : ℕ) (G : ADMG n) (π : List (Fin n)) (O S : Finset (Fin n)),
      Rep G0 G y S → y ∉ S → (∀ v ∈ π, v ≠ y) → X ∈ subPOMISsAux y k G π O →
        ∃ S' : Finset (Fin n), y ∉ S' ∧ X = ib G0 y S' := by
  intro k
  induction k with
  | zero =>
    intro G π O S _ _ _ hX
    simp [subPOMISsAux] at hX
  | succ k ih =>
    intro G π O S hR hyS hπ hX
    simp only [subPOMISsAux] at hX
    rw [mem_forLoop] at hX
    obtain ⟨pre, q, post, hsplit, hX⟩ := hX
    have hqy : q ≠ y := hπ q (by rw [hsplit]; simp)
    have hyq : y ∉ ({q} : Finset (Fin n)) := fun h => hqy (Finset.mem_singleton.mp h).symm
    have hyS' : y ∉ S ∪ {q} := by
      rw [Finset.mem_union]
      rintro (h | h)
      · exact hyS h
      · exact hyq h
    have hR' : Rep G0 (compress G (muct G y {q}) (ib G y {q})) y (S ∪ {q}) :=
      hR.trans (rep_compress (diagMUCT_muct hyq))
    have hib : ib G y {q} = ib G0 y (S ∪ {q}) := (hR {q}).2
    unfold pomisBody pomisStep at hX
    by_cases h1 : ib G y {q} ∩ (O ∪ pre.toFinset) = ∅
    · rw [ite_eq_left h1] at hX
      rcases List.mem_cons.mp hX with hX | hX
      · exact ⟨S ∪ {q}, hyS', hX.trans hib⟩
      · by_cases h2 : post.filter (fun v => decide (v ∈ muct G y {q})) = []
        · rw [ite_eq_left h2] at hX
          simp at hX
        · rw [ite_eq_right h2] at hX
          refine ih _ _ _ _ hR' hyS' (fun v hv => hπ v ?_) hX
          have hv' := (List.mem_filter.mp hv).1
          rw [hsplit]
          simp [hv']
    · rw [ite_eq_right h1] at hX
      simp at hX

/-! ## Completeness -/

/-- The invariants of the completeness proof (App. C) at a call `subPOMISs(G, Y, π, O)`
reached by the pivots `S`, for the target `W` and the order `ord`. -/
structure Inv (G0 G : ADMG n) (y : Fin n) (W : Finset (Fin n)) (ord : List (Fin n))
    (π : List (Fin n)) (O S : Finset (Fin n)) : Prop where
  rep : Rep G0 G y S
  sub : S ⊆ W
  cand : ∀ v ∈ π, v ∈ muct G0 y S ∧ v ≠ y
  cover : ∀ v ∈ W, v ∈ muct G0 y S → v ∈ π
  excl : ∀ v ∈ O, v ∉ W ∧ v ∈ muct G0 y ∅
  sorted : π.Pairwise (fun a b => ord.idxOf a < ord.idxOf b)
  before : ∀ o ∈ O, ∀ c ∈ π, ord.idxOf o < ord.idxOf c

/-- The branch of the earliest target pivot `Q` is not pruned: `IB(G_{\overline{S ∪ {Q}}}, Y)`
misses `O ∪ π^{1:i-1}` (App. C, proof of Thm. 9). -/
theorem prune_ok {G0 G : ADMG n} {y : Fin n} {W TW : Finset (Fin n)} {ord : List (Fin n)}
    (hW : DiagMUCT G0 y W TW) (hB : diagBorder G0 TW = W)
    (hord : ∀ a ∈ muct G0 y ∅, ∀ b ∈ muct G0 y ∅, G0.dir a b = true →
      ord.idxOf b < ord.idxOf a)
    {π pre post : List (Fin n)} {Q : Fin n} {O S : Finset (Fin n)}
    (hI : Inv G0 G y W ord π O S) (hsplit : π = pre ++ Q :: post) (hQW : Q ∈ W)
    (hpre : ∀ v ∈ pre, ¬ (v ∈ W ∧ v ∈ muct G0 y S))
    (hpost : ∀ v ∈ W, v ∈ muct G0 y (S ∪ {Q}) → v ∈ post) :
    ib G0 y (S ∪ {Q}) ∩ (O ∪ pre.toFinset) = ∅ := by
  have hyW : y ∉ W := y_not_mem_of_diagMUCT hW
  have hS'W : S ∪ {Q} ⊆ W := Finset.union_subset hI.sub (Finset.singleton_subset_iff.mpr hQW)
  have hyS' : y ∉ S ∪ {Q} := fun h => hyW (hS'W h)
  have hT'0 : muct G0 y (S ∪ {Q}) ⊆ muct G0 y ∅ := muct_anti (Finset.empty_subset _)
  have hTS0 : muct G0 y S ⊆ muct G0 y ∅ := muct_anti (Finset.empty_subset _)
  have hsorted := hI.sorted
  rw [hsplit, List.pairwise_append, List.pairwise_cons] at hsorted
  obtain ⟨-, ⟨hQpost, -⟩, hprepost⟩ := hsorted
  have hQπ : Q ∈ π := by rw [hsplit]; simp
  rw [Finset.eq_empty_iff_forall_notMem]
  intro R hR
  rw [Finset.mem_inter, Finset.mem_union, List.mem_toFinset] at hR
  obtain ⟨hRB, hRO⟩ := hR
  have hRfacts : R ∉ W ∧ R ∈ muct G0 y ∅ ∧ ord.idxOf R < ord.idxOf Q := by
    rcases hRO with hRO | hRpre
    · exact ⟨(hI.excl R hRO).1, (hI.excl R hRO).2, hI.before R hRO Q hQπ⟩
    · have hRπ : R ∈ π := by rw [hsplit]; simp [hRpre]
      have hRT := (hI.cand R hRπ).1
      exact ⟨fun h => hpre R hRpre ⟨h, hRT⟩, hTS0 hRT, hprepost R hRpre Q (by simp)⟩
  obtain ⟨hRW, hR0, hRQ⟩ := hRfacts
  have hT' : DiagMUCT G0 y (S ∪ {Q}) (muct G0 y (S ∪ {Q})) := diagMUCT_muct hyS'
  obtain ⟨hRT', t, ht, hRt⟩ := mem_diagBorder.mp hRB
  have hlate : ∀ u ∈ muct G0 y (S ∪ {Q}), u ∈ W → ord.idxOf Q < ord.idxOf u :=
    fun u hu huW => hQpost u (hpost u huW hu)
  have htR : ord.idxOf t < ord.idxOf R := hord R hR0 t (hT'0 ht) hRt
  have key : ∀ u, Reach G0.dir (Finset.univ \ muct G0 y (S ∪ {Q})) t u →
      u ∈ muct G0 y (S ∪ {Q}) ∧ ord.idxOf u < ord.idxOf R ∧ Reach G0.dir W t u := by
    intro u hu
    induction hu with
    | refl => exact ⟨ht, htR, Reach.refl _⟩
    | @tail b c _ hbc ih =>
      obtain ⟨hbT, hbR, hr⟩ := ih
      have hcT : c ∈ muct G0 y (S ∪ {Q}) := by simpa using hbc.2
      have hcb : ord.idxOf c < ord.idxOf b := hord b (hT'0 hbT) c (hT'0 hcT) hbc.1
      refine ⟨hcT, by omega, Reach.tail hr hbc.1 (fun hcW => ?_)⟩
      have := hlate c hcT hcW
      omega
  have htW : t ∉ W := fun htW => by
    have := hlate t ht htW
    omega
  have hpath : G0.AncAvoid W R y :=
    Reach.head hRt htW (key y (reach_in_terr hT'.1 ht)).2.2
  have hTW : TW ⊆ muct G0 y (S ∪ {Q}) := by
    rw [eq_muct hW]
    exact muct_anti hS'W
  have hRTW : R ∈ TW := by
    refine diag_mem_of_avoid_border hW.1 (by rw [hB]; exact hRW) ?_
    rw [hB]
    exact hpath
  exact hRT' (hTW hRTW)

theorem complete_aux {G0 : ADMG n} {y : Fin n} {W TW : Finset (Fin n)} {ord : List (Fin n)}
    (hW : DiagMUCT G0 y W TW) (hB : diagBorder G0 TW = W)
    (hord : ∀ a ∈ muct G0 y ∅, ∀ b ∈ muct G0 y ∅, G0.dir a b = true →
      ord.idxOf b < ord.idxOf a) :
    ∀ (k : ℕ) (G : ADMG n) (π : List (Fin n)) (O S : Finset (Fin n)), π.length ≤ k →
      Inv G0 G y W ord π O S → (∃ v ∈ W, v ∈ muct G0 y S) →
        W ∈ subPOMISsAux y k G π O := by
  have hyW : y ∉ W := y_not_mem_of_diagMUCT hW
  intro k
  induction k with
  | zero =>
    intro G π O S hlen hI hex
    obtain ⟨v, hvW, hvT⟩ := hex
    have hvπ := hI.cover v hvW hvT
    rw [List.length_eq_zero_iff.mp (Nat.le_zero.mp hlen)] at hvπ
    simp at hvπ
  | succ k ih =>
    intro G π O S hlen hI hex
    obtain ⟨pre, Q, post, hsplit, ⟨hQW, hQT⟩, hpre⟩ :=
      exists_first (fun v => v ∈ W ∧ v ∈ muct G0 y S) π
        (by
          obtain ⟨v, hvW, hvT⟩ := hex
          exact ⟨v, hI.cover v hvW hvT, hvW, hvT⟩)
    have hQπ : Q ∈ π := by rw [hsplit]; simp
    have hQy : Q ≠ y := (hI.cand Q hQπ).2
    have hyQ : y ∉ ({Q} : Finset (Fin n)) := fun h => hQy (Finset.mem_singleton.mp h).symm
    have hS'W : S ∪ {Q} ⊆ W :=
      Finset.union_subset hI.sub (Finset.singleton_subset_iff.mpr hQW)
    have hyS' : y ∉ S ∪ {Q} := fun h => hyW (hS'W h)
    have hT'S : muct G0 y (S ∪ {Q}) ⊆ muct G0 y S := muct_anti Finset.subset_union_left
    have hTS0 : muct G0 y S ⊆ muct G0 y ∅ := muct_anti (Finset.empty_subset _)
    have hQT' : Q ∉ muct G0 y (S ∪ {Q}) := fun h =>
      (mem_diagH.mp (muct_subset_diagH hyS' h)).1
        (Finset.mem_union_right _ (Finset.mem_singleton_self Q))
    have hpre' : ∀ v ∈ pre, ¬ (v ∈ W ∧ v ∈ muct G0 y S) := hpre
    have hpost : ∀ v ∈ W, v ∈ muct G0 y (S ∪ {Q}) → v ∈ post := by
      intro v hvW hvT'
      have hvπ := hI.cover v hvW (hT'S hvT')
      rw [hsplit, List.mem_append, List.mem_cons] at hvπ
      rcases hvπ with hv | hv | hv
      · exact absurd ⟨hvW, hT'S hvT'⟩ (hpre' v hv)
      · rw [hv] at hvT'
        exact absurd hvT' hQT'
      · exact hv
    have hprune := prune_ok hW hB hord hI hsplit hQW hpre' hpost
    have hsorted := hI.sorted
    rw [hsplit, List.pairwise_append, List.pairwise_cons] at hsorted
    obtain ⟨-, ⟨-, hpostsorted⟩, hprepost⟩ := hsorted
    obtain ⟨hmQ, hiQ⟩ := hI.rep {Q}
    have hR' : Rep G0 (compress G (muct G y {Q}) (ib G y {Q})) y (S ∪ {Q}) :=
      hI.rep.trans (rep_compress (diagMUCT_muct hyQ))
    rw [hmQ, hiQ] at hR'
    simp only [subPOMISsAux]
    rw [mem_forLoop]
    refine ⟨pre, Q, post, hsplit, ?_⟩
    unfold pomisBody pomisStep
    rw [hmQ, hiQ, ite_eq_left hprune]
    by_cases hcase : ∃ v ∈ W, v ∈ muct G0 y (S ∪ {Q})
    · obtain ⟨v, hvW, hvT'⟩ := hcase
      have hne : post.filter (fun v => decide (v ∈ muct G0 y (S ∪ {Q}))) ≠ [] :=
        List.ne_nil_of_mem (List.mem_filter.mpr ⟨hpost v hvW hvT', decide_eq_true hvT'⟩)
      rw [ite_eq_right hne]
      apply List.mem_cons_of_mem
      have hlen' : (post.filter (fun v => decide (v ∈ muct G0 y (S ∪ {Q})))).length ≤ k := by
        have h1 := List.length_filter_le (fun v => decide (v ∈ muct G0 y (S ∪ {Q}))) post
        have h2 : π.length = pre.length + (post.length + 1) := by
          rw [hsplit, List.length_append, List.length_cons]
        omega
      refine ih _ _ _ _ hlen' ⟨hR', hS'W, ?_, ?_, ?_, hpostsorted.filter _, ?_⟩
        ⟨v, hvW, hvT'⟩
      · intro u hu
        simp only [List.mem_filter, decide_eq_true_eq] at hu
        refine ⟨hu.2, (hI.cand u ?_).2⟩
        rw [hsplit]
        simp [hu.1]
      · intro u huW huT
        exact List.mem_filter.mpr ⟨hpost u huW huT, decide_eq_true huT⟩
      · intro u hu
        rw [Finset.mem_union, List.mem_toFinset] at hu
        rcases hu with hu | hu
        · exact hI.excl u hu
        · have huπ : u ∈ π := by rw [hsplit]; simp [hu]
          have huT := (hI.cand u huπ).1
          exact ⟨fun h => hpre' u hu ⟨h, huT⟩, hTS0 huT⟩
      · intro o ho c hc
        have hcpost := (List.mem_filter.mp hc).1
        rw [Finset.mem_union, List.mem_toFinset] at ho
        rcases ho with ho | ho
        · exact hI.before o ho c (by rw [hsplit]; simp [hcpost])
        · exact hprepost o ho c (by simp [hcpost])
    · apply List.mem_cons.mpr
      left
      have hdisj : Disjoint W (muct G0 y (S ∪ {Q})) :=
        Finset.disjoint_left.mpr (fun v hvW hvT => hcase ⟨v, hvW, hvT⟩)
      have h1 := diagMUCT_union (diagMUCT_muct (G := G0) hyS') hdisj
      rw [Finset.union_eq_right.mpr hS'W] at h1
      have h2 : muct G0 y (S ∪ {Q}) = TW := by rw [eq_muct h1, ← eq_muct hW]
      unfold ib
      rw [h2, hB]

/-! ## Theorem 9 -/

/-- `ord` is a reverse topological order of `H` on the vertex set `S`: it repeats no vertex,
lists every vertex of `S`, and lists every vertex of `S` after its children in `H[S]`. -/
def RevTopo (H : ADMG n) (S : Finset (Fin n)) (ord : List (Fin n)) : Prop :=
  ord.Nodup ∧ (∀ v ∈ S, v ∈ ord) ∧
    ∀ a ∈ S, ∀ b ∈ S, H.dir a b = true → ord.idxOf b < ord.idxOf a

instance (H : ADMG n) (S : Finset (Fin n)) (ord : List (Fin n)) :
    Decidable (RevTopo H S ord) := by
  unfold RevTopo
  infer_instance

theorem pairwise_idxOf :
    ∀ l : List (Fin n), l.Nodup → l.Pairwise (fun a b => l.idxOf a < l.idxOf b)
  | [], _ => List.Pairwise.nil
  | a :: l, h => by
    rw [List.nodup_cons] at h
    rw [List.pairwise_cons]
    refine ⟨fun b hb => ?_, (pairwise_idxOf l h.2).imp_of_mem (fun {b c} hb hc hbc => ?_)⟩
    · have hab : a ≠ b := fun e => h.1 (e ▸ hb)
      rw [List.idxOf_cons, List.idxOf_cons]
      simp [hab]
    · have hab : a ≠ b := fun e => h.1 (e ▸ hb)
      have hac : a ≠ c := fun e => h.1 (e ▸ hc)
      have hbc' : l.idxOf b < l.idxOf c := hbc
      rw [List.idxOf_cons, List.idxOf_cons]
      simp [hab, hac]
      omega

/-- The hypothesis of the pseudocode, a reverse topological order of `H = G_{\overline X}[T ∪ X]`
on its vertices, gives a reverse topological order of `G` on `T = MUCT(G, Y)`. -/
theorem revTopo_of_compress {G : ADMG n} {y : Fin n} {ord : List (Fin n)}
    (h : RevTopo (compress G (muct G y ∅) (ib G y ∅)) (muct G y ∅ ∪ ib G y ∅) ord) :
    RevTopo G (muct G y ∅) ord := by
  refine ⟨h.1, fun v hv => h.2.1 v (Finset.mem_union_left _ hv), fun a ha b hb hab => ?_⟩
  exact h.2.2 a (Finset.mem_union_left _ ha) b (Finset.mem_union_left _ hb)
    ((compress_dir_iff ha hb).mpr hab)

end Alg

open Alg

/-- **Thm. 9**, graphical form: `POMISs` returns exactly the sets `X` with
`IB(G_{\overline X}, Y) = X`. -/
theorem thm9_graphical {G : ADMG n} {y : Fin n} {ord : List (Fin n)}
    (hord : RevTopo G (muct G y ∅) ord) (X : Finset (Fin n)) :
    X ∈ POMISs G y ord ↔ ∃ T, DiagMUCT G y X T ∧ diagBorder G T = X := by
  have hy0 : y ∉ (∅ : Finset (Fin n)) := Finset.notMem_empty y
  have hR0 : Rep G (compress G (muct G y ∅) (ib G y ∅)) y ∅ := rep_compress (diagMUCT_muct hy0)
  constructor
  · intro hX
    have hS : ∃ S : Finset (Fin n), y ∉ S ∧ X = ib G y S := by
      rcases List.mem_cons.mp hX with hX | hX
      · exact ⟨∅, hy0, hX⟩
      · refine sound_aux _ _ _ _ _ hR0 hy0 (fun v hv => ?_) hX
        simp only [List.mem_filter, decide_eq_true_eq] at hv
        exact hv.2.2
    obtain ⟨S, hyS, hXS⟩ := hS
    rw [hXS]
    exact ⟨muct G y S, (ib_self hyS).1, (ib_self hyS).2⟩
  · rintro ⟨T, hT, hB⟩
    have hyX : y ∉ X := y_not_mem_of_diagMUCT hT
    by_cases hcase : ∃ v ∈ X, v ∈ muct G y ∅
    · apply List.mem_cons_of_mem
      refine complete_aux hT hB hord.2.2 _ _ _ _ _ le_rfl
        ⟨hR0, Finset.empty_subset _, ?_, ?_, ?_, (pairwise_idxOf ord hord.1).filter _, ?_⟩ hcase
      · intro v hv
        simp only [List.mem_filter, decide_eq_true_eq] at hv
        exact hv.2
      · intro v hvX hvT
        exact List.mem_filter.mpr ⟨hord.2.1 v hvT, decide_eq_true ⟨hvT, fun h => hyX (h ▸ hvX)⟩⟩
      · intro v hv
        simp at hv
      · intro o ho
        simp at ho
    · apply List.mem_cons.mpr
      left
      have hdisj : Disjoint X (muct G y ∅) :=
        Finset.disjoint_left.mpr (fun v hvX hvT => hcase ⟨v, hvX, hvT⟩)
      have h1 := diagMUCT_union (diagMUCT_muct (G := G) hy0) hdisj
      rw [Finset.empty_union] at h1
      have h2 : muct G y ∅ = T := by rw [eq_muct h1, ← eq_muct hT]
      unfold ib
      rw [h2, hB]

/-- **Thm. 9** (soundness and completeness of `POMISs`): for a well-formed causal diagram `G`
and a reverse topological order `ord` of `G` on `MUCT(G, Y)`, `POMISs` returns all, and only,
the POMISs. -/
theorem thm9 {G : ADMG n} (hG : G.WF) {y : Fin n} {ord : List (Fin n)}
    (hord : RevTopo G (muct G y ∅) ord) (X : Finset (Fin n)) :
    X ∈ POMISs G y ord ↔ POMIS (ConfDiag G) y X := by
  rw [thm6 hG]
  exact thm9_graphical hord X

/-- **Thm. 9** under the hypothesis of the pseudocode: `ord` is a reverse topological order of
`H = G_{\overline X}[T ∪ X]` on its vertices `T ∪ X`, for `T = MUCT(G, Y)`, `X = IB(G, Y)`. -/
theorem thm9_of_revTopo_compress {G : ADMG n} (hG : G.WF) {y : Fin n} {ord : List (Fin n)}
    (hord : RevTopo (compress G (muct G y ∅) (ib G y ∅)) (muct G y ∅ ∪ ib G y ∅) ord)
    (X : Finset (Fin n)) : X ∈ POMISs G y ord ↔ POMIS (ConfDiag G) y X :=
  thm9 hG (revTopo_of_compress hord) X

end Mec.Scb2018
