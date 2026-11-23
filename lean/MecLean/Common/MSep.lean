import MecLean.Common.Graph

set_option linter.unusedSectionVars false

/-!
# m-connecting paths in ADMGs

A *route* from a start vertex is a list of steps `(kind, next vertex)`; the kind records which
edge is used (`fwd`: `a → b`, `bwd`: `a ← b`, `bi`: `a ↔ b`), so a route is a path even when two
vertices are joined by both a directed and a bidirected edge.  A path is m-connecting given `S`
when its vertices are distinct, every internal collider is an ancestor of a vertex of `S`, every
internal non-collider lies outside `S`, and the endpoints lie outside `S`.

`routes G k vis a` enumerates the valid routes by depth-first search; `mconn_iff` proves that it
finds every simple route, which makes `MConn` decidable by kernel reduction.
-/

open Finset

namespace Mec

variable {n : ℕ}

/-- The kind of edge used by a step. -/
inductive EK
  | fwd
  | bwd
  | bi
  deriving DecidableEq, Repr

namespace EK

/-- The step from `a` to `b` uses an edge of `G`. -/
def ok (G : ADMG n) (a b : Fin n) : EK → Bool
  | fwd => G.dir a b
  | bwd => G.dir b a
  | bi => G.bi a b

/-- Arrowhead at the far endpoint of the step. -/
def headFar : EK → Bool
  | fwd => true
  | bwd => false
  | bi => true

/-- Arrowhead at the near endpoint of the step. -/
def headNear : EK → Bool
  | fwd => false
  | bwd => true
  | bi => true

end EK

/-- A route: steps `(edge kind, next vertex)` from an implicit start vertex. -/
abbrev Route (n : ℕ) := List (EK × Fin n)

namespace Route

/-- The vertices visited, starting with the start vertex. -/
def verts : Fin n → Route n → List (Fin n)
  | a, [] => [a]
  | a, (_, v) :: r => a :: verts v r

/-- The final vertex. -/
def last : Fin n → Route n → Fin n
  | a, [] => a
  | _, (_, v) :: r => last v r

/-- Every step uses an edge of `G`. -/
def valid (G : ADMG n) : Fin n → Route n → Bool
  | _, [] => true
  | a, (k, v) :: r => k.ok G a v && valid G v r

/-- Conditions at the internal vertices from `v` on; `h` records an arrowhead at `v` on the step
into `v`.  Colliders must satisfy `A`, non-colliders must avoid `C`. -/
def active (C A : Fin n → Bool) : Bool → Fin n → Route n → Bool
  | _, _, [] => true
  | h, v, (k, w) :: r => (if h && k.headNear then A v else !C v) && active C A k.headFar w r

/-- Conditions at all internal vertices of a route. -/
def activeFrom (C A : Fin n → Bool) : Route n → Bool
  | [] => true
  | (k, v) :: r => active C A k.headFar v r

theorem verts_eq (a : Fin n) (p : Route n) : verts a p = a :: p.map Prod.snd := by
  induction p generalizing a with
  | nil => rfl
  | cons s r ih =>
    obtain ⟨k, v⟩ := s
    simp [verts, ih]

theorem length_lt {a : Fin n} {p : Route n} (h : (verts a p).Nodup) : p.length < n := by
  rw [verts_eq] at h
  have := h.length_le_card
  simp only [List.length_cons, List.length_map, Fintype.card_fin] at this
  omega

theorem last_mem : ∀ (a : Fin n) (p : Route n), p ≠ [] → last a p ∈ p.map Prod.snd
  | _, [], h => absurd rfl h
  | _, [(_, v)], _ => by simp [last]
  | _, (_, v) :: (k', w) :: r, _ =>
    List.mem_cons_of_mem v (last_mem v ((k', w) :: r) (by simp))

end Route

/-- `MConn G S a b`: some path between `a` and `b` is m-connecting given `S`. -/
def MConn (G : ADMG n) (S : Finset (Fin n)) (a b : Fin n) : Prop :=
  a ∉ S ∧ b ∉ S ∧ ∃ p : Route n, p ≠ [] ∧ (Route.verts a p).Nodup ∧ Route.valid G a p = true ∧
    Route.last a p = b ∧
    Route.activeFrom (fun v => decide (v ∈ S)) (fun v => decide (∃ s ∈ S, G.Anc v s)) p = true

/-- Valid simple routes of at most `k` steps from `a` avoiding the visited vertices `vis`. -/
def routes (G : ADMG n) : ℕ → List (Fin n) → Fin n → List (Route n)
  | 0, _, _ => [[]]
  | k + 1, vis, a => [] :: (List.finRange n).flatMap (fun v =>
      if v ∈ vis ∨ v = a then [] else
        [EK.fwd, EK.bwd, EK.bi].flatMap (fun e =>
          if e.ok G a v = true then (routes G k (a :: vis) v).map (fun r => (e, v) :: r)
          else []))

theorem mem_routes (G : ADMG n) : ∀ (k : ℕ) (vis : List (Fin n)) (a : Fin n) (p : Route n),
    p.length ≤ k → Route.valid G a p = true → (a :: p.map Prod.snd).Nodup →
    (∀ v ∈ p.map Prod.snd, v ∉ vis) → p ∈ routes G k vis a := by
  intro k
  induction k with
  | zero =>
    intro vis a p hlen _ _ _
    have hp : p = [] := List.eq_nil_of_length_eq_zero (Nat.le_zero.mp hlen)
    subst hp
    simp [routes]
  | succ k ih =>
    intro vis a p hlen hval hnd hvis
    cases p with
    | nil => simp [routes]
    | cons s r =>
      obtain ⟨e, v⟩ := s
      simp only [Route.valid, Bool.and_eq_true] at hval
      simp only [List.map_cons, List.nodup_cons, List.mem_cons, not_or] at hnd
      obtain ⟨⟨hav, har⟩, hvr, hnd'⟩ := hnd
      have hvvis : v ∉ vis := hvis v (by simp)
      have hcond : ¬ (v ∈ vis ∨ v = a) := by
        rintro (h | h)
        · exact hvvis h
        · exact hav h.symm
      have hr : r ∈ routes G k (a :: vis) v := by
        refine ih (a :: vis) v r (by simpa using hlen) hval.2 (List.nodup_cons.mpr ⟨hvr, hnd'⟩) ?_
        intro w hw
        simp only [List.mem_cons, not_or]
        refine ⟨fun h => har (h ▸ hw), hvis w (by simp [hw])⟩
      have key : (e, v) :: r ∈ (List.finRange n).flatMap (fun w =>
          if w ∈ vis ∨ w = a then [] else
            [EK.fwd, EK.bwd, EK.bi].flatMap (fun e' =>
              if e'.ok G a w = true then (routes G k (a :: vis) w).map (fun r => (e', w) :: r)
              else [])) := by
        rw [List.mem_flatMap]
        refine ⟨v, List.mem_finRange v, ?_⟩
        rw [if_neg hcond, List.mem_flatMap]
        refine ⟨e, by cases e <;> simp, ?_⟩
        rw [if_pos hval.1]
        exact List.mem_map.mpr ⟨r, hr, rfl⟩
      exact List.mem_cons_of_mem _ key

theorem mconn_iff (G : ADMG n) (S : Finset (Fin n)) (a b : Fin n) :
    MConn G S a b ↔ a ∉ S ∧ b ∉ S ∧ ∃ p ∈ routes G n [] a, p ≠ [] ∧ (Route.verts a p).Nodup ∧
      Route.valid G a p = true ∧ Route.last a p = b ∧
      Route.activeFrom (fun v => decide (v ∈ S)) (fun v => decide (∃ s ∈ S, G.Anc v s)) p = true := by
  constructor
  · rintro ⟨ha, hb, p, hp, hnd, hval, hlast, hact⟩
    refine ⟨ha, hb, p, ?_, hp, hnd, hval, hlast, hact⟩
    have hnd' := hnd
    rw [Route.verts_eq] at hnd'
    exact mem_routes G n [] a p (Route.length_lt hnd).le hval hnd' (by simp)
  · rintro ⟨ha, hb, p, -, hp, hnd, hval, hlast, hact⟩
    exact ⟨ha, hb, p, hp, hnd, hval, hlast, hact⟩

instance decMConn (G : ADMG n) (S : Finset (Fin n)) (a b : Fin n) : Decidable (MConn G S a b) :=
  decidable_of_iff _ (mconn_iff G S a b).symm

/-- A single edge is m-connecting given any set avoiding its endpoints. -/
theorem mconn_of_edge (G : ADMG n) {S : Finset (Fin n)} {a b : Fin n} (hab : a ≠ b)
    (ha : a ∉ S) (hb : b ∉ S) (e : EK) (he : e.ok G a b = true) : MConn G S a b := by
  refine ⟨ha, hb, [(e, b)], by simp, ?_, ?_, rfl, ?_⟩
  · simp [Route.verts, hab]
  · simp [Route.valid, he]
  · simp [Route.activeFrom, Route.active]

/-- A two-edge path `a – c – b` is m-connecting when its middle vertex is active. -/
theorem mconn_of_two (G : ADMG n) {S : Finset (Fin n)} {a c b : Fin n} (hac : a ≠ c)
    (hab : a ≠ b) (hcb : c ≠ b) (ha : a ∉ S) (hb : b ∉ S) (e₁ e₂ : EK)
    (h₁ : e₁.ok G a c = true) (h₂ : e₂.ok G c b = true)
    (hact : (if e₁.headFar && e₂.headNear then decide (∃ s ∈ S, G.Anc c s)
      else !decide (c ∈ S)) = true) : MConn G S a b := by
  refine ⟨ha, hb, [(e₁, c), (e₂, b)], by simp, ?_, ?_, rfl, ?_⟩
  · simp [Route.verts, hac, hab, hcb]
  · simp [Route.valid, h₁, h₂]
  · simp only [Route.activeFrom, Route.active, Bool.and_true]
    exact hact

/-- No m-connecting path joins a vertex to itself. -/
theorem not_mconn_self (G : ADMG n) (S : Finset (Fin n)) (a : Fin n) : ¬ MConn G S a a := by
  rintro ⟨_, _, p, hp, hnd, _, hlast, _⟩
  rw [Route.verts_eq] at hnd
  have hmem := Route.last_mem a p hp
  rw [hlast] at hmem
  exact (List.nodup_cons.mp hnd).1 hmem

end Mec
