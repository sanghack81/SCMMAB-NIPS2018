import MecLean.Common.SCM

set_option linter.unusedSectionVars false

/-!
# Bundled finite models and realizations of ADMGs

* `Model n`: a finite SCM together with its domains and exogenous spaces, so that statements can
  quantify over "every SCM" or assert "some SCM" with domains of any finite size.
* `Model.mu`, `Model.best`: the mean reward `E[r(Y) | do(S = x)]` and its maximum over `x`.
* `Mech.ofADMG`: an SCM whose causal diagram is a prescribed acyclic ADMG, with one private root
  per vertex and one root per ordered pair of vertices joined by a bidirected edge.  Every MAG is
  therefore the induced MAG of some SCM (`ADMG.mag_self`).
-/

open Finset

namespace Mec

variable {n : ℕ}

/-- A finite SCM bundled with its domains and exogenous spaces. -/
structure Model (n : ℕ) where
  X : Fin n → Type
  L : Type
  U : L → Type
  [fX : ∀ i, Fintype (X i)]
  [dX : ∀ i, DecidableEq (X i)]
  [iX : ∀ i, Inhabited (X i)]
  [fL : Fintype L]
  [dL : DecidableEq L]
  [fU : ∀ l, Fintype (U l)]
  scm : SCM n X L U

attribute [instance] Model.fX Model.dX Model.iX Model.fL Model.dL Model.fU

namespace Model

variable (m : Model n)

/-- Mean reward `E[r(Y) | do(S = x)]`. -/
noncomputable def mu (y : Fin n) (r : m.X y → ℝ) (S : Finset (Fin n)) (x : ∀ i, m.X i) : ℝ :=
  m.scm.E (fun u => r (m.scm.toMech.resp S x u y))

/-- Best mean reward among the arms of `S`. -/
noncomputable def best (y : Fin n) (r : m.X y → ℝ) (S : Finset (Fin n)) : ℝ :=
  Finset.univ.sup' Finset.univ_nonempty (fun x : (∀ i, m.X i) => m.mu y r S x)

theorem mu_le_best (y : Fin n) (r : m.X y → ℝ) (S : Finset (Fin n)) (x : ∀ i, m.X i) :
    m.mu y r S x ≤ m.best y r S :=
  Finset.le_sup' (fun x : (∀ i, m.X i) => m.mu y r S x) (Finset.mem_univ x)

theorem exists_best (y : Fin n) (r : m.X y → ℝ) (S : Finset (Fin n)) :
    ∃ x, m.mu y r S x = m.best y r S := by
  obtain ⟨x, -, hx⟩ := Finset.exists_mem_eq_sup' Finset.univ_nonempty
    (fun x : (∀ i, m.X i) => m.mu y r S x)
  exact ⟨x, hx.symm⟩

/-- Equal arm means give equal best values. -/
theorem best_congr {y : Fin n} {r : m.X y → ℝ} {S T : Finset (Fin n)}
    (h : ∀ x, m.mu y r S x = m.mu y r T x) : m.best y r S = m.best y r T := by
  unfold best
  congr 1
  funext x
  exact h x

/-- Pointwise equal responses give equal means. -/
theorem mu_congr {y : Fin n} {r : m.X y → ℝ} {S T : Finset (Fin n)} {x x' : ∀ i, m.X i}
    (h : ∀ u, m.scm.toMech.resp S x u y = m.scm.toMech.resp T x' u y) : m.mu y r S x = m.mu y r T x' := by
  unfold mu
  exact m.scm.E_congr (fun u => by rw [h u])

end Model

/-! ## Realizing an acyclic ADMG -/

/-- Exogenous roots of a realization: a private root per vertex, a root per ordered pair. -/
abbrev Roots (n : ℕ) := Fin n ⊕ (Fin n × Fin n)

/-- The roots read by vertex `i`. -/
def rootScope (G : ADMG n) (i : Fin n) : Finset (Roots n) :=
  insert (Sum.inl i) ((Finset.univ.filter (fun p : Fin n × Fin n =>
    G.bi p.1 p.2 = true ∧ (p.1 = i ∨ p.2 = i))).map ⟨Sum.inr, Sum.inr_injective⟩)

/-- Number of proper ancestors: a topological rank of an acyclic ADMG. -/
def topRank (G : ADMG n) (i : Fin n) : ℕ :=
  (Finset.univ.filter (fun j => j ≠ i ∧ G.Anc j i)).card

theorem topRank_lt {G : ADMG n} (hG : G.WF) {i j : Fin n} (h : G.dir j i = true) :
    topRank G j < topRank G i := by
  unfold topRank
  apply Finset.card_lt_card
  have hji : j ≠ i := fun e => by rw [e, hG.dir_irrefl] at h; exact Bool.false_ne_true h
  refine (Finset.ssubset_iff_of_subset ?_).mpr ⟨j, ?_, ?_⟩
  · intro k hk
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hk ⊢
    obtain ⟨hkj, hk⟩ := hk
    refine ⟨fun hki => ?_, Reach.tail hk h (Finset.notMem_empty i)⟩
    subst hki
    exact hG.acyclic h hk
  · simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact ⟨hji, ADMG.anc_of_dir h⟩
  · simp

/-- An SCM skeleton whose causal diagram is `G`. -/
def Mech.ofADMG (G : ADMG n) (hG : G.WF) {X : Fin n → Type} {U : Roots n → Type}
    (f : ∀ i, (∀ j, X j) → (∀ l, U l) → X i)
    (f_pa : ∀ i (x x' : ∀ j, X j) u, (∀ j, G.dir j i = true → x j = x' j) → f i x u = f i x' u)
    (f_ex : ∀ i x (u u' : ∀ l, U l), (∀ l ∈ rootScope G i, u l = u' l) → f i x u = f i x u') :
    Mech n X (Roots n) U where
  pa i := Finset.univ.filter (fun j => G.dir j i = true)
  ex i := rootScope G i
  rank := topRank G
  pa_rank := fun {i j} h => topRank_lt hG (by simpa using h)
  f := f
  f_pa := fun i x x' u h => f_pa i x x' u (fun j hj => h j (by simpa using hj))
  f_ex := f_ex

theorem mem_rootScope {G : ADMG n} {i : Fin n} {l : Roots n} :
    l ∈ rootScope G i ↔
      l = Sum.inl i ∨ ∃ a b, l = Sum.inr (a, b) ∧ G.bi a b = true ∧ (a = i ∨ b = i) := by
  unfold rootScope
  constructor
  · intro h
    rw [Finset.mem_insert, Finset.mem_map] at h
    rcases h with h | ⟨⟨a, b⟩, hab, rfl⟩
    · exact Or.inl h
    · simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hab
      exact Or.inr ⟨a, b, rfl, hab⟩
  · rintro (h | ⟨a, b, rfl, hab⟩)
    · rw [h]
      exact Finset.mem_insert_self _ _
    · rw [Finset.mem_insert, Finset.mem_map]
      right
      exact ⟨(a, b), by simpa using hab, rfl⟩

theorem rootScope_inter {G : ADMG n} (hG : G.WF) {i j : Fin n} (hij : i ≠ j) :
    (rootScope G i ∩ rootScope G j).Nonempty ↔ G.bi i j = true := by
  constructor
  · rintro ⟨l, hl⟩
    rw [Finset.mem_inter, mem_rootScope, mem_rootScope] at hl
    obtain ⟨hli, hlj⟩ := hl
    rcases hli with hli | ⟨a, b, hl, hab, hai⟩
    · rcases hlj with hlj | ⟨a, b, hl', -, -⟩
      · rw [hli] at hlj
        exact absurd (Sum.inl_injective hlj) hij
      · rw [hli] at hl'
        exact absurd hl' (by simp)
    · rcases hlj with hlj | ⟨a', b', hl', -, hbj⟩
      · rw [hl] at hlj
        exact absurd hlj (by simp)
      · rw [hl] at hl'
        simp only [Sum.inr.injEq, Prod.mk.injEq] at hl'
        obtain ⟨h1, h2⟩ := hl'
        subst h1 h2
        rcases hai with hai | hai <;> rcases hbj with hbj | hbj
        · exact absurd (hai.symm.trans hbj) hij
        · rw [← hai, ← hbj]
          exact hab
        · rw [← hai, ← hbj, hG.bi_symm]
          exact hab
        · exact absurd (hai.symm.trans hbj) hij
  · intro hb
    refine ⟨Sum.inr (i, j), Finset.mem_inter.mpr ⟨?_, ?_⟩⟩
    · rw [mem_rootScope]
      exact Or.inr ⟨i, j, rfl, hb, Or.inl rfl⟩
    · rw [mem_rootScope]
      exact Or.inr ⟨i, j, rfl, hb, Or.inr rfl⟩

theorem Mech.ofADMG_diag (G : ADMG n) (hG : G.WF) {X : Fin n → Type} {U : Roots n → Type}
    (f : ∀ i, (∀ j, X j) → (∀ l, U l) → X i) (f_pa) (f_ex) :
    (Mech.ofADMG G hG f f_pa f_ex).diag = G := by
  rw [ADMG.ext_iff']
  constructor
  · funext j i
    simp [Mech.diag, Mech.ofADMG]
  · funext i j
    show decide (i ≠ j ∧ (rootScope G i ∩ rootScope G j).Nonempty) = G.bi i j
    by_cases hij : i = j
    · subst hij
      simp [hG.bi_irrefl]
    · rw [Bool.eq_iff_iff, decide_eq_true_iff]
      exact ⟨fun h => (rootScope_inter hG hij).mp h.2,
        fun h => ⟨hij, (rootScope_inter hG hij).mpr h⟩⟩

/-- A deterministic model: every root is `Unit` with probability one. -/
noncomputable def Model.ofDet {X : Fin n → Type} [∀ i, Fintype (X i)] [∀ i, DecidableEq (X i)]
    [∀ i, Inhabited (X i)] (M : Mech n X (Roots n) (fun _ => Unit)) : Model n where
  X := X
  L := Roots n
  U := fun _ => Unit
  scm := { M with
    p := fun _ _ => 1
    p_nonneg := fun _ _ => zero_le_one
    p_sum := fun _ => by simp }

theorem Model.ofDet_mu {X : Fin n → Type} [∀ i, Fintype (X i)] [∀ i, DecidableEq (X i)]
    [∀ i, Inhabited (X i)] (M : Mech n X (Roots n) (fun _ => Unit)) (y : Fin n) (r : X y → ℝ)
    (S : Finset (Fin n)) (x : ∀ i, X i) :
    (Model.ofDet M).mu y r S x = r (M.resp S x (fun _ : Roots n => ()) y) := by
  have hs : Subsingleton (∀ l : Roots n, Unit) := ⟨fun a b => funext (fun _ => rfl)⟩
  unfold Model.mu
  exact SCM.E_of_subsingleton (Model.ofDet M).scm (hsub := hs) _ (fun _ : Roots n => ())

/-- `Mech.resp_core` for the mean reward of a bundled model. -/
theorem Model.mu_eq_of_core (m : Model n) {y : Fin n} (r : m.X y → ℝ) {S T : Finset (Fin n)}
    (hTS : T ⊆ S) (hT : ∀ z ∈ S, m.scm.toMech.diag.AncAvoid S z y → z ∈ T)
    (x : ∀ i, m.X i) : m.mu y r S x = m.mu y r T x :=
  m.mu_congr (fun u => Mech.resp_core hTS hT x u)


end Mec
