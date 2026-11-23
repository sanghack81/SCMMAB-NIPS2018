import MecLean.Scb2018.Semantics
import MecLean.ScbMec.Helpers

set_option linter.unusedSectionVars false

/-!
# Structural Causal Bandits (NeurIPS 2018): Fig. 1 and the models of Appendix A

Vertices `Z = 0`, `X = 1`, `Y = 2`.  Fig. 1 (labels `fig:be-a`, `be-b`, `be-d`, `be-e`, displayed
(a)–(d)): `Z → X → Y` with no confounder, with `Z ↔ Y`, with `X ↔ Y`, and with both.

* `mis_sets`: in all four graphs the MISs are `∅`, `{X}`, `{Z}`.
* `pomis_*`: the POMISs, i.e. the non-dominated arm sets of Fig. 1(e): (a) `{X}`; (b) `∅`, `{X}`;
  (c) `{X}`, `{Z}`; (d) `∅`, `{X}`, `{Z}` (via Thm. 6).
* The four models of Appendix A, with their parameters kept symbolic:
  - `B1` on (b): `U ~ Bern(α)`, `Z = 1 − U`, `X = Z`, `Y = X ⊕ U`.  `μ_∅ = 1` and
    `μ_{x*} = μ_{z*} = max(α, 1 − α)`.
  - `B2` on (b): the same with `Y = X + U`.  `μ_∅ = 1` and `μ_{x*} = μ_{z*} = 1 + α`.
  - `C1` on `fig:be-d` (displayed (c), `X ↔ Y`): `P(U = 1) = 1/2`, private noises with
    `P(U_V = 1) = ε`, `Y = X ⊕ (1 − U) ⊕ U_Y`, `X = Z ⊕ U_X ⊕ U`, `Z = U_Z`.  `μ_x = 1/2` for every
    `x` and `μ_{z*} = (1 − ε)² + ε²`.
  - `C2` on the same graph: all noises fair, `Y = X + U + U_Y`, `X = Z ⊕ U_X ⊕ U`.  `μ_x = x + 1` and
    `μ_z = 3/2` for every `z`.
  Domains: Boolean in `B1` and `C1`.  In `B2` and `C2` all vertices share the domain `Fin 4` and
  every non-reward equation reads its inputs as bits (`v ≠ 0`), so intervening with a value
  `≥ 2` acts as intervening with `1`; the paper's binary `X`, `Z` are the values `0, 1`.
-/

open Finset

namespace Mec.Scb2018.Fig1

open Mec ScbMec

/-! ## Expectations of functions of two root coordinates -/

theorem scmE_coord2 {n : ℕ} {Xd : Fin n → Type} {L : Type} [Fintype L] [DecidableEq L]
    {V : Type} [Fintype V] (M : SCM n Xd L (fun _ => V)) {l₁ l₂ : L} (h12 : l₁ ≠ l₂)
    (g : V → V → ℝ) {F : (L → V) → ℝ} (hF : ∀ u, F u = g (u l₁) (u l₂)) :
    M.E F = ∑ a, ∑ b, M.p l₁ a * M.p l₂ b * g a b := by
  have hsplit := M.E_split {l₁} F (fun a b => g (a l₁) (b l₂)) (fun a b => by
    rw [hF]
    simp [SCM.merge, Ne.symm h12])
  rw [hsplit]
  have inner : ∀ a : L → V, M.E (fun b => g (a l₁) (b l₂)) = ∑ c, M.p l₂ c * g (a l₁) c :=
    fun a => scmE_coord M l₂ (fun c => g (a l₁) c) (fun _ => rfl)
  simp_rw [inner, Finset.mul_sum]
  rw [Finset.sum_comm]
  have outer : ∀ c, ∑ a : L → V, M.prob a * (M.p l₂ c * g (a l₁) c) =
      M.p l₂ c * ∑ d, M.p l₁ d * g d c := by
    intro c
    have := scmE_coord M l₁ (fun d => g d c) (F := fun a => g (a l₁) c) (fun _ => rfl)
    unfold SCM.E at this
    rw [← this, Finset.mul_sum]
    refine Finset.sum_congr rfl (fun a _ => by ring)
  simp_rw [outer, Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl (fun d _ => Finset.sum_congr rfl (fun c _ => by ring))

/-! ## The graphs -/

abbrev Z : Fin 3 := 0
abbrev X : Fin 3 := 1
abbrev Y : Fin 3 := 2

/-- `Z → X → Y`. -/
def dirZXY (i j : Fin 3) : Bool := decide ((i = Z ∧ j = X) ∨ (i = X ∧ j = Y))

def Ga : ADMG 3 where
  dir := dirZXY
  bi _ _ := false

def Gb : ADMG 3 where
  dir := dirZXY
  bi i j := decide ((i = Z ∧ j = Y) ∨ (i = Y ∧ j = Z))

def Gc : ADMG 3 where
  dir := dirZXY
  bi i j := decide ((i = X ∧ j = Y) ∨ (i = Y ∧ j = X))

def Gd : ADMG 3 where
  dir := dirZXY
  bi i j := decide ((i = Z ∧ j = Y) ∨ (i = Y ∧ j = Z) ∨ (i = X ∧ j = Y) ∨ (i = Y ∧ j = X))

theorem wf : Ga.WF ∧ Gb.WF ∧ Gc.WF ∧ Gd.WF := by decide +kernel

/-- In all four graphs the sets satisfying the MIS criterion are `∅`, `{X}`, `{Z}`. -/
theorem ancCrit_sets : ∀ G ∈ [Ga, Gb, Gc, Gd], ∀ S : Finset (Fin 3), Y ∉ S →
    (AncCrit G Y S ↔ S ∈ ({∅, {X}, {Z}} : Finset (Finset (Fin 3)))) := by
  decide +kernel

/-- Fig. 1: the MISs are `∅`, `{X}`, `{Z}` in all four graphs. -/
theorem mis_sets {G : ADMG 3} (hG : G ∈ [Ga, Gb, Gc, Gd]) {S : Finset (Fin 3)} (hy : Y ∉ S) :
    MIS (ConfDiag G) Y S ↔ S ∈ ({∅, {X}, {Z}} : Finset (Finset (Fin 3))) := by
  have hwf : G.WF := by
    simp only [List.mem_cons, List.mem_nil_iff, or_false] at hG
    rcases hG with rfl | rfl | rfl | rfl
    · exact wf.1
    · exact wf.2.1
    · exact wf.2.2.1
    · exact wf.2.2.2
  rw [mis_diag_iff hwf hy]
  exact ancCrit_sets G hG S hy

/-- From a statement about every MUCT to the graphical condition of Thm. 6. -/
theorem graphical_iff {G : ADMG 3} {P : Finset (Fin 3) → Prop}
    (h : ∀ S T : Finset (Fin 3), Y ∉ S → DiagMUCT G Y S T → (diagBorder G T = S ↔ P S))
    {S : Finset (Fin 3)} (hy : Y ∉ S) :
    (∃ T, DiagMUCT G Y S T ∧ diagBorder G T = S) ↔ P S := by
  constructor
  · rintro ⟨T, hT, hB⟩
    exact (h S T hy hT).mp hB
  · intro hP
    have hyH : Y ∈ diagH G Y S := mem_diagH.mpr ⟨hy, Reach.refl _⟩
    obtain ⟨T, -, hT⟩ := exists_diagMUCT (diagUCT_whole hyH)
    exact ⟨T, hT, (h S T hy hT).mpr hP⟩

/-- The graphical POMIS condition of Thm. 6 on the four graphs, for every MUCT. -/
theorem border_a : ∀ S T : Finset (Fin 3), Y ∉ S → DiagMUCT Ga Y S T →
    (diagBorder Ga T = S ↔ S = {X}) := by
  decide +kernel

theorem border_b : ∀ S T : Finset (Fin 3), Y ∉ S → DiagMUCT Gb Y S T →
    (diagBorder Gb T = S ↔ S = ∅ ∨ S = {X}) := by
  decide +kernel

theorem border_c : ∀ S T : Finset (Fin 3), Y ∉ S → DiagMUCT Gc Y S T →
    (diagBorder Gc T = S ↔ S = {X} ∨ S = {Z}) := by
  decide +kernel

theorem border_d : ∀ S T : Finset (Fin 3), Y ∉ S → DiagMUCT Gd Y S T →
    (diagBorder Gd T = S ↔ S = ∅ ∨ S = {X} ∨ S = {Z}) := by
  decide +kernel

/-- Fig. 1(e): the POMISs of the four graphs. -/
theorem pomis_sets {S : Finset (Fin 3)} (hy : Y ∉ S) :
    (POMIS (ConfDiag Ga) Y S ↔ S = {X}) ∧
    (POMIS (ConfDiag Gb) Y S ↔ S = ∅ ∨ S = {X}) ∧
    (POMIS (ConfDiag Gc) Y S ↔ S = {X} ∨ S = {Z}) ∧
    (POMIS (ConfDiag Gd) Y S ↔ S = ∅ ∨ S = {X} ∨ S = {Z}) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [thm6 wf.1]; exact graphical_iff border_a hy
  · rw [thm6 wf.2.1]; exact graphical_iff border_b hy
  · rw [thm6 wf.2.2.1]; exact graphical_iff border_c hy
  · rw [thm6 wf.2.2.2]; exact graphical_iff border_d hy

/-! ## Helpers for the models -/

theorem scmE_add {n : ℕ} {Xd : Fin n → Type} {L : Type} {U : L → Type} [Fintype L]
    [DecidableEq L] [∀ l, Fintype (U l)] (M : SCM n Xd L U) (f g : (∀ l, U l) → ℝ) :
    M.E (fun u => f u + g u) = M.E f + M.E g := by
  unfold SCM.E
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_congr rfl (fun u _ => by ring)

/-- A best arm value that some arm attains and every arm respects. -/
theorem best_eq_of {n : ℕ} (m : Model n) {y : Fin n} {r : m.X y → ℝ} {S : Finset (Fin n)}
    {c : ℝ} (hle : ∀ x, m.mu y r S x ≤ c) (x₀ : ∀ i, m.X i) (h₀ : m.mu y r S x₀ = c) :
    m.best y r S = c :=
  le_antisymm (best_le_of_forall m hle) (h₀ ▸ m.mu_le_best y r S x₀)

/-- The confounder root shared by `Z` and `Y` in Fig. 1(b), and by `X` and `Y` in Fig. 1(c). -/
abbrev uZY : Roots 3 := Sum.inr (Z, Y)
abbrev uXY : Roots 3 := Sum.inr (X, Y)
abbrev uZ : Roots 3 := Sum.inl Z
abbrev uX : Roots 3 := Sum.inl X
abbrev uY : Roots 3 := Sum.inl Y

theorem uZY_Z : uZY ∈ rootScope Gb Z := mem_rootScope.mpr (Or.inr ⟨Z, Y, rfl, by decide, Or.inl rfl⟩)
theorem uZY_Y : uZY ∈ rootScope Gb Y := mem_rootScope.mpr (Or.inr ⟨Z, Y, rfl, by decide, Or.inr rfl⟩)
theorem uXY_X : uXY ∈ rootScope Gc X := mem_rootScope.mpr (Or.inr ⟨X, Y, rfl, by decide, Or.inl rfl⟩)
theorem uXY_Y : uXY ∈ rootScope Gc Y := mem_rootScope.mpr (Or.inr ⟨X, Y, rfl, by decide, Or.inr rfl⟩)
theorem uZ_Z : uZ ∈ rootScope Gc Z := mem_rootScope.mpr (Or.inl rfl)
theorem uX_X : uX ∈ rootScope Gc X := mem_rootScope.mpr (Or.inl rfl)
theorem uY_Y : uY ∈ rootScope Gc Y := mem_rootScope.mpr (Or.inl rfl)

/-! ## Model `B1` (Appendix A, first model for Fig. 1(b)) -/

/-- `Z = 1 − U`, `X = Z`, `Y = X ⊕ U`. -/
def fB1 (i : Fin 3) (z : Fin 3 → Bool) (u : Roots 3 → Bool) : Bool :=
  if i = Z then !(u uZY) else if i = X then z Z else xor (z X) (u uZY)

def mechB1 : Mech 3 (fun _ => Bool) (Roots 3) (fun _ => Bool) :=
  Mech.ofADMG Gb wf.2.1 fB1
    (by
      intro i x x' u h
      fin_cases i
      · rfl
      · show x Z = x' Z
        exact h Z (by decide)
      · show xor (x X) (u uZY) = xor (x' X) (u uZY)
        rw [h X (by decide)])
    (by
      intro i x u u' h
      fin_cases i
      · show (!(u uZY)) = !(u' uZY)
        rw [h uZY uZY_Z]
      · rfl
      · show xor (x X) (u uZY) = xor (x X) (u' uZY)
        rw [h uZY uZY_Y])

theorem mechB1_diag : mechB1.diag = Gb := Mech.ofADMG_diag Gb wf.2.1 fB1 _ _

/-- `P(U = 1) = α`; the other roots are unused. -/
noncomputable def scmB1 (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1) :
    SCM 3 (fun _ => Bool) (Roots 3) (fun _ => Bool) where
  toMech := mechB1
  p l b := if l = uZY then (if b then α else 1 - α) else 1 / 2
  p_nonneg l b := by split_ifs <;> linarith
  p_sum l := by
    rw [Fintype.sum_bool]
    by_cases hl : l = uZY <;> simp [hl] <;> norm_num

noncomputable def modelB1 (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1) : Model 3 where
  X := fun _ => Bool
  L := Roots 3
  U := fun _ => Bool
  scm := scmB1 α h0 h1

section B1

variable (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1)

theorem confB1 : ConfDiag Gb (modelB1 α h0 h1) := mechB1_diag

theorem respB1_Z {S : Finset (Fin 3)} (hZ : Z ∉ S) (x : Fin 3 → Bool) (u : Roots 3 → Bool) :
    mechB1.resp S x u Z = !(u uZY) := by
  rw [Mech.resp_of_not_mem hZ]; rfl

theorem respB1_X {S : Finset (Fin 3)} (hX : X ∉ S) (x : Fin 3 → Bool) (u : Roots 3 → Bool) :
    mechB1.resp S x u X = mechB1.resp S x u Z := by
  rw [Mech.resp_of_not_mem hX]; rfl

theorem respB1_Y {S : Finset (Fin 3)} (hY : Y ∉ S) (x : Fin 3 → Bool) (u : Roots 3 → Bool) :
    mechB1.resp S x u Y = xor (mechB1.resp S x u X) (u uZY) := by
  rw [Mech.resp_of_not_mem hY]; rfl

theorem B1_mu (S : Finset (Fin 3)) (x : Fin 3 → Bool) :
    (modelB1 α h0 h1).mu Y boolReward S x =
      (scmB1 α h0 h1).E (fun u => boolReward (mechB1.resp S x u Y)) := rfl

/-- `μ_∅ = 1`. -/
theorem B1_empty (x : Fin 3 → Bool) : (modelB1 α h0 h1).mu Y boolReward ∅ x = 1 := by
  rw [B1_mu]
  apply scmE_const
  intro u
  rw [respB1_Y (by decide), respB1_X (by decide), respB1_Z (by decide)]
  cases u uZY <;> simp [boolReward]

theorem B1_coord (b : Bool) : (scmB1 α h0 h1).p uZY b = if b then α else 1 - α := by
  show (if uZY = uZY then (if b then α else 1 - α) else 1 / 2) = _
  rw [if_pos rfl]

/-- `μ_x = 1 − α` for `x = 1` and `α` for `x = 0`. -/
theorem B1_X (x : Fin 3 → Bool) :
    (modelB1 α h0 h1).mu Y boolReward {X} x = if x X then 1 - α else α := by
  have hF : ∀ u : Roots 3 → Bool, boolReward (mechB1.resp {X} x u Y) =
      boolReward (xor (x X) (u uZY)) := by
    intro u
    rw [respB1_Y (by decide), Mech.resp_of_mem (Finset.mem_singleton_self X)]
  rw [B1_mu, scmE_coord (scmB1 α h0 h1) uZY (fun b => boolReward (xor (x X) b)) hF,
    Fintype.sum_bool, B1_coord, B1_coord]
  cases x X <;> simp [boolReward]

/-- `μ_z = μ_x` with `x = z`. -/
theorem B1_Z (x : Fin 3 → Bool) :
    (modelB1 α h0 h1).mu Y boolReward {Z} x = if x Z then 1 - α else α := by
  have hF : ∀ u : Roots 3 → Bool, boolReward (mechB1.resp {Z} x u Y) =
      boolReward (xor (x Z) (u uZY)) := by
    intro u
    rw [respB1_Y (by decide), respB1_X (by decide), Mech.resp_of_mem (Finset.mem_singleton_self Z)]
  rw [B1_mu, scmE_coord (scmB1 α h0 h1) uZY (fun b => boolReward (xor (x Z) b)) hF,
    Fintype.sum_bool, B1_coord, B1_coord]
  cases x Z <;> simp [boolReward]

theorem B1_best_X : (modelB1 α h0 h1).best Y boolReward {X} = max α (1 - α) := by
  apply le_antisymm
  · apply best_le_of_forall
    intro x
    rw [B1_X α h0 h1 x]
    split_ifs
    · exact le_max_right _ _
    · exact le_max_left _ _
  · apply max_le
    · have h := (modelB1 α h0 h1).mu_le_best Y boolReward {X} (fun _ => false)
      rw [B1_X α h0 h1 (fun _ => false)] at h
      simpa using h
    · have h := (modelB1 α h0 h1).mu_le_best Y boolReward {X} (fun _ => true)
      rw [B1_X α h0 h1 (fun _ => true)] at h
      simpa using h

theorem B1_best_Z : (modelB1 α h0 h1).best Y boolReward {Z} = max α (1 - α) := by
  apply le_antisymm
  · apply best_le_of_forall
    intro x
    rw [B1_Z α h0 h1 x]
    split_ifs
    · exact le_max_right _ _
    · exact le_max_left _ _
  · apply max_le
    · have h := (modelB1 α h0 h1).mu_le_best Y boolReward {Z} (fun _ => false)
      rw [B1_Z α h0 h1 (fun _ => false)] at h
      simpa using h
    · have h := (modelB1 α h0 h1).mu_le_best Y boolReward {Z} (fun _ => true)
      rw [B1_Z α h0 h1 (fun _ => true)] at h
      simpa using h

theorem B1_best_empty : (modelB1 α h0 h1).best Y boolReward ∅ = 1 :=
  best_eq_of _ (fun x => (B1_empty α h0 h1 x).le) (fun _ => false) (B1_empty α h0 h1 _)

/-- Appendix A: observation beats intervening on `X` or `Z` when `0 < α < 1`. -/
theorem B1_compare (ha : 0 < α) (hb : α < 1) :
    (modelB1 α h0 h1).best Y boolReward {X} < (modelB1 α h0 h1).best Y boolReward ∅ ∧
    (modelB1 α h0 h1).best Y boolReward {Z} < (modelB1 α h0 h1).best Y boolReward ∅ := by
  rw [B1_best_X, B1_best_Z, B1_best_empty]
  exact ⟨max_lt hb (by linarith), max_lt hb (by linarith)⟩

end B1

/-- The main-text instance (`α = 1/2`): `μ_∅ = 1` and `μ_{x*} = 1/2`. -/
theorem B1_half :
    (modelB1 (1/2) (by norm_num) (by norm_num)).best Y boolReward ∅ = 1 ∧
    (modelB1 (1/2) (by norm_num) (by norm_num)).best Y boolReward {X} = 1 / 2 := by
  refine ⟨B1_best_empty _ _ _, ?_⟩
  rw [B1_best_X]
  norm_num

/-! ## Model `C1` (Appendix A, first model for `fig:be-d`, displayed Fig. 1(c)) -/

/-- `Z = U_Z`, `X = Z ⊕ U_X ⊕ U`, `Y = X ⊕ (1 − U) ⊕ U_Y`. -/
def fC1 (i : Fin 3) (z : Fin 3 → Bool) (u : Roots 3 → Bool) : Bool :=
  if i = Z then u uZ else if i = X then xor (z Z) (xor (u uX) (u uXY))
  else xor (z X) (xor (!(u uXY)) (u uY))

def mechC1 : Mech 3 (fun _ => Bool) (Roots 3) (fun _ => Bool) :=
  Mech.ofADMG Gc wf.2.2.1 fC1
    (by
      intro i x x' u h
      fin_cases i
      · rfl
      · show xor (x Z) (xor (u uX) (u uXY)) = xor (x' Z) (xor (u uX) (u uXY))
        rw [h Z (by decide)]
      · show xor (x X) (xor (!(u uXY)) (u uY)) = xor (x' X) (xor (!(u uXY)) (u uY))
        rw [h X (by decide)])
    (by
      intro i x u u' h
      fin_cases i
      · show u uZ = u' uZ
        exact h uZ uZ_Z
      · show xor (x Z) (xor (u uX) (u uXY)) = xor (x Z) (xor (u' uX) (u' uXY))
        rw [h uX uX_X, h uXY uXY_X]
      · show xor (x X) (xor (!(u uXY)) (u uY)) = xor (x X) (xor (!(u' uXY)) (u' uY))
        rw [h uXY uXY_Y, h uY uY_Y])

theorem mechC1_diag : mechC1.diag = Gc := Mech.ofADMG_diag Gc wf.2.2.1 fC1 _ _

/-- `P(U = 1) = 1/2` and `P(U_V = 1) = ε` for the other roots. -/
noncomputable def scmC1 (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1) :
    SCM 3 (fun _ => Bool) (Roots 3) (fun _ => Bool) where
  toMech := mechC1
  p l b := if l = uXY then 1 / 2 else (if b then ε else 1 - ε)
  p_nonneg l b := by split_ifs <;> linarith
  p_sum l := by
    rw [Fintype.sum_bool]
    by_cases hl : l = uXY <;> simp [hl] <;> norm_num

noncomputable def modelC1 (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1) : Model 3 where
  X := fun _ => Bool
  L := Roots 3
  U := fun _ => Bool
  scm := scmC1 ε h0 h1

section C1

variable (ε : ℝ) (h0 : 0 ≤ ε) (h1 : ε ≤ 1)

theorem confC1 : ConfDiag Gc (modelC1 ε h0 h1) := mechC1_diag

theorem C1_mu (S : Finset (Fin 3)) (x : Fin 3 → Bool) :
    (modelC1 ε h0 h1).mu Y boolReward S x =
      (scmC1 ε h0 h1).E (fun u => boolReward (mechC1.resp S x u Y)) := rfl

theorem C1_pXY (b : Bool) : (scmC1 ε h0 h1).p uXY b = 1 / 2 := by
  show (if uXY = uXY then (1 / 2 : ℝ) else (if b then ε else 1 - ε)) = 1 / 2
  rw [if_pos rfl]

theorem C1_pV {l : Roots 3} (hl : l ≠ uXY) (b : Bool) :
    (scmC1 ε h0 h1).p l b = if b then ε else 1 - ε := by
  show (if l = uXY then (1 / 2 : ℝ) else (if b then ε else 1 - ε)) = (if b then ε else 1 - ε)
  rw [if_neg hl]

/-- `μ_x = 1/2` for every `x`. -/
theorem C1_X (x : Fin 3 → Bool) : (modelC1 ε h0 h1).mu Y boolReward {X} x = 1 / 2 := by
  have hF : ∀ u : Roots 3 → Bool, boolReward (mechC1.resp {X} x u Y) =
      boolReward (xor (x X) (xor (!(u uXY)) (u uY))) := by
    intro u
    rw [Mech.resp_of_not_mem (show Y ∉ ({X} : Finset (Fin 3)) by decide)]
    show boolReward (xor (mechC1.resp {X} x u X) (xor (!(u uXY)) (u uY))) = _
    rw [Mech.resp_of_mem (Finset.mem_singleton_self X)]
  rw [C1_mu, scmE_coord2 (scmC1 ε h0 h1) (show uXY ≠ uY by decide)
    (fun a b => boolReward (xor (x X) (xor (!a) b))) hF]
  simp only [Fintype.sum_bool, C1_pXY ε h0 h1, C1_pV ε h0 h1 (show uY ≠ uXY by decide)]
  cases x X <;> simp [boolReward] <;> ring

/-- `μ_z = (1 − ε)² + ε²` for `z = 0` and `2ε(1 − ε)` for `z = 1`. -/
theorem C1_Z (x : Fin 3 → Bool) : (modelC1 ε h0 h1).mu Y boolReward {Z} x =
    if x Z then 2 * ε * (1 - ε) else (1 - ε) ^ 2 + ε ^ 2 := by
  have hF : ∀ u : Roots 3 → Bool, boolReward (mechC1.resp {Z} x u Y) =
      boolReward (xor (x Z) (xor (u uX) (!(u uY)))) := by
    intro u
    rw [Mech.resp_of_not_mem (show Y ∉ ({Z} : Finset (Fin 3)) by decide)]
    show boolReward (xor (mechC1.resp {Z} x u X) (xor (!(u uXY)) (u uY))) = _
    rw [Mech.resp_of_not_mem (show X ∉ ({Z} : Finset (Fin 3)) by decide)]
    show boolReward (xor (xor (mechC1.resp {Z} x u Z) (xor (u uX) (u uXY)))
      (xor (!(u uXY)) (u uY))) = _
    rw [Mech.resp_of_mem (Finset.mem_singleton_self Z)]
    cases x Z <;> cases u uX <;> cases u uXY <;> cases u uY <;> rfl
  rw [C1_mu, scmE_coord2 (scmC1 ε h0 h1) (show uX ≠ uY by decide)
    (fun a b => boolReward (xor (x Z) (xor a (!b)))) hF]
  simp only [Fintype.sum_bool, C1_pV ε h0 h1 (show uX ≠ uXY by decide),
    C1_pV ε h0 h1 (show uY ≠ uXY by decide)]
  cases x Z <;> simp [boolReward] <;> ring

theorem C1_best_X : (modelC1 ε h0 h1).best Y boolReward {X} = 1 / 2 :=
  best_eq_of _ (fun x => (C1_X ε h0 h1 x).le) (fun _ => false) (C1_X ε h0 h1 _)

theorem C1_best_Z : (modelC1 ε h0 h1).best Y boolReward {Z} = (1 - ε) ^ 2 + ε ^ 2 := by
  refine best_eq_of _ (fun x => ?_) (fun _ => false) (by rw [C1_Z ε h0 h1 (fun _ => false)]; rfl)
  rw [C1_Z ε h0 h1 x]
  split_ifs
  · nlinarith [sq_nonneg (1 - 2 * ε)]
  · exact le_rfl

/-- Appendix A: intervening on `Z` beats intervening on `X` whenever `ε ≠ 1/2`. -/
theorem C1_compare (hε : ε ≠ 1 / 2) :
    (modelC1 ε h0 h1).best Y boolReward {X} < (modelC1 ε h0 h1).best Y boolReward {Z} := by
  rw [C1_best_X, C1_best_Z]
  have : 0 < (1 - 2 * ε) ^ 2 := by
    have : 1 - 2 * ε ≠ 0 := fun h => hε (by linarith)
    positivity
  nlinarith

end C1

/-! ## Models `B2` and `C2` (integer-valued reward) -/

/-- The bit read from a value: `0` stays `0`, everything else is `1`. -/
def bit (v : Fin 4) : Fin 4 := if v = 0 then 0 else 1

def ind (b : Bool) : Fin 4 := if b then 1 else 0

/-- The reward `r(y) = y`. -/
def valReward (v : Fin 4) : ℝ := ((v : ℕ) : ℝ)

theorem bit_ind (b : Bool) : bit (ind b) = ind b := by cases b <;> decide

theorem val_add2 (v : Fin 4) (a : Bool) :
    valReward (bit v + ind a) = valReward (bit v) + valReward (ind a) := by
  unfold valReward
  by_cases hv : v = 0 <;> cases a <;> simp [bit, ind, hv] <;> norm_num [Fin.val_add]

theorem val_add3 (c a b : Bool) :
    valReward (ind c + ind a + ind b) =
      valReward (ind c) + valReward (ind a) + valReward (ind b) := by
  unfold valReward
  have h : ((ind c + ind a + ind b : Fin 4) : ℕ) = (ind c : ℕ) + (ind a : ℕ) + (ind b : ℕ) := by
    cases c <;> cases a <;> cases b <;> decide
  rw [h]
  push_cast
  ring

theorem valReward_ind (b : Bool) : valReward (ind b) = if b then 1 else 0 := by
  cases b <;> simp [valReward, ind]

theorem valReward_bit (v : Fin 4) : valReward (bit v) = if v = 0 then 0 else 1 := by
  by_cases hv : v = 0 <;> simp [valReward, bit, hv]

/-- `B2`: `Z = 1 − U`, `X = Z`, `Y = X + U` (the second model for Fig. 1(b)). -/
def fB2 (i : Fin 3) (z : Fin 3 → Fin 4) (u : Roots 3 → Bool) : Fin 4 :=
  if i = Z then ind (!(u uZY)) else if i = X then bit (z Z) else bit (z X) + ind (u uZY)

def mechB2 : Mech 3 (fun _ => Fin 4) (Roots 3) (fun _ => Bool) :=
  Mech.ofADMG Gb wf.2.1 fB2
    (by
      intro i x x' u h
      fin_cases i
      · rfl
      · show bit (x Z) = bit (x' Z)
        rw [h Z (by decide)]
      · show bit (x X) + ind (u uZY) = bit (x' X) + ind (u uZY)
        rw [h X (by decide)])
    (by
      intro i x u u' h
      fin_cases i
      · show ind (!(u uZY)) = ind (!(u' uZY))
        rw [h uZY uZY_Z]
      · rfl
      · show bit (x X) + ind (u uZY) = bit (x X) + ind (u' uZY)
        rw [h uZY uZY_Y])

theorem mechB2_diag : mechB2.diag = Gb := Mech.ofADMG_diag Gb wf.2.1 fB2 _ _

noncomputable def scmB2 (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1) :
    SCM 3 (fun _ => Fin 4) (Roots 3) (fun _ => Bool) where
  toMech := mechB2
  p l b := if l = uZY then (if b then α else 1 - α) else 1 / 2
  p_nonneg l b := by split_ifs <;> linarith
  p_sum l := by
    rw [Fintype.sum_bool]
    by_cases hl : l = uZY <;> simp [hl] <;> norm_num

noncomputable def modelB2 (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1) : Model 3 where
  X := fun _ => Fin 4
  L := Roots 3
  U := fun _ => Bool
  scm := scmB2 α h0 h1

section B2

variable (α : ℝ) (h0 : 0 ≤ α) (h1 : α ≤ 1)

theorem confB2 : ConfDiag Gb (modelB2 α h0 h1) := mechB2_diag

theorem B2_mu (S : Finset (Fin 3)) (x : Fin 3 → Fin 4) :
    (modelB2 α h0 h1).mu Y valReward S x =
      (scmB2 α h0 h1).E (fun u => valReward (mechB2.resp S x u Y)) := rfl

theorem B2_coord (b : Bool) : (scmB2 α h0 h1).p uZY b = if b then α else 1 - α := by
  show (if uZY = uZY then (if b then α else 1 - α) else 1 / 2) = _
  rw [if_pos rfl]

/-- `μ_∅ = 1`. -/
theorem B2_empty (x : Fin 3 → Fin 4) : (modelB2 α h0 h1).mu Y valReward ∅ x = 1 := by
  rw [B2_mu]
  apply scmE_const
  intro u
  rw [Mech.resp_of_not_mem (show Y ∉ (∅ : Finset (Fin 3)) by decide)]
  show valReward (bit (mechB2.resp ∅ x u X) + ind (u uZY)) = 1
  rw [Mech.resp_of_not_mem (show X ∉ (∅ : Finset (Fin 3)) by decide)]
  show valReward (bit (bit (mechB2.resp ∅ x u Z)) + ind (u uZY)) = 1
  rw [Mech.resp_of_not_mem (show Z ∉ (∅ : Finset (Fin 3)) by decide)]
  show valReward (bit (bit (ind (!(u uZY)))) + ind (u uZY)) = 1
  have h : ∀ b : Bool, ((bit (bit (ind (!b))) + ind b : Fin 4) : ℕ) = 1 := by decide
  unfold valReward
  rw [h]
  norm_num

/-- `μ_x = α` for `x = 0` and `1 + α` otherwise. -/
theorem B2_X (x : Fin 3 → Fin 4) :
    (modelB2 α h0 h1).mu Y valReward {X} x = (if x X = 0 then 0 else 1) + α := by
  have hF : ∀ u : Roots 3 → Bool, valReward (mechB2.resp {X} x u Y) =
      valReward (bit (x X)) + valReward (ind (u uZY)) := by
    intro u
    rw [Mech.resp_of_not_mem (show Y ∉ ({X} : Finset (Fin 3)) by decide)]
    show valReward (bit (mechB2.resp {X} x u X) + ind (u uZY)) = _
    rw [Mech.resp_of_mem (Finset.mem_singleton_self X), val_add2]
  rw [B2_mu, scmE_coord (scmB2 α h0 h1) uZY
    (fun b => valReward (bit (x X)) + valReward (ind b)) hF, Fintype.sum_bool, B2_coord,
    B2_coord, valReward_bit, valReward_ind, valReward_ind]
  simp only [if_true, if_false, Bool.false_eq_true]
  ring

/-- `μ_z = μ_x` with `x = z`. -/
theorem B2_Z (x : Fin 3 → Fin 4) :
    (modelB2 α h0 h1).mu Y valReward {Z} x = (if x Z = 0 then 0 else 1) + α := by
  have hF : ∀ u : Roots 3 → Bool, valReward (mechB2.resp {Z} x u Y) =
      valReward (bit (x Z)) + valReward (ind (u uZY)) := by
    intro u
    rw [Mech.resp_of_not_mem (show Y ∉ ({Z} : Finset (Fin 3)) by decide)]
    show valReward (bit (mechB2.resp {Z} x u X) + ind (u uZY)) = _
    rw [Mech.resp_of_not_mem (show X ∉ ({Z} : Finset (Fin 3)) by decide)]
    show valReward (bit (bit (mechB2.resp {Z} x u Z)) + ind (u uZY)) = _
    rw [Mech.resp_of_mem (Finset.mem_singleton_self Z), val_add2]
    have hb : bit (bit (x Z)) = bit (x Z) := by unfold bit; split_ifs <;> simp_all
    rw [hb]
  rw [B2_mu, scmE_coord (scmB2 α h0 h1) uZY
    (fun b => valReward (bit (x Z)) + valReward (ind b)) hF, Fintype.sum_bool, B2_coord,
    B2_coord, valReward_bit, valReward_ind, valReward_ind]
  simp only [if_true, if_false, Bool.false_eq_true]
  ring

theorem B2_best_X : (modelB2 α h0 h1).best Y valReward {X} = 1 + α := by
  refine best_eq_of _ (fun x => ?_) (fun _ => (1 : Fin 4)) ?_
  · rw [B2_X α h0 h1 x]
    split_ifs <;> linarith
  · rw [B2_X α h0 h1 (fun _ => (1 : Fin 4))]
    simp

theorem B2_best_Z : (modelB2 α h0 h1).best Y valReward {Z} = 1 + α := by
  refine best_eq_of _ (fun x => ?_) (fun _ => (1 : Fin 4)) ?_
  · rw [B2_Z α h0 h1 x]
    split_ifs <;> linarith
  · rw [B2_Z α h0 h1 (fun _ => (1 : Fin 4))]
    simp

theorem B2_best_empty : (modelB2 α h0 h1).best Y valReward ∅ = 1 :=
  best_eq_of _ (fun x => (B2_empty α h0 h1 x).le) (fun _ => (0 : Fin 4)) (B2_empty α h0 h1 _)

/-- Appendix A: intervening on `X` or `Z` beats observation when `α > 0`. -/
theorem B2_compare (ha : 0 < α) :
    (modelB2 α h0 h1).best Y valReward ∅ < (modelB2 α h0 h1).best Y valReward {X} ∧
    (modelB2 α h0 h1).best Y valReward ∅ < (modelB2 α h0 h1).best Y valReward {Z} := by
  rw [B2_best_X, B2_best_Z, B2_best_empty]
  constructor <;> linarith

end B2

/-- `C2`: `Z = U_Z`, `X = Z ⊕ U_X ⊕ U`, `Y = X + U + U_Y`, all noises fair (the second model for
`fig:be-d`). -/
def fC2 (i : Fin 3) (z : Fin 3 → Fin 4) (u : Roots 3 → Bool) : Fin 4 :=
  if i = Z then ind (u uZ) else if i = X then ind (xor (decide (z Z ≠ 0)) (xor (u uX) (u uXY)))
  else bit (z X) + ind (u uXY) + ind (u uY)

def mechC2 : Mech 3 (fun _ => Fin 4) (Roots 3) (fun _ => Bool) :=
  Mech.ofADMG Gc wf.2.2.1 fC2
    (by
      intro i x x' u h
      fin_cases i
      · rfl
      · show ind (xor (decide (x Z ≠ 0)) (xor (u uX) (u uXY))) =
          ind (xor (decide (x' Z ≠ 0)) (xor (u uX) (u uXY)))
        rw [h Z (by decide)]
      · show bit (x X) + ind (u uXY) + ind (u uY) = bit (x' X) + ind (u uXY) + ind (u uY)
        rw [h X (by decide)])
    (by
      intro i x u u' h
      fin_cases i
      · show ind (u uZ) = ind (u' uZ)
        rw [h uZ uZ_Z]
      · show ind (xor (decide (x Z ≠ 0)) (xor (u uX) (u uXY))) =
          ind (xor (decide (x Z ≠ 0)) (xor (u' uX) (u' uXY)))
        rw [h uX uX_X, h uXY uXY_X]
      · show bit (x X) + ind (u uXY) + ind (u uY) = bit (x X) + ind (u' uXY) + ind (u' uY)
        rw [h uXY uXY_Y, h uY uY_Y])

theorem mechC2_diag : mechC2.diag = Gc := Mech.ofADMG_diag Gc wf.2.2.1 fC2 _ _

noncomputable def scmC2 : SCM 3 (fun _ => Fin 4) (Roots 3) (fun _ => Bool) where
  toMech := mechC2
  p _ _ := 1 / 2
  p_nonneg _ _ := by norm_num
  p_sum _ := by norm_num [Fintype.sum_bool]

noncomputable def modelC2 : Model 3 where
  X := fun _ => Fin 4
  L := Roots 3
  U := fun _ => Bool
  scm := scmC2

theorem confC2 : ConfDiag Gc modelC2 := mechC2_diag

theorem C2_mu (S : Finset (Fin 3)) (x : Fin 3 → Fin 4) :
    modelC2.mu Y valReward S x = scmC2.E (fun u => valReward (mechC2.resp S x u Y)) := rfl

theorem bit_add2 (v : Fin 4) (a b : Bool) :
    valReward (bit v + ind a + ind b) =
      valReward (bit v) + valReward (ind a) + valReward (ind b) := by
  by_cases hv : v = 0
  · have : bit v = ind false := by simp [bit, ind, hv]
    rw [this, val_add3]
  · have : bit v = ind true := by simp [bit, ind, hv]
    rw [this, val_add3]

/-- `μ_x = 1` for `x = 0` and `2` otherwise. -/
theorem C2_X (x : Fin 3 → Fin 4) :
    modelC2.mu Y valReward {X} x = (if x X = 0 then 0 else 1) + 1 := by
  have hF : ∀ u : Roots 3 → Bool, valReward (mechC2.resp {X} x u Y) =
      valReward (bit (x X)) + valReward (ind (u uXY)) + valReward (ind (u uY)) := by
    intro u
    rw [Mech.resp_of_not_mem (show Y ∉ ({X} : Finset (Fin 3)) by decide)]
    show valReward (bit (mechC2.resp {X} x u X) + ind (u uXY) + ind (u uY)) = _
    rw [Mech.resp_of_mem (Finset.mem_singleton_self X), bit_add2]
  rw [C2_mu, scmE_coord2 scmC2 (show uXY ≠ uY by decide)
    (fun a b => valReward (bit (x X)) + valReward (ind a) + valReward (ind b)) hF]
  simp only [Fintype.sum_bool, valReward_bit, valReward_ind, if_true, if_false,
    Bool.false_eq_true]
  show (1 / 2 : ℝ) * (1 / 2) * _ + 1 / 2 * (1 / 2) * _ + (1 / 2 * (1 / 2) * _ +
    1 / 2 * (1 / 2) * _) = _
  ring

/-- `μ_z = 3/2` for every `z`. -/
theorem C2_Z (x : Fin 3 → Fin 4) : modelC2.mu Y valReward {Z} x = 3 / 2 := by
  have hF : ∀ u : Roots 3 → Bool, valReward (mechC2.resp {Z} x u Y) =
      valReward (ind (xor (decide (x Z ≠ 0)) (xor (u uX) (u uXY)))) +
        valReward (ind (u uXY)) + valReward (ind (u uY)) := by
    intro u
    rw [Mech.resp_of_not_mem (show Y ∉ ({Z} : Finset (Fin 3)) by decide)]
    show valReward (bit (mechC2.resp {Z} x u X) + ind (u uXY) + ind (u uY)) = _
    rw [Mech.resp_of_not_mem (show X ∉ ({Z} : Finset (Fin 3)) by decide)]
    show valReward (bit (ind (xor (decide (mechC2.resp {Z} x u Z ≠ 0)) (xor (u uX) (u uXY))))
      + ind (u uXY) + ind (u uY)) = _
    rw [Mech.resp_of_mem (Finset.mem_singleton_self Z), bit_ind, val_add3]
  rw [C2_mu, show (fun u => valReward (mechC2.resp {Z} x u Y)) =
      (fun u => (valReward (ind (xor (decide (x Z ≠ 0)) (xor (u uX) (u uXY)))) +
        valReward (ind (u uXY))) + valReward (ind (u uY))) from funext hF,
    scmE_add, scmE_add,
    scmE_coord2 scmC2 (show uX ≠ uXY by decide)
      (fun a b => valReward (ind (xor (decide (x Z ≠ 0)) (xor a b)))) (fun _ => rfl),
    scmE_coord scmC2 uXY (fun b => valReward (ind b)) (fun _ => rfl),
    scmE_coord scmC2 uY (fun b => valReward (ind b)) (fun _ => rfl)]
  simp only [Fintype.sum_bool, valReward_ind]
  show (1 / 2 : ℝ) * (1 / 2) * _ + 1 / 2 * (1 / 2) * _ + (1 / 2 * (1 / 2) * _ +
    1 / 2 * (1 / 2) * _) + (1 / 2 * _ + 1 / 2 * _) + (1 / 2 * _ + 1 / 2 * _) = _
  cases decide (x Z ≠ 0) <;> simp <;> norm_num

theorem C2_best_X : modelC2.best Y valReward {X} = 2 := by
  refine best_eq_of _ (fun x => ?_) (fun _ => (1 : Fin 4)) ?_
  · rw [C2_X x]
    split_ifs <;> norm_num
  · rw [C2_X (fun _ => (1 : Fin 4))]
    norm_num

theorem C2_best_Z : modelC2.best Y valReward {Z} = 3 / 2 :=
  best_eq_of _ (fun x => (C2_Z x).le) (fun _ => (0 : Fin 4)) (C2_Z _)

/-- Appendix A: intervening on `X` beats intervening on `Z`. -/
theorem C2_compare : modelC2.best Y valReward {Z} < modelC2.best Y valReward {X} := by
  rw [C2_best_X, C2_best_Z]
  norm_num

end Mec.Scb2018.Fig1
