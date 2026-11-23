# Lean formalization of the graphical results

This directory holds machine-checked proofs, in Lean 4 with Mathlib, of the graphical results of

> Sanghack Lee and Elias Bareinboim. *Structural Causal Bandits: Where to Intervene?*
> Advances in Neural Information Processing Systems 31 (NeurIPS 2018).

Statement numbers follow the paper as stated in its
[unofficial AI-corrected version](https://www.sanghacklee.me/ai-corrected/SCB2018_AI_corrected_2026-11-23.pdf)
(2026-11-23; author review pending). That version keeps the paper's results and corrects
proofs and explanations; its last page lists the changes. All statements below were formalized,
and none needed a change.

No proof uses `sorry`, `admit` or `native_decide`, and no axiom is added.
`MecLean/AxiomCheck.lean` prints the axioms of every result below. Each depends only on Lean's
standard axioms `propext`, `Classical.choice` and `Quot.sound`.

## Results

| Paper | Lean | File |
| --- | --- | --- |
| Prop. 1 (minimality): `X` is an MIS iff `X ⊆ An(Y)` in `G` after intervening on `X` | `Mec.ScbMec.mis_diag_iff` | `ScbMec/MIS.lean` |
| Prop. 2, Cor. 3: if `Y` is not confounded with its ancestors (in particular, if `G` is Markovian), `pa(Y)` is the only POMIS | `Mec.Scb2018.prop2`, `cor3` | `Scb2018/Semantics.lean` |
| `pa(Y)` is always a POMIS (text of Sec. 3) | `Mec.Scb2018.pa_pomis` | `Scb2018/Semantics.lean` |
| Props. 4, 5: `IB(G_W̄, Y)` is a POMIS for every `W` | `Mec.Scb2018.prop4`, `prop5` | `Scb2018/Semantics.lean` |
| Thm. 6: `X` is a POMIS iff `IB(G_X̄, Y) = X` | `Mec.Scb2018.thm6`; "if" is `Mec.ScbMec.pomis_diag` | `Scb2018/Thm6.lean`, `ScbMec/LB2018.lean` |
| Prop. 7 | `Mec.Scb2018.prop7` | `Scb2018/Recursion.lean` |
| Prop. 8 (also for every `W'`) | `Mec.Scb2018.prop8`, `prop8_general` | `Scb2018/Recursion.lean` |
| Thm. 9: algorithm `POMISs` returns all and only the POMISs | `Mec.Scb2018.thm9`, `thm9_of_revTopo_compress` | `Scb2018/Algorithm.lean` |
| Prop. 10 (subsumption), for every UC-territory | `Mec.Scb2018.subsumption` | `Scb2018/Semantics.lean` |
| Cor. 11 | `Mec.Scb2018.cor11` | `Scb2018/Semantics.lean` |
| Prop. 12 (also for `W' = W`) | `Mec.Scb2018.prop12` | `Scb2018/Recursion.lean` |
| App. A, Alg. 4: algorithm `MISs` returns exactly the MISs | `Mec.Scb2018.mem_MISs_iff_mis` | `Scb2018/MISs.lean` |
| Fig. 1 and the four models of App. A | `Mec.Scb2018.Fig1.*` | `Scb2018/Examples.lean` |
| Fig. 2 and the worked example of App. C | `Mec.Scb2018.Fig2.*` | `Scb2018/Fig2.lean` |

For example, Thm. 6 reads

```lean
theorem Mec.Scb2018.thm6 (hG : G.WF) :
    POMIS (ConfDiag G) y W ↔ ∃ T, DiagMUCT G y W T ∧ diagBorder G T = W
```

## How the paper is read

- **Graphs.** Vertices are `Fin n`. A causal diagram `G : ADMG n` has directed edges and
  bidirected edges, and may carry both between two vertices. `G.WF` excludes self-loops and
  directed cycles and makes bidirected edges symmetric.
- **Models.** `Model n` is a structural causal model with finitely many independent exogenous
  roots and arbitrary finite domains. Two vertices share a root exactly when `G` has a
  bidirected edge between them. `ConfDiag G m` says that the causal diagram of `m` is `G`.
  An equation may ignore some of its listed arguments, so faithfulness is not assumed.
  Existential witnesses may use freely chosen finite domains, as Sec. 1 of the corrected
  version states.
- **Arms.** The reward is `r(Y)` for an arbitrary `r : Ω_Y → ℝ`. `m.mu y r S x` is the mean
  reward of `do(S = x)`, and `m.best y r S` is its maximum over `x`.
- **MIS and POMIS.** `MIS`, `DMIS` and `POMIS` are defined for any class of models
  (`ScbMec/Basic.lean`). For a causal diagram an MIS is a DMIS (`dmis_of_mis_diag`), so
  `POMIS (ConfDiag G)` is the paper's Def. 2.
- **Territories.** `DiagMUCT G y W T` says that `T` is the minimal UC-territory of `G` after
  intervention on `W`; `diagBorder G T = Pa(T) \ T` is the interventional border.
  `Scb2018/Recursion.lean` proves that these are the MUCT and IB of the mutilated graph
  `G_W̄` (`diagMUCT_cut_iff`).
- **Algorithms.** `POMISs` and `MISs` follow the pseudocode line by line. The topological order
  is an input satisfying `RevTopo`, and outputs are lists of sets; the theorems concern
  membership.

The proofs of Thm. 6 ("only if") and Prop. 10 do not use do-calculus. Instead they use
composition, exclusion and the independence of disjoint blocks of exogenous roots. The proof of
Thm. 6 ("if") builds its own witness model; the module docstring of `ScbMec/LB2018.lean`
describes it.

## Not formalized

- The observational identity `μ_x = E[Y | x]` for the IB (Sec. 3); the library has
  interventional means only.
- The remark in App. A that the inequalities survive small perturbations.
- The kl-UCB regret bound, the experiments and the counts of Task 3.
- Running times, such as `O(mn²)` for Alg. 4.

## Building

The toolchain is `leanprover/lean4:v4.36.0` (`lean-toolchain`) with Mathlib `v4.36.0`
(`lakefile.toml`). Run, from this directory:

```
lake exe cache get
lake build
lake env lean MecLean/AxiomCheck.lean
```

## Provenance

These files are a subset of a larger Lean library that we use for other reviews. Only the
modules needed here are included; comments that referred to other work were rewritten, and the
four helper facts of `ScbMec/Helpers.lean` were moved from another module unchanged.

The formalization was written with AI assistance (Claude Code, `claude-opus-5-5`) in November
2026, during a review of the unofficial AI-corrected version. The Lean kernel checks every
proof. Whether a Lean statement expresses the paper's statement is a matter of reading the
definitions above and the module docstrings. The review also found proof steps left implicit
in the written proofs of Props. 7–8 and Thms. 6 and 9; the 2026-11-23 corrected version states
them.
