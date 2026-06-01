# RLQAS-CHEM Innovation Design

**Date**: 2026-03-31
**Status**: Draft

---

## 1. Overview

RLQAS-CHEM applies Reinforcement Learning Quantum Architecture Search to molecular ground state problems. The existing codebase supports UCC/HEA/Hybrid search with PPO, DQN, A2C, SAC agents and multiple operator pools (FOP, QOP).

This document defines four innovation directions that differentiate RLQAS-CHEM from the current state of the art.

**One-sentence thesis**: RLQAS-CHEM introduces GRPO-based circuit search over FOP/QOP operator pools with multi-objective Pareto optimization, achieving hardware-efficient molecular ground state circuits that ADAPT-VQE variants and LLM-driven approaches cannot match.

---

## 2. Positioning Against Existing Literature

| Method | Approach | Limitation |
|--------|----------|------------|
| ADAPT-VQE (all variants) | Greedy gradient-based operator selection | Single objective; no learned policy; restarts per molecule |
| K-ADAPT-VQE (2026) | Batch operator addition | Still gradient-based; no RL generalization |
| Pruned-ADAPT-VQE (2025) | Post-hoc redundancy removal | Reactive, not proactive; no reward shaping |
| BenchRL-QAS | RL benchmark on generic gate circuits | Generic search space; no chemistry-motivated pools |
| Hive/Hiverge (2026) | LLM + evolutionary program synthesis | Evolutionary, not RL; outputs explicit code, not policy |
| **RLQAS-CHEM** | RL policy over chemistry-motivated pools | **This work** |

---

## 3. Innovation Point 1: Multi-Operator-Pool Comparison (FOP / QOP)

### What
Systematically compare two operator pools within the same RL framework:
- **FOP** (Fermionic Operator Pool): standard JW-mapped UCC single/double excitations
- **QOP** (Qubit Operator Pool): Qubit-ADAPT-style direct Pauli string operators

### Why This Is a Contribution
No existing paper compares FOP and QOP within an RL search framework. ADAPT-VQE papers compare them under gradient-based greedy selection—the RL search dynamics (exploration, reward shaping, policy learning) may favor a different pool than gradient-based results suggest.

### Research Questions
1. Which pool leads to faster RL convergence (fewer episodes to chemical accuracy)?
2. Which pool produces circuits with the best energy/depth Pareto trade-off?
3. Does the best pool depend on molecule size or correlation strength?

### Experimental Protocol
- Molecules: H₂, LiH, BeH₂, H₂O (increasing correlation)
- Fixed agent: PPO (baseline)
- Fixed metric: episodes to chemical accuracy (1.6 mHa) and final CNOT count
- Compare: FOP vs QOP across all molecules

---

## 4. Innovation Point 2: GRPO for Quantum Circuit Search

### What
Introduce Group Relative Policy Optimization (GRPO, DeepSeek 2024) as a new RL algorithm for UCC circuit construction, replacing the standard actor-critic paradigm.

### Why GRPO Fits Circuit Search
Circuit construction is a sequential decision process with sparse, outcome-based reward (VQE energy only known after circuit completion)—the same structure GRPO was designed for in LLM reasoning tasks.

| LLM Reasoning (GRPO origin) | RLQAS-CHEM |
|-----------------------------|------------|
| Input: reasoning problem | Input: molecular Hamiltonian |
| Output: token sequence | Output: operator sequence (circuit) |
| Reward: answer correctness | Reward: VQE energy |
| Sparse: known only at end | Sparse: known after VQE optimization |
| Group sampling: G answers/problem | Group sampling: G circuits/molecule |
| Advantage: within-group relative | Advantage: which circuit is lower energy |

**Key advantage over PPO/A2C/SAC**: No critic network required. Critic networks for partially-built quantum circuits are difficult to design. GRPO eliminates this by comparing G sampled circuits for the same molecule.

### GRPO Advantage Estimation
For a molecule m, sample G circuits {c₁, ..., c_G}. Run VQE on each. Compute advantages:

```
A_i = (E_FCI - E_i) - mean_j(E_FCI - E_j)
         ─────────────────────────────────
              std_j(E_FCI - E_j) + ε
```

Policy update: maximize sum of clipped importance-weighted advantages (same as GRPO paper, PPO-style clip).

### Implementation Notes
- Group size G: start with G=4, tune to G=8
- VQE cost: G × (VQE calls per update)—acceptable given that each VQE is cheap for small molecules via TenCirChem
- New file: `src/rlqas_chem/rl/grpo_agent.py`

### Research Questions
1. Does GRPO converge faster than PPO for UCC circuit search (fewer VQE calls to chemical accuracy)?
2. Is GRPO more stable under sparse reward (long circuits before any energy signal)?
3. Does GRPO find lower-energy circuits than PPO given the same computational budget?

---

## 5. Innovation Point 3: Multi-Objective Pareto Optimization

### What
Simultaneously optimize energy accuracy (ΔE = |E_RL - E_FCI|) and circuit depth (CNOT count) using a scalarized multi-objective reward. Sweep the trade-off parameter α to generate a Pareto frontier.

### Reward Formulation
```
reward = α × (energy_improvement / energy_scale) - (1 - α) × (n_cnots / cnot_scale)
```

Where:
- `energy_scale`: molecule-specific normalization (e.g., UCCSD energy error)
- `cnot_scale`: maximum pool-dependent CNOT budget
- `α ∈ {0.3, 0.5, 0.7, 0.9, 1.0}`: trade-off parameter, train one agent per α

### Why This Differentiates From ADAPT-VQE
ADAPT-VQE and all variants (Pruned, K-ADAPT, Counterdiabatic) optimize a single objective: energy. They have no mechanism to explicitly control circuit depth during the search. RLQAS-CHEM can produce a Pareto frontier showing the achievable energy-depth trade-offs, which is directly useful for hardware deployment.

### Visualization
Plot Pareto frontier: x-axis = CNOT count, y-axis = energy error (mHa). Overlay ADAPT-VQE result as a single point. Show that RL discovers points ADAPT-VQE cannot reach (lower depth at acceptable energy loss, or lower energy at comparable depth).

### Implementation Notes
- `UCCRewardFunction` already supports `complexity_penalty` (default 0.01, `reward_function.py:36`). Extend to accept an `alpha` parameter that sets `energy_weight = alpha` and `complexity_penalty = (1 - alpha) / cnot_scale`.
- Train 5 agents per (molecule, pool) combination across α values
- No structural changes to `reward_function.py` needed—only parameterization of existing fields via constructor

---

## 6. Innovation Point 4: Cross-Geometry Policy Transfer (Exploratory)

### What
Train a single RL policy on multiple bond lengths of one molecule, then test its generalization to unseen bond lengths (without retraining). This tests whether the learned policy captures transferable circuit-building heuristics.

### Motivation and Caveat
**Hypothesis**: at different bond lengths of the same molecule, similar excitation patterns matter (same orbital structure, varying correlation strength). A policy trained across geometries may learn to recognize which excitations are relevant from the Hamiltonian features in the state.

**Skepticism**: Molecules may differ enough that cross-molecule transfer does not work. Cross-geometry transfer within one molecule is a weaker and more testable claim.

### Experimental Protocol (Conservative)
1. Train on H₂ at bond lengths [0.5, 0.7, 0.9, 1.1, 1.3] Å
2. Test zero-shot on H₂ at [1.5, 1.7, 2.0] Å (dissociation regime)
3. Compare: (a) zero-shot, (b) fine-tune 50 episodes, (c) train from scratch
4. Success criterion: fine-tune reaches chemical accuracy 3× faster than from scratch

If cross-geometry transfer works: report it as a result. If it does not: report the negative result honestly—this is still scientifically useful and differentiates from ADAPT-VQE (which has no transfer mechanism at all).

---

## 7. Experimental Plan Summary

| Experiment | Molecules | Agents | Pools | Primary Metric |
|------------|-----------|--------|-------|----------------|
| E1: Pool comparison | H₂, LiH, BeH₂, H₂O | PPO | FOP, QOP | Episodes to 1.6 mHa; CNOT count |
| E2: GRPO vs baselines | LiH, BeH₂ | PPO, GRPO | Pool with fewest episodes to 1.6 mHa in E1 | VQE calls to 1.6 mHa; stability |
| E3: Pareto frontier | LiH, BeH₂ | Agent with fewest VQE calls in E2 | Same pool as E2 | Pareto curve vs ADAPT-VQE |
| E4: Transfer (exploratory) | H₂ multi-geometry | Agent from E2 | Same pool as E2 | Zero-shot vs fine-tune vs scratch |

### Baseline Comparisons
- **ADAPT-VQE** (FOP pool): reference greedy method
- **BenchRL-QAS** algorithms: RL baselines on same task

---

## 8. Code Architecture Changes

### New Files
- `src/rlqas_chem/rl/grpo_agent.py`: GRPO implementation

### Modified Files
- `src/rlqas_chem/search/ucc/reward_function.py`: add `alpha` parameter for multi-objective reward
- `src/rlqas_chem/rl/agent_factory.py`: register GRPO agent
- `src/rlqas_chem/api.py`: expose `operator_pool` and `alpha` parameters

### No Changes Needed
- `search/hea/`, `search/hybrid/`: out of scope for this work
- RL base infrastructure (PPO, DQN, A2C, SAC): keep as-is for comparison

---

## 9. Differentiation Summary

| Claim | vs ADAPT-VQE | vs BenchRL-QAS | vs Hive (2026) |
|-------|-------------|----------------|----------------|
| Chemistry-motivated search space (FOP/QOP) | ✓ same pools, different search | ✗ generic gates only | ✗ arbitrary code |
| GRPO algorithm | ✗ gradient-based | ✗ not evaluated | ✗ not RL |
| Multi-objective Pareto | ✗ single-objective | ✗ not chemistry | ✗ not explicit |
| Cross-geometry transfer | ✗ no policy | ✗ no chemistry | ~ code reuse only |
