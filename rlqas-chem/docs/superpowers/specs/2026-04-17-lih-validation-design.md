# RLQAS LiH Learning Validation — Design Spec

**Date**: 2026-04-17  
**Status**: Approved

---

## Problem

The RLQAS slurm run (compare_63701) showed `explained_variance` staying near 0 or negative throughout training, meaning the value network is not learning. Before extending to HEA/Hybrid or running large benchmarks, we need to confirm whether PPO is actually learning a policy or performing sophisticated random search.

---

## Goal

Validate that the PPO agent in RLQAS is genuinely learning — i.e., the policy improves over training — using **LiH as the test molecule** (12 qubits, UCC ansatz, FCI energy = -7.784460 Ha).

---

## Pass Criteria (all must be satisfied)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| `explained_variance` (final 20% of training) | > 0.1 | Value network has predictive power |
| Policy entropy | Decreasing trend over training | Policy becoming more focused |
| Best energy per checkpoint | Monotonically decreasing trend | Agent is improving |
| Chemical accuracy (1.6 mHa) | Reached within 2000 episodes | End-to-end correctness |

---

## Changes to rlqas-chem

### 1. New file: `src/rlqas_chem/rl/diagnostics_callback.py`

An SB3-compatible callback that captures per-update training metrics:
- `explained_variance`, `entropy_loss`, `policy_gradient_loss`, `approx_kl`, `value_loss`
- Best energy from env at each update checkpoint (via `env.global_best_energy`)
- Saves to JSON on completion

### 2. Modified: `src/rlqas_chem/search/ucc/controller.py`

Add optional `callbacks` parameter to `UCCSearchController.search()`.  
Pass through to `agent.learn(total_timesteps, callback=callbacks)`.

### 3. Modified: `src/rlqas_chem/rl/__init__.py`

Export `DiagnosticsCallback`.

---

## Test Script

**Location**: `rlqas_test/validate_lih_learning.py`

**Steps**:
1. Run LiH UCC PPO, 2000 episodes, with `DiagnosticsCallback`
2. Load diagnostics JSON
3. Evaluate all 4 pass criteria
4. Print PASS / FAIL with evidence

**Slurm script**: `rlqas_test/submit_validate.sh`

---

## Directory

All new files are in:
- `rlqas-chem/src/rlqas_chem/rl/diagnostics_callback.py` (new)
- `rlqas-chem/src/rlqas_chem/search/ucc/controller.py` (modified)
- `rlqas-chem/src/rlqas_chem/rl/__init__.py` (modified)
- `rlqas_test/validate_lih_learning.py` (new)
- `rlqas_test/submit_validate.sh` (new)
