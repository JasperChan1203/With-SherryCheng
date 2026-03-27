# Ralph Agent Prompt: RLQAS Phase 2 Fix001 — Correct Partial-Circuit Architecture Search

You are Ralph, an autonomous AI agent. Your job is to fix a critical bug in the RLQAS codebase and re-validate the Phase 2 benchmarks.

## WHAT YOU ARE FIXING

A critical bug makes `UCCSearchEnv.step()` report near-FCI energy after ANY single action, regardless of which excitation operator the RL agent selects. This means the four-algorithm benchmark (PPO/DQN/A2C/SAC-Discrete) in `Phase2/full/results/algorithm_comparison/ppo_dqn_a2c_sacd_lih_10q.json` is invalid — the agents were not doing genuine architecture search.

Read `prd.json` for the full root-cause analysis. The short version:

**Bug A — `environment.py`:** When `run_classical_opt=True`, the L-BFGS-B optimizer minimizes `ucc.energy(p)` over ALL `n_params` parameters, including those for operators the agent has NOT selected. Starting from all-zeros, it always converges to the full-UCCSD minimum.

**Bug B — `tencirchem.py` (simulator):** `compute_energy()` short-circuits to `circuit.ucc.energy(circuit.params)` when `circuit.ucc` is attached, returning full-UCCSD energy regardless of the partial params.

**Physical fact:** `ucc.energy(partial_params)` where `partial_params[j]=0` for non-selected operators IS the correct partial-circuit energy. The only fix needed is to ensure the optimizer never writes non-zero values into inactive parameter slots.

---

## YOUR TASKS (in order)

### Task 1 — Fix Bug A: Constrain the Optimizer to Active Parameters

**File:** `../../Phase1/006/src/rlqas/phase1/search/environment.py`

Find the `run_classical_opt` block inside `step()`. Replace it with code that:

1. Computes `active_param_indices` = unique list of `self.excitation_to_param_idx[exc]` for each `exc` in `self.current_excitations`.
2. Defines `energy_func_partial(theta)`:
   - Creates `p = self.current_params.copy()`
   - Sets `p[active_param_indices[i]] = theta[i]` for each i
   - Returns `self.circuit_builder.evaluate_energy(None, p)`
3. Calls `minimize(energy_func_partial, x0=[self.current_params[idx] for idx in active_param_indices], method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})`
4. Writes `result.x[i]` back ONLY into `self.current_params[active_param_indices[i]]` — all other entries of `self.current_params` remain at 0.
5. Rebuild the circuit with the updated (partial) `self.current_params`.

**Important:** `evaluate_energy(circuit, params)` calls `self.ucc.energy(params)`. When `params` has zeros for non-selected operators, this correctly computes the energy of the partial circuit. Do NOT change `evaluate_energy` itself.

### Task 2 — Fix Bug B: Remove the Simulator Shortcut

**File:** `../../Phase1/006/src/rlqas/phase1/simulator/tencirchem.py`

In `compute_energy()`, find the block that does:
```python
if hasattr(circuit, 'ucc') and hasattr(circuit, 'params'):
    if hasattr(circuit.ucc, 'energy'):
        return circuit.ucc.energy(circuit.params)
```

Replace this shortcut with a call to `circuit.ucc.energy(circuit.params)` ONLY after asserting that the energy is being evaluated with the correct partial params. The safest fix is to remove the shortcut entirely and fall through to the standard CI-vector / statevector path. Alternatively, keep the shortcut but document that it is safe ONLY because Bug A guarantees `circuit.params` has zeros for non-active operators — add an assert or a clear comment.

The preferred implementation is to keep the shortcut (it is efficient) but add a comment explaining the invariant: "circuit.params must have zeros for non-selected operators; Bug A fix in environment.py guarantees this."

### Task 3 — Write Tests Confirming the Fix

**File:** `../../Phase1/006/tests/` or `Phase2/full/tests/`

Write the following tests (add to an appropriate existing test file or create `tests/integration/test_circuit_fix.py`):

1. **`test_single_operator_does_not_cheat`**: Create `UCCSearchEnv` for LiH active_space=(2,5) with `run_classical_opt=True, complexity_penalty=0.0, param_init_strategy='zeros'`. Take exactly ONE step with any action. Assert `abs(env.current_energy - fci_energy) > 1.6e-3` (one operator should NOT give FCI accuracy).

2. **`test_optimizer_only_touches_active_params`**: After the step in test 1, assert that `env.current_params` has exactly 1 non-zero entry (the active operator's parameter). Assert all other entries are zero (or very close to 0, within 1e-10).

3. **`test_full_uccsd_matches_fci`**: Directly call `ucc.energy(optimal_full_params)` (using scipy.optimize on all params) for LiH active_space=(2,5) and assert energy_error < 1.6e-3. This confirms the environment/molecule setup is still correct.

Run these tests and confirm they pass before proceeding.

### Task 4 — Re-Run Honest Benchmarks

With the corrected environment, re-run the multi-algorithm comparison:

**Using `ExplorationFramework.run_benchmarks()`** from `Phase2/full/src/rlqas/phase2/adaptation/exploration_framework.py`:
- molecule: LiH active_space=(2,5), bond_length=1.6, basis_set='sto-3g', transform='jordan_wigner'
- n_episodes=500 (increase to 2000 if chemical accuracy is not reached)
- env_config: `{'complexity_penalty': 0.0, 'param_init_strategy': 'zeros', 'max_depth': 10, 'run_classical_opt': True}`
- agent_types: `['ppo', 'dqn', 'a2c', 'sac_discrete']`

Save results to:
`Phase2/full/results/algorithm_comparison/honest_ppo_dqn_a2c_sacd_lih_10q.json`

The JSON should include per-algorithm: `best_energy, fci_energy, energy_error_ha, energy_error_mha, chemical_accuracy_reached, operator_count, best_operators`.

**Also update** `Phase2/full/results/algorithm_comparison/ppo_dqn_a2c_sacd_lih_10q.json`: add a top-level key `"WARNING"` with value `"This file contains results from a buggy implementation. See honest_ppo_dqn_a2c_sacd_lih_10q.json for correct results."` and a key `"superseded_by": "honest_ppo_dqn_a2c_sacd_lih_10q.json"`.

### Task 5 — Update Integration Tests

**File:** `Phase2/full/tests/integration/test_multi_algorithm.py`

Update `test_four_way_comparison_lih_10q` and related tests to:
1. Use the honest benchmark results (not hardcoded expectations of operator_count=1 for all agents)
2. Assert that `chemical_accuracy_reached` is True for at least ONE algorithm (not necessarily all four)
3. Add `test_single_operator_does_not_cheat` from Task 3 to the integration test suite

If `test_lih_10q_chemical_accuracy_with_full_training` fails because the agent can no longer trivially reach chemical accuracy in 500 episodes, increase `n_episodes` to 2000 and `max_depth` to 10 (the agent needs more exploration time now that it must genuinely search).

### Task 6 — Document in progress.txt

Append a section to `progress.txt`:

```
## Fix001: Partial-Circuit Architecture Search Bug — FIXED

Root cause: ...
Files changed: ...
Before (buggy): energy_error ≈ 4e-9 Ha with 1 operator for ALL algorithms
After (fixed): [actual results from honest benchmark]
```

---

## RULES

1. **Always verify your fix works before proceeding to the next task.** After fixing Bug A, run `test_single_operator_does_not_cheat` immediately.
2. **Do NOT change the Phase 2 package structure** (tasks 002-006 code is correct). Only fix the Phase 1 environment and simulator files and the Phase 2 exploration framework's benchmark runner.
3. **Do NOT change `evaluate_energy`** — `ucc.energy(partial_params)` with zeros for inactive operators is physically correct and efficient.
4. **The honest benchmarks may show worse results** (higher energy errors, more operators needed). This is expected and desirable — it means the RL agents are genuinely searching.
5. All tests that were passing before must continue to pass (except tests that were silently relying on the buggy behavior).

---

## HOW TO VERIFY YOU ARE DONE

Run these checks:
```bash
# Test 1: Single operator should NOT give chemical accuracy
# Note: rlqas.phase1 is installed as an editable package — no sys.path needed for phase1.
# Phase2/full is NOT installed; add its src to sys.path before importing phase2.adaptation:
#   import sys; sys.path.insert(0, '/curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full/src')
python3 -c "
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.environment import UCCSearchEnv
mol = process_molecule({'formula':'LiH','bond_length':1.6,'active_space':[2,5],'basis_set':'sto-3g','transform':'jordan_wigner'})
env = UCCSearchEnv(mol, {'run_classical_opt':True,'complexity_penalty':0.0,'param_init_strategy':'zeros','max_depth':10})
obs, _ = env.reset()
obs, r, term, trunc, info = env.step(0)
err = abs(env.current_energy - mol.fci_energy)
print(f'1-operator error: {err*1000:.4f} mHa')
assert err > 1.6e-3, f'BUG STILL PRESENT: 1 operator achieved chemical accuracy ({err*1000:.4f} mHa)'
nz = sum(1 for x in env.current_params if abs(x) > 1e-10)
print(f'Non-zero params: {nz} (should be 1)')
assert nz == 1, f'BUG STILL PRESENT: {nz} params non-zero after 1 action'
print('Bug A: FIXED')
"
```

Record the honest benchmark results in `progress.txt` and confirm the JSON file exists at `Phase2/full/results/algorithm_comparison/honest_ppo_dqn_a2c_sacd_lih_10q.json`.
