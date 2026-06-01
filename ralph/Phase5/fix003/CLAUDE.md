# Ralph Agent Instructions — RLQAS Phase 5 Fix 003: QOP + Hybrid GRPO + HEA Cleanup

You are an autonomous coding agent fixing four bugs in the rlqas-chem package.
All bugs are independent (no inter-story dependencies except US-004 depending on US-001).
The acceptance system has pre-written tests that are **currently failing** — your job is to make them pass.

## Your Task

1. Read `prd.json` in this directory
2. Read `progress.txt` if it exists
3. Pick the **lowest-numbered** user story where `passes: false`
4. Implement the fix following the specifications in prd.json AND the detailed context below
5. Run the acceptance commands from prd.json to verify
6. If all pass, commit ALL changes with message: `fix: [Story ID] - [Story Title]`
7. Update `prd.json` to set `passes: true` for the completed story
8. Append progress to `progress.txt`
9. Continue until ALL stories have `passes: true`, then output: `<promise>COMPLETE</promise>`

---

## Codebase Layout

```
rlqas-chem/              ← main package (editable install, changes take effect immediately)
  src/rlqas_chem/
    api.py               ← top-level search() function
    search/
      ucc/
        circuit_builder.py   ← UCCCircuitBuilder: wraps TenCirChem UCCSD
        environment.py       ← UCCSearchEnv: Gymnasium env for UCC search
        controller.py        ← UCCSearchController: training loop + GRPO dispatch
      hea/
        controller.py        ← HEASearchController: HEA training loop
      hybrid/
        controller.py        ← HybridSearchController: Hybrid training loop
      qop/
        controller.py        ← QubitUCCSearchController: BROKEN (see US-001)
    molecule/
      processor.py           ← process_molecule() → MoleculeData

rlqas_acceptance_system/ ← test suite (run from RLQAS repo root)
  rl_algorithms/
    test_acceptance_level1_5.py  ← 4 new QOP tests already written and currently FAILING
  hea_algorithms/
    test_acceptance_hea_level1_5.py  ← EXISTING_AGENTS needs updating
```

All paths in this file are relative to `Phase5/fix003/` (this file's directory).

---

## US-001: QOP Fix — Detailed Technical Specification

### Problem

`api.py` lines 105-113 route `operator_pool='qop'` to `QubitUCCSearchController`.
That controller builds a `QubitOperatorPool` from the Hamiltonian but then creates
`UCCSearchEnv` with the **default** fermion `UCCCircuitBuilder`. The qubit pool is
used only for `performance_metrics['qubit_pool_size']`. The RL agent trains on
fermion circuits. QOP == FOP + one fake number.

### Correct TenCirChem Interface

```python
# FOP (default)
from tencirchem import UCCSD
u = UCCSD(mol, mode="fermion", active_space=(2, 5), init_method="zeros")
u.kernel()
n_fop_cnots = u.get_circuit(decompose_multicontrol=True).gate_summary().get("cnot", 0)

# QOP — just change mode
u = UCCSD(mol, mode="qubit", active_space=(2, 5), init_method="zeros")
u.kernel()
n_qop_cnots = u.get_circuit(decompose_multicontrol=True).gate_summary().get("cnot", 0)
# n_qop_cnots < n_fop_cnots for molecules with >= 4 qubits
```

Both modes have the **same** `ex_ops`, `param_ids`, and `n_params`.
The difference is purely in circuit implementation (qubit mode drops Z strings from JW-transformed Pauli rotations).
This means the RL action space is identical — the same UCCSearchEnv logic works for both modes.

### Implementation Plan

#### 1. UCCCircuitBuilder (`search/ucc/circuit_builder.py`)

In `__init__`, after reading `config`, add:

```python
self._ucc_mode = (config or {}).get("ucc_mode", "fermion")
```

Then add a helper method `_build_ucc_for_mode(self, mode: str)` that:
1. Reconstructs the PySCF mol object from `molecule_data.molecular_info`
   (the formula/bond_length/basis_set parsing block already exists in the fallback branch — reuse it)
2. Gets `active_space = molecule_data.molecular_info.get("active_space", None)`
3. Returns `UCCSD(mol, mode=mode, active_space=active_space, init_method="zeros")`

Modify the `__init__` branching:
- If `self._ucc_mode == "qubit"`: **always** call `self.ucc = self._build_ucc_for_mode("qubit")`,
  regardless of whether `ucc_sd_object` exists (the existing object was built with mode="fermion").
- If `self._ucc_mode == "fermion"`: keep existing logic unchanged.

Add `get_cnot_count()` method:
```python
def get_cnot_count(self) -> int:
    try:
        circ = self.ucc.get_circuit(decompose_multicontrol=True)
        return int(circ.gate_summary().get("cnot", 0))
    except Exception:
        return 0
```

#### 2. UCCSearchController (`search/ucc/controller.py`)

In `search()`, `_grpo_search()`, and `_double_dqn_search()`, add to the results dict before returning:
```python
self.results['cnot_count'] = self.env.circuit_builder.get_cnot_count()
```

**Where to add it**: all three return paths write to `self.results` and then `return self.results`.
Add the cnot_count line immediately before each `return self.results`.

#### 3. api.py

Replace lines 105-113 (the QOP branch):
```python
# BEFORE (broken):
if ansatz_type == "UCC" and operator_pool == "qop":
    from rlqas_chem.search.qop import QubitUCCSearchController
    ctrl = QubitUCCSearchController(...)
    ...

# AFTER (correct):
if ansatz_type == "UCC" and operator_pool == "qop":
    ctrl_config["ucc_mode"] = "qubit"
    ctrl = UCCSearchController(mol, agent_type=agent_type, config=ctrl_config)
    result = ctrl.search(n_episodes=n_episodes, early_stop_threshold=early_stop_threshold)
    best_energy = float(_extract(result, "best_energy"))
    n_ops = _extract(result, "best_excitations")
    n_operators = len(n_ops) if n_ops else None
    fusion_template = None
```

In both the FOP and QOP return paths (the `elif ansatz_type == "UCC" and agent_type in _UCC_AGENTS:` block AND the new QOP block), extract cnot_count:
```python
cnot_count = _extract(result, "cnot_count", default=None)
```

Add `"cnot_count": cnot_count` to the returned dict.

---

## US-002: Hybrid GRPO Fix — Detailed Technical Specification

### Problem

`HybridSearchController.search()` (lines 255-357) uses a per-episode loop.
The agent is `_AgentAdapter` which implements `train()` via REINFORCE (actually present in the code).
GRPO/GiGPO/Tree-GRPO implement learning via `train_one_group(env)`, not REINFORCE.
The base agent's `learn()` is a stub. So Hybrid+GiGPO = REINFORCE policy over GiGPO's network weights,
which is wrong — GiGPO's network gets no gradient updates from its group-relative reward normalization.

### Implementation Plan

#### 1. HybridSearchController.search()

Add dispatch at the start of `search()`, before the episode loop:

```python
_GRPO_TYPES = ('grpo', 'gigppo', 'tree_grpo')
if self.agent_type.lower() in _GRPO_TYPES:
    return self._run_grpo_loop(n_eps=n_eps, threshold=threshold)
```

#### 2. Add `_run_grpo_loop` to HybridSearchController

```python
def _run_grpo_loop(self, n_eps: int, threshold: float) -> SearchResult:
    """Group-based training loop for GRPO-family agents."""
    base_agent = self.agent._agent  # unwrap _AgentAdapter
    group_size = getattr(base_agent, 'group_size', 8)
    n_groups = max(1, n_eps // group_size)
    log_freq = self.ctrl_config.get("log_frequency", 10)

    print(
        f"[HybridSearchController] GRPO loop: {n_groups} groups × {group_size} "
        f"= {n_groups * group_size} episodes"
    )

    for g in range(n_groups):
        group_result = base_agent.train_one_group(self.env)

        env_best = getattr(self.env, 'global_best_energy', float('inf'))
        if self.env.global_best_energy < self._best_energy:
            self._best_energy = self.env.global_best_energy
            self._best_excitations = list(self.env.global_best_excitations)

        self._training_history.append({
            "group": g,
            "best_energy": env_best if env_best != float('inf') else None,
        })

        if g % log_freq == 0:
            best_str = f"{env_best:.6f}" if env_best != float('inf') else "N/A"
            print(f"  Group {g:4d}: best={best_str}")

        if (
            self.molecule_data.fci_energy is not None
            and env_best != float('inf')
            and abs(env_best - self.molecule_data.fci_energy) < threshold
        ):
            print(f"  Converged at group {g}: error={abs(env_best - self.molecule_data.fci_energy)*1000:.4f} mHa")
            break

    # Build result
    best_err = None
    converged = False
    if self.molecule_data.fci_energy is not None and self._best_energy != float('inf'):
        best_err = abs(self._best_energy - self.molecule_data.fci_energy)
        converged = best_err < 1.6e-3

    fusion_template = self.fusion_strategy.generate_fusion_template()

    return SearchResult(
        best_circuit=None,
        best_energy=self._best_energy if self._best_energy != float('inf') else None,
        best_error=best_err,
        training_history=self._training_history,
        performance_metrics={"n_groups": n_groups, "convergence_reached": converged},
        fusion_template=fusion_template,
        convergence_reached=converged,
    )
```

---

## US-003: HEA Tree-GRPO Block — Detailed Technical Specification

### Problem

`HEASearchController._run_grpo_loop` includes `'tree_grpo'` in `_grpo_types`.
Tree-GRPO's prefix cache requires deterministic UCC operator semantics — not present in HEA.

### Implementation Plan

#### 1. HEASearchController.search() (`search/hea/controller.py`)

Add a guard near the top of `search()`, before `setup_agent()`:
```python
if agent_type.lower() == 'tree_grpo':
    raise ValueError(
        "Tree-GRPO is UCC-only: its prefix-sharing cache requires deterministic "
        "operator sequences that only exist in UCC search. "
        "Use agent_type='gigppo' for HEA sparse-reward problems instead."
    )
```

#### 2. HEASearchController._run_grpo_loop()

Change:
```python
_grpo_types = ('grpo', 'gigppo', 'tree_grpo')
```
to:
```python
_grpo_types = ('grpo', 'gigppo')
```

#### 3. api.py

Add guard in the HEA branch:
```python
elif ansatz_type == "HEA":
    if agent_type.lower() == 'tree_grpo':
        raise ValueError("Tree-GRPO is UCC-only. Use gigppo for HEA.")
    ...
```

#### 4. test_acceptance_hea_level1_5.py

Change the line:
```python
EXISTING_AGENTS = ["ppo", "dqn", "a2c", "sac_discrete", "grpo"]
```
to:
```python
EXISTING_AGENTS = ["ppo", "dqn", "a2c", "sac_discrete", "grpo", "gigppo", "double_dqn"]
```

---

## US-004: QOP Acceptance Test Verification

No code changes required. Just run:

```bash
PYTHON=/curie-home/jpchen/.conda/envs/llm/bin/python3
cd /curie-home/jpchen/scratch/LLM/code/RLQAS

$PYTHON -m pytest rlqas_acceptance_system/rl_algorithms/test_acceptance_level1_5.py -k 'qop' -v
```

All 4 QOP tests must pass. If `test_chemical_accuracy_qop` fails with LiH > 1.6 mHa at 300 episodes,
try 500 episodes by editing the test file temporarily and re-running. Document results in progress.txt.

---

## Python Environment

```bash
PYTHON=/curie-home/jpchen/.conda/envs/llm/bin/python3
```

The `rlqas-chem` package is installed as **editable** (`pip install -e .`).
All edits to files under `rlqas-chem/src/` take effect immediately — no reinstall needed.

---

## Validation — Quick Smoke Test After Each Story

```bash
PYTHON=/curie-home/jpchen/.conda/envs/llm/bin/python3
REPO=/curie-home/jpchen/scratch/LLM/code/RLQAS
cd $REPO

# US-001: QOP gives valid energy + cnot_count field
$PYTHON -c "
import rlqas_chem
r = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', operator_pool='qop', n_episodes=20)
print('QOP result:', r)
assert r['energy_error_mha'] < 10.0, f'QOP silent failure: {r[\"energy_error_mha\"]:.2f} mHa > 10 mHa'
assert r.get('cnot_count') is not None, 'Missing cnot_count'
print('US-001 smoke: PASS')
"

# US-001: QOP CNOT < FOP CNOT
$PYTHON -c "
import rlqas_chem
r_f = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', operator_pool='fop', n_episodes=5)
r_q = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', operator_pool='qop', n_episodes=5)
print(f'FOP cnot={r_f[\"cnot_count\"]}  QOP cnot={r_q[\"cnot_count\"]}')
assert r_q['cnot_count'] < r_f['cnot_count'], 'QOP must have fewer CNOTs'
print('US-001 CNOT check: PASS')
"

# US-002: Hybrid GiGPO runs without error and takes real time
$PYTHON -c "
import rlqas_chem, time
t = time.time()
r = rlqas_chem.search('H2', 0.74, ansatz_type='HYBRID', agent_type='gigppo', n_episodes=40)
elapsed = time.time() - t
print(f'Hybrid GiGPO: energy={r[\"best_energy\"]}, elapsed={elapsed:.1f}s')
assert r['best_energy'] is not None
assert elapsed > 5, f'Suspiciously fast ({elapsed:.1f}s) — may be a no-op'
print('US-002 smoke: PASS')
"

# US-003: HEA tree_grpo raises ValueError
$PYTHON -c "
import rlqas_chem
try:
    rlqas_chem.search('H2', 0.74, ansatz_type='HEA', agent_type='tree_grpo', n_episodes=5)
    print('FAIL: expected ValueError')
except ValueError as e:
    print(f'US-003 tree_grpo block: PASS ({e})')
"

# US-003: HEA gigppo works
$PYTHON -c "
import rlqas_chem
r = rlqas_chem.search('H2', 0.74, ansatz_type='HEA', agent_type='gigppo', n_episodes=20)
print('HEA GiGPO energy:', r.get('best_energy'))
assert r.get('best_energy') is not None
print('US-003 HEA gigppo: PASS')
"

# US-004: Full QOP acceptance tests
$PYTHON -m pytest rlqas_acceptance_system/rl_algorithms/test_acceptance_level1_5.py -k 'qop' -v
```

---

## Progress Report Format

Create or append to `progress.txt`:

```
## [Date/Time] - [Story ID]: [Story Title]
- Status: COMPLETE
- Files modified: [list]
- Acceptance results: [paste key command output]
- Notes: [any surprises or edge cases]
---
```

---

## Important Notes

- `run_classical_opt=True` must stay True — disabling breaks energy evaluation (reward = HF energy forever)
- `complexity_penalty=0.0` must stay 0.0 — non-zero is 62× too large vs chemical accuracy threshold
- Chemical accuracy threshold: **1.6 mHa = 1.6e-3 Ha**
- TenCirChem UCCSD `mode="qubit"` and `mode="fermion"` have the **same** `ex_ops` and `n_params` — the RL action space does not change, only the circuit implementation differs
- When building UCCSD with `mode="qubit"`, always use `init_method="zeros"` (not "mp2") to avoid convergence issues
- The `active_space` for the fresh UCCSD must come from `molecule_data.molecular_info["active_space"]`
- Do NOT modify `rlqas_acceptance_system/rl_algorithms/test_acceptance_level1_5.py` — those tests are the acceptance criteria and must pass as-is
