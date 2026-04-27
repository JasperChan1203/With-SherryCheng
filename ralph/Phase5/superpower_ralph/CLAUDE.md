# Ralph Agent Instructions — RLQAS Phase 6: Innovation Extensions

You are an autonomous coding agent extending **rlqas-chem** with four innovation directions:
QOP operator pool, GRPO algorithm, multi-objective Pareto reward, and experiment runners.

## Your Task

1. Read `prd.json` in this directory
2. Read `progress.txt` if it exists
3. Pick the **lowest-numbered** user story where `passes: false`
4. Implement the story following the specifications in prd.json AND the context below
5. Run the acceptance criteria checks listed in the story
6. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
7. Update `prd.json` to set `passes: true` for the completed story
8. Append progress to `progress.txt`
9. Continue until ALL stories have `passes: true`, then output: `<promise>COMPLETE</promise>`

---

## Package Location

The existing rlqas-chem package is at: `../pack/rlqas-chem/`
It is already installed as editable: `pip install -e ../pack/rlqas-chem/ -q`

**Python environment**: `/curie-home/jpchen/.conda/envs/llm/bin/python3`

If you need to re-install after changes: `pip install -e ../pack/rlqas-chem/ -q`

---

## Source Paths (read-only — copy from, do not modify)

All paths are relative to this CLAUDE.md file's directory (`Phase5/superpower_ralph/`):

```
Phase 1 source : ../../Phase1/006/src/rlqas/phase1/
Phase 2 source : ../../Phase2/full/src/rlqas/phase2/
Phase 3 source : ../../Phase3/full/src/rlqas/phase3/
```

**Key source for US-009:**
```
Phase 3 qubit_ops : ../../Phase3/full/src/rlqas/phase3/qubit_ops/
```

**Target (write here):**
```
../pack/rlqas-chem/src/rlqas_chem/
```

---

## Import Rewriting Rules

When copying files from Phase 1–3, replace ALL occurrences:

| From | To |
|------|----|
| `from rlqas.phase1.molecule.processor` | `from rlqas_chem.molecule` |
| `from rlqas.phase1.molecule` | `from rlqas_chem.molecule` |
| `from rlqas.phase1.search.environment` | `from rlqas_chem.search.ucc.environment` |
| `from rlqas.phase1.search` | `from rlqas_chem.search.ucc` |
| `from rlqas.phase2.rl.agent_factory` | `from rlqas_chem.rl.agent_factory` |
| `from rlqas.phase2.rl` | `from rlqas_chem.rl` |
| `from rlqas.phase3.qubit_ops` | `from rlqas_chem.search.qop` |
| `import rlqas.phase1` | `import rlqas_chem` |
| `import rlqas.phase2` | `import rlqas_chem` |
| `import rlqas.phase3` | `import rlqas_chem` |

---

## Critical Constants — Do NOT Change

- `run_classical_opt=True` — must remain True in all search controllers
- `complexity_penalty=0.0` — base default (alpha parameter controls this in US-011)
- `ent_coef=0.01` — UCC controller default (Phase 5 fix, must not regress)
- Chemical accuracy threshold: **1.6 mHa = 1.6e-3 Ha**

---

## Existing rlqas-chem API (do not break)

```python
import rlqas_chem

# Existing API — must still work after all changes
result = rlqas_chem.search('LiH', 1.6, ansatz_type='UCC', agent_type='ppo', n_episodes=300)
result = rlqas_chem.search('H2', 0.74, ansatz_type='HEA', agent_type='ppo', n_episodes=100)
result = rlqas_chem.search('H2', 0.74, ansatz_type='HYBRID', agent_type='ppo', n_episodes=100)
```

New parameters added by this Phase are all **optional with backward-compatible defaults**:
- `operator_pool='fop'` (default 'fop' = existing behavior)
- `alpha=1.0` (default 1.0 = existing behavior)

---

## Anti-Hollow Checks

### US-009 (QOP):
```python
# QOP pool must return non-trivial operators
from rlqas_chem.molecule import process_molecule
from rlqas_chem.search.qop import QubitOperatorPool
mol = process_molecule('LiH', 1.6, 'UCC', active_space=(2,5))
pool = QubitOperatorPool(mol)
assert pool.get_pool_size() >= 2, f"QOP pool too small: {pool.get_pool_size()}"
```

### US-010 (GRPO):
```python
# GRPO must produce different energies across group members (real sampling)
import rlqas_chem
r = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type='grpo', n_episodes=20)
assert r['best_energy'] < -1.0, "GRPO energy suspiciously high — check VQE is running"
```

### US-011 (alpha):
```python
# alpha=1.0 must be backward compatible
from rlqas_chem.search.ucc.reward_function import UCCRewardFunction
rf_default = UCCRewardFunction()
rf_alpha1  = UCCRewardFunction({'alpha': 1.0})
e1 = rf_default.compute_reward(-1.1, 3)
e2 = rf_alpha1.compute_reward(-1.1, 3)
assert abs(e1 - e2) < 1e-10, f"alpha=1.0 breaks backward compat: {e1} vs {e2}"
```

---

## Progress Report Format

Create `progress.txt` if it does not exist, then APPEND:

```
## [Date/Time] - [Story ID]: [Story Title]
- Status: COMPLETE
- Files created/modified: [list]
- Acceptance criteria results: [paste actual output]
- Notes: [any issues encountered]
---
```

---

## Story Execution Order

```
US-009 (QOP)  →  US-010 (GRPO)  →  US-011 (alpha)  →  US-012 (Optuna)
                                                      →  US-013 (Experiments, depends on US-009/010/011)
                                                      →  US-014 (Transfer, depends on US-009/011)
```

US-013 and US-014 can only run after US-009, US-010, US-011 are complete.
