# Ralph Agent Instructions — RLQAS Phase 5: RL Training Bugfix

You are an autonomous coding agent fixing two confirmed RL training bugs in the RLQAS codebase.

## Your Task

1. Read `prd.json` in this directory
2. Read `progress.txt` (check Codebase Patterns section first)
3. Pick the **highest priority** user story where `passes: false`
4. Implement the fix
5. Run quality checks (pytest)
6. If checks pass, commit ALL changes with message: `fix: [Story ID] - [Story Title]`
7. Update `prd.json` to set `passes: true` for the completed story
8. Append progress to `progress.txt`

## Stop Condition

If ALL stories have `passes: true`, output:
```
<promise>COMPLETE</promise>
```

---

## Critical Context: What Is Broken and Why

### Bug 1 — UCCSearchController: PPO Never Trains (US-001, HIGH)

**File:** `../../Phase1/006/src/rlqas/phase1/search/controller.py`

**Root cause:** `UCCPPOAgent.train()` (lines 38-42) is a no-op:
```python
def train(self):
    pass  # no-op — PPO policy is NEVER updated
```
The manual episode loop (lines 127-214) calls `self.agent.train()` each episode but nothing happens.
After 500 episodes the PPO policy is still the same random initialization.

**Consequence:** The agent always selects the same operator (or random operators). After step 1 adds the first unique operator, steps 2-20 all try to add the same one again (duplicate → -1 reward). Episode ends with only 1 unique operator. For LiH 12q, 1 operator is insufficient for chemical accuracy (6.7 mHa vs. threshold 1.6 mHa).

**How to fix:** Replace the broken manual episode loop in `UCCSearchController.search()` with a call to `self.agent.learn()`, which already correctly calls SB3's `model.learn(total_timesteps=...)`. Then read results from `self.env`.

```python
def search(self, n_episodes: int = 1000, early_stop_threshold: float = 1.6e-3):
    n_episodes = self.config.get("n_episodes", n_episodes)
    early_stop_threshold = self.config.get("early_stop_threshold", early_stop_threshold)
    max_steps = self.config.get("max_excitations", 20)
    total_timesteps = n_episodes * max_steps

    print(f"Starting UCC search for {n_episodes} episodes ({total_timesteps} timesteps)")
    print(f"Early stop threshold: {early_stop_threshold} Hartree")

    self.agent.learn(total_timesteps=total_timesteps)

    # Read results from env's global tracking (updated by env.step() during learn())
    self.best_overall_energy = self.env.global_best_energy
    self.best_overall_excitations = (
        self.env.global_best_excitations.copy()
        if self.env.global_best_excitations else []
    )
    self.best_overall_params = (
        self.env.global_best_params.copy()
        if self.env.global_best_params is not None else None
    )

    self.results['best_energy'] = self.best_overall_energy
    self.results['best_excitations'] = self.best_overall_excitations
    self.results['best_params'] = self.best_overall_params
    self.results['convergence_reached'] = self._check_convergence(early_stop_threshold)

    print(f"Search completed. Best energy: {self.best_overall_energy:.6f} Hartree")
    print(f"Best excitations: {len(self.best_overall_excitations)} operators")
    return self.results
```

**Key detail:** `PPOAgent.learn()` at `../../Phase1/006/src/rlqas/phase1/rl/ppo_agent.py` (lines 169-201) already calls `self.model.learn(total_timesteps=total_timesteps)` correctly. `UCCPPOAgent` inherits this method. The env's `global_best_energy` / `global_best_excitations` / `global_best_params` are updated by `UCCSearchEnv.step()` (lines 269-275 of environment.py) during SB3's internal episode loop.

**Also remove** the now-unused `UCCPPOAgent.train()` and `UCCPPOAgent.store_experience()` no-op methods, or keep them if removing risks breaking other callers.

---

### Bug 2 — HEASearchController: _best_energy stays float('inf') (US-002, LOW)

**File:** `../../Phase2/full/src/rlqas/phase2/hea_search/controller.py`

**Root cause:** `_best_energy` (line 59) is initialized to `float('inf')` and is only updated inside `_run_episode()` (line 227). But `search()` calls `self._agent.learn(total_timesteps=...)` which uses SB3's internal episode loop — `_run_episode()` is never called.

**How to fix:**
1. Inspect `../../Phase2/full/src/rlqas/phase2/hea_search/environment.py` to see if `HEASearchEnv` tracks `best_energy` (similar to Phase 1's `global_best_energy`)
2. If yes: after `self._agent.learn(...)`, set `self._best_energy = self._env.best_energy`
3. If no: add `best_energy` tracking to `HEASearchEnv.step()` (same pattern as Phase 1) and then read it back

**Note:** Phase 4 already routes `ansatz_type='HEA'` to Phase 3 `HybridSearchController` as a workaround, so this bug does not affect `rlqas.search()`. Fix it directly in Phase 2 for correctness.

---

## File Paths

```
Phase 1 controller  : ../../Phase1/006/src/rlqas/phase1/search/controller.py
Phase 1 ppo_agent   : ../../Phase1/006/src/rlqas/phase1/rl/ppo_agent.py
Phase 1 environment : ../../Phase1/006/src/rlqas/phase1/search/environment.py
Phase 1 tests       : ../../Phase1/006/tests/
Phase 2 controller  : ../../Phase2/full/src/rlqas/phase2/hea_search/controller.py
Phase 2 environment : ../../Phase2/full/src/rlqas/phase2/hea_search/environment.py
Phase 2 tests       : ../../Phase2/full/tests/
```

All paths are relative to this CLAUDE.md file's directory (`Phase5/bugfix001/`).

---

## Validation Commands

```bash
# Use the conda llm environment
PYTHON=/curie-home/jpchen/.conda/envs/llm/bin/python3

# Anti-hollow check for US-001
$PYTHON -c "
import rlqas
r = rlqas.search('H2', 0.74, ansatz_type='UCC', agent_type='ppo', n_episodes=50)
assert r['energy_error_mha'] > 0 and r['energy_error_mha'] < 50
print(f'H2 UCC: {r[\"energy_error_mha\"]:.3f} mHa, operators={r[\"n_operators\"]}')
r2 = rlqas.search('LiH', 1.6, ansatz_type='UCC', agent_type='ppo', n_episodes=300)
assert r2['n_operators'] >= 2, f'Expected n_operators >= 2, got {r2[\"n_operators\"]}'
print(f'LiH UCC: {r2[\"energy_error_mha\"]:.3f} mHa, operators={r2[\"n_operators\"]}')
print('US-001 PASS')
"

# Run Phase 1 tests
cd ../../Phase1/006 && $PYTHON -m pytest tests/ -x -q

# Anti-hollow check for US-002
$PYTHON -c "
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase2.hea_search.controller import HEASearchController
mol = process_molecule('H2', 0.74, 'HEA')
ctrl = HEASearchController(mol)
result = ctrl.search(n_episodes=50, total_timesteps=500)
assert result['best_energy'] != float('inf'), f'best_energy is still inf!'
print(f'HEA H2: best_energy={result[\"best_energy\"]:.6f} Ha')
print('US-002 PASS')
"

# Run Phase 2 tests
cd ../../Phase2/full && $PYTHON -m pytest tests/ -x -q
```

---

## Progress Report Format

APPEND to `progress.txt` (never replace):
```
## [Date/Time] - [Story ID]: [Story Title]
- Status: COMPLETE
- Files modified: [list]
- Anti-hollow results: [output]
- Learnings:
  - [any patterns or gotchas]
---
```

## Codebase Patterns

- Python env: `/curie-home/jpchen/.conda/envs/llm/bin/python3`
- Phase 4 unified package: install with `pip install -e ../../Phase4/full/` from conda env
- After modifying Phase 1/2 source, no reinstall needed (installed as editable `-e`)
- `run_classical_opt=True` must stay True — disabling breaks energy evaluation
- `complexity_penalty=0.0` must stay 0.0 — non-zero is 62x too large vs chemical accuracy
- Chemical accuracy threshold: 1.6 mHa = 1.6e-3 Ha
- Valid ansatz_type: UCC, HEA, HYBRID
- Valid agent_type: ppo, dqn, a2c, sac_discrete
