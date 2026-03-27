# Ralph Agent Instructions — RLQAS Phase 5: Genuine RL Circuit Search

You are an autonomous coding agent implementing genuine reinforcement learning in the RLQAS codebase.
The previous bugfix (Phase5/bugfix001) made PPO training run, but deep analysis of LiH benchmark results
reveals the RL agent still cannot learn effectively. This task fixes the root causes.

## Your Task

1. Read `prd.json` in this directory
2. Read `progress.txt` if it exists (check Codebase Patterns section)
3. Pick the **lowest-numbered** user story where `passes: false`
4. Implement the fix following the specifications in prd.json AND the detailed context below
5. Run quality checks (pytest + anti-hollow checks)
6. If checks pass, commit ALL changes with message: `fix: [Story ID] - [Story Title]`
7. Update `prd.json` to set `passes: true` for the completed story
8. Append progress to `progress.txt`
9. Continue until ALL stories have `passes: true`, then output: `<promise>COMPLETE</promise>`

---

## Critical Context: Why RLQAS Does Not Actually Learn

### Evidence From Benchmark Tests

The benchmark test at `../../test/rlqas_lih/` ran 5 LiH configurations through Phase 1 UCCSearchController:

```
[FAIL] LiH_UCC_PPO_baseline_12q    error=4.747 mHa  (threshold: 1.6 mHa)
[FAIL] LiH_UCC_PPO_baseline_10q    error=3.217 mHa
[FAIL] LiH_UCC_PPO_tuned_12q       error=6.374 mHa  (energy_weight=100, ent_coef=0.05)
[FAIL] LiH_UCC_PPO_tuned_10q       error=3.217 mHa
[FAIL] LiH_UCC_PPO_tuned_10q_highent error=3.217 mHa
```

Every Phase 1 run returns exactly 2 operators. Meanwhile, Phase 3 HybridSearchController passes LiH in 35
episodes with 1.56 mHa — but its agent.train() is a no-op. Phase 3 succeeds via **random search**, not RL.

### Root Cause 1 — Phase 1 MDP Structural Flaw (US-001)

**File:** `../../Phase1/006/src/rlqas/phase1/search/environment.py`

The episode terminates when `step_count >= max_excitations` (default 20), counting ALL steps including
duplicates. Here is what happens in a typical episode after the PPO policy finds 2 good operators:

```
Step 1: select op_A  → reward = +0.008  (energy drops)
Step 2: select op_B  → reward = +0.003  (energy drops more)
Step 3: select op_A  → reward = -1.0    (duplicate!)
Step 4: select op_A  → reward = -1.0    (duplicate!)
...
Step 20: select op_A → reward = -1.0    (duplicate!)
```

Total return ≈ 0.011 + 18 × (−1.0) = **−17.989**

If the agent tried op_C at step 3 instead:
```
Step 3: select op_C  → reward ≈ +0.001 or −0.002 (uncertain, maybe negative)
Steps 4-20: more duplicates → −17 penalty
```

The risk-adjusted expected return of exploring op_C is WORSE than repeating op_A (which ends with the same
duplicate penalties). The PPO value network correctly predicts that exploration does not pay off under
this MDP structure. **The agent is rational; the MDP is broken.**

**Fix:** In `UCCSearchEnv.step()`, when `action in self.selected_excitations`:
- Set `terminated = True` (end episode immediately)
- Give `reward = -1.0` (still penalizes, but does not waste remaining budget)
- Do NOT increment step_count or add to selected_excitations

With this fix, selecting a duplicate immediately ends the episode. The agent learns: "repeat an operator
and the episode ends; select a new one and the episode continues." This creates a clean incentive to
explore new operators each step.

```python
# In UCCSearchEnv.step():
if action in self.selected_excitations:
    # Duplicate action — terminate episode immediately
    reward = -1.0
    terminated = True
    return self._get_observation(), reward, terminated, truncated, info
```

Also update `_check_termination()` to terminate when `len(self.selected_excitations) >= max_excitations`
(unique operators reached the limit). The default `max_excitations=20` is fine — in practice episodes
will terminate on unique-limit or first-duplicate, rarely reaching 20 unique operators.

---

### Root Cause 2 — Phase 1 ent_coef=0.0 (US-002)

**File:** `../../Phase1/006/src/rlqas/phase1/search/controller.py`

Find the line that reads `self.config.get("ent_coef", 0.0)` and change the default to `0.01`:

```python
# Before:
"ent_coef": self.config.get("ent_coef", 0.0),
# After:
"ent_coef": self.config.get("ent_coef", 0.01),
```

With `ent_coef=0.0`, PPO's entropy bonus term is zero. The policy rapidly becomes deterministic and
stops exploring. `ent_coef=0.01` (the SB3 default and AgentConfig.DEFAULT_CONFIG value) adds a small
entropy regularization that keeps the policy from collapsing.

---

### Root Cause 3 — Phase 3 agent.train() is No-Op (US-004)

**Files:** `../../Phase3/full/src/rlqas/phase3/hybrid_search/`

Phase 3 HybridSearchController calls `self.agent.train(states, actions, rewards)` after each episode,
but this method is a no-op. The agent runs a pure random policy for all 200+ episodes.

Before implementing the fix, READ the Phase 3 source files:
1. `controller.py` — understand what self.agent is
2. The agent class file (likely `agent.py` or similar) — understand what train() should do
3. The policy network structure (if any)

Implement REINFORCE (Monte Carlo Policy Gradient):

```python
def train(self, states, actions, rewards):
    """REINFORCE: Monte Carlo policy gradient update."""
    import torch

    # Compute discounted returns
    gamma = 0.99
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize returns for stability
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Policy gradient loss: L = -E[G_t * log pi(a_t|s_t)]
    self.optimizer.zero_grad()
    total_loss = torch.tensor(0.0, requires_grad=True)
    for state, action, G_t in zip(states, actions, returns):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.policy(state_tensor)  # shape: [1, n_actions]
        log_prob = torch.log(action_probs[0, action] + 1e-8)
        total_loss = total_loss - G_t * log_prob

    total_loss.backward()
    self.optimizer.step()
```

If the agent has no policy network (`policy` attribute), add one:
```python
import torch.nn as nn
self.policy = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.ReLU(),
    nn.Linear(64, n_actions),
    nn.Softmax(dim=-1)
)
self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)
```

The `obs_dim` and `n_actions` can be read from the environment passed to the agent.

---

## File Paths

```
Phase 1 controller   : ../../Phase1/006/src/rlqas/phase1/search/controller.py
Phase 1 environment  : ../../Phase1/006/src/rlqas/phase1/search/environment.py
Phase 1 tests        : ../../Phase1/006/tests/
Phase 3 hybrid_search: ../../Phase3/full/src/rlqas/phase3/hybrid_search/
Phase 3 tests        : ../../Phase3/full/tests/
```

All paths relative to this CLAUDE.md file's directory (`Phase5/fix002/`).

---

## Validation Commands

```bash
PYTHON=/curie-home/jpchen/.conda/envs/llm/bin/python3

# ── US-001: Verify duplicate action terminates episode ─────────────────────
$PYTHON -c "
import sys
sys.path.insert(0, '../../Phase1/006/src')
sys.path.insert(0, '../../Phase4/full/src')
import rlqas
from rlqas.phase1.search.environment import UCCSearchEnv
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.config import UCCSearchConfig

mol = process_molecule('H2', 0.74, 'UCC')
config = UCCSearchConfig.DEFAULT_CONFIG.copy()
env = UCCSearchEnv(mol, config)
obs, _ = env.reset()
n_actions = env.action_space.n

# Select first action
obs, r1, term1, trunc1, _ = env.step(0)
assert not term1, 'Episode should not terminate on first unique action'

# Select same action again (duplicate)
obs, r2, term2, trunc2, _ = env.step(0)
assert term2, f'Duplicate action should terminate episode immediately, got term={term2}'
assert r2 == -1.0, f'Duplicate reward should be -1.0, got {r2}'
print('US-001 duplicate-terminates-episode: PASS')
"

# ── US-001 + US-002 combined: Phase 1 LiH exploration check ───────────────
$PYTHON -c "
import rlqas
r = rlqas.search('LiH', 1.6, ansatz_type='UCC', agent_type='ppo', n_episodes=300)
print(f'LiH UCC PPO: energy_error={r[\"energy_error_mha\"]:.3f} mHa, n_operators={r[\"n_operators\"]}')
assert r['n_operators'] >= 4, f'Expected >= 4 operators, got {r[\"n_operators\"]}'
print('n_operators >= 4: PASS')
"

# ── US-003: Chemical accuracy check ────────────────────────────────────────
$PYTHON -c "
import rlqas
r = rlqas.search('LiH', 1.6, ansatz_type='UCC', agent_type='ppo', n_episodes=300)
err = r['energy_error_mha']
print(f'LiH energy_error = {err:.3f} mHa (threshold: 1.6 mHa)')
assert err < 1.6, f'Chemical accuracy not reached: {err:.3f} mHa'
print('US-003 chemical accuracy: PASS')
"

# ── Phase 1 unit tests ──────────────────────────────────────────────────────
cd ../../Phase1/006 && $PYTHON -m pytest tests/ -x -q

# ── US-004: Phase 3 policy changes over training ───────────────────────────
$PYTHON -c "
import sys, numpy as np
sys.path.insert(0, '../../Phase3/full/src')
sys.path.insert(0, '../../Phase4/full/src')
from rlqas.phase3.hybrid_search.controller import HybridSearchController
from rlqas.phase1.molecule.processor import process_molecule
mol = process_molecule('H2', 0.74, 'HYBRID')
ctrl = HybridSearchController(mol, config={})

# Capture initial policy distribution (if accessible)
# After training 200 episodes, policy should differ from initial
result = ctrl.search(n_episodes=200)
print(f'H2 HYBRID: best_energy={result[\"best_energy\"]:.6f} Ha')
assert result['best_energy'] < -1.0, f'Energy too high: {result[\"best_energy\"]}'
print('US-004 Phase 3 energy check: PASS')
"

# ── Phase 3 integration tests ──────────────────────────────────────────────
cd ../../Phase3/full && $PYTHON -m pytest tests/ -x -q
```

---

## Important Notes

- **Python env:** `/curie-home/jpchen/.conda/envs/llm/bin/python3`
- Phase 1 package is installed as **editable** (`pip install -e .`) — edits take effect immediately
- Phase 3 package is installed as **editable** — edits take effect immediately
- `run_classical_opt=True` must stay True — disabling breaks energy evaluation
- `complexity_penalty=0.0` must stay 0.0 — non-zero is 62× too large vs chemical accuracy
- Chemical accuracy threshold: **1.6 mHa = 1.6e-3 Ha**
- US-003 depends on US-001 and US-002 being complete first — implement in priority order
- US-004 (Phase 3) is independent of US-001/002/003 (Phase 1) — can be done in any order relative to US-003

---

## Progress Report Format

Create `progress.txt` if it does not exist, then APPEND (never replace):

```
## [Date/Time] - [Story ID]: [Story Title]
- Status: COMPLETE
- Files modified: [list]
- Anti-hollow results: [paste actual command output]
- Learnings:
  - [any patterns or gotchas discovered]
---
```
