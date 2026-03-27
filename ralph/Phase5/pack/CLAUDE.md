# Ralph Agent Instructions — RLQAS Phase 5: Pack (RLQAS-CHEM Standalone Package)

You are an autonomous coding agent building **rlqas-chem**, a standalone, pip-installable Python library
that consolidates all RLQAS functionality from Phase 1–5 into a single, self-contained package.

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

## Package Goal

Build `./rlqas-chem/` — a standalone Python package with:
- **No runtime dependency** on Phase 1–4 source paths
- All Phase 5 bug fixes baked directly into the source code
- Public API: `rlqas_chem.search()`, `rlqas_chem.Experiment`, CLI `rlqas-chem`
- Installable via `pip install -e ./rlqas-chem/`

---

## Source Paths (read-only — copy from, do not modify)

All paths are relative to this CLAUDE.md file's directory (`Phase5/pack/`):

```
Phase 1 source : ../../../Phase1/006/src/rlqas/phase1/
Phase 2 source : ../../../Phase2/full/src/rlqas/phase2/
Phase 3 source : ../../../Phase3/full/src/rlqas/phase3/
Phase 4 source : ../../../Phase4/full/src/rlqas/
```

**Target (write here):**
```
./rlqas-chem/src/rlqas_chem/
```

---

## Phase 5 Fixes — MUST be baked into rlqas-chem source directly

### Fix A — UCC MDP (applies to search/ucc/environment.py)

In `UCCSearchEnv.step()`, when `action in self.selected_excitations` (duplicate):
```python
if action in self.selected_excitations:
    reward = -1.0
    terminated = True
    return self._get_observation(), reward, terminated, False, {}
```
Do NOT call this Phase 1's environment directly — copy the file and apply this edit in the copy.

### Fix B — ent_coef default (applies to search/ucc/controller.py)

Change:
```python
"ent_coef": self.config.get("ent_coef", 0.0),
```
To:
```python
"ent_coef": self.config.get("ent_coef", 0.01),
```

### Fix C — REINFORCE in HybridSearchAgent (applies to search/hybrid/agent.py)

The `train(self, states, actions, rewards)` method must NOT be a no-op. Implement:
```python
def train(self, states, actions, rewards):
    import torch
    gamma = 0.99
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    self.optimizer.zero_grad()
    total_loss = torch.tensor(0.0, requires_grad=True)
    for state, action, G_t in zip(states, actions, returns):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs = self.policy(state_tensor)
        log_prob = torch.log(action_probs[0, action] + 1e-8)
        total_loss = total_loss - G_t * log_prob
    total_loss.backward()
    self.optimizer.step()
```
If no `self.policy` exists, add a 2-layer MLP in `__init__`.

---

## Import Rewriting Rules

When copying any file from Phase 1–4, replace all occurrences of:
- `from rlqas.phase1.` → `from rlqas_chem.`
- `from rlqas.phase2.` → `from rlqas_chem.`
- `from rlqas.phase3.` → `from rlqas_chem.`
- `import rlqas.phase1` → `import rlqas_chem`
- `import rlqas.phase2` → `import rlqas_chem`
- `import rlqas.phase3` → `import rlqas_chem`

Module path mappings (Phase source → rlqas_chem target):
```
phase1.molecule.processor    → rlqas_chem.molecule.processor
phase1.simulator.tencirchem  → rlqas_chem.simulator.tencirchem
phase1.simulator.factory     → rlqas_chem.simulator.factory
phase1.rl.base_agent         → rlqas_chem.rl.base_agent
phase1.rl.ppo_agent          → rlqas_chem.rl.ppo_agent
phase1.rl.config             → rlqas_chem.rl.config
phase2.rl.dqn_agent          → rlqas_chem.rl.dqn_agent
phase2.rl.a2c_agent          → rlqas_chem.rl.a2c_agent
phase2.rl.sac_discrete_agent → rlqas_chem.rl.sac_discrete_agent
phase2.rl.agent_factory      → rlqas_chem.rl.agent_factory
phase1.search.*              → rlqas_chem.search.ucc.*
phase2.hea_search.*          → rlqas_chem.search.hea.*
phase3.hybrid_search.*       → rlqas_chem.search.hybrid.*
phase1.utils.*               → rlqas_chem.utils.*
```

---

## pyproject.toml Template

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "rlqas-chem"
version = "1.0.0"
description = "Reinforcement Learning Quantum Architecture Search for Chemistry"
requires-python = ">=3.9"
dependencies = [
    "numpy",
    "scipy",
    "torch",
    "stable-baselines3>=2.0",
    "gymnasium",
    "click",
    "tencirchem",
    "openfermion",
    "openfermionpyscf",
]

[project.scripts]
rlqas-chem = "rlqas_chem.cli:main"

[tool.setuptools.packages.find]
where = ["src"]
```

---

## Important Constants

- `run_classical_opt=True` — MUST stay True in all search controllers
- `complexity_penalty=0.0` — MUST stay 0.0 in all reward functions
- `ent_coef=0.01` — UCC controller default (Fix B)
- Chemical accuracy threshold: **1.6 mHa = 1.6e-3 Ha**

---

## Python Environment

```
/curie-home/jpchen/.conda/envs/llm/bin/python3
```

The rlqas-chem package is installed as editable: `pip install -e ./rlqas-chem/ -q`

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
