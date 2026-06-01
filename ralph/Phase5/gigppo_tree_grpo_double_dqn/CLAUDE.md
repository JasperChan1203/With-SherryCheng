# Ralph Agent Instructions — RLQAS Phase 5: New Algorithms (GiGPO, Tree-GRPO, Double-DQN)

You are an autonomous coding agent adding three new RL algorithms to the **rlqas-chem** package.

## Your Task Loop

1. Read `prd.json` in this directory
2. Read `progress.txt` if it exists
3. Pick the **lowest-priority** user story where `passes: false`
4. Implement the story following the spec in `prd.json` AND the context below
5. Run **all** acceptance criteria listed in the story
6. If all checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
7. Update `prd.json` to set `passes: true` for the completed story
8. Append progress to `progress.txt`
9. Continue until ALL stories have `passes: true`, then output: `<promise>COMPLETE</promise>`

---

## Working Directory

```
/curie-home/jpchen/scratch/LLM/code/RLQAS
```

All paths in `prd.json` are relative to this working directory.

---

## Package Location

```
rlqas-chem/src/rlqas_chem/
```

The package is installed as editable. After adding new files, you do NOT need to reinstall — just restart Python imports.

---

## Python Environment

```
/curie-home/jpchen/.conda/envs/llm/bin/python3
```

---

## Acceptance System

Every story's primary acceptance criterion is:
```
bash rlqas_acceptance_system/run_acceptance.sh --agent <agentType>
```
This must exit with code 0 for the story to be considered complete.

The acceptance system runs:
- **Level 0**: Interface contract (AgentFactory registration, act() format, save/load, minimal training)
- **Level 1**: Environment stability (monotonic global_best, reset preservation, obs types)
- **Level 2**: Serialization (JSON-serializable results, convergence_reached is Python bool)
- **Level 3**: Search functionality (UCC and HEA basic search, energy physical reasonability)
- **Level 4**: Hyperparameter constraints (n_episodes respected, max_excitations respected)
- **Level 5**: Chemical accuracy (< 1.6 mHa on LiH UCC and H2 HEA, 300 episodes)

---

## Agent Interface Contract

Every new agent MUST implement:

```python
class MyAgent(RLAgent):
    def __init__(self, observation_space, action_space, config: dict):
        ...

    def act(self, obs) -> Tuple[int, dict]:
        """Return (action_index, info_dict)"""

    def learn(self, total_timesteps: int, callback=None):
        """Train the agent. callback is optional."""

    def save(self, path: str):
        """Save model to path (no extension needed)."""

    def load(self, path: str):
        """Load model from path."""
```

Register in `rlqas-chem/src/rlqas_chem/rl/agent_factory.py`:
```python
from rlqas_chem.rl.my_agent import MyAgent
_AGENT_REGISTRY['my_agent'] = MyAgent
```

Export from `rlqas-chem/src/rlqas_chem/rl/__init__.py`.

---

## Reference: Existing Agent Pattern

Look at `rlqas-chem/src/rlqas_chem/rl/grpo_agent.py` as a reference for GRPO-based agents.
Look at `rlqas-chem/src/rlqas_chem/rl/dqn_agent.py` as a reference for DQN-based agents.
Look at `rlqas-chem/src/rlqas_chem/rl/agent_factory.py` to see how agents are registered.

---

## Implementation Order

Implement stories in priority order: **US-ALG-001 → US-ALG-002 → US-ALG-003**

Do not skip ahead. Each story builds on confirmed working infrastructure.

---

## Algorithm Reference

See `/curie-home/jpchen/scratch/LLM/code/RLQAS/ideas_pool/RLQAS_Algorithm_Selection_Improved_20260428.md` for the full algorithm selection rationale, mathematical details, and design decisions.

---

## Important Constants

- Chemical accuracy threshold: **1.6 mHa = 1.6e-3 Ha**
- `run_classical_opt=True` — must stay True in all search controllers
- `ent_coef=0.01` — default for policy-gradient agents
- Do NOT change any existing agents or controllers unless strictly necessary for integration

---

## Progress Report Format

Create `progress.txt` if it does not exist, then APPEND:

```
## [Date/Time] - [Story ID]: [Story Title]
- Status: COMPLETE
- Files created/modified: [list]
- Acceptance criteria results: [paste actual output of run_acceptance.sh]
- Notes: [any issues encountered]
---
```
