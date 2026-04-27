# DQN LiH Validation — Design Spec

**Date:** 2026-04-19  
**Status:** Approved  
**Context:** PPO test (job 68273) failed on explained_variance (0.0957 < 0.1) and operator efficiency (9 ops > 6 target). This spec defines a parallel DQN test to determine if an off-policy algorithm can meet the same pass criteria.

---

## Goals

Test whether a DQN agent can:
1. Reach chemical accuracy (< 1.6 mHa error vs FCI) on LiH @ 1.6 Å, active space (4,6)
2. Do so using ≤ 6 excitation operators

Pass criteria are identical to the PPO test for direct comparison.

---

## Scope

**In scope:**
- New `DQNDiagnosticsCallback` in `rlqas_chem/rl/`
- Add `dqn` branch to `UCCSearchController`
- New standalone `validate_lih_dqn.py` test script
- New `submit_validate_dqn.sh` Slurm script

**Out of scope:**
- Modifying the existing PPO test or `DiagnosticsCallback`
- Hyperparameter tuning (use reasonable DQN defaults)
- Multi-seed runs

---

## Architecture

```
Files changed:
rlqas-chem/src/rlqas_chem/rl/dqn_diagnostics_callback.py   [NEW]
rlqas-chem/src/rlqas_chem/search/ucc/controller.py          [MODIFIED]
rlqas_test/validate_lih_dqn.py                              [NEW]
rlqas_test/submit_validate_dqn.sh                           [NEW]
```

---

## Component Designs

### 1. `DQNDiagnosticsCallback`

**Location:** `rlqas-chem/src/rlqas_chem/rl/dqn_diagnostics_callback.py`

DQN is off-policy and does not trigger `_on_rollout_end`. Instead, metrics are sampled in `_on_step` every `sample_freq` steps (default: 2048, matching PPO rollout size for comparable data density).

**Metrics collected per sample:**
- `train/loss` — Q-network MSE loss (from `model.logger`)
- `train/exploration_rate` — current ε value
- `global_best_energy` — best energy seen across all episodes (from `env.global_best_energy`)

**`summary()` pass criteria:**

| Metric | Pass Condition | Rationale |
|--------|---------------|-----------|
| `q_loss_trend` | mean(last 20%) < mean(first 20%) | Proxy for "learning" (replaces explained_variance) |
| `exploration_decay` | final ε < initial ε | ε-greedy decaying as expected |
| `energy_trend` | final best_energy < first best_energy | Agent improving |

**`_save()`:** writes JSON to `output_path` (same format as `DiagnosticsCallback`).

### 2. `UCCSearchController` modification

**Location:** `rlqas-chem/src/rlqas_chem/search/ucc/controller.py`

Add `elif agent_type.lower() == 'dqn':` branch in `__init__` that instantiates `DQNAgent` with the following defaults (tuned for RLQAS episodic structure):

```python
dqn_defaults = {
    "learning_rate": 1e-3,
    "buffer_size": 50000,
    "batch_size": 64,
    "exploration_fraction": 0.3,    # 30% of timesteps for epsilon decay
    "exploration_final_eps": 0.05,
    "learning_starts": 1000,        # fill buffer before learning
    "train_freq": 4,
    "target_update_interval": 500,
    "verbose": 1,
    "use_gpu": False,
}
```

The existing `search()` method's PPO path (`self.agent.learn(total_timesteps=..., callback=callbacks)`) works for DQN as SB3's DQN also accepts callbacks. No changes to `search()` needed.

### 3. `validate_lih_dqn.py`

**Location:** `rlqas_test/validate_lih_dqn.py`

Mirror of `validate_lih_learning.py` with:
- `agent_type='dqn'` passed to `UCCSearchController`
- `DQNDiagnosticsCallback` instead of `DiagnosticsCallback`
- DQN-specific config dict (overrides defaults above)
- `summary()` uses `q_loss_trend` + `exploration_decay` instead of `explained_variance` + `entropy_slope`
- PASS criteria unchanged: chemical accuracy AND operator efficiency

DQN config in script:
```python
config = {
    'max_excitations': 30,
    'run_classical_opt': True,
    'param_init_strategy': 'zeros',
    'use_early_stop': True,
    'learning_rate': 1e-3,
    'buffer_size': 50000,
    'batch_size': 64,
    'exploration_fraction': 0.3,
    'exploration_final_eps': 0.05,
    'learning_starts': 1000,
    'train_freq': 4,
    'target_update_interval': 500,
    'verbose': 1,
}
```

### 4. `submit_validate_dqn.sh`

**Location:** `rlqas_test/submit_validate_dqn.sh`

Same SLURM parameters as `submit_validate.sh` (4V100, 32G, 6h). Changes:
- `--job-name=rlqas-dqn-validate`
- Output: `slurm_logs/dqn_validate_%j.{out,err}`
- Calls `validate_lih_dqn.py`
- Output files: `results/dqn_lih_validation_${SLURM_JOB_ID}.json` and `results/dqn_lih_diagnostics_${SLURM_JOB_ID}.json`

---

## Data Flow

```
submit_validate_dqn.sh
  └─► validate_lih_dqn.py
        ├─ process_molecule(LiH, 1.6Å, active_space=(4,6))
        ├─ UCCSearchController(mol, agent_type='dqn', config)
        │    └─ DQNAgent(env=UCCSearchEnv)
        ├─ DQNDiagnosticsCallback(output_path, sample_freq=2048)
        ├─ controller.search(n_episodes=2000, callbacks=callback)
        │    └─ agent.learn(total_timesteps=60000, callback)
        │         └─ [every 2048 steps] callback._on_step → record metrics
        └─ summary() → Pass/Fail report + JSON output
```

---

## Error Handling

- DQN's `DQNConfig` validates keys strictly; unknown keys raise `KeyError`. The config dict in `validate_lih_dqn.py` must only contain keys recognized by `DQNConfig`. Keys like `max_excitations`, `run_classical_opt`, `param_init_strategy`, `use_early_stop` are consumed by the controller/environment, not passed to `DQNConfig`.
- If `train/loss` is not yet in `model.logger` (before `learning_starts`), `DQNDiagnosticsCallback` stores `None` for that sample.

---

## Testing

No automated tests are added. Validation is the test itself. The JSON output files serve as the artifact for result comparison.

---

## Success Criteria

The implementation is complete when:
1. `submit_validate_dqn.sh` can be submitted to SLURM without errors
2. The script runs to completion and produces both JSON output files
3. The PASS/FAIL report is printed with the same structure as the PPO test
