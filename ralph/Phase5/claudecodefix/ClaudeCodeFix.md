# RLQAS-Chem Bug Fix Report
**Date:** 2026-04-26  
**Scope:** `rlqas-chem/src/rlqas_chem/search/ucc/` and `search/hea/`  
**Author:** Claude Code (automated audit + fix)

---

## Summary

Six bug categories were identified and fixed across five source files. The fixes address correctness issues that directly impaired DQN/PPO training quality, agent exploration efficiency, and result completeness.

---

## Fix 1 — Action Masking (UCC Environment)

**File:** `search/ucc/environment.py`  
**Severity:** High

### Problem
Invalid actions (selecting an operator already in the current circuit) were handled reactively: the environment accepted the action, then terminated the episode with a −1 penalty. With no upfront masking, DQN's epsilon-greedy exploration wasted a large fraction of episodes on these immediately-terminal actions, providing no useful gradient signal.

### Fix
Added `get_valid_action_mask()` which returns a boolean array of shape `(n_actions,)`: `True` for operators not yet in the circuit, `False` for duplicates and for all actions when the circuit is already full. Callers (e.g. a masked-DQN wrapper or custom training loop) can use this mask to restrict the agent's action set before each step.

Additionally, the order of checks in `step()` was corrected: the max-depth guard now runs **before** the duplicate check (previously, a duplicate action could slip through when the circuit was already at capacity).

The penalty for exceeding max depth was unified to −1.0 (previously −10.0 for max-depth vs. −1.0 for duplicates, which created inconsistent incentives).

---

## Fix 2 — `step_count` Termination Semantics (UCC Environment)

**File:** `search/ucc/environment.py`  
**Severity:** High

### Problem
`_check_termination()` contained:
```python
if self.step_count >= max_excitations:  # WRONG
    return True
```
`step_count` increments on every call to `step()`, including penalised invalid steps that do not add an operator. Using `max_excitations` as the cap for `step_count` therefore terminated episodes prematurely whenever the agent had taken any invalid actions, even if fewer than `max_excitations` operators had actually been added.

### Fix
```python
max_steps = max(max_depth, max_excitations)
if self.step_count >= max_steps:
    return True
```
`max_steps` now acts as a safety cap against infinite loops from repeated invalid actions, while the circuit-depth conditions (`len(current_excitations) >= max_depth/max_excitations`) remain the primary termination triggers. The same correction was applied in `_get_termination_reason()`.

---

## Fix 3 — Reward Baseline Cross-Episode Preservation (UCC Environment)

**File:** `search/ucc/environment.py`  
**Severity:** Medium

### Problem
`reset()` always called `self.reward_function.update_baseline(E_HF)`, which reset the `current_best` baseline to the Hartree–Fock energy on every new episode. This defeated the purpose of the `current_best` baseline type, whose intent is to track the global best energy found so far and reward the agent only for beating it. Resetting it every episode caused reward magnitudes to be inconsistent across episodes and made the agent re-learn the same improvements from scratch.

Additionally, the `_first_evaluation` flag and `last_energy` / `consecutive_improvements` shaping state were never reset between episodes, causing the reward function to treat the first step of a new episode as a continuation of the previous one.

### Fix
```python
if self.reward_function.baseline_type == "hartree_fock":
    self.reward_function.update_baseline(self._get_hf_energy())
elif self.reward_function.hf_energy is None:
    self.reward_function.hf_energy = self._get_hf_energy()
# Always reset per-episode state
self.reward_function._first_evaluation = True
self.reward_function.last_energy = None
self.reward_function.consecutive_improvements = 0
```
The `current_best` baseline is now preserved across episodes as intended. The shaping state is correctly reset each episode.

---

## Fix 4 — Observation Parameter Encoding (UCC Environment)

**File:** `search/ucc/environment.py`  
**Severity:** Medium

### Problem
The `params_padded` component of the observation was built by copying `current_params[0:max_depth]` — a pool-order slice. Because `current_params` is a fixed-length array indexed by operator pool position, any operator with pool index ≥ `max_depth` had its optimised θ value invisible to the agent, even though the agent had explicitly selected it. With `max_depth=10` and a pool of 42 operators, parameters for operators at indices 10–41 were always zero in the observation.

### Fix
Parameters are now packed in **selection order** rather than pool order:
```python
params_padded = np.zeros(max_depth, dtype=np.float32)
for slot, exc in enumerate(self.current_excitations[:max_depth]):
    p_idx = self.excitation_to_param_idx[exc]
    params_padded[slot] = self.current_params[p_idx]
```
The agent now always sees the θ values for the operators it has actually selected, regardless of their pool index.

---

## Fix 5 — Controller: n_episodes Override, max_steps Calculation, DQN Early Stopping

**File:** `search/ucc/controller.py`  
**Severity:** High

### Problem A — n_episodes silently overridden
`UCCSearchConfig` defaults to `controller.n_episodes = 1000`. The controller's `search()` method unconditionally read this default, so a caller passing `n_episodes=2000` always had it silently replaced with 1000. Slurm logs confirmed this: jobs submitted with `--episodes 2000` reported `"Starting UCC search for 1000 episodes"`.

### Problem B — total_timesteps used wrong max_steps
```python
max_steps = self.config.get("max_excitations", 20)  # reads controller section → 20
total_timesteps = n_episodes * max_steps
```
`max_excitations` lives in the **environment** section, not the controller section. The controller always computed `total_timesteps` using the default 20 regardless of the actual environment cap.

### Problem C — DQN early stopping ineffective
`agent.learn(total_timesteps=...)` is a blocking call. The `early_stop_threshold` was only checked **after** `learn()` returned, meaning DQN always trained for the full budget even if chemical accuracy was achieved halfway through.

### Fix
```python
# A: Only override n_episodes when controller config explicitly sets a non-default value
controller_n_episodes = self.config.get("n_episodes", None)
if controller_n_episodes is not None and controller_n_episodes != 1000:
    n_episodes = controller_n_episodes

# B: Read max_excitations from the environment section
env_max_excitations = (
    raw_config.get("environment", {}).get("max_excitations")
    or raw_config.get("max_excitations")
    or self.env.config.get("max_excitations", 20)
)
total_timesteps = n_episodes * env_max_excitations

# C: Inject an SB3 EarlyStopCallback that terminates training mid-run
class EarlyStopCallback(BaseCallback):
    def _on_step(self) -> bool:
        best = getattr(self._env, 'global_best_energy', None)
        if best is not None and abs(best - self._fci_energy) < self._threshold:
            return False  # stops training
        return True
```
The `EarlyStopCallback` is automatically prepended to any user-supplied callbacks so it works transparently for both DQN and PPO.

---

## Fix 6 — Complexity Penalty Normalisation (Reward Function)

**File:** `search/ucc/reward_function.py`  
**Severity:** Medium

### Problem
The default complexity penalty was:
```python
complexity_penalty = self.complexity_penalty * circuit_complexity  # linear, unbounded
```
With `complexity_penalty=0.01` and 20 operators, the cumulative penalty was 0.20 Hartree — comparable to or larger than the energy improvement signal for complex molecules. This biased the agent against adding operators even when doing so would improve accuracy, preventing convergence on molecules like H6 that genuinely require many operators.

### Fix
```python
max_ops = max(1, self.max_operators)
complexity_penalty = self.complexity_penalty * (circuit_complexity / max_ops)
```
The penalty is now normalised by `max_operators`, keeping it in the range `[0, complexity_penalty]` regardless of circuit depth. This is already the same form used by the alpha-weighted path; the two branches are now consistent.

---

## Fix 7 — HEA Best Circuit Tracking

**Files:** `search/hea/environment.py`, `search/hea/controller.py`  
**Severity:** High (missing feature)

### Problem
`HEASearchEnv` tracked `best_energy` across episodes but never stored the corresponding circuit configuration. As a result, `HEASearchController.search()` always returned `"best_circuit": None` in its results — the best architecture found during training was permanently lost.

Additionally, the energy-tracking update in `step()` was gated on `molecule_data is not None`, meaning unit-test environments (without a real molecule) never updated `best_energy`.

### Fix (environment)
Added `best_circuit_config: Optional[Dict]` attribute initialised to `None`. Updated `step()` to always track best:
```python
if new_energy < self.best_energy:
    self.best_energy = new_energy
    self.best_circuit_config = self.get_circuit_config()
```
Removed the `molecule_data is not None` guard so test environments are also tracked.

### Fix (controller)
```python
if hasattr(self._env, 'best_energy'):
    self._best_energy = self._env.best_energy
if hasattr(self._env, 'best_circuit_config') and self._env.best_circuit_config is not None:
    self._best_circuit = self._env.best_circuit_config
```

---

## Change Summary Table

| # | File | Lines affected | Category | Severity |
|---|------|---------------|----------|----------|
| 1 | `ucc/environment.py` | `step()`, new `get_valid_action_mask()` | Action masking | High |
| 2 | `ucc/environment.py` | `_check_termination()`, `_get_termination_reason()` | Termination logic | High |
| 3 | `ucc/environment.py` | `reset()` | Reward baseline | Medium |
| 4 | `ucc/environment.py` | `_get_observation()` | Observation encoding | Medium |
| 5 | `ucc/controller.py` | `search()` | n_episodes, max_steps, DQN early stop | High |
| 6 | `ucc/reward_function.py` | `compute_reward()` | Complexity penalty | Medium |
| 7 | `hea/environment.py` | `__init__()`, `step()` | Best circuit tracking | High |
| 7 | `hea/controller.py` | `search()` | Best circuit collection | High |
