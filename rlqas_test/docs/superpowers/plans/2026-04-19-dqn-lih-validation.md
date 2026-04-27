# DQN LiH Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DQN agent support to RLQAS and run the same LiH learning validation that PPO failed (job 68273), to determine if an off-policy algorithm can achieve chemical accuracy with ≤ 6 operators.

**Architecture:** Minimal-touch approach — new `DQNDiagnosticsCallback` (off-policy compatible, samples on `_on_step`), add `callback` parameter to `DQNAgent.learn()`, wire a `dqn` branch into `UCCSearchController`, then create standalone validation script and Slurm submit script.

**Tech Stack:** Python 3.10, Stable-Baselines3 DQN, Gymnasium, RLQAS-chem (local install), SLURM

---

## File Map

| Action | Path | Purpose |
|--------|------|---------|
| CREATE | `rlqas-chem/src/rlqas_chem/rl/dqn_diagnostics_callback.py` | DQN-compatible diagnostics callback |
| MODIFY | `rlqas-chem/src/rlqas_chem/rl/dqn_agent.py:341-375` | Add `callback` param to `learn()` |
| MODIFY | `rlqas-chem/src/rlqas_chem/search/ucc/controller.py:44-98` | Add `dqn` branch in `__init__` |
| CREATE | `rlqas_test/validate_lih_dqn.py` | Standalone DQN validation script |
| CREATE | `rlqas_test/submit_validate_dqn.sh` | SLURM submission script |

---

## Task 1: `DQNDiagnosticsCallback`

**Files:**
- Create: `rlqas-chem/src/rlqas_chem/rl/dqn_diagnostics_callback.py`
- Test: `rlqas-chem/tests/test_dqn_diagnostics_callback.py`

- [ ] **Step 1: Write the failing test**

```python
# rlqas-chem/tests/test_dqn_diagnostics_callback.py
"""Tests for DQNDiagnosticsCallback."""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from rlqas_chem.rl.dqn_diagnostics_callback import DQNDiagnosticsCallback


class FakeModel:
    """Minimal SB3-like model mock."""
    def __init__(self):
        self.logger = MagicMock()
        self.logger.name_to_value = {
            "train/loss": 0.5,
            "train/exploration_rate": 0.8,
        }
        self.num_timesteps = 0


class FakeEnv:
    """Minimal env mock with global_best_energy."""
    def __init__(self, energy):
        self.global_best_energy = energy

    @property
    def unwrapped(self):
        return self


class FakeTrainingEnv:
    """Minimal DummyVecEnv mock."""
    def __init__(self, energies):
        self.envs = [FakeEnv(e) for e in energies]


def test_callback_records_samples_on_step():
    """DQNDiagnosticsCallback records a sample every sample_freq steps."""
    cb = DQNDiagnosticsCallback(output_path="/tmp/test_dqn_diag.json",
                                 sample_freq=3, verbose=0)
    cb.model = FakeModel()
    cb.training_env = FakeTrainingEnv([-7.88])

    # First 2 steps: no sample yet
    cb.model.num_timesteps = 1
    cb._on_step()
    cb.model.num_timesteps = 2
    cb._on_step()
    assert len(cb.samples) == 0

    # Third step triggers sample
    cb.model.num_timesteps = 3
    cb._on_step()
    assert len(cb.samples) == 1
    assert cb.samples[0]["q_loss"] == pytest.approx(0.5)
    assert cb.samples[0]["exploration_rate"] == pytest.approx(0.8)
    assert cb.samples[0]["global_best_energy"] == pytest.approx(-7.88)


def test_summary_energy_trend_pass():
    """summary() returns energy_trend_pass=True when energy improves."""
    cb = DQNDiagnosticsCallback(output_path="/tmp/test_dqn_diag.json", verbose=0)
    cb.samples = [
        {"step": i, "q_loss": 1.0 - i * 0.05, "exploration_rate": 1.0 - i * 0.05,
         "global_best_energy": -7.87 - i * 0.001}
        for i in range(10)
    ]
    s = cb.summary()
    assert s["energy_trend_pass"] is True
    assert s["exploration_decay_pass"] is True
    assert s["q_loss_trend_pass"] is True


def test_summary_energy_trend_fail():
    """summary() returns energy_trend_pass=False when energy does not improve."""
    cb = DQNDiagnosticsCallback(output_path="/tmp/test_dqn_diag.json", verbose=0)
    cb.samples = [
        {"step": i, "q_loss": 0.5, "exploration_rate": 0.1,
         "global_best_energy": -7.87}
        for i in range(10)
    ]
    s = cb.summary()
    assert s["energy_trend_pass"] is False


def test_save_writes_json():
    """_save() writes samples to the output path as valid JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        cb = DQNDiagnosticsCallback(output_path=path, verbose=0)
        cb.samples = [{"step": 1, "q_loss": 0.4, "exploration_rate": 0.9,
                       "global_best_energy": -7.88}]
        cb._save()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["q_loss"] == pytest.approx(0.4)
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
/curie-home/jpchen/.conda/envs/llm/bin/python -m pytest tests/test_dqn_diagnostics_callback.py -v
```

Expected: `ImportError` or `ModuleNotFoundError: No module named 'rlqas_chem.rl.dqn_diagnostics_callback'`

- [ ] **Step 3: Create `dqn_diagnostics_callback.py`**

```python
# rlqas-chem/src/rlqas_chem/rl/dqn_diagnostics_callback.py
"""SB3 callback for capturing DQN learning diagnostics during RLQAS training.

DQN is off-policy and does not trigger _on_rollout_end. Instead, this
callback samples metrics in _on_step every `sample_freq` timesteps.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DQNDiagnosticsCallback(BaseCallback):
    """Captures per-step training metrics from SB3 DQN for learning validation.

    Samples every `sample_freq` timesteps:
    - Q-network loss (train/loss)
    - Exploration rate / epsilon (train/exploration_rate)
    - Best energy across all episodes (env.global_best_energy)

    Args:
        output_path: Path to save diagnostics JSON on training end.
        sample_freq: Collect a sample every this many timesteps.
        checkpoint_freq: Save running JSON every N samples (0 = only on end).
        verbose: Verbosity level.
    """

    def __init__(self, output_path: str, sample_freq: int = 2048,
                 checkpoint_freq: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.output_path = output_path
        self.sample_freq = sample_freq
        self.checkpoint_freq = checkpoint_freq
        self.samples: List[Dict[str, Any]] = []
        self._sample_count = 0

    def _on_training_start(self) -> None:
        self.samples = []
        self._sample_count = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.sample_freq != 0:
            return True

        record: Dict[str, Any] = {"step": self.num_timesteps}

        # SB3 DQN logger keys
        log = self.model.logger.name_to_value
        record["q_loss"] = float(log["train/loss"]) if "train/loss" in log else None
        record["exploration_rate"] = (
            float(log["train/exploration_rate"])
            if "train/exploration_rate" in log
            else None
        )

        # Best energy from vectorised env
        try:
            envs = self.training_env.envs
            best_energies = [
                getattr(e.unwrapped if hasattr(e, "unwrapped") else e,
                        "global_best_energy", None)
                for e in envs
            ]
            valid = [e for e in best_energies if e is not None]
            record["global_best_energy"] = float(min(valid)) if valid else None
        except Exception:
            record["global_best_energy"] = None

        self.samples.append(record)
        self._sample_count += 1

        if self.checkpoint_freq > 0 and self._sample_count % self.checkpoint_freq == 0:
            self._save()

        return True

    def _on_training_end(self) -> None:
        self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.samples, f, indent=2)
        if self.verbose:
            print(f"[DQNDiagnosticsCallback] saved {len(self.samples)} samples → {self.output_path}")

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def q_loss_series(self) -> List[Optional[float]]:
        return [r["q_loss"] for r in self.samples]

    def exploration_series(self) -> List[Optional[float]]:
        return [r["exploration_rate"] for r in self.samples]

    def best_energy_series(self) -> List[Optional[float]]:
        return [r["global_best_energy"] for r in self.samples]

    def summary(self) -> Dict[str, Any]:
        """Return a pass/fail summary of DQN learning diagnostics."""
        losses = [v for v in self.q_loss_series() if v is not None]
        epsilons = [v for v in self.exploration_series() if v is not None]
        energies = [v for v in self.best_energy_series() if v is not None]

        n = len(losses)
        tail = max(1, n // 5)  # last 20%
        head = max(1, n // 5)  # first 20%

        # Q-loss: last 20% mean < first 20% mean
        q_loss_trend_pass = False
        if n >= 2:
            q_loss_trend_pass = float(np.mean(losses[-tail:])) < float(np.mean(losses[:head]))

        # Exploration rate: final < initial (epsilon decaying)
        exploration_decay_pass = (
            len(epsilons) >= 2 and epsilons[-1] < epsilons[0]
        )

        # Energy trend: final best < first best
        energy_trend_pass = (
            len(energies) >= 2 and energies[-1] < energies[0]
        )

        return {
            "n_samples": len(self.samples),
            "q_loss_first": float(np.mean(losses[:head])) if losses else None,
            "q_loss_last": float(np.mean(losses[-tail:])) if losses else None,
            "q_loss_trend_pass": q_loss_trend_pass,
            "exploration_rate_first": epsilons[0] if epsilons else None,
            "exploration_rate_last": epsilons[-1] if epsilons else None,
            "exploration_decay_pass": exploration_decay_pass,
            "best_energy_first": energies[0] if energies else None,
            "best_energy_last": energies[-1] if energies else None,
            "energy_trend_pass": energy_trend_pass,
            "overall_pass": q_loss_trend_pass and exploration_decay_pass and energy_trend_pass,
        }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
/curie-home/jpchen/.conda/envs/llm/bin/python -m pytest tests/test_dqn_diagnostics_callback.py -v
```

Expected:
```
PASSED tests/test_dqn_diagnostics_callback.py::test_callback_records_samples_on_step
PASSED tests/test_dqn_diagnostics_callback.py::test_summary_energy_trend_pass
PASSED tests/test_dqn_diagnostics_callback.py::test_summary_energy_trend_fail
PASSED tests/test_dqn_diagnostics_callback.py::test_save_writes_json
4 passed
```

- [ ] **Step 5: Commit**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
git add src/rlqas_chem/rl/dqn_diagnostics_callback.py tests/test_dqn_diagnostics_callback.py
git commit -m "feat: add DQNDiagnosticsCallback for off-policy learning validation"
```

---

## Task 2: Add `callback` parameter to `DQNAgent.learn()`

**Files:**
- Modify: `rlqas-chem/src/rlqas_chem/rl/dqn_agent.py:341-375`

The existing `DQNAgent.learn()` does not pass `callback` to `self.model.learn()`. This means the DQNDiagnosticsCallback cannot be used. Fix by adding `callback=None`.

- [ ] **Step 1: Write the failing test**

Add to `rlqas-chem/tests/test_dqn_diagnostics_callback.py`:

```python
def test_dqn_agent_learn_accepts_callback():
    """DQNAgent.learn() must accept and pass a callback without error."""
    from rlqas_chem.rl.dqn_agent import DQNAgent
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    agent = DQNAgent(config={"learning_starts": 10, "buffer_size": 100,
                              "batch_size": 10, "verbose": 0}, env=env)

    callback_called = []

    class CountCB:
        def __init__(self):
            self.n_calls = 0
        def init_callback(self, model):
            pass
        def on_step(self):
            self.n_calls += 1
            return True

    # Must not raise TypeError about unexpected keyword argument
    agent.learn(total_timesteps=50, callback=None)  # callback=None is the minimal test
    env.close()
```

Run:
```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
/curie-home/jpchen/.conda/envs/llm/bin/python -m pytest tests/test_dqn_diagnostics_callback.py::test_dqn_agent_learn_accepts_callback -v
```

Expected: FAIL with `TypeError: DQNAgent.learn() got an unexpected keyword argument 'callback'`

- [ ] **Step 2: Modify `DQNAgent.learn()`**

In `rlqas-chem/src/rlqas_chem/rl/dqn_agent.py`, find the `learn()` method (around line 341) and change:

```python
    def learn(self, experience: Optional[Dict] = None, total_timesteps: int = 10000) -> Dict:
```

to:

```python
    def learn(self, experience: Optional[Dict] = None, total_timesteps: int = 10000,
              callback=None) -> Dict:
```

And change:

```python
        # Train on environment
        self.model.learn(total_timesteps=total_timesteps)
```

to:

```python
        # Train on environment
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
```

- [ ] **Step 3: Run the test to confirm it passes**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
/curie-home/jpchen/.conda/envs/llm/bin/python -m pytest tests/test_dqn_diagnostics_callback.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
git add src/rlqas_chem/rl/dqn_agent.py tests/test_dqn_diagnostics_callback.py
git commit -m "fix: add callback parameter to DQNAgent.learn()"
```

---

## Task 3: Add `dqn` branch to `UCCSearchController`

**Files:**
- Modify: `rlqas-chem/src/rlqas_chem/search/ucc/controller.py:64-98`

- [ ] **Step 1: Write the failing test**

```python
# Add to rlqas-chem/tests/test_dqn_diagnostics_callback.py

def test_ucc_controller_accepts_dqn_agent_type():
    """UCCSearchController should instantiate a DQNAgent when agent_type='dqn'."""
    from unittest.mock import patch, MagicMock
    from rlqas_chem.search.ucc.controller import UCCSearchController
    from rlqas_chem.rl.dqn_agent import DQNAgent

    mol = MagicMock()
    mol.n_qubits = 4
    mol.fci_energy = -7.882324
    mol.hamiltonian = MagicMock()
    mol.reference_state = None
    mol.molecular_info = {"hf_energy": -7.8}

    # Patch _init_model to prevent SB3 from trying to inspect the mock env
    with patch("rlqas_chem.search.ucc.controller.UCCSearchEnv"), \
         patch("rlqas_chem.search.ucc.controller.SimulatorFactory"), \
         patch.object(DQNAgent, "_init_model"):
        controller = UCCSearchController(mol, agent_type='dqn', config={})
        assert isinstance(controller.agent, DQNAgent)
```

Run:
```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
/curie-home/jpchen/.conda/envs/llm/bin/python -m pytest tests/test_dqn_diagnostics_callback.py::test_ucc_controller_accepts_dqn_agent_type -v
```

Expected: FAIL with `ValueError: Unsupported agent type: dqn`

- [ ] **Step 2: Add `dqn` branch to `UCCSearchController.__init__()`**

In `rlqas-chem/src/rlqas_chem/search/ucc/controller.py`, after the imports at the top of the file add (if not already present):

```python
from rlqas_chem.rl.dqn_agent import DQNAgent
```

Then find the agent creation block (around line 64) that ends with:
```python
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")
```

Insert before that `else:`:

```python
        elif agent_type.lower() == 'dqn':
            raw = config or {}
            dqn_config = {
                "learning_rate": raw.get("learning_rate", 1e-3),
                "buffer_size": raw.get("buffer_size", 50000),
                "batch_size": raw.get("batch_size", 64),
                "exploration_fraction": raw.get("exploration_fraction", 0.3),
                "exploration_initial_eps": raw.get("exploration_initial_eps", 1.0),
                "exploration_final_eps": raw.get("exploration_final_eps", 0.05),
                "learning_starts": raw.get("learning_starts", 1000),
                "train_freq": raw.get("train_freq", 4),
                "gradient_steps": raw.get("gradient_steps", 1),
                "target_update_interval": raw.get("target_update_interval", 500),
                "gamma": raw.get("gamma", 0.99),
                "tau": raw.get("tau", 1.0),
                "max_grad_norm": raw.get("max_grad_norm", 10.0),
                "verbose": raw.get("verbose", 1),
                "use_gpu": raw.get("use_gpu", False),
                "seed": raw.get("seed", 42),
            }
            self.agent = DQNAgent(config=dqn_config, env=self.env)
```

- [ ] **Step 3: Run the test**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
/curie-home/jpchen/.conda/envs/llm/bin/python -m pytest tests/test_dqn_diagnostics_callback.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 4: Commit**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
git add src/rlqas_chem/search/ucc/controller.py tests/test_dqn_diagnostics_callback.py
git commit -m "feat: add DQN agent type to UCCSearchController"
```

---

## Task 4: Create `validate_lih_dqn.py`

**Files:**
- Create: `rlqas_test/validate_lih_dqn.py`

No additional unit test — this script is itself the integration test (its exit code is the result).

- [ ] **Step 1: Create the script**

```python
#!/usr/bin/env python3
"""
LiH UCC DQN Learning Validation
=================================
Validates that the DQN agent in RLQAS can achieve the same goals as the
PPO baseline: chemical accuracy + ≤ 6 excitation operators on LiH @ 1.6 Å.

Pass criteria (ALL must be satisfied):
  1. Chemical accuracy: |best_energy - FCI| < 1.6 mHa
  2. Operator efficiency: ≤ 6 excitation operators (matching ADAPT-VQE)
  3. Q-loss trend: mean(last 20% losses) < mean(first 20% losses)
  4. Exploration decay: final epsilon < initial epsilon
  5. Energy trend: final best_energy < first recorded best_energy

Usage:
  python validate_lih_dqn.py [--episodes 2000] [--output results/dqn_lih_validation.json]
"""

import argparse
import datetime
import json
import os
import sys

RLQAS_CHEM = os.path.join(os.path.dirname(__file__), '..', 'rlqas-chem', 'src')
sys.path.insert(0, os.path.abspath(RLQAS_CHEM))

import numpy as np
from rlqas_chem.molecule.processor import process_molecule
from rlqas_chem.search.ucc.controller import UCCSearchController
from rlqas_chem.rl.dqn_diagnostics_callback import DQNDiagnosticsCallback

MOLECULE     = 'LiH'
BOND_LENGTH  = 1.6
ACTIVE_SPACE = (4, 6)
CHEM_ACC     = 1.6e-3   # Hartree
TARGET_OPS   = 6        # ADAPT-VQE needs 5 @ 1.6 Å; we target ≤ 6


def run_validation(n_episodes: int, output_path: str, diag_path: str):
    print("=" * 60)
    print("RLQAS LiH DQN Learning Validation")
    print(f"Molecule    : {MOLECULE} @ {BOND_LENGTH} Å  active_space={ACTIVE_SPACE}")
    print(f"Episodes    : {n_episodes}")
    print(f"Started     : {datetime.datetime.now()}")
    print("=" * 60)

    # --- Setup ---
    mol = process_molecule(MOLECULE, BOND_LENGTH, 'UCC', active_space=ACTIVE_SPACE)
    fci_energy = mol.fci_energy
    print(f"FCI energy  : {fci_energy:.6f} Ha  (from processor)")
    print(f"Target ops  : ≤ {TARGET_OPS} (ADAPT-VQE @ 1.6 Å needs 5)")

    # DQN config — only keys recognized by DQNConfig + env/controller keys
    config = {
        # Environment / controller keys (consumed before reaching DQNConfig)
        'max_excitations': 30,
        'run_classical_opt': True,
        'param_init_strategy': 'zeros',
        'use_early_stop': True,
        # DQN-specific keys (extracted by controller dqn branch)
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

    controller = UCCSearchController(mol, agent_type='dqn', config=config)

    # sample_freq=2048 matches PPO rollout size for comparable data density
    callback = DQNDiagnosticsCallback(
        output_path=diag_path,
        sample_freq=2048,
        checkpoint_freq=5,
        verbose=1,
    )

    # --- Run ---
    results = controller.search(
        n_episodes=n_episodes,
        early_stop_threshold=CHEM_ACC,
        callbacks=callback,
    )

    # --- Evaluate ---
    summary = callback.summary()

    best_energy   = results.get('best_energy', float('inf'))
    best_ops      = len(results.get('best_excitations') or [])
    chem_acc_pass = abs(best_energy - fci_energy) < CHEM_ACC if best_energy != float('inf') else False
    ops_pass      = chem_acc_pass and best_ops <= TARGET_OPS
    energy_error  = abs(best_energy - fci_energy) * 1000  # mHa

    print("\n" + "=" * 60)
    print("DQN LEARNING DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"  Samples collected      : {summary['n_samples']}")
    print(f"  Q-loss trend           : {summary['q_loss_first']:.4f} → {summary['q_loss_last']:.4f}"
          f"  {'✓ PASS' if summary['q_loss_trend_pass'] else '✗ FAIL (loss should decrease)'}")
    print(f"  Exploration decay      : ε {summary['exploration_rate_first']:.3f} → {summary['exploration_rate_last']:.3f}"
          f"  {'✓ PASS' if summary['exploration_decay_pass'] else '✗ FAIL (epsilon should decrease)'}")
    print(f"  Energy trend           : {summary['best_energy_first']:.6f} → {summary['best_energy_last']:.6f} Ha"
          f"  {'✓ PASS' if summary['energy_trend_pass'] else '✗ FAIL'}")
    print(f"  Chemical accuracy      : error={energy_error:.3f} mHa"
          f"  {'✓ PASS' if chem_acc_pass else '✗ FAIL (need < 1.6 mHa)'}")
    print(f"  Operator efficiency    : {best_ops} ops (target ≤ {TARGET_OPS})"
          f"  {'✓ PASS' if ops_pass else f'✗ FAIL (need ≤ {TARGET_OPS} ops with chem acc)'}")

    overall = summary['overall_pass'] and chem_acc_pass and ops_pass
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'✓ PASS — DQN is learning' if overall else '✗ FAIL — DQN may not be learning'}")
    print("=" * 60)

    # --- Save full report ---
    report = {
        'timestamp': str(datetime.datetime.now()),
        'agent': 'dqn',
        'molecule': MOLECULE,
        'bond_length': BOND_LENGTH,
        'n_episodes': n_episodes,
        'fci_energy': fci_energy,
        'best_energy': best_energy,
        'energy_error_mha': energy_error,
        'best_ops': best_ops,
        'target_ops': TARGET_OPS,
        'chemical_accuracy_pass': chem_acc_pass,
        'operator_efficiency_pass': ops_pass,
        'diagnostics': summary,
        'overall_pass': overall,
    }

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved → {output_path}")
    print(f"DQN diagnostics → {diag_path}")

    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--output',   default='results/dqn_lih_validation.json')
    parser.add_argument('--diag',     default='results/dqn_lih_diagnostics.json')
    args = parser.parse_args()

    passed = run_validation(args.episodes, args.output, args.diag)
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke-test the script can be imported**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test
/curie-home/jpchen/.conda/envs/llm/bin/python -c "
import sys, os
sys.path.insert(0, os.path.abspath('../rlqas-chem/src'))
import validate_lih_dqn
print('Import OK')
"
```

Expected: `Import OK` (no errors)

- [ ] **Step 3: Commit**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test
git add validate_lih_dqn.py
git commit -m "feat: add DQN LiH validation script"
```

---

## Task 5: Create `submit_validate_dqn.sh`

**Files:**
- Create: `rlqas_test/submit_validate_dqn.sh`

- [ ] **Step 1: Create the SLURM script**

```bash
#!/bin/bash
#SBATCH --job-name=rlqas-dqn-validate
#SBATCH --output=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/slurm_logs/dqn_validate_%j.out
#SBATCH --error=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/slurm_logs/dqn_validate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=4V100
#SBATCH --qos=normal

WORKDIR="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test"
RLQAS_CHEM="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem"
PYTHON="/curie-home/jpchen/.conda/envs/llm/bin/python3"

cd "$WORKDIR"
mkdir -p "$WORKDIR/slurm_logs" "$WORKDIR/results"

echo "=== RLQAS LiH DQN Learning Validation ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started : $(date)"

export PYTHONUNBUFFERED=1

# Install latest rlqas-chem
$PYTHON -m pip install -e "$RLQAS_CHEM" -q 2>&1 | tail -1
$PYTHON -c "import rlqas_chem; print('rlqas_chem ready')" || { echo "ERROR: import failed"; exit 1; }

EPISODES="${RLQAS_N_EPISODES:-2000}"

$PYTHON validate_lih_dqn.py \
    --episodes "$EPISODES" \
    --output "results/dqn_lih_validation_${SLURM_JOB_ID}.json" \
    --diag   "results/dqn_lih_diagnostics_${SLURM_JOB_ID}.json"

EXIT_CODE=$?
echo ""
echo "=== Finished at $(date) — exit code: $EXIT_CODE ==="
exit $EXIT_CODE
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/submit_validate_dqn.sh
```

- [ ] **Step 3: Verify syntax**

```bash
bash -n /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/submit_validate_dqn.sh
```

Expected: no output (syntax OK)

- [ ] **Step 4: Commit**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test
git add submit_validate_dqn.sh
git commit -m "feat: add SLURM submit script for DQN LiH validation"
```

---

## Final Check: Reinstall and Verify

- [ ] **Step 1: Reinstall rlqas-chem so new modules are picked up**

```bash
/curie-home/jpchen/.conda/envs/llm/bin/python -m pip install -e \
    /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem -q
```

- [ ] **Step 2: Run all tests**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem
/curie-home/jpchen/.conda/envs/llm/bin/python -m pytest tests/test_dqn_diagnostics_callback.py -v
```

Expected: 6 tests pass.

- [ ] **Step 3: Submit to SLURM**

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test
sbatch submit_validate_dqn.sh
```

Expected: `Submitted batch job <JOB_ID>`

Monitor with:
```bash
squeue -u jpchen
tail -f slurm_logs/dqn_validate_<JOB_ID>.out
```
