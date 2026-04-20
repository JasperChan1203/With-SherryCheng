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
        self._training_env_override = None

    # Allow tests (and real usage) to set training_env directly by overriding
    # the read-only property from BaseCallback.
    @property  # type: ignore[override]
    def training_env(self):  # type: ignore[override]
        if self._training_env_override is not None:
            return self._training_env_override
        return super().training_env

    @training_env.setter
    def training_env(self, value):
        self._training_env_override = value

    def _on_training_start(self) -> None:
        self.samples = []
        self._sample_count = 0

    def _on_step(self) -> bool:
        # Use model.num_timesteps directly so the test can set it on the model
        # without needing to go through the full on_step() sync path.
        current_step = self.model.num_timesteps
        if current_step % self.sample_freq != 0:
            return True

        record: Dict[str, Any] = {"step": current_step}

        # SB3 DQN logger keys
        log = self.model.logger.name_to_value
        record["q_loss"] = float(log["train/loss"]) if "train/loss" in log else None
        record["exploration_rate"] = (
            float(log["rollout/exploration_rate"])
            if "rollout/exploration_rate" in log
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
            "n_loss_samples": n,
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
