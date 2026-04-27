"""SB3 callback for capturing PPO learning diagnostics during RLQAS training."""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class DiagnosticsCallback(BaseCallback):
    """Captures per-update training metrics from SB3 PPO for learning validation.

    Collects at each training update:
    - explained_variance, entropy_loss, policy_gradient_loss, approx_kl, value_loss
    - Best energy from the environment (via env.global_best_energy)

    Args:
        output_path: Path to save diagnostics JSON on training end.
        checkpoint_freq: Save a running checkpoint every N updates (0 = only on end).
        verbose: Verbosity level.
    """

    def __init__(self, output_path: str, checkpoint_freq: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.output_path = output_path
        self.checkpoint_freq = checkpoint_freq

        self.updates: List[Dict[str, Any]] = []
        self._update_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        self.updates = []
        self._update_count = 0

    def _on_rollout_end(self) -> None:
        """Called after each rollout collection + policy update."""
        self._update_count += 1

        record: Dict[str, Any] = {"update": self._update_count}

        # --- SB3 training metrics (logged in model.logger) ---
        log = self.model.logger.name_to_value
        for key in (
            "train/explained_variance",
            "train/entropy_loss",
            "train/policy_gradient_loss",
            "train/approx_kl",
            "train/value_loss",
            "train/loss",
            "time/total_timesteps",
        ):
            short = key.split("/")[-1]
            record[short] = float(log[key]) if key in log else None

        # --- Best energy from the vectorised env ---
        try:
            envs = self.training_env.envs  # DummyVecEnv
            best_energies = [
                getattr(e.unwrapped if hasattr(e, "unwrapped") else e,
                        "global_best_energy", None)
                for e in envs
            ]
            valid = [e for e in best_energies if e is not None]
            record["global_best_energy"] = float(min(valid)) if valid else None
        except Exception:
            record["global_best_energy"] = None

        self.updates.append(record)

        if self.checkpoint_freq > 0 and self._update_count % self.checkpoint_freq == 0:
            self._save()

    def _on_training_end(self) -> None:
        self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.updates, f, indent=2)
        if self.verbose:
            print(f"[DiagnosticsCallback] saved {len(self.updates)} updates → {self.output_path}")

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def explained_variance_series(self) -> List[Optional[float]]:
        return [r["explained_variance"] for r in self.updates]

    def entropy_series(self) -> List[Optional[float]]:
        return [r["entropy_loss"] for r in self.updates]

    def best_energy_series(self) -> List[Optional[float]]:
        return [r["global_best_energy"] for r in self.updates]

    def summary(self) -> Dict[str, Any]:
        """Return a pass/fail summary of learning diagnostics."""
        ev = [v for v in self.explained_variance_series() if v is not None]
        ent = [v for v in self.entropy_series() if v is not None]
        energies = [v for v in self.best_energy_series() if v is not None]

        n = len(ev)
        tail = max(1, n // 5)  # last 20%

        ev_final = float(np.mean(ev[-tail:])) if ev else None
        ev_pass = ev_final is not None and ev_final > 0.1

        # Entropy decreasing: SB3 logs entropy_loss = -entropy * ent_coef (negative values).
        # As policy becomes more focused, entropy decreases → entropy_loss increases toward 0.
        # So a positive slope in entropy_loss means entropy is decreasing — that's PASS.
        ent_pass = False
        if len(ent) >= 4:
            x = np.arange(len(ent))
            slope = float(np.polyfit(x, ent, 1)[0])
            ent_pass = slope > 0  # entropy_loss rising toward 0 = entropy decreasing

        # Energy trend: last value < first value
        energy_pass = (
            len(energies) >= 2 and energies[-1] < energies[0]
        )

        return {
            "n_updates": n,
            "explained_variance_final": ev_final,
            "ev_pass": bool(ev_pass),
            "entropy_slope": float(np.polyfit(np.arange(len(ent)), ent, 1)[0]) if len(ent) >= 4 else None,
            "entropy_pass": bool(ent_pass),
            "best_energy_first": energies[0] if energies else None,
            "best_energy_last": energies[-1] if energies else None,
            "energy_trend_pass": bool(energy_pass),
            "overall_pass": bool(ev_pass and ent_pass and energy_pass),
        }
