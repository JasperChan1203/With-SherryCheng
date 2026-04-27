"""Diagnostics tracker for SAC-Discrete training in RLQAS.

SAC is a custom (non-SB3) agent, so this tracker is a plain callable
rather than an SB3 BaseCallback.  It is passed as ``step_callback`` to
``SACDiscreteAgent.learn()``, which invokes it every ``sample_freq`` steps.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class SACDiagnosticsTracker:
    """Collects per-sample training metrics from SAC-Discrete for learning validation.

    Called every ``sample_freq`` timesteps with (step, env, losses) where
    ``losses`` is the dict returned by ``SACDiscreteAgent._update()``:
      - c1_loss, c2_loss: critic losses
      - actor_loss: policy loss
      - alpha: current temperature coefficient

    Also reads ``env.global_best_energy`` at each sample point.

    Args:
        output_path: Path to save diagnostics JSON on training end.
        sample_freq: Passed to SACDiscreteAgent.learn() to control call frequency.
        checkpoint_freq: Save a running checkpoint every N samples (0 = only on end).
        verbose: Verbosity level.
    """

    def __init__(self, output_path: str, sample_freq: int = 2048,
                 checkpoint_freq: int = 0, verbose: int = 0):
        self.output_path = output_path
        self.sample_freq = sample_freq
        self.checkpoint_freq = checkpoint_freq
        self.verbose = verbose
        self.samples: List[Dict[str, Any]] = []
        self._sample_count = 0

    def __call__(self, step: int, env, losses: Dict[str, float]) -> bool:
        """Record one diagnostic sample.  Return True to continue training."""
        record: Dict[str, Any] = {
            "step": step,
            "c1_loss": losses.get("c1_loss"),
            "c2_loss": losses.get("c2_loss"),
            "actor_loss": losses.get("actor_loss"),
            "alpha": losses.get("alpha"),
        }

        # Best energy from environment
        try:
            record["global_best_energy"] = float(
                getattr(env, "global_best_energy", None) or float("inf")
            )
            if record["global_best_energy"] == float("inf"):
                record["global_best_energy"] = None
        except Exception:
            record["global_best_energy"] = None

        self.samples.append(record)
        self._sample_count += 1

        if self.checkpoint_freq > 0 and self._sample_count % self.checkpoint_freq == 0:
            self._save()

        return True  # continue training

    def finish(self) -> None:
        """Call after training ends to persist final diagnostics."""
        self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.samples, f, indent=2)
        if self.verbose:
            print(f"[SACDiagnosticsTracker] saved {len(self.samples)} samples → {self.output_path}")

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def actor_loss_series(self) -> List[Optional[float]]:
        return [r["actor_loss"] for r in self.samples]

    def alpha_series(self) -> List[Optional[float]]:
        return [r["alpha"] for r in self.samples]

    def best_energy_series(self) -> List[Optional[float]]:
        return [r["global_best_energy"] for r in self.samples]

    def summary(self) -> Dict[str, Any]:
        """Return a pass/fail summary of SAC learning diagnostics."""
        losses = [v for v in self.actor_loss_series() if v is not None]
        alphas = [v for v in self.alpha_series() if v is not None]
        energies = [v for v in self.best_energy_series() if v is not None]

        n = len(losses)
        tail = max(1, n // 5)
        head = max(1, n // 5)

        # Actor loss: last 20% mean < first 20% mean (policy improving)
        actor_loss_pass = False
        if n >= 2:
            actor_loss_pass = float(np.mean(losses[-tail:])) < float(np.mean(losses[:head]))

        # Alpha tuning: alpha changed from initial value (auto-tuning active)
        alpha_tuned = False
        if len(alphas) >= 2:
            alpha_tuned = bool(abs(alphas[-1] - alphas[0]) > 1e-4)

        # Energy trend: best_energy improved over training
        energy_trend_pass = (
            len(energies) >= 2 and energies[-1] < energies[0]
        )

        return {
            "n_samples": len(self.samples),
            "n_loss_samples": n,
            "actor_loss_first": float(np.mean(losses[:head])) if losses else None,
            "actor_loss_last": float(np.mean(losses[-tail:])) if losses else None,
            "actor_loss_pass": bool(actor_loss_pass),
            "alpha_first": alphas[0] if alphas else None,
            "alpha_last": alphas[-1] if alphas else None,
            "alpha_tuned": bool(alpha_tuned),
            "best_energy_first": energies[0] if energies else None,
            "best_energy_last": energies[-1] if energies else None,
            "energy_trend_pass": bool(energy_trend_pass),
            "overall_pass": bool(actor_loss_pass and energy_trend_pass),
        }
