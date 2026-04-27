"""Diagnostics tracker for GRPO training in RLQAS.

GRPO runs in discrete groups rather than individual timesteps, so this
tracker is a plain object with an ``on_group_end`` hook rather than an
SB3 BaseCallback.  It is passed as ``callbacks`` to
``UCCSearchController.search()`` → ``_grpo_search()``.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np


class GRPODiagnosticsTracker:
    """Collects per-group training metrics from GRPO for learning validation.

    ``on_group_end`` is called by ``_grpo_search`` after each group update
    with the dict returned by ``GRPOAgent.train_one_group()``:
      - best_energy: best VQE energy within the group
      - mean_energy: mean VQE energy within the group
      - loss: policy update loss
      - advantages: within-group relative advantages

    Also reads ``env.global_best_energy`` at each group boundary.

    Args:
        output_path: Path to save diagnostics JSON on training end.
        checkpoint_freq: Save a running checkpoint every N groups (0 = only on end).
        verbose: Verbosity level.
    """

    def __init__(self, output_path: str, checkpoint_freq: int = 0, verbose: int = 0):
        self.output_path = output_path
        self.checkpoint_freq = checkpoint_freq
        self.verbose = verbose
        self.records: List[Dict[str, Any]] = []
        self._group_count = 0

    def on_group_end(self, group_idx: int, group_result: Dict[str, Any], env) -> None:
        """Record diagnostics for one completed group."""
        record: Dict[str, Any] = {
            "group": group_idx,
            "best_energy": group_result.get("best_energy"),
            "mean_energy": group_result.get("mean_energy"),
            "loss": group_result.get("loss"),
        }

        # Advantage spread: measure of within-group diversity
        adv = group_result.get("advantages")
        if adv is not None:
            record["advantage_std"] = float(np.std(adv)) if len(adv) > 1 else 0.0
        else:
            record["advantage_std"] = None

        # Global best energy from environment
        try:
            gbe = getattr(env, "global_best_energy", None)
            record["global_best_energy"] = float(gbe) if gbe is not None else None
        except Exception:
            record["global_best_energy"] = None

        self.records.append(record)
        self._group_count += 1

        if self.checkpoint_freq > 0 and self._group_count % self.checkpoint_freq == 0:
            self._save()

    def finish(self) -> None:
        """Call after training ends to persist final diagnostics."""
        self._save()

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self.records, f, indent=2)
        if self.verbose:
            print(f"[GRPODiagnosticsTracker] saved {len(self.records)} groups → {self.output_path}")

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def loss_series(self) -> List[Optional[float]]:
        return [r["loss"] for r in self.records]

    def best_energy_series(self) -> List[Optional[float]]:
        return [r["global_best_energy"] for r in self.records]

    def group_best_series(self) -> List[Optional[float]]:
        return [r["best_energy"] for r in self.records]

    def summary(self) -> Dict[str, Any]:
        """Return a pass/fail summary of GRPO learning diagnostics."""
        losses = [v for v in self.loss_series() if v is not None]
        energies = [v for v in self.best_energy_series() if v is not None]

        n = len(losses)
        tail = max(1, n // 5)
        head = max(1, n // 5)

        # Loss trend: last 20% mean < first 20% mean
        loss_trend_pass = False
        if n >= 2:
            loss_trend_pass = float(np.mean(losses[-tail:])) < float(np.mean(losses[:head]))

        # Energy trend: global best improved over training
        energy_trend_pass = (
            len(energies) >= 2 and energies[-1] < energies[0]
        )

        return {
            "n_groups": len(self.records),
            "n_loss_samples": n,
            "loss_first": float(np.mean(losses[:head])) if losses else None,
            "loss_last": float(np.mean(losses[-tail:])) if losses else None,
            "loss_trend_pass": bool(loss_trend_pass),
            "best_energy_first": energies[0] if energies else None,
            "best_energy_last": energies[-1] if energies else None,
            "energy_trend_pass": bool(energy_trend_pass),
            "overall_pass": bool(loss_trend_pass and energy_trend_pass),
        }
