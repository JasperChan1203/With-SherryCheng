"""Reward function for UCC search based on energy improvement.

Copied from Task 004, with imports adapted for integration.
"""

from typing import Dict, Any, Optional
import numpy as np

from rlqas_chem.search.ucc.config import UCCSearchConfig


class UCCRewardFunction:
    """Computes rewards for UCC search."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize reward function.

        Args:
            config: Reward configuration (can be full config dict)
        """
        # Extract reward-specific configuration using UCCSearchConfig
        self.config = UCCSearchConfig(config).get_section("reward_function")
        _raw = config or {}

        # Baseline energy tracking
        self.best_energy = None  # for 'current_best' baseline type
        self.hf_energy = None    # for 'hartree_fock' baseline type
        self.rolling_energy = None  # for 'rolling_average' baseline type
        self.rolling_window = []
        self.window_size = self.config.get("rolling_window_size", 10)

        # Reward parameters — check flat raw config first (takes priority over section defaults)
        self.energy_weight = _raw.get("energy_weight",
                                      self.config.get("energy_weight", 1.0))
        self.complexity_penalty = _raw.get("complexity_penalty",
                                           self.config.get("complexity_penalty", 0.01))
        self.baseline_type = self.config.get("baseline_type", "current_best")
        self.use_shaping = self.config.get("shaping_rewards", False)

        # Alpha parameter: trade-off between energy accuracy and circuit complexity.
        # alpha=1.0 (default) means pure energy optimization — backward compatible.
        self.alpha = _raw.get("alpha", 1.0)
        self.max_operators = _raw.get("max_operators", 20)

        # If alpha < 1.0, use alpha-weighted complexity formula
        if self.alpha < 1.0:
            self.energy_weight = self.alpha
            self._use_alpha_complexity = True
        else:
            self._use_alpha_complexity = False

        # State for shaping rewards
        self.last_energy = None
        self.consecutive_improvements = 0
        # First evaluation flag
        self._first_evaluation = True

    def _get_baseline_energy(self) -> Optional[float]:
        """Get baseline energy based on baseline_type."""
        if self.baseline_type == "hartree_fock":
            return self.hf_energy
        elif self.baseline_type == "current_best":
            return self.best_energy
        elif self.baseline_type == "rolling_average":
            return self.rolling_energy
        else:
            raise ValueError(f"Unknown baseline_type: {self.baseline_type}")

    def _update_baseline_energy(self, current_energy: float):
        """Update baseline energy based on baseline_type."""
        if self.baseline_type == "current_best":
            if self.best_energy is None or current_energy < self.best_energy:
                self.best_energy = current_energy
        elif self.baseline_type == "rolling_average":
            self.rolling_window.append(current_energy)
            if len(self.rolling_window) > self.window_size:
                self.rolling_window.pop(0)
            self.rolling_energy = np.mean(self.rolling_window) if self.rolling_window else current_energy
        # For hartree_fock baseline, hf_energy is set via update_baseline method

    def _compute_shaping_reward(self, current_energy: float) -> float:
        """Compute optional shaping rewards."""
        if not self.use_shaping:
            return 0.0

        shaping_reward = 0.0
        # Reward for consecutive improvements
        if self.last_energy is not None and current_energy < self.last_energy:
            self.consecutive_improvements += 1
            shaping_reward += 0.01 * self.consecutive_improvements
        else:
            self.consecutive_improvements = 0

        # Small penalty for energy increase
        if self.last_energy is not None and current_energy > self.last_energy:
            shaping_reward -= 0.005

        self.last_energy = current_energy
        return shaping_reward

    def compute_reward(self, current_energy: float, circuit_complexity: int) -> float:
        """Compute reward for current energy and circuit.

        Args:
            current_energy: Current circuit energy
            circuit_complexity: Number of excitation operators in circuit

        Returns:
            Computed reward value
        """
        # First evaluation: initialize baseline and return 0.0
        if self._first_evaluation:
            self._first_evaluation = False
            # Initialize baseline based on baseline_type
            if self.baseline_type == "hartree_fock":
                # HF energy should have been set via update_baseline before first compute
                # If not set, use current energy as baseline
                if self.hf_energy is None:
                    self.hf_energy = current_energy
            elif self.baseline_type == "current_best":
                self.best_energy = current_energy
            elif self.baseline_type == "rolling_average":
                self.rolling_window = [current_energy]
                self.rolling_energy = current_energy
            # Initialize shaping reward state
            self.last_energy = current_energy
            self.consecutive_improvements = 0
            # Return 0.0 for first evaluation (no improvement, no penalty)
            return 0.0

        # Get baseline energy
        baseline = self._get_baseline_energy()
        if baseline is None:
            # Should not happen after first evaluation, but fallback
            baseline = current_energy

        # Compute energy improvement relative to baseline
        energy_improvement = baseline - current_energy

        # Apply energy weight
        weighted_improvement = energy_improvement * self.energy_weight

        # Compute complexity penalty (per excitation operator).
        # FIX: Replace unbounded linear penalty (complexity_penalty * n_ops) with
        # a normalised form so that the penalty stays proportional regardless of
        # max_operators.  This prevents the agent from refusing to add operators
        # for molecules that genuinely need 10+ excitations (e.g. H6).
        if self._use_alpha_complexity:
            max_ops = max(1, self.max_operators)
            complexity_penalty = (1.0 - self.alpha) * (circuit_complexity / max_ops)
        else:
            max_ops = max(1, self.max_operators)
            complexity_penalty = self.complexity_penalty * (circuit_complexity / max_ops)

        # Compute shaping reward
        shaping_reward = self._compute_shaping_reward(current_energy)

        # Total reward
        reward = weighted_improvement - complexity_penalty + shaping_reward

        # Update baseline tracking based on baseline type
        self._update_baseline_energy(current_energy)

        return reward

    def update_baseline(self, new_baseline: float):
        """Update baseline energy for reward computation.

        Args:
            new_baseline: New baseline energy
        """
        if self.baseline_type == "hartree_fock":
            self.hf_energy = new_baseline
        elif self.baseline_type == "current_best":
            self.best_energy = new_baseline
        elif self.baseline_type == "rolling_average":
            self.rolling_window = [new_baseline]
            self.rolling_energy = new_baseline
        # Also update best_energy for backward compatibility
        self.best_energy = new_baseline