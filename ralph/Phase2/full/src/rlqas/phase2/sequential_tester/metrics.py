"""
Metrics collection for Sequential Testing Framework.

This module provides standardized metrics collection across RL algorithms
for quantum architecture search problems.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MetricsCollector:
    """Collects and aggregates metrics from RL training runs.

    This class provides standardized metrics collection for comparing
    different RL algorithms on quantum architecture search tasks.

    Attributes:
        metrics: Dictionary storing collected metrics
        history: List of metric snapshots over time
    """

    metrics: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def record_step(
        self,
        episode: int,
        reward: float,
        energy: Optional[float] = None,
        excitation_count: Optional[int] = None,
        **kwargs,
    ):
        """Record metrics for a single training step.

        Args:
            episode: Episode number
            reward: Reward received
            energy: Optional energy value
            excitation_count: Optional number of excitation operators
            **kwargs: Additional metrics to record
        """
        step_metrics = {
            "episode": episode,
            "reward": reward,
        }

        if energy is not None:
            step_metrics["energy"] = energy

        if excitation_count is not None:
            step_metrics["excitation_count"] = excitation_count

        step_metrics.update(kwargs)
        self.history.append(step_metrics)

    def record_final_metrics(
        self,
        final_energy: float,
        final_reward: float,
        total_episodes: int,
        convergence_episode: Optional[int] = None,
        excitation_operators_used: Optional[int] = None,
        **kwargs,
    ):
        """Record final metrics after training completes.

        Args:
            final_energy: Final energy achieved
            final_reward: Final reward achieved
            total_episodes: Total number of episodes trained
            convergence_episode: Episode where convergence was reached
            excitation_operators_used: Number of excitation operators used
            **kwargs: Additional metrics to record
        """
        self.metrics.update(
            {
                "final_energy": final_energy,
                "final_reward": final_reward,
                "total_episodes": total_episodes,
                "convergence_episode": convergence_episode,
                "excitation_operators_used": excitation_operators_used,
                **kwargs,
            }
        )

    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from training history.

        Returns:
            Dictionary containing aggregated metrics
        """
        if not self.history:
            return {}

        rewards = [step["reward"] for step in self.history]
        aggregate = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "total_steps": len(self.history),
        }

        # Check for energy in history
        if "energy" in self.history[0]:
            energies = [step.get("energy", 0) for step in self.history]
            aggregate["mean_energy"] = float(np.mean(energies))
            aggregate["final_energy"] = energies[-1] if energies else 0

        # Check for excitation count
        if "excitation_count" in self.history[0]:
            excitation_counts = [step.get("excitation_count", 0) for step in self.history]
            aggregate["mean_excitation_count"] = float(np.mean(excitation_counts))
            aggregate["max_excitation_count"] = int(np.max(excitation_counts))

        return aggregate

    def check_chemical_accuracy(
        self,
        target_energy: float,
        threshold: float = 1.6e-3,
    ) -> Dict[str, Any]:
        """Check if chemical accuracy was achieved.

        Args:
            target_energy: Target (ground truth) energy
            threshold: Chemical accuracy threshold in Hartree (default: 1.6 mHa)

        Returns:
            Dictionary containing accuracy metrics
        """
        final_energy = self.metrics.get("final_energy")
        if final_energy is None:
            # Try to get from history
            if self.history and "energy" in self.history[-1]:
                final_energy = self.history[-1]["energy"]

        if final_energy is None:
            return {
                "achieved": False,
                "error": None,
                "threshold": threshold,
                "message": "No energy data available",
            }

        energy_error = abs(final_energy - target_energy)
        achieved = energy_error < threshold

        return {
            "achieved": achieved,
            "final_energy": final_energy,
            "target_energy": target_energy,
            "energy_error": energy_error,
            "threshold": threshold,
            "message": (
                "Chemical accuracy achieved!" if achieved
                else f"Energy error {energy_error:.6f} Ha exceeds threshold {threshold:.6f} Ha"
            ),
        }

    def get_convergence_info(self) -> Dict[str, Any]:
        """Get information about convergence during training.

        Returns:
            Dictionary containing convergence metrics
        """
        if not self.history:
            return {"converged": False, "message": "No training history"}

        rewards = [step["reward"] for step in self.history]

        # Simple convergence detection: check if reward stabilized
        if len(rewards) < 10:
            return {"converged": False, "message": "Insufficient data for convergence analysis"}

        # Check last 10% of training
        window_size = max(1, len(rewards) // 10)
        recent_rewards = rewards[-window_size:]
        recent_std = np.std(recent_rewards)

        converged = recent_std < 0.1  # Threshold for stability
        convergence_episode = len(rewards) - window_size if converged else None

        return {
            "converged": converged,
            "convergence_episode": convergence_episode,
            "recent_reward_std": float(recent_std),
            "final_reward": rewards[-1],
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary.

        Returns:
            Dictionary containing all collected metrics
        """
        return {
            "metrics": self.metrics,
            "aggregate": self.get_aggregate_metrics(),
            "convergence": self.get_convergence_info(),
            "history_length": len(self.history),
        }


def create_metrics_collector() -> MetricsCollector:
    """Factory function to create a MetricsCollector instance.

    Returns:
        New MetricsCollector instance
    """
    return MetricsCollector()


def compare_excitation_efficiency(
    results: Dict[str, MetricsCollector],
) -> Dict[str, Any]:
    """Compare excitation operator efficiency across algorithms.

    Args:
        results: Dictionary mapping agent names to their MetricsCollectors

    Returns:
        Dictionary containing efficiency comparison
    """
    comparison = {
        "agents": {},
        "ranking": [],
    }

    for name, collector in results.items():
        excitation_count = collector.metrics.get("excitation_operators_used")
        if excitation_count is None:
            # Try to get from history
            if collector.history and "excitation_count" in collector.history[-1]:
                excitation_count = collector.history[-1]["excitation_count"]

        if excitation_count is not None:
            comparison["agents"][name] = {
                "excitation_operators": excitation_count,
            }
            comparison["ranking"].append((name, excitation_count))

    # Sort by excitation count (fewer is better)
    comparison["ranking"].sort(key=lambda x: x[1])

    if comparison["ranking"]:
        comparison["most_efficient"] = comparison["ranking"][0][0]
        comparison["most_efficient_count"] = comparison["ranking"][0][1]

    return comparison
