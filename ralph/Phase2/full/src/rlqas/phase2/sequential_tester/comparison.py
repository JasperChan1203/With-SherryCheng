"""
Sequential Testing Framework for RLQAS Phase 2.

This module provides utilities for comparing results from multiple RL algorithms
and generating comparison reports.
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime


class ComparisonUtilities:
    """Utilities for comparing RL algorithm performance.

    This class provides methods for analyzing and comparing results
    from multiple RL algorithms trained on the same task.
    """

    def __init__(self, results: Dict[str, Dict]):
        """Initialize with results dictionary.

        Args:
            results: Dictionary mapping agent names to their results
        """
        self.results = results

    def get_training_times(self) -> Dict[str, float]:
        """Get training times for all agents.

        Returns:
            Dictionary mapping agent names to training times
        """
        times = {}
        for name, result in self.results.items():
            if "avg_train_time" in result:
                times[name] = result["avg_train_time"]
            elif "train_time" in result:
                times[name] = result["train_time"]
        return times

    def get_final_metrics(self) -> Dict[str, Any]:
        """Get final metrics for all agents.

        Returns:
            Dictionary mapping agent names to final metrics
        """
        metrics = {}
        for name, result in self.results.items():
            metrics[name] = result.get("final_metrics", {})
        return metrics

    def rank_by_metric(
        self, metric_name: str, ascending: bool = True
    ) -> List[Tuple[str, float]]:
        """Rank agents by a specific metric.

        Args:
            metric_name: Name of the metric to rank by
            ascending: Whether lower values are better

        Returns:
            List of (agent_name, metric_value) tuples sorted by metric
        """
        rankings = []
        for name, result in self.results.items():
            metrics = result.get("final_metrics", {})
            if metric_name in metrics:
                rankings.append((name, metrics[metric_name]))

        rankings.sort(key=lambda x: x[1], reverse=not ascending)
        return rankings

    def generate_comparison_table(self) -> str:
        """Generate a text comparison table.

        Returns:
            Formatted string table comparing all agents
        """
        if not self.results:
            return "No results to compare"

        lines = []
        lines.append("=" * 70)
        lines.append("RL Algorithm Comparison")
        lines.append("=" * 70)
        lines.append("")

        # Header
        header = f"{'Agent':<20} {'Type':<10} {'Timesteps':<12} {'Train Time':<12}"
        lines.append(header)
        lines.append("-" * 70)

        # Data rows
        for name, result in self.results.items():
            agent_type = result.get("agent_type", "unknown")
            timesteps = result.get("total_timesteps", "N/A")
            train_time = result.get("avg_train_time", result.get("train_time", "N/A"))

            if isinstance(train_time, float):
                train_time_str = f"{train_time:.2f}s"
            else:
                train_time_str = str(train_time)

            row = f"{name:<20} {agent_type:<10} {timesteps:<12} {train_time_str:<12}"
            lines.append(row)

        lines.append("=" * 70)
        return "\n".join(lines)

    def find_best_agent(
        self, metric_name: str = "total_timesteps", ascending: bool = True
    ) -> Optional[str]:
        """Find the best performing agent by a metric.

        Args:
            metric_name: Name of the metric to compare
            ascending: Whether lower values are better

        Returns:
            Name of the best agent, or None if no data
        """
        rankings = self.rank_by_metric(metric_name, ascending)
        if rankings:
            return rankings[0][0]
        return None


def compare_energy_convergence(
    results: Dict[str, Dict],
    target_energy: float,
    chemical_accuracy: float = 1.6e-3,
) -> Dict:
    """Compare energy convergence across algorithms.

    Args:
        results: Dictionary mapping agent names to their results
        target_energy: Target (ground truth) energy value
        chemical_accuracy: Chemical accuracy threshold in Hartree (default: 1.6 mHa)

    Returns:
        Dictionary containing comparison metrics
    """
    comparison = {
        "target_energy": target_energy,
        "chemical_accuracy": chemical_accuracy,
        "agents": {},
    }

    for name, result in results.items():
        metrics = result.get("final_metrics", {})
        final_energy = metrics.get("final_energy", metrics.get("final_reward", 0))
        energy_error = abs(final_energy - target_energy)
        achieved_chemical_accuracy = energy_error < chemical_accuracy

        comparison["agents"][name] = {
            "final_energy": final_energy,
            "energy_error": energy_error,
            "achieved_chemical_accuracy": achieved_chemical_accuracy,
        }

    # Find best agent
    best_agent = None
    best_error = float("inf")
    for name, agent_data in comparison["agents"].items():
        if agent_data["energy_error"] < best_error:
            best_error = agent_data["energy_error"]
            best_agent = name

    comparison["best_agent"] = best_agent
    comparison["best_energy_error"] = best_error

    return comparison


def compare_training_efficiency(
    results: Dict[str, Dict],
    target_metric: str = "total_timesteps",
    threshold: Optional[float] = None,
) -> Dict:
    """Compare training efficiency across algorithms.

    Args:
        results: Dictionary mapping agent names to their results
        target_metric: Metric to use for efficiency comparison
        threshold: Optional threshold value for convergence

    Returns:
        Dictionary containing efficiency comparison
    """
    efficiency = {
        "agents": {},
        "ranking": [],
    }

    for name, result in results.items():
        train_time = result.get("avg_train_time", result.get("train_time", 0))
        timesteps = result.get("total_timesteps", 0)

        # Calculate efficiency (timesteps per second)
        if train_time > 0:
            efficiency_score = timesteps / train_time
        else:
            efficiency_score = float("inf")

        efficiency["agents"][name] = {
            "train_time": train_time,
            "timesteps": timesteps,
            "efficiency_score": efficiency_score,
        }
        efficiency["ranking"].append((name, efficiency_score))

    # Sort by efficiency (higher is better)
    efficiency["ranking"].sort(key=lambda x: x[1], reverse=True)
    efficiency["most_efficient"] = efficiency["ranking"][0][0] if efficiency["ranking"] else None

    return efficiency


def generate_summary_report(
    results: Dict[str, Dict],
    test_name: str = "Unnamed Test",
    output_path: Optional[str] = None,
) -> str:
    """Generate a summary report of test results.

    Args:
        results: Dictionary mapping agent names to their results
        test_name: Name of the test
        output_path: Optional path to save the report

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"RLQAS Phase 2 - Sequential Testing Summary")
    lines.append(f"Test: {test_name}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Comparison table
    comparator = ComparisonUtilities(results)
    lines.append(comparator.generate_comparison_table())
    lines.append("")

    # Best agents
    lines.append("Performance Summary:")
    lines.append("-" * 40)

    # Best by training time
    fastest = comparator.find_best_agent("train_time", ascending=True)
    if fastest:
        lines.append(f"Fastest training: {fastest}")

    # Count agents
    lines.append(f"Total agents tested: {len(results)}")
    lines.append("")

    # Detailed results
    lines.append("Detailed Results:")
    lines.append("-" * 40)
    for name, result in results.items():
        lines.append(f"\n{name}:")
        lines.append(f"  Type: {result.get('agent_type', 'unknown')}")
        lines.append(f"  Timesteps: {result.get('total_timesteps', 'N/A')}")
        train_time = result.get("avg_train_time", result.get("train_time"))
        if isinstance(train_time, float):
            lines.append(f"  Training time: {train_time:.2f}s")
        else:
            lines.append(f"  Training time: {train_time}")

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

    return report
