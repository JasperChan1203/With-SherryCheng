"""
RL Algorithm Exploration Framework for RLQAS Phase 2.

This module provides a framework for exploring and evaluating new RL algorithms
for quantum architecture search problems.
"""

import os
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import json

from rlqas.phase2.rl import RLAgent, AgentFactory


class ExplorationFramework:
    """Framework for exploring new RL algorithms.

    This class provides utilities for:
    - Discovering and cataloging RL algorithms
    - Evaluating algorithm compatibility with quantum architecture search
    - Comparing new algorithms against baseline methods (PPO, DQN)
    - Documenting findings for future reference

    Args:
        output_dir: Directory for storing exploration results
        verbose: Verbosity level (0, 1, or 2)
    """

    def __init__(self, output_dir: str = "results/exploration", verbose: int = 1):
        """Initialize exploration framework."""
        self.output_dir = output_dir
        self.verbose = verbose
        self.discovered_algorithms: Dict[str, Dict] = {}
        self.evaluation_results: Dict[str, Dict] = {}

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Register baseline algorithms
        self._register_baselines()

    def _register_baselines(self):
        """Register baseline algorithms for comparison."""
        self.discovered_algorithms["ppo"] = {
            "name": "PPO",
            "full_name": "Proximal Policy Optimization",
            "type": "policy_gradient",
            "action_space": ["discrete", "continuous"],
            "sample_efficiency": "medium",
            "stability": "high",
            "sparse_rewards": "good",
            "baseline": True,
            "registered_at": datetime.now().isoformat(),
        }

        self.discovered_algorithms["dqn"] = {
            "name": "DQN",
            "full_name": "Deep Q-Network",
            "type": "value_based",
            "action_space": ["discrete"],
            "sample_efficiency": "low",
            "stability": "medium",
            "sparse_rewards": "fair",
            "baseline": True,
            "registered_at": datetime.now().isoformat(),
        }

    def discover_algorithm(
        self,
        name: str,
        algorithm_class: Optional[Type[RLAgent]] = None,
        description: str = "",
        properties: Optional[Dict] = None,
    ) -> str:
        """Discover and register a new RL algorithm.

        Args:
            name: Algorithm name/identifier
            algorithm_class: Optional algorithm class
            description: Algorithm description
            properties: Algorithm properties

        Returns:
            Algorithm ID
        """
        algorithm_id = name.lower().replace(" ", "_")

        properties = properties or {}

        self.discovered_algorithms[algorithm_id] = {
            "name": name,
            "full_name": properties.get("full_name", name),
            "type": properties.get("type", "unknown"),
            "action_space": properties.get("action_space", []),
            "sample_efficiency": properties.get("sample_efficiency", "unknown"),
            "stability": properties.get("stability", "unknown"),
            "sparse_rewards": properties.get("sparse_rewards", "unknown"),
            "description": description,
            "baseline": False,
            "algorithm_class": algorithm_class,
            "registered_at": datetime.now().isoformat(),
        }

        if self.verbose >= 1:
            print(f"Discovered algorithm: {algorithm_id} ({name})")

        return algorithm_id

    def evaluate_compatibility(
        self,
        algorithm_id: str,
        env_properties: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Evaluate algorithm compatibility with quantum architecture search.

        Evaluation criteria:
        - Action space compatibility (discrete/continuous)
        - Sample efficiency
        - Stability in high-dimensional state spaces
        - Ability to handle sparse rewards

        Args:
            algorithm_id: Algorithm identifier
            env_properties: Optional environment properties

        Returns:
            Compatibility evaluation dictionary
        """
        if algorithm_id not in self.discovered_algorithms:
            return {"error": f"Algorithm not found: {algorithm_id}"}

        algo = self.discovered_algorithms[algorithm_id]
        env_properties = env_properties or {}

        # Default environment properties for quantum architecture search
        default_env = {
            "action_space": "discrete",  # Excitation operator selection
            "state_dim": "high",  # Circuit parameters + energy
            "reward_sparsity": "sparse",  # Energy improvements are sparse
            "episode_length": "medium",
        }
        default_env.update(env_properties)

        # Evaluate compatibility
        evaluation = {
            "algorithm_id": algorithm_id,
            "algorithm_name": algo["name"],
            "criteria": {},
            "overall_score": 0,
            "recommendation": "",
        }

        # Action space compatibility
        action_compat = self._evaluate_action_space(
            algo.get("action_space", []),
            default_env["action_space"],
        )
        evaluation["criteria"]["action_space_compatibility"] = action_compat

        # Sample efficiency
        efficiency_score = self._evaluate_sample_efficiency(
            algo.get("sample_efficiency", "unknown"),
        )
        evaluation["criteria"]["sample_efficiency"] = efficiency_score

        # Stability
        stability_score = self._evaluate_stability(
            algo.get("stability", "unknown"),
            default_env["state_dim"],
        )
        evaluation["criteria"]["stability"] = stability_score

        # Sparse rewards
        sparse_score = self._evaluate_sparse_rewards(
            algo.get("sparse_rewards", "unknown"),
        )
        evaluation["criteria"]["sparse_rewards_handling"] = sparse_score

        # Calculate overall score
        scores = [
            action_compat["score"],
            efficiency_score["score"],
            stability_score["score"],
            sparse_score["score"],
        ]
        evaluation["overall_score"] = sum(scores) / len(scores)

        # Generate recommendation
        if evaluation["overall_score"] >= 0.8:
            evaluation["recommendation"] = "Highly recommended for quantum architecture search"
        elif evaluation["overall_score"] >= 0.6:
            evaluation["recommendation"] = "Recommended with minor modifications"
        elif evaluation["overall_score"] >= 0.4:
            evaluation["recommendation"] = "May require significant modifications"
        else:
            evaluation["recommendation"] = "Not recommended for this use case"

        self.evaluation_results[algorithm_id] = evaluation

        return evaluation

    def _evaluate_action_space(
        self,
        algo_action_space: List[str],
        env_action_space: str,
    ) -> Dict[str, Any]:
        """Evaluate action space compatibility.

        Args:
            algo_action_space: Algorithm supported action spaces
            env_action_space: Environment action space type

        Returns:
            Evaluation dictionary
        """
        if env_action_space in algo_action_space:
            return {
                "compatible": True,
                "score": 1.0,
                "message": f"Algorithm supports {env_action_space} action space",
            }
        else:
            return {
                "compatible": False,
                "score": 0.0,
                "message": f"Algorithm does not support {env_action_space} action space",
            }

    def _evaluate_sample_efficiency(
        self,
        efficiency: str,
    ) -> Dict[str, Any]:
        """Evaluate sample efficiency.

        Args:
            efficiency: Efficiency rating (low, medium, high)

        Returns:
            Evaluation dictionary
        """
        scores = {"high": 1.0, "medium": 0.7, "low": 0.3, "unknown": 0.5}
        score = scores.get(efficiency.lower(), 0.5)

        return {
            "efficiency": efficiency,
            "score": score,
            "message": f"Sample efficiency: {efficiency}",
        }

    def _evaluate_stability(
        self,
        stability: str,
        state_dim: str,
    ) -> Dict[str, Any]:
        """Evaluate stability in high-dimensional spaces.

        Args:
            stability: Stability rating (low, medium, high)
            state_dim: State dimension (low, medium, high)

        Returns:
            Evaluation dictionary
        """
        base_scores = {"high": 1.0, "medium": 0.7, "low": 0.3, "unknown": 0.5}
        score = base_scores.get(stability.lower(), 0.5)

        # Reduce score for high-dimensional states if stability is low
        if state_dim == "high" and stability.lower() in ["low", "medium"]:
            score *= 0.8

        return {
            "stability": stability,
            "state_dimension": state_dim,
            "score": score,
            "message": f"Stability ({stability}) in {state_dim}-dimensional state space",
        }

    def _evaluate_sparse_rewards(
        self,
        sparse_handling: str,
    ) -> Dict[str, Any]:
        """Evaluate sparse rewards handling.

        Args:
            sparse_handling: Rating (poor, fair, good, excellent)

        Returns:
            Evaluation dictionary
        """
        scores = {
            "excellent": 1.0,
            "good": 0.8,
            "fair": 0.5,
            "poor": 0.2,
            "unknown": 0.5,
        }
        score = scores.get(sparse_handling.lower(), 0.5)

        return {
            "sparse_rewards_handling": sparse_handling,
            "score": score,
            "message": f"Sparse rewards handling: {sparse_handling}",
        }

    def compare_algorithms(
        self,
        algorithm_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare multiple algorithms.

        Args:
            algorithm_ids: List of algorithm IDs to compare

        Returns:
            Comparison dictionary
        """
        if algorithm_ids is None:
            algorithm_ids = list(self.discovered_algorithms.keys())

        comparison = {
            "algorithms": {},
            "ranking": [],
            "best_overall": None,
            "best_by_category": {},
        }

        for algo_id in algorithm_ids:
            if algo_id not in self.discovered_algorithms:
                continue

            algo = self.discovered_algorithms[algo_id]
            evaluation = self.evaluation_results.get(
                algo_id,
                self.evaluate_compatibility(algo_id),
            )

            comparison["algorithms"][algo_id] = {
                "name": algo["name"],
                "type": algo.get("type", "unknown"),
                "overall_score": evaluation.get("overall_score", 0),
                "criteria": evaluation.get("criteria", {}),
            }

            comparison["ranking"].append(
                (algo_id, evaluation.get("overall_score", 0))
            )

        # Sort by overall score
        comparison["ranking"].sort(key=lambda x: x[1], reverse=True)

        if comparison["ranking"]:
            comparison["best_overall"] = comparison["ranking"][0][0]

        # Find best by category
        categories = [
            "action_space_compatibility",
            "sample_efficiency",
            "stability",
            "sparse_rewards_handling",
        ]

        for category in categories:
            best_score = 0
            best_algo = None
            for algo_id, data in comparison["algorithms"].items():
                criteria = data.get("criteria", {})
                score = criteria.get(category, {}).get("score", 0)
                if score > best_score:
                    best_score = score
                    best_algo = algo_id
            if best_algo:
                comparison["best_by_category"][category] = best_algo

        return comparison

    def get_algorithm_info(self, algorithm_id: str) -> Optional[Dict]:
        """Get information about a discovered algorithm.

        Args:
            algorithm_id: Algorithm identifier

        Returns:
            Algorithm information dictionary
        """
        return self.discovered_algorithms.get(algorithm_id)

    def get_evaluation(self, algorithm_id: str) -> Optional[Dict]:
        """Get evaluation results for an algorithm.

        Args:
            algorithm_id: Algorithm identifier

        Returns:
            Evaluation dictionary
        """
        return self.evaluation_results.get(algorithm_id)

    def save_exploration_results(self, filename: Optional[str] = None) -> str:
        """Save exploration results to disk.

        Args:
            filename: Optional filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"exploration_results_{timestamp}.json"

        output_path = os.path.join(self.output_dir, filename)

        # Prepare serializable data
        export_data = {
            "discovered_algorithms": {},
            "evaluation_results": self.evaluation_results,
            "comparison": self.compare_algorithms(),
            "exported_at": datetime.now().isoformat(),
        }

        for algo_id, algo in self.discovered_algorithms.items():
            export_data["discovered_algorithms"][algo_id] = {
                k: v for k, v in algo.items()
                if k != "algorithm_class"  # Skip non-serializable
            }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        if self.verbose >= 1:
            print(f"Exploration results saved to {output_path}")

        return output_path


def create_exploration_framework(
    output_dir: str = "results/exploration",
    verbose: int = 1,
) -> ExplorationFramework:
    """Create an exploration framework instance.

    Args:
        output_dir: Output directory
        verbose: Verbosity level

    Returns:
        ExplorationFramework instance
    """
    return ExplorationFramework(output_dir=output_dir, verbose=verbose)
