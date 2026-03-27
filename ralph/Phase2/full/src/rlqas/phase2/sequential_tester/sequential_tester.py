"""
Sequential Testing Framework for RLQAS Phase 2.

This module implements the SequentialRLTester class for managing sequential
tests of multiple RL algorithms on quantum architecture search problems.
"""

import os
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np

from rlqas.phase2.rl import RLAgent, AgentFactory


class SequentialRLTester:
    """Sequential testing framework for comparing multiple RL algorithms.

    This class manages the execution of multiple RL agents in sequence on
    the same environment, collecting and storing results for comparison.

    Features:
    - Run multiple agents sequentially on the same task
    - Standardized metrics collection across algorithms
    - Support for configurable test sequences
    - Result storage and retrieval

    Args:
        output_dir: Directory for storing test results (default: "results")
        verbose: Verbosity level (0, 1, or 2)
    """

    def __init__(self, output_dir: str = "results", verbose: int = 1):
        """Initialize the sequential tester."""
        self.output_dir = output_dir
        self.verbose = verbose
        self.results: Dict[str, Dict] = {}
        self.metrics_history: Dict[str, List[Dict]] = {}
        self._current_test_id: Optional[str] = None

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

    def run_sequential_test(
        self,
        agent_configs: List[Dict[str, Any]],
        env_fn,
        test_name: str,
        total_timesteps: int = 10000,
        n_seeds: int = 3,
    ) -> Dict[str, Dict]:
        """Run sequential tests for multiple agents.

        Args:
            agent_configs: List of agent configuration dictionaries.
                Each dict should have:
                - "agent_type": "ppo" or "dqn"
                - "config": Optional agent configuration dict
            env_fn: Function that creates the environment
            test_name: Name for this test run
            total_timesteps: Number of timesteps to train each agent
            n_seeds: Number of random seeds to average over

        Returns:
            Dictionary mapping agent names to their results
        """
        self._current_test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results = {}
        self.metrics_history = {}

        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Sequential RL Test: {test_name}")
            print(f"Test ID: {self._current_test_id}")
            print(f"Agents to test: {[c['agent_type'] for c in agent_configs]}")
            print(f"Timesteps per agent: {total_timesteps}")
            print(f"Number of seeds: {n_seeds}")
            print(f"{'='*60}\n")

        for agent_config in agent_configs:
            agent_type = agent_config["agent_type"]
            agent_name = agent_config.get("name", agent_type)
            config = agent_config.get("config", {})

            if self.verbose >= 1:
                print(f"\n--- Testing {agent_name} ({agent_type}) ---")

            # Run test with multiple seeds
            seed_results = []
            for seed in range(n_seeds):
                if self.verbose >= 2:
                    print(f"  Running seed {seed + 1}/{n_seeds}...")

                # Create environment with seed
                env = env_fn(seed=seed)

                # Create agent
                agent = AgentFactory.create_agent(
                    agent_type=agent_type,
                    config=config,
                    env=env,
                )

                # Train agent
                start_time = time.time()
                metrics = agent.learn(total_timesteps=total_timesteps)
                train_time = time.time() - start_time

                # Collect results
                seed_result = {
                    "seed": seed,
                    "total_timesteps": total_timesteps,
                    "train_time": train_time,
                    "final_metrics": metrics,
                    "config": config,
                }
                seed_results.append(seed_result)

            # Aggregate results across seeds
            avg_train_time = np.mean([r["train_time"] for r in seed_results])
            self.results[agent_name] = {
                "agent_type": agent_type,
                "config": config,
                "n_seeds": n_seeds,
                "total_timesteps": total_timesteps,
                "avg_train_time": avg_train_time,
                "seed_results": seed_results,
            }

            if self.verbose >= 1:
                print(f"  Average training time: {avg_train_time:.2f}s")

        # Save results
        self._save_results()

        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"Sequential test complete. Results saved to {self.output_dir}")
            print(f"{'='*60}\n")

        return self.results

    def run_single_agent(
        self,
        agent_type: str,
        env_fn,
        agent_name: Optional[str] = None,
        config: Optional[Dict] = None,
        total_timesteps: int = 10000,
        seed: int = 42,
    ) -> Dict:
        """Run a single agent test.

        Args:
            agent_type: Type of agent ("ppo" or "dqn")
            env_fn: Function that creates the environment
            agent_name: Optional name for the agent
            config: Optional agent configuration
            total_timesteps: Number of timesteps to train
            seed: Random seed

        Returns:
            Dictionary containing test results
        """
        if agent_name is None:
            agent_name = agent_type

        if config is None:
            config = {}

        # Create environment
        env = env_fn(seed=seed)

        # Create agent
        agent = AgentFactory.create_agent(
            agent_type=agent_type,
            config=config,
            env=env,
        )

        # Train agent
        start_time = time.time()
        metrics = agent.learn(total_timesteps=total_timesteps)
        train_time = time.time() - start_time

        # Store results
        result = {
            "agent_type": agent_type,
            "agent_name": agent_name,
            "config": config,
            "seed": seed,
            "total_timesteps": total_timesteps,
            "train_time": train_time,
            "final_metrics": metrics,
        }

        self.results[agent_name] = result

        if self.verbose >= 1:
            print(f"Agent {agent_name} trained in {train_time:.2f}s")

        return result

    def get_results(self, agent_name: Optional[str] = None) -> Union[Dict, Optional[Dict]]:
        """Get test results.

        Args:
            agent_name: Optional specific agent name to get results for

        Returns:
            Results dictionary (all results or specific agent results)
        """
        if agent_name:
            return self.results.get(agent_name)
        return self.results

    def get_metrics(self, agent_name: str) -> Optional[Dict]:
        """Get final metrics for a specific agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Metrics dictionary or None if not found
        """
        result = self.results.get(agent_name)
        if result:
            return result.get("final_metrics")
        return None

    def compare_results(self) -> Dict:
        """Compare results across all tested agents.

        Returns:
            Dictionary containing comparison metrics
        """
        if not self.results:
            return {}

        # Build ranking based on training time (lower is better)
        ranking = []
        for agent_name, result in self.results.items():
            train_time = result.get("avg_train_time", result.get("train_time", float('inf')))
            ranking.append({
                "rank": None,  # Will be set after sorting
                "agent_name": agent_name,
                "agent_type": result["agent_type"],
                "avg_train_time": train_time,
                "total_timesteps": result["total_timesteps"],
            })

        # Sort by training time (lower is better)
        ranking.sort(key=lambda x: x["avg_train_time"])
        for i, item in enumerate(ranking):
            item["rank"] = i + 1

        comparison = {
            "test_id": self._current_test_id,
            "n_agents": len(self.results),
            "agents": {},
            "ranking": ranking,
        }

        for agent_name, result in self.results.items():
            comparison["agents"][agent_name] = {
                "agent_type": result["agent_type"],
                "avg_train_time": result.get("avg_train_time", result.get("train_time")),
                "total_timesteps": result["total_timesteps"],
            }

        return comparison

    def _save_results(self):
        """Save results to disk."""
        if not self._current_test_id:
            return

        # Save full results
        results_path = os.path.join(
            self.output_dir, f"{self._current_test_id}_results.json"
        )

        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        serializable_results = convert_to_serializable(self.results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        if self.verbose >= 2:
            print(f"Results saved to {results_path}")

    def save_comparison_report(self, filename: Optional[str] = None) -> str:
        """Save a comparison report.

        Args:
            filename: Optional filename for the report

        Returns:
            Path to the saved report
        """
        comparison = self.compare_results()

        if filename is None:
            filename = f"{self._current_test_id}_comparison.json"

        report_path = os.path.join(self.output_dir, filename)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        serializable_comparison = convert_to_serializable(comparison)

        with open(report_path, "w") as f:
            json.dump(serializable_comparison, f, indent=2)

        if self.verbose >= 1:
            print(f"Comparison report saved to {report_path}")

        return report_path
