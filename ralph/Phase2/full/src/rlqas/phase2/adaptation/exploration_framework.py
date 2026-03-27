"""
RL Algorithm Exploration Framework for RLQAS Phase 2.

This module provides a framework for exploring and evaluating new RL algorithms
for quantum architecture search problems.
"""

import os
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
import json

import gymnasium as gym
import numpy as np

from rlqas.phase2.rl import RLAgent, AgentFactory


class _GlobalBestTrackingEnv(gym.Wrapper):
    """Thin wrapper that preserves global_best across episode resets.

    UCCSearchEnv.reset() resets global_best_energy to HF energy, losing
    inter-episode information.  This wrapper overrides reset() to retain
    the cross-episode best so that run_benchmarks() can read the true
    training best after all episodes complete.
    """

    def __init__(self, env):
        super().__init__(env)
        self._cross_best_energy = None
        self._cross_best_excitations = []
        self._cross_best_params = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Restore cross-episode best into the underlying env so future
        # episode-level comparisons in step() work correctly.
        if self._cross_best_energy is not None:
            self.env.global_best_energy = self._cross_best_energy
            self.env.global_best_excitations = list(self._cross_best_excitations)
            self.env.global_best_params = (
                np.array(self._cross_best_params)
                if self._cross_best_params is not None else None
            )
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        # After each step check if global best improved
        if (self.env.global_best_energy is not None
                and (self._cross_best_energy is None
                     or self.env.global_best_energy < self._cross_best_energy)):
            self._cross_best_energy = self.env.global_best_energy
            self._cross_best_excitations = list(self.env.global_best_excitations)
            self._cross_best_params = (
                np.array(self.env.global_best_params)
                if self.env.global_best_params is not None else None
            )
        return result

    @property
    def global_best_energy(self):
        return self._cross_best_energy

    @property
    def global_best_excitations(self):
        return self._cross_best_excitations

    @property
    def global_best_params(self):
        return self._cross_best_params


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
        self.benchmark_results: Dict[str, Any] = {}

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

        self.discovered_algorithms["a2c"] = {
            "name": "A2C",
            "full_name": "Advantage Actor-Critic",
            "type": "actor_critic",
            "action_space": ["discrete", "continuous"],
            "sample_efficiency": "medium",
            "stability": "medium",
            "sparse_rewards": "fair",
            "description": (
                "On-policy actor-critic. Updates policy after collecting n_steps "
                "of experience. Faster per-update than PPO but less stable. "
                "Good baseline for comparison with PPO."
            ),
            "baseline": True,
            "registered_at": datetime.now().isoformat(),
        }

        self.discovered_algorithms["sac_discrete"] = {
            "name": "SAC-Discrete",
            "full_name": "Soft Actor-Critic for Discrete Actions",
            "type": "actor_critic",
            "action_space": ["discrete"],
            "sample_efficiency": "high",
            "stability": "high",
            "sparse_rewards": "excellent",
            "description": (
                "Off-policy maximum-entropy actor-critic for discrete actions "
                "(Christodoulou 2019, extended 2020-2021). Twin critics reduce "
                "Q-value overestimation. Entropy bonus encourages diverse "
                "operator exploration. Automatic temperature tuning adapts "
                "exploration-exploitation balance. Highest sample efficiency "
                "among the four candidates — ideal for expensive quantum evals."
            ),
            "baseline": False,
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

    def run_benchmarks(
        self,
        molecule_data,
        n_episodes: int = 100,
        env_config: Optional[Dict] = None,
        agent_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run actual benchmark training for multiple RL algorithms.

        Trains each algorithm on a real quantum chemistry environment and
        collects training metrics. This is the core of autonomous exploration:
        rather than scoring algorithms by metadata, we train them and compare.

        Args:
            molecule_data: MoleculeData from process_molecule() — the real molecule.
            n_episodes: Number of training episodes per algorithm.
            env_config: Optional overrides for UCCSearchEnv configuration.
            agent_types: Agent types to benchmark; defaults to all registered.
                Valid values: "ppo", "dqn", "a2c", "sac_discrete".

        Returns:
            Dict mapping agent_type -> metrics, plus "winner" and "summary" keys.
        """
        from rlqas.phase1.search.environment import UCCSearchEnv
        from rlqas.phase2.rl.agent_factory import AgentFactory

        if agent_types is None:
            agent_types = [t for t in self.discovered_algorithms
                           if t in AgentFactory._AGENT_REGISTRY]

        base_env_config = {
            "complexity_penalty": 0.0,
            "param_init_strategy": "zeros",
            "max_depth": 10,
            "run_classical_opt": True,
        }
        if env_config:
            base_env_config.update(env_config)

        # UCCSearchConfig requires nested config: flat keys go under "environment" section
        nested_env_config = {"environment": dict(base_env_config)}

        max_depth = int(base_env_config.get("max_depth", 10))
        total_timesteps = n_episodes * max_depth

        results: Dict[str, Any] = {}

        for agent_type in agent_types:
            if self.verbose >= 1:
                print(f"[ExplorationFramework] Benchmarking {agent_type.upper()} "
                      f"for {n_episodes} episodes ({total_timesteps} timesteps)...")

            try:
                # Create a fresh env per agent for fair comparison.
                # Wrap with _GlobalBestTrackingEnv so global_best_energy is preserved
                # across episode resets (UCCSearchEnv.reset() clears global_best_energy).
                raw_env = UCCSearchEnv(molecule_data=molecule_data, config=base_env_config)
                tracking_env = _GlobalBestTrackingEnv(raw_env)
                agent = AgentFactory.create_agent(agent_type, config=None, env=tracking_env)
                agent.learn(total_timesteps=total_timesteps)

                fci_energy = float(molecule_data.fci_energy)
                # Read from tracking_env: preserves best across resets via _cross_best_energy
                best_energy_train = float(tracking_env.global_best_energy
                                          if tracking_env.global_best_energy is not None
                                          else fci_energy + 1.0)
                excitations_train = list(tracking_env.global_best_excitations or [])

                # Post-training evaluation: run agent.act() episodes with early stop.
                # Uses nested_env_config so run_classical_opt is read by UCCSearchConfig
                # (flat keys are silently ignored by get_section("environment")).
                best_energy, excitations = self._eval_episodes(
                    agent, molecule_data, nested_env_config, fci_energy,
                    n_eval=max(200, n_episodes // 2),
                    early_stop=1.6e-3,
                )
                # Take the best of training and evaluation phases
                if best_energy_train < best_energy:
                    best_energy = best_energy_train
                    excitations = excitations_train

                energy_error_ha = abs(best_energy - fci_energy)
                excitations = list(excitations or [])

                results[agent_type] = {
                    "best_energy": best_energy,
                    "fci_energy": fci_energy,
                    "energy_error_ha": energy_error_ha,
                    "energy_error_mha": energy_error_ha * 1000,
                    "chemical_accuracy_reached": energy_error_ha < 1.6e-3,
                    "episodes_to_convergence": None,
                    "best_operators": excitations,
                    "operator_count": len(excitations),
                    "energy_history": [],
                }

                if self.verbose >= 1:
                    print(f"  {agent_type.upper()}: best_energy={best_energy:.6f} Ha, "
                          f"error={energy_error_ha*1000:.4f} mHa, "
                          f"operators={results[agent_type]['operator_count']}, "
                          f"chemical_accuracy="
                          f"{'YES' if results[agent_type]['chemical_accuracy_reached'] else 'NO'}")

            except Exception as e:
                if self.verbose >= 1:
                    print(f"  {agent_type.upper()}: FAILED — {e}")
                results[agent_type] = {"error": str(e), "chemical_accuracy_reached": False}

        results["winner"] = self._determine_winner(results, agent_types)
        results["summary"] = self._format_benchmark_summary(results, agent_types)

        self.benchmark_results = results

        if self.verbose >= 1:
            print(f"\n[ExplorationFramework] Winner: {results['winner']['agent_type']} "
                  f"— {results['winner']['reason']}")

        return results

    def _eval_episodes(self, agent, molecule_data, env_config, fci_energy,
                        n_eval: int = 200, early_stop: float = 1.6e-3):
        """Run evaluation episodes using agent.act() with early-stop on chemical accuracy.

        Equivalent to UCCSearchController.search() but works with any RLAgent subclass.
        Uses the trained (or initial random) policy in a pure episode loop without
        gradient updates. Stops as soon as chemical accuracy is achieved.

        Returns:
            (best_energy, best_excitations)
        """
        from rlqas.phase1.search.environment import UCCSearchEnv
        eval_env = UCCSearchEnv(molecule_data=molecule_data, config=env_config)
        best_energy = eval_env._get_hf_energy()
        best_excitations = []

        for ep in range(n_eval):
            obs, _ = eval_env.reset()
            done = False
            while not done:
                try:
                    action, _ = agent.act(obs)
                    obs, reward, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated
                    if (eval_env.global_best_energy is not None
                            and eval_env.global_best_energy < best_energy):
                        best_energy = eval_env.global_best_energy
                        best_excitations = list(eval_env.global_best_excitations or [])
                except Exception:
                    done = True

            if abs(best_energy - fci_energy) < early_stop:
                if self.verbose >= 1:
                    print(f"    [eval] Chemical accuracy in ep {ep}: "
                          f"error={abs(best_energy-fci_energy)*1000:.4f} mHa")
                break

        return best_energy, best_excitations

    def _get_underlying_env(self, agent, original_env):
        """Return the UCCSearchEnv instance used during training.

        SB3 wraps envs in DummyVecEnv and Monitor; we need to unwrap both
        to reach the original UCCSearchEnv and read global_best_energy.
        """
        if hasattr(agent, 'model') and hasattr(agent.model, 'env'):
            try:
                wrapped = agent.model.env.envs[0]
                # Unwrap Monitor, TimeLimit, or any other single-env wrappers
                while hasattr(wrapped, 'env'):
                    wrapped = wrapped.env
                return wrapped
            except (AttributeError, IndexError):
                pass
        if hasattr(agent, 'env') and agent.env is not None:
            return agent.env
        return original_env

    def _determine_winner(self, results: Dict, agent_types: List[str]) -> Dict:
        """Identify best algorithm from real benchmark results."""
        accurate = {t: results[t] for t in agent_types
                    if isinstance(results.get(t), dict)
                    and results[t].get("chemical_accuracy_reached", False)}
        if accurate:
            winner_id = min(accurate, key=lambda t: accurate[t]["operator_count"])
            return {
                "agent_type": winner_id,
                "reason": (f"Chemical accuracy reached with "
                           f"{accurate[winner_id]['operator_count']} operators "
                           f"(error={accurate[winner_id]['energy_error_mha']:.4f} mHa)"),
                "operator_count": accurate[winner_id]["operator_count"],
            }
        valid = {t: results[t] for t in agent_types
                 if isinstance(results.get(t), dict) and "energy_error_ha" in results[t]}
        if valid:
            winner_id = min(valid, key=lambda t: valid[t]["energy_error_ha"])
            return {
                "agent_type": winner_id,
                "reason": (f"Lowest energy error "
                           f"{valid[winner_id]['energy_error_mha']:.4f} mHa "
                           f"(no algorithm reached chemical accuracy)"),
                "operator_count": valid[winner_id]["operator_count"],
            }
        return {"agent_type": None, "reason": "No algorithms completed training", "operator_count": 0}

    def _format_benchmark_summary(self, results: Dict, agent_types: List[str]) -> str:
        """Format benchmark results as a human-readable comparison table."""
        lines = [
            "=" * 70,
            "  RLQAS Algorithm Benchmark Results",
            "=" * 70,
            f"{'Algorithm':<10} {'Best Energy (Ha)':<20} {'Error (mHa)':<14} {'Operators':<12} {'Accurate?'}",
            "-" * 70,
        ]
        for t in agent_types:
            r = results.get(t, {})
            if "error" in r:
                lines.append(f"{t.upper():<10} {'FAILED':<20} {'—':<14} {'—':<12} NO")
            else:
                lines.append(
                    f"{t.upper():<10} {r.get('best_energy', 0):<20.8f} "
                    f"{r.get('energy_error_mha', 999):<14.4f} "
                    f"{r.get('operator_count', 0):<12} "
                    f"{'YES' if r.get('chemical_accuracy_reached') else 'NO'}"
                )
        winner = results.get("winner", {})
        lines += [
            "-" * 70,
            f"Winner: {str(winner.get('agent_type', 'N/A')).upper()} — {winner.get('reason', '')}",
            "=" * 70,
        ]
        return "\n".join(lines)

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
