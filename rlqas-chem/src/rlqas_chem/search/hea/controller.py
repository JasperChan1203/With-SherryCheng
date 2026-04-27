"""
HEA Search Controller for RLQAS Phase 2.

This module implements the HEASearchController class for managing
HEA search processes with RL agents.
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np

from rlqas_chem.rl import RLAgent, AgentFactory
from rlqas_chem.search.hea.environment import HEASearchEnv
from rlqas_chem.search.hea.circuit_builder import HEACircuitBuilder


class HEASearchController:
    """Controller for HEA architecture search.

    This class manages the complete HEA search process, including:
    - Environment setup and configuration
    - RL agent integration
    - Training loop execution
    - Result collection and analysis
    - Checkpointing and resumption

    Args:
        n_qubits: Number of qubits for the HEA
        max_layers: Maximum number of layers to search
        entanglement_patterns: List of allowed entanglement patterns
        rotation_gates: List of allowed rotation gate types
        output_dir: Directory for saving results and checkpoints
        verbose: Verbosity level (0, 1, or 2)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        max_layers: int = 4,
        entanglement_patterns: Optional[List[str]] = None,
        rotation_gates: Optional[List[str]] = None,
        output_dir: str = "results/hea_search",
        verbose: int = 1,
        config: Optional[Dict] = None,
    ):
        """Initialize HEA search controller."""
        # Support new-style: HEASearchController(molecule_data)
        self._molecule_data = None
        if hasattr(n_qubits, 'n_qubits'):
            self._molecule_data = n_qubits
            n_qubits = int(n_qubits.n_qubits)

        self.n_qubits = n_qubits
        self.max_layers = max_layers
        self.entanglement_patterns = entanglement_patterns or ["linear", "circular", "full"]
        self.rotation_gates = rotation_gates or ["rx", "ry", "rz"]
        self.output_dir = output_dir
        self.verbose = verbose

        # Search state
        self._env: Optional[HEASearchEnv] = None
        self._agent: Optional[RLAgent] = None
        self._best_circuit: Optional[Dict] = None
        self._best_energy: float = float("inf")
        self._training_history: List[Dict] = []
        self._current_episode: int = 0

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def setup_environment(
        self,
        molecule_data: Optional[Any] = None,
        target_energy: Optional[float] = None,
        parameter_sharing: str = "layer_wise",
    ) -> HEASearchEnv:
        """Set up the HEA search environment.

        Args:
            molecule_data: Optional molecule data for real energy computation
            target_energy: Optional target energy for the search
            parameter_sharing: Parameter sharing strategy

        Returns:
            Configured HEASearchEnv instance
        """
        self._env = HEASearchEnv(
            n_qubits=self.n_qubits,
            max_layers=self.max_layers,
            entanglement_patterns=self.entanglement_patterns,
            rotation_gates=self.rotation_gates,
            parameter_sharing=parameter_sharing,
            target_energy=target_energy,
            molecule_data=molecule_data,
        )

        if self.verbose >= 2:
            print(f"HEA Search Environment created:")
            print(f"  Qubits: {self.n_qubits}")
            print(f"  Max layers: {self.max_layers}")
            print(f"  Entanglement patterns: {self.entanglement_patterns}")
            print(f"  Rotation gates: {self.rotation_gates}")

        return self._env

    def setup_agent(
        self,
        agent_type: str = "ppo",
        config: Optional[Dict] = None,
    ) -> RLAgent:
        """Set up the RL agent for HEA search.

        Args:
            agent_type: Type of agent ("ppo" or "dqn")
            config: Optional agent configuration

        Returns:
            Configured RLAgent instance
        """
        if self._env is None:
            raise RuntimeError("Environment not set up. Call setup_environment() first.")

        self._agent = AgentFactory.create_agent(
            agent_type=agent_type,
            config=config,
            env=self._env,
        )

        if self.verbose >= 2:
            print(f"RL Agent created: {agent_type}")

        return self._agent

    def search(
        self,
        agent_type: str = "ppo",
        agent_config: Optional[Dict] = None,
        n_episodes: int = 100,
        total_timesteps: int = 10000,
        target_energy: Optional[float] = None,
        checkpoint_interval: int = 10,
        molecule_data: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run the HEA search process.

        Args:
            agent_type: Type of RL agent to use
            agent_config: Optional agent configuration
            n_episodes: Number of episodes to run
            total_timesteps: Total training timesteps
            target_energy: Optional target energy
            checkpoint_interval: Interval for saving checkpoints

        Returns:
            Dictionary containing search results
        """
        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"HEA Search Starting")
            print(f"  Agent: {agent_type}")
            print(f"  Episodes: {n_episodes}")
            print(f"  Timesteps: {total_timesteps}")
            print(f"{'='*60}\n")

        # Set up environment and agent
        mol_data = molecule_data if molecule_data is not None else self._molecule_data
        self.setup_environment(molecule_data=mol_data, target_energy=target_energy)
        self.setup_agent(agent_type=agent_type, config=agent_config)

        # Training loop
        self._training_history = []
        self._best_energy = float("inf")
        self._best_circuit = None

        # Use total_timesteps for training
        train_metrics = self._agent.learn(total_timesteps=total_timesteps)

        # FIX: Read both best_energy AND best_circuit_config from the environment.
        # Previously only best_energy was read; best_circuit was always None in results.
        if self._env is not None:
            if hasattr(self._env, 'best_energy'):
                self._best_energy = self._env.best_energy
            if hasattr(self._env, 'best_circuit_config') and self._env.best_circuit_config is not None:
                self._best_circuit = self._env.best_circuit_config

        # Collect final results
        results = {
            "n_qubits": self.n_qubits,
            "max_layers": self.max_layers,
            "agent_type": agent_type,
            "n_episodes": n_episodes,
            "total_timesteps": total_timesteps,
            "best_energy": self._best_energy,
            "best_circuit": self._best_circuit,
            "training_metrics": train_metrics,
            "training_history": self._training_history,
            "timestamp": datetime.now().isoformat(),
        }

        # Save results
        self._save_results(results)

        if self.verbose >= 1:
            print(f"\n{'='*60}")
            print(f"HEA Search Complete")
            print(f"  Best energy: {self._best_energy:.6f}")
            print(f"  Results saved to: {self.output_dir}")
            print(f"{'='*60}\n")

        return results

    def run_episode(self, episode_idx: int) -> Dict[str, Any]:
        """Run a single episode of HEA search.

        Args:
            episode_idx: Episode index

        Returns:
            Episode results dictionary
        """
        if self._env is None or self._agent is None:
            raise RuntimeError("Environment and agent must be set up first")

        obs, info = self._env.reset()
        episode_reward = 0.0
        done = False
        step = 0

        while not done:
            # Select action
            action, action_info = self._agent.act(obs)

            # Execute action
            next_obs, reward, done, truncated, step_info = self._env.step(action)

            episode_reward += reward
            step += 1

            # Check for new best
            current_energy = step_info.get("energy", 0)
            if current_energy < self._best_energy:
                self._best_energy = current_energy
                self._best_circuit = self._env.get_circuit_config()

            obs = next_obs

        episode_result = {
            "episode": episode_idx,
            "reward": episode_reward,
            "steps": step,
            "final_energy": self._env._current_energy,
            "circuit_config": self._env.get_circuit_config(),
        }

        self._training_history.append(episode_result)

        if self.verbose >= 2 and episode_idx % 10 == 0:
            print(f"  Episode {episode_idx}: reward={episode_reward:.4f}, energy={episode_result['final_energy']:.6f}")

        return episode_result

    def get_best_circuit(self) -> Optional[Dict]:
        """Get the best circuit found so far.

        Returns:
            Best circuit configuration or None
        """
        return self._best_circuit

    def get_best_energy(self) -> float:
        """Get the best energy found so far.

        Returns:
            Best energy value
        """
        return self._best_energy

    def get_training_history(self) -> List[Dict]:
        """Get training history.

        Returns:
            List of episode results
        """
        return self._training_history

    def _save_results(self, results: Dict):
        """Save search results to disk.

        Args:
            results: Results dictionary to save
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.output_dir, f"hea_search_{timestamp}.json")

        # Convert numpy types for JSON serialization
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

        serializable_results = convert_to_serializable(results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        if self.verbose >= 2:
            print(f"Results saved to {results_path}")

    def save_checkpoint(self, checkpoint_path: Optional[str] = None) -> str:
        """Save current search state as checkpoint.

        Args:
            checkpoint_path: Optional path for checkpoint file

        Returns:
            Path to saved checkpoint
        """
        if checkpoint_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{timestamp}.json")

        checkpoint = {
            "best_energy": self._best_energy,
            "best_circuit": self._best_circuit,
            "training_history": self._training_history,
            "current_episode": self._current_episode,
            "config": {
                "n_qubits": self.n_qubits,
                "max_layers": self.max_layers,
                "entanglement_patterns": self.entanglement_patterns,
                "rotation_gates": self.rotation_gates,
            },
        }

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

        with open(checkpoint_path, "w") as f:
            json.dump(convert_to_serializable(checkpoint), f, indent=2)

        if self.verbose >= 1:
            print(f"Checkpoint saved to {checkpoint_path}")

        return checkpoint_path

    def build_final_circuit(self) -> Optional[HEACircuitBuilder]:
        """Build the final circuit using best parameters found.

        Returns:
            HEACircuitBuilder instance or None if no search completed
        """
        if self._best_circuit is None:
            return None

        builder = HEACircuitBuilder(
            n_qubits=self.n_qubits,
            n_layers=self.max_layers,
            entanglement_pattern=self._best_circuit.get("entanglement_history", ["linear"])[-1] if self._best_circuit.get("entanglement_history") else "linear",
            rotation_gates=self.rotation_gates,
        )

        return builder


def run_hea_search(
    n_qubits: int,
    max_layers: int,
    agent_type: str = "ppo",
    total_timesteps: int = 10000,
    output_dir: str = "results/hea_search",
    verbose: int = 1,
) -> Dict[str, Any]:
    """Convenience function to run HEA search.

    Args:
        n_qubits: Number of qubits
        max_layers: Maximum number of layers
        agent_type: RL agent type
        total_timesteps: Training timesteps
        output_dir: Output directory
        verbose: Verbosity level

    Returns:
        Search results dictionary
    """
    controller = HEASearchController(
        n_qubits=n_qubits,
        max_layers=max_layers,
        output_dir=output_dir,
        verbose=verbose,
    )

    return controller.search(
        agent_type=agent_type,
        total_timesteps=total_timesteps,
    )
