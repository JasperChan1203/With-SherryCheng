"""UCC search controller for managing the complete search process."""

import os
import datetime
import json
import pickle
from typing import Dict, Any, Optional, List
import numpy as np

from ..molecule.processor import MoleculeData
from ..simulator.factory import SimulatorFactory
from ..rl.ppo_agent import PPOAgent
from .environment import UCCSearchEnv
from .config import UCCSearchConfig


class UCCPPOAgent(PPOAgent):
    """PPOAgent adapter for UCC search controller interface."""

    def __init__(self, config: Dict[str, Any] = None, env=None):
        """Initialize UCC PPO agent."""
        super().__init__(config=config, env=env)
        self.experience_buffer = []

    def select_action(self, state: np.ndarray) -> int:
        """Select action given state (compatible with controller)."""
        # Add batch dimension for vectorized environment
        if state.ndim == 1:
            state = state[np.newaxis, :]
        action, _ = self.act(state)
        return action

    def store_experience(self, state, reward, done, info):
        """Store experience (no-op for now)."""
        # Could store for later training, but PPO is on-policy
        pass

    def train(self):
        """Train agent (no-op for now)."""
        # PPO training happens through learn() with environment interaction
        pass


class UCCSearchController:
    """Manages the complete UCC search process."""

    def __init__(self, molecule_data: MoleculeData,
                 agent_type: str = 'ppo',
                 config: Dict[str, Any] = None):
        """Initialize search controller.

        Args:
            molecule_data: MoleculeData object from Task 001
            agent_type: Type of RL agent ('ppo' from Task 003)
            config: Controller configuration
        """
        self.molecule_data = molecule_data
        self.agent_type = agent_type
        self.config = UCCSearchConfig(config).get_section("controller")

        # Initialize components
        self.env = UCCSearchEnv(molecule_data, config)

        # Create RL agent
        if agent_type.lower() == 'ppo':
            agent_config = {
                "use_gpu": self.config.get("use_gpu", False),
                "seed": self.config.get("seed", 42),
                "policy_type": self.config.get("policy_type", "MlpPolicy"),
                "learning_rate": self.config.get("learning_rate", 3e-4),
                "n_steps": self.config.get("n_steps", 2048),
                "batch_size": self.config.get("batch_size", 64),
                "n_epochs": self.config.get("n_epochs", 10),
                "gamma": self.config.get("gamma", 0.99),
                "gae_lambda": self.config.get("gae_lambda", 0.95),
                "clip_range": self.config.get("clip_range", 0.2),
                "ent_coef": self.config.get("ent_coef", 0.0),
                "vf_coef": self.config.get("vf_coef", 0.5),
                "max_grad_norm": self.config.get("max_grad_norm", 0.5),
                "verbose": self.config.get("verbose", 1),
                "tensorboard_log": self.config.get("tensorboard_log", None),
                "n_envs": self.config.get("n_envs", 1),
                "monitor_dir": self.config.get("monitor_dir", None),
                "wrapper_class": self.config.get("wrapper_class", None),
            }
            self.agent = UCCPPOAgent(config=agent_config, env=self.env)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        # Initialize results storage
        self.results = {
            'best_energy': None,
            'best_circuit': None,
            'best_excitations': None,
            'best_params': None,
            'training_history': [],
            'convergence_reached': False,
            'episode_rewards': [],
            'episode_energies': [],
            'episode_depths': [],
        }

        # Track best overall
        self.best_overall_energy = float('inf')
        self.best_overall_circuit = None
        self.best_overall_excitations = None
        self.best_overall_params = None

    def search(self, n_episodes: int = 1000,
               early_stop_threshold: float = 1.6e-3) -> Dict[str, Any]:
        """Run UCC search.

        Args:
            n_episodes: Maximum number of episodes
            early_stop_threshold: Convergence threshold (Hartree)

        Returns:
            Dictionary containing search results
        """
        # Override config with function arguments if provided
        n_episodes = self.config.get("n_episodes", n_episodes)
        early_stop_threshold = self.config.get("early_stop_threshold", early_stop_threshold)

        print(f"Starting UCC search for {n_episodes} episodes")
        print(f"Early stop threshold: {early_stop_threshold} Hartree")

        for episode in range(n_episodes):
            # Reset environment (gymnasium returns observation and info)
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0.0
            episode_energy = self.env.current_energy
            episode_depth = 0

            while not done:
                try:
                    # Agent selects action
                    action = self.agent.select_action(obs)

                    # Environment step (gymnasium returns 5 values)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated

                except Exception as e:
                    # Handle unexpected errors (e.g., simulator crash, agent failure)
                    print(f"Error during episode {episode}: {e}")
                    reward = -10.0
                    done = True
                    info = {"error": str(e), "termination_reason": "unexpected_error"}
                    # Break out of while loop
                    break

                # Update agent with experience
                self.agent.store_experience(obs, reward, done, info)

                # Accumulate episode metrics
                episode_reward += reward
                episode_energy = info.get('energy', episode_energy)
                episode_depth = len(info.get('excitations', []))

                # Check for terminal state due to error (negative reward)
                if reward <= -10.0 and done:
                    # Invalid action or simulator failure - skip episode
                    break

            # End of episode
            # Train agent on collected experiences
            train_freq = self.config.get("train_frequency", 1)
            if train_freq > 0 and episode % train_freq == 0:
                try:
                    self.agent.train()
                except Exception as e:
                    print(f"Error during agent training at episode {episode}: {e}")

            # Record episode results
            self.results['episode_rewards'].append(episode_reward)
            self.results['episode_energies'].append(episode_energy)
            self.results['episode_depths'].append(episode_depth)

            # Update best overall results from environment's global best
            if self.env.global_best_energy is not None and (self.best_overall_energy is None or self.env.global_best_energy < self.best_overall_energy):
                self.best_overall_energy = self.env.global_best_energy
                self.best_overall_excitations = self.env.global_best_excitations.copy()
                self.best_overall_params = self.env.global_best_params.copy() if self.env.global_best_params is not None else None
                # Store circuit? Could store the circuit builder's circuit
                # For simplicity, store excitations and parameters
                self.results['best_energy'] = self.best_overall_energy
                self.results['best_excitations'] = self.best_overall_excitations.copy()
                self.results['best_params'] = self.best_overall_params.copy() if self.best_overall_params is not None else None
                self.results['best_circuit'] = None  # placeholder
                # Save checkpoint for new best
                self._save_checkpoint(episode, force=True)

            # Record training history
            self.results['training_history'].append({
                'episode': episode,
                'reward': episode_reward,
                'energy': episode_energy,
                'depth': episode_depth,
                'best_energy': self.best_overall_energy,
            })

            # Log progress
            if episode % self.config.get("log_frequency", 10) == 0:
                print(f"Episode {episode}: reward={episode_reward:.3f}, "
                      f"energy={episode_energy:.6f}, best={self.best_overall_energy:.6f}")
            # Save checkpoint if configured
            self._save_checkpoint(episode)

            # Check early stopping condition
            if self._check_convergence(early_stop_threshold):
                self.results['convergence_reached'] = True
                print(f"Convergence reached at episode {episode}")
                break

        # Finalize
        print(f"Search completed. Best energy: {self.best_overall_energy:.6f} Hartree")
        self.results['convergence_reached'] = self._check_convergence(early_stop_threshold)

        return self.results

    def _check_convergence(self, threshold: float) -> bool:
        """Check if energy convergence threshold is met.

        Args:
            threshold: Convergence threshold in Hartree

        Returns:
            True if converged
        """
        if self.best_overall_energy is None:
            return False

        # Compare with FCI energy from molecule data
        fci_energy = self.molecule_data.fci_energy
        if fci_energy is not None:
            energy_error = abs(self.best_overall_energy - fci_energy)
            return energy_error < threshold

        # If no FCI energy, check if improvement stagnated
        # Simple implementation: check last N episodes
        recent_energies = self.results['episode_energies'][-10:]
        if len(recent_energies) >= 10:
            # Check if standard deviation is small
            return np.std(recent_energies) < threshold

        return False

    def _save_checkpoint(self, episode: int, force: bool = False):
        """Save checkpoint of agent and results if checkpoint frequency reached.

        Args:
            episode: Current episode number
            force: If True, save checkpoint regardless of frequency
        """
        checkpoint_frequency = self.config.get("checkpoint_frequency", 0)
        if checkpoint_frequency <= 0:
            if not force:
                return
            # force=True, proceed without frequency check
            should_save = True
        else:
            should_save = force or episode % checkpoint_frequency == 0

        if should_save:
            checkpoint_dir = self.config.get("checkpoint_dir", "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Create timestamp or episode-based filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = "best" if force else "regular"
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_ep{episode}_{timestamp}_{suffix}")

            # Save agent
            agent_path = f"{checkpoint_path}_agent.pkl"
            self.agent.save(agent_path)

            # Save results snapshot
            results_path = f"{checkpoint_path}_results.json"
            self.save_results(results_path)

            print(f"Checkpoint saved for episode {episode} at {checkpoint_path}")

    def save_results(self, path: str):
        """Save search results to disk.

        Args:
            path: File path to save results
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Prepare serializable results
        serializable_results = self.results.copy()

        # Helper function to convert numpy types to Python types
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_for_json(item) for item in obj)
            else:
                return obj

        # Apply conversion to entire results dictionary
        serializable_results = convert_for_json(serializable_results)

        # Save as JSON
        with open(path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Optionally save agent model
        agent_path = path.replace('.json', '_agent.pkl')
        self.agent.save(agent_path)

        print(f"Results saved to {path}")

    def load_results(self, path: str):
        """Load search results from disk.

        Args:
            path: File path to load results from
        """
        with open(path, 'r') as f:
            loaded_results = json.load(f)

        # Restore numpy arrays
        for key in ['episode_rewards', 'episode_energies', 'episode_depths']:
            if key in loaded_results:
                loaded_results[key] = np.array(loaded_results[key])

        if loaded_results['best_params'] is not None:
            loaded_results['best_params'] = np.array(loaded_results['best_params'])

        # Convert best_excitations from lists to tuples (JSON serializes tuples as lists)
        if 'best_excitations' in loaded_results and loaded_results['best_excitations'] is not None:
            loaded_results['best_excitations'] = [tuple(exc) if isinstance(exc, list) else exc for exc in loaded_results['best_excitations']]

        self.results = loaded_results

        # Update best overall
        self.best_overall_energy = loaded_results['best_energy']
        self.best_overall_excitations = loaded_results['best_excitations']
        self.best_overall_params = loaded_results['best_params']

        print(f"Results loaded from {path}")