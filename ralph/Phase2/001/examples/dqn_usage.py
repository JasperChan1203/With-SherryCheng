"""
Example usage of DQNAgent for RLQAS Phase 2.

This script demonstrates how to use the DQN agent for quantum architecture
search tasks, showing basic training, evaluation, and checkpoint management.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add Phase 1 and Phase 2 src to path
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(EXAMPLES_DIR)
# Phase 1 is at ../../Phase1/006/src from project root
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

# Insert Phase 1 FIRST to ensure it's loaded before Phase 2
sys.path.insert(0, PHASE1_SRC)
import rlqas.phase1.rl  # Pre-import Phase 1

sys.path.insert(0, PHASE2_SRC)
from rlqas.phase2.rl import DQNAgent, DQNConfig, AgentFactory, create_agent


class SimpleQuantumEnv(gym.Env):
    """
    Simplified quantum architecture search environment for demonstration.

    This environment simulates the key aspects of quantum circuit search:
    - State: Current circuit energy + parameters
    - Action: Select excitation operator to add
    - Reward: Energy improvement (negative = better for minimization)
    """

    def __init__(self, n_qubits=4, max_circuit_depth=10):
        super().__init__()
        self.n_qubits = n_qubits
        self.max_circuit_depth = max_circuit_depth
        self.n_actions = n_qubits * (n_qubits - 1)  # Pairwise excitations

        # State: [energy] + [circuit_params] (simplified)
        state_dim = 1 + n_qubits * 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.n_actions)

        self._current_energy = 0.0
        self._circuit_params = np.zeros(n_qubits * 2, dtype=np.float32)
        self._step_count = 0
        self._best_energy = np.inf

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._current_energy = self.np_random.uniform(-1.0, 0.0)
        self._circuit_params = self.np_random.uniform(
            -0.5, 0.5, size=len(self._circuit_params)
        ).astype(np.float32)
        self._best_energy = self._current_energy
        return self._get_state(), {}

    def _get_state(self):
        """Get current state as numpy array."""
        return np.concatenate([[self._current_energy], self._circuit_params])

    def step(self, action):
        self._step_count += 1

        # Simulate energy change based on action
        # In real application, this would involve quantum simulation
        energy_change = self.np_random.uniform(-0.1, 0.05)
        self._current_energy += energy_change

        # Update circuit params slightly
        self._circuit_params += self.np_random.uniform(
            -0.1, 0.1, size=len(self._circuit_params)
        ).astype(np.float32)

        # Reward is negative energy change (we want to minimize energy)
        reward = -energy_change

        # Track best energy
        if self._current_energy < self._best_energy:
            self._best_energy = self._current_energy
            reward += 0.5  # Bonus for finding new best

        done = self._step_count >= self.max_circuit_depth
        truncated = False

        info = {
            "energy": self._current_energy,
            "best_energy": self._best_energy,
            "excitation_idx": action,
        }

        return self._get_state(), reward, done, truncated, info


def train_dqn_agent(env, total_timesteps=1000, verbose=1):
    """
    Train a DQN agent on the given environment.

    Args:
        env: Gym environment
        total_timesteps: Number of timesteps to train
        verbose: Verbosity level

    Returns:
        Trained DQNAgent
    """
    print(f"Creating DQN agent...")

    # Custom config optimized for this environment
    config = {
        "learning_rate": 0.001,
        "gamma": 0.99,
        "buffer_size": 5000,
        "batch_size": 64,
        "exploration_fraction": 0.2,
        "exploration_final_eps": 0.05,
        "target_update_interval": 100,
        "train_freq": 4,
        "verbose": verbose,
        "seed": 42,
    }

    agent = DQNAgent(config=config, env=env)
    print(f"Agent created with config: {agent.config}")

    print(f"\nTraining DQN for {total_timesteps} timesteps...")
    metrics = agent.learn(total_timesteps=total_timesteps)
    print(f"Training complete! Metrics: {metrics}")

    return agent


def evaluate_agent(agent, env, n_episodes=5):
    """
    Evaluate a trained agent.

    Args:
        agent: Trained DQNAgent
        env: Gym environment
        n_episodes: Number of evaluation episodes
    """
    print(f"\nEvaluating agent over {n_episodes} episodes...")

    episode_rewards = []
    episode_energies = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=ep + 100)
        total_reward = 0.0
        done = False

        while not done:
            action, info = agent.act(state)
            state, reward, done, truncated, info = env.step(action)
            total_reward += reward

        episode_rewards.append(total_reward)
        episode_energies.append(info.get("best_energy", float("inf")))
        print(f"  Episode {ep + 1}: Reward = {total_reward:.3f}, Best Energy = {info.get('best_energy', 'N/A'):.4f}")

    print(f"\nEvaluation Summary:")
    print(f"  Avg Reward: {np.mean(episode_rewards):.3f} +/- {np.std(episode_rewards):.3f}")
    print(f"  Avg Best Energy: {np.mean(episode_energies):.4f} +/- {np.std(episode_energies):.4f}")


def save_and_load_demo(agent, save_path="dqn_checkpoint.zip"):
    """
    Demonstrate save/load functionality.

    Args:
        agent: Trained DQNAgent
        save_path: Path to save checkpoint
    """
    print(f"\nSaving agent to {save_path}...")
    agent.save(save_path)
    print(f"Agent saved successfully!")

    print(f"\nLoading agent from {save_path}...")
    loaded_agent = DQNAgent()
    loaded_agent.load(save_path)
    print(f"Agent loaded successfully!")

    return loaded_agent


def compare_ppo_dqn():
    """
    Compare PPO and DQN agents on the same environment.
    """
    print("\n" + "=" * 60)
    print("COMPARING PPO vs DQN")
    print("=" * 60)

    env = SimpleQuantumEnv(n_qubits=4, max_circuit_depth=10)

    # Train DQN
    print("\n--- Training DQN ---")
    dqn_agent = create_agent(
        "dqn",
        env=env,
        config={
            "verbose": 0,
            "buffer_size": 1000,
            "learning_rate": 0.001,
        },
    )
    dqn_agent.learn(total_timesteps=500)
    print("DQN training complete!")

    # Train PPO
    print("\n--- Training PPO ---")
    ppo_agent = create_agent(
        "ppo",
        env=env,
        config={
            "verbose": 0,
            "n_steps": 64,
            "learning_rate": 0.0003,
        },
    )
    ppo_agent.learn(total_timesteps=500)
    print("PPO training complete!")

    # Evaluate both
    print("\n--- Evaluation ---")
    state, _ = env.reset(seed=42)

    dqn_action, _ = dqn_agent.act(state)
    ppo_action, _ = ppo_agent.act(state)

    print(f"DQN action: {dqn_action}")
    print(f"PPO action: {ppo_action}")
    print(f"Actions match: {dqn_action == ppo_action}")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("RLQAS Phase 2 - DQN Agent Example")
    print("=" * 60)

    # Create environment
    env = SimpleQuantumEnv(n_qubits=4, max_circuit_depth=10)
    print(f"\nEnvironment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Number of actions: {env.n_actions}")

    # Train agent
    agent = train_dqn_agent(env, total_timesteps=1000, verbose=0)

    # Evaluate
    evaluate_agent(agent, env, n_episodes=5)

    # Save/load demo
    save_and_load_demo(agent, "dqn_example_checkpoint.zip")

    # Compare with PPO
    compare_ppo_dqn()

    # Cleanup
    if os.path.exists("dqn_example_checkpoint.zip"):
        os.remove("dqn_example_checkpoint.zip")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
