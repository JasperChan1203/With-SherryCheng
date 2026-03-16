"""
Multi-agent comparison example for RLQAS Phase 2.

This script demonstrates how to compare PPO and DQN agents
on the same quantum architecture search task.
"""

import os
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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
from rlqas.phase2.rl import AgentFactory, create_agent


class BenchmarkEnv(gym.Env):
    """Benchmark environment for algorithm comparison."""

    def __init__(self, state_dim=10, n_actions=6, reward_sparsity=0.3):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(n_actions)
        self._max_steps = 100
        self._step_count = 0
        self._reward_sparsity = reward_sparsity

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self.np_random.uniform(-1.0, 1.0, size=self.observation_space.shape[0]).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        # Sparse reward with some noise
        if self.np_random.random() < self._reward_sparsity:
            reward = self.np_random.uniform(0.5, 1.0)
        else:
            reward = self.np_random.uniform(-0.1, 0.1)

        obs = self.np_random.uniform(-1.0, 1.0, size=self.observation_space.shape[0]).astype(np.float32)
        done = self._step_count >= self._max_steps
        return obs, reward, done, False, {}


def train_and_evaluate(agent_type, env, total_timesteps=2000, n_eval_episodes=10):
    """
    Train an agent and evaluate its performance.

    Args:
        agent_type: "ppo" or "dqn"
        env: Gym environment
        total_timesteps: Training timesteps
        n_eval_episodes: Number of evaluation episodes

    Returns:
        Dictionary with training metrics and evaluation results
    """
    print(f"\nTraining {agent_type.upper()} agent...")
    start_time = time.time()

    # Configure agent
    if agent_type == "dqn":
        config = {
            "verbose": 0,
            "buffer_size": 5000,
            "learning_rate": 0.001,
            "exploration_fraction": 0.2,
        }
    else:  # ppo
        config = {
            "verbose": 0,
            "n_steps": 128,
            "learning_rate": 0.0003,
        }

    agent = create_agent(agent_type, env=env, config=config)

    # Train
    agent.learn(total_timesteps=total_timesteps)
    train_time = time.time() - start_time

    # Evaluate
    episode_rewards = []
    for ep in range(n_eval_episodes):
        state, _ = env.reset(seed=ep + 100)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = agent.act(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
        episode_rewards.append(total_reward)

    results = {
        "agent_type": agent_type,
        "train_time": train_time,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
    }

    print(f"{agent_type.upper()} Results:")
    print(f"  Training time: {train_time:.2f}s")
    print(f"  Mean reward: {results['mean_reward']:.3f} +/- {results['std_reward']:.3f}")
    print(f"  Range: [{results['min_reward']:.3f}, {results['max_reward']:.3f}]")

    return results


def plot_comparison(results, output_path="algorithm_comparison.png"):
    """
    Plot comparison results.

    Args:
        results: List of result dictionaries
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    agent_types = [r["agent_type"].upper() for r in results]
    mean_rewards = [r["mean_reward"] for r in results]
    std_rewards = [r["std_reward"] for r in results]
    train_times = [r["train_time"] for r in results]

    # Plot 1: Mean rewards
    ax1 = axes[0]
    bars1 = ax1.bar(agent_types, mean_rewards, yerr=std_rewards, capsize=5, color=['#3498db', '#e74c3c'])
    ax1.set_ylabel('Mean Episode Reward')
    ax1.set_title('Comparison of Mean Rewards')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    for bar, reward in zip(bars1, mean_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{reward:.2f}', ha='center', va='bottom')

    # Plot 2: Training time
    ax2 = axes[1]
    bars2 = ax2.bar(agent_types, train_times, color=['#3498db', '#e74c3c'])
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    for bar, time_val in zip(bars2, train_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{time_val:.1f}s', ha='center', va='bottom')

    # Plot 3: Reward distribution
    ax3 = axes[2]
    ax3.plot([0], [0], 'o', color='white', alpha=0)  # Dummy plot for legend
    for r in results:
        rewards = [r["mean_reward"] - r["std_reward"], r["mean_reward"], r["mean_reward"] + r["std_reward"]]
        label = f"{r['agent_type'].upper()}: {r['mean_reward']:.2f} +/- {r['std_reward']:.2f}"
        ax3.errorbar(r['agent_type'].upper(), r['mean_reward'], yerr=r['std_reward'],
                    fmt='o', capsize=10, label=label)
    ax3.set_ylabel('Episode Reward')
    ax3.set_title('Reward Comparison with Std Dev')
    ax3.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nComparison plot saved to: {output_path}")


def main():
    """Run multi-agent comparison."""
    print("=" * 60)
    print("RLQAS Phase 2 - Multi-Agent Comparison")
    print("=" * 60)

    # Create benchmark environment
    env = BenchmarkEnv(state_dim=10, n_actions=6, reward_sparsity=0.3)
    print(f"\nEnvironment: {env.observation_space}, Actions: {env.action_space.n}")

    # Train and evaluate both agents
    results = []

    for agent_type in ["dqn", "ppo"]:
        result = train_and_evaluate(
            agent_type, env,
            total_timesteps=2000,
            n_eval_episodes=10
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        print(f"\n{r['agent_type'].upper()}:")
        print(f"  Mean Reward: {r['mean_reward']:.3f} +/- {r['std_reward']:.3f}")
        print(f"  Training Time: {r['train_time']:.2f}s")

    # Plot
    plot_comparison(results)

    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
