"""GRPO (Group Relative Policy Optimization) Agent for quantum circuit search."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from rlqas_chem.rl.base_agent import RLAgent


class _PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class GRPOAgent(RLAgent):
    """Group Relative Policy Optimization agent.

    Samples G complete circuit rollouts per group, computes within-group
    relative advantages from VQE energies, and updates policy with clipped
    importance-weighted objective. No critic network needed.
    """

    def __init__(self, config: Optional[Dict] = None, env=None):
        super().__init__()
        cfg = config or {}
        self.lr = cfg.get("lr", 3e-4)
        self.clip_range = cfg.get("clip_range", 0.2)
        self.group_size = cfg.get("group_size", 4)
        self.gamma = cfg.get("gamma", 0.99)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)

        self._obs_dim = None
        self._action_dim = None
        self.policy = None
        self.optimizer = None
        self.env = env

        # Initialize if env is provided
        if env is not None:
            self._init_from_env(env)

    def _init_from_env(self, env):
        """Initialize policy network from environment."""
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        self._init_policy(obs_dim, action_dim)

    def _init_policy(self, obs_dim: int, action_dim: int):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.policy = _PolicyNet(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def _ensure_initialized(self, obs=None):
        """Lazily initialize policy from observation if not yet done."""
        if self.policy is None and obs is not None:
            obs_dim = obs.shape[-1] if hasattr(obs, 'shape') else len(obs)
            # Use env if available, otherwise we need action_dim from somewhere
            if self.env is not None:
                action_dim = self.env.action_space.n
            else:
                raise RuntimeError("GRPOAgent: must provide env or call _init_policy() before use")
            self._init_policy(obs_dim, action_dim)

    def act(self, obs: np.ndarray) -> Tuple[int, float]:
        """Sample action from policy.

        Returns:
            (action, log_prob)
        """
        self._ensure_initialized(obs)
        obs_t = torch.FloatTensor(np.array(obs)).flatten()
        with torch.no_grad():
            probs = self.policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def collect_episode(self, env) -> Dict:
        """Run one complete episode.

        Returns:
            dict with keys: states, actions, log_probs, final_energy
        """
        self._ensure_initialized()
        if self.policy is None:
            self._init_from_env(env)

        obs, _ = env.reset()
        states, actions, log_probs = [], [], []
        final_energy = float('inf')
        done = False

        while not done:
            action, log_prob = self.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(np.array(obs).flatten().tolist())
            actions.append(action)
            log_probs.append(log_prob)

            final_energy = info.get("energy", final_energy)
            obs = next_obs

        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "final_energy": final_energy,
        }

    def compute_group_advantages(self, group_results: List[Dict]) -> torch.Tensor:
        """Compute normalized within-group advantages.

        Energy is minimized (lower = better), so we negate to get rewards.
        """
        energy_rewards = [r["final_energy"] for r in group_results]
        rewards = [-e if e != float('inf') else -1000.0 for e in energy_rewards]
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        mean_r = rewards_t.mean()
        std_r = rewards_t.std() + 1e-8
        advantages = (rewards_t - mean_r) / std_r
        return advantages

    def update(self, group_results: List[Dict], advantages: torch.Tensor):
        """Update policy using clipped importance-weighted objective."""
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        total_steps = 0

        for i, episode in enumerate(group_results):
            A_i = advantages[i]
            states = episode["states"]
            actions = episode["actions"]
            old_log_probs = episode["log_probs"]

            for state, action, old_lp in zip(states, actions, old_log_probs):
                state_t = torch.FloatTensor(state)
                action_t = torch.tensor(action)

                probs = self.policy(state_t)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(action_t)

                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.min(ratio * A_i, clipped * A_i)

                # Entropy bonus
                entropy = dist.entropy()
                total_loss = total_loss + policy_loss - self.entropy_coef * entropy
                total_steps += 1

        if total_steps > 0:
            loss = total_loss / total_steps
            loss.backward()
            self.optimizer.step()
            return float(loss.item())
        return 0.0

    def train_one_group(self, env) -> Dict:
        """Collect G episodes and perform one policy update.

        Returns:
            dict with mean_energy, best_energy, advantages
        """
        group_results = [self.collect_episode(env) for _ in range(self.group_size)]
        advantages = self.compute_group_advantages(group_results)
        loss = self.update(group_results, advantages)

        energies = [r["final_energy"] for r in group_results if r["final_energy"] != float('inf')]
        mean_energy = float(np.mean(energies)) if energies else float('inf')
        best_energy = float(min(energies)) if energies else float('inf')

        return {
            "mean_energy": mean_energy,
            "best_energy": best_energy,
            "advantages": advantages.tolist(),
            "loss": loss,
        }

    def learn(self, experience=None, total_timesteps: int = 10000) -> Dict:
        """Satisfy RLAgent ABC. For GRPO, use train_one_group() instead."""
        return {"total_timesteps": total_timesteps}

    def save(self, path: str) -> None:
        if self.policy is not None:
            import os
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save({"policy": self.policy.state_dict(), "config": {
                "obs_dim": self._obs_dim, "action_dim": self._action_dim,
                "lr": self.lr, "clip_range": self.clip_range,
                "group_size": self.group_size,
            }}, path)

    def load(self, path: str) -> None:
        data = torch.load(path)
        cfg = data["config"]
        self._init_policy(cfg["obs_dim"], cfg["action_dim"])
        self.policy.load_state_dict(data["policy"])
