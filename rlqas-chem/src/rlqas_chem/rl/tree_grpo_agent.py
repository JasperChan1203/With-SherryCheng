"""Tree-GRPO: GRPO with prefix sharing and VQE caching."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import numpy as np

from rlqas_chem.rl.base_agent import RLAgent
from rlqas_chem.search.ucc.prefix_cache import PrefixCache


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


class TreeGRPOAgent(RLAgent):
    """Tree-GRPO agent: GRPO with prefix sharing and VQE caching.

    Exploits deterministic prefix semantics of UCC operator sequences.
    Episodes that share the same final operator sequence are detected via
    a PrefixCache, enabling energy reuse and reducing redundant VQE calls.

    Supports two instantiation conventions:
    - TreeGRPOAgent(obs_space, act_space, config)  — algorithm test style
    - TreeGRPOAgent(config=dict, env=env)          — AgentFactory style
    """

    def __init__(self, observation_space_or_config=None, action_space=None,
                 config=None, env=None):
        super().__init__()

        if action_space is not None:
            # Convention 1: TreeGRPOAgent(obs_space, act_space, config_dict)
            obs_space = observation_space_or_config
            act_space = action_space
            cfg = config if config is not None else {}
            self.env = None
        else:
            # Convention 2: TreeGRPOAgent(config=dict, env=env)
            cfg = (observation_space_or_config
                   if isinstance(observation_space_or_config, dict)
                   else (config or {}))
            self.env = env
            obs_space = env.observation_space if env is not None else None
            act_space = env.action_space if env is not None else None

        self.lr = cfg.get("lr", 3e-4)
        self.gamma = cfg.get("gamma", 0.99)
        self.group_size = cfg.get("group_size", 8)
        self.clip_range = cfg.get("clip_range", 0.2)
        self.entropy_coef = cfg.get("entropy_coef", 0.01)

        self._obs_dim = None
        self._action_dim = None
        self.policy = None
        self.optimizer = None
        self.prefix_cache = PrefixCache()

        if obs_space is not None and act_space is not None:
            self._init_policy(obs_space.shape[0], act_space.n)

    def _init_policy(self, obs_dim: int, action_dim: int):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.policy = _PolicyNet(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def _ensure_initialized(self, env=None):
        if self.policy is None:
            ref_env = env or self.env
            if ref_env is None:
                raise RuntimeError(
                    "TreeGRPOAgent: must provide env or initialize with obs/action spaces"
                )
            self._init_policy(
                ref_env.observation_space.shape[0],
                ref_env.action_space.n,
            )

    def act(self, obs: np.ndarray) -> Tuple[int, Dict]:
        """Deterministic (argmax) action selection."""
        self._ensure_initialized()
        obs_t = torch.FloatTensor(np.array(obs)).flatten()
        with torch.no_grad():
            probs = self.policy(obs_t)
            action = torch.argmax(probs)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
        return int(action.item()), {"log_prob": float(log_prob.item())}

    def _sample_action(self, obs_t: torch.Tensor) -> Tuple[int, float]:
        """Stochastic action sampling for training."""
        with torch.no_grad():
            probs = self.policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def collect_episode_with_cache(self, env) -> Dict:
        """Run one complete episode with prefix caching.

        After the episode, checks if the final operator sequence (prefix) is
        already cached. If found, uses the cached energy; otherwise stores
        the new result.

        Returns:
            dict with states, actions, log_probs, rewards, final_energy, cache_hits
        """
        self._ensure_initialized(env)
        obs, _ = env.reset()
        states, actions, log_probs, rewards = [], [], [], []
        actions_so_far = []
        final_energy = float("inf")
        done = False

        while not done:
            obs_t = torch.FloatTensor(np.array(obs)).flatten()
            action, log_prob = self._sample_action(obs_t)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            states.append(np.array(obs).flatten().tolist())
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(float(reward))
            actions_so_far.append(action)

            if "energy" in info:
                final_energy = info["energy"]
            obs = next_obs

        # Prefix caching: check if we've seen this operator sequence before
        final_prefix = tuple(actions_so_far)
        cached_e = self.prefix_cache.get(final_prefix)
        if cached_e is not None:
            # Use cached energy (cache.get() already incremented hit counter)
            final_energy = cached_e
        elif final_energy != float("inf"):
            self.prefix_cache.put(final_prefix, final_energy)

        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "final_energy": final_energy,
            "cache_hits": self.prefix_cache.hits(),
        }

    def collect_group(self, env) -> List[Dict]:
        """Collect group_size episodes."""
        return [self.collect_episode_with_cache(env) for _ in range(self.group_size)]

    def compute_group_advantages(self, group_episodes: List[Dict]) -> List[float]:
        """Compute normalized within-group advantages from episode-level energies."""
        energies = [ep["final_energy"] for ep in group_episodes]
        rewards = [-e if e != float("inf") else -1000.0 for e in energies]
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards)) + 1e-8
        return [(r - mean_r) / std_r for r in rewards]

    def update(self, group_episodes: List[Dict], advantages: List[float]) -> float:
        """Standard GRPO clipped objective at episode level."""
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        total_steps = 0

        for episode, A_i in zip(group_episodes, advantages):
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
        """Collect group, compute advantages, update policy, return metrics.

        Returns:
            dict with mean_energy, best_energy, cache_hits
        """
        self._ensure_initialized(env)
        group = self.collect_group(env)
        advantages = self.compute_group_advantages(group)
        loss = self.update(group, advantages)

        energies = [
            ep["final_energy"] for ep in group
            if ep["final_energy"] != float("inf")
        ]
        mean_energy = float(np.mean(energies)) if energies else float("inf")
        best_energy = float(min(energies)) if energies else float("inf")

        return {
            "mean_energy": mean_energy,
            "best_energy": best_energy,
            "cache_hits": self.prefix_cache.hits(),
            "loss": loss,
        }

    def learn(self, experience=None, total_timesteps: int = 10000,
              callback=None) -> Dict:
        """Train by running collect_group() + update() until total_timesteps used."""
        env = self.env
        if env is None:
            return {"total_timesteps": total_timesteps}

        self._ensure_initialized(env)
        steps_done = 0

        while steps_done < total_timesteps:
            group = self.collect_group(env)
            group_steps = sum(len(ep["actions"]) for ep in group)
            advantages = self.compute_group_advantages(group)
            self.update(group, advantages)
            steps_done += group_steps

        return {"total_timesteps": total_timesteps}

    def save(self, path: str) -> None:
        if self.policy is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(
                {
                    "policy": self.policy.state_dict(),
                    "config": {
                        "obs_dim": self._obs_dim,
                        "action_dim": self._action_dim,
                        "lr": self.lr,
                        "clip_range": self.clip_range,
                        "group_size": self.group_size,
                        "gamma": self.gamma,
                        "entropy_coef": self.entropy_coef,
                    },
                },
                path,
            )

    def load(self, path: str) -> None:
        data = torch.load(path, weights_only=False)
        cfg = data["config"]
        self._init_policy(cfg["obs_dim"], cfg["action_dim"])
        self.policy.load_state_dict(data["policy"])
