"""GiGPO: Group-relative Policy Optimization with intra-episode credit assignment."""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
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


class GiGPOAgent(RLAgent):
    """GiGPO agent: per-step advantages with anchor-group normalization.

    Unlike standard GRPO (one advantage per episode), GiGPO computes
    per-step advantages using Monte Carlo returns and anchor-group
    normalization, addressing sparse reward problems in QAS tasks.

    Supports two instantiation conventions:
    - GiGPOAgent(obs_space, act_space, config)  — algorithm test style
    - GiGPOAgent(config=dict, env=env)          — AgentFactory style
    """

    def __init__(self, observation_space_or_config=None, action_space=None,
                 config=None, env=None):
        super().__init__()

        # Support both calling conventions
        if action_space is not None:
            # Convention 1: GiGPOAgent(obs_space, act_space, config_dict)
            obs_space = observation_space_or_config
            act_space = action_space
            cfg = config if config is not None else {}
            self.env = None
        else:
            # Convention 2: GiGPOAgent(config=dict, env=env)
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
        self.anchor_ratio = cfg.get("anchor_ratio", 0.33)

        self._obs_dim = None
        self._action_dim = None
        self.policy = None
        self.optimizer = None

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
                    "GiGPOAgent: must provide env or initialize with obs/action spaces"
                )
            self._init_policy(
                ref_env.observation_space.shape[0],
                ref_env.action_space.n,
            )

    def act(self, obs: np.ndarray) -> Tuple[int, Dict]:
        """Deterministic (argmax) action selection. Returns (action, {'log_prob': lp})."""
        self._ensure_initialized()
        obs_t = torch.FloatTensor(np.array(obs)).flatten()
        with torch.no_grad():
            probs = self.policy(obs_t)
            action = torch.argmax(probs)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)
        return int(action.item()), {"log_prob": float(log_prob.item())}

    def _sample_action(self, obs_t: torch.Tensor) -> Tuple[int, float]:
        """Stochastic action sampling for training rollouts."""
        with torch.no_grad():
            probs = self.policy(obs_t)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item())

    def collect_episode(self, env) -> Dict:
        """Run one complete episode with stochastic sampling.

        Returns:
            dict with states, actions, log_probs, rewards, final_energy
        """
        self._ensure_initialized(env)
        obs, _ = env.reset()
        states, actions, log_probs, rewards = [], [], [], []
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

            if "energy" in info:
                final_energy = info["energy"]
            obs = next_obs

        return {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "rewards": rewards,
            "final_energy": final_energy,
        }

    def compute_step_advantages(self, episode_data: Dict) -> List[float]:
        """Compute per-step advantages using MC returns + anchor-group normalization.

        Steps:
        1. Monte Carlo returns: G_t = sum_{k=t}^T gamma^{k-t} * r_k
        2. Subtract mean(G) as baseline
        3. Anchor-Group normalization: subtract mean of first anchor_ratio steps
           from the remaining steps
        4. Normalize: (A - mean(A)) / (std(A) + 1e-8)
        """
        rewards = episode_data["rewards"]
        T = len(rewards)
        if T == 0:
            return []

        # Step 1: Monte Carlo returns
        G = np.zeros(T)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            G[t] = running

        # Step 2: Subtract mean as baseline
        advs = G - G.mean()

        # Step 3: Anchor-Group normalization
        anchor_size = max(1, T // 3)
        if anchor_size < T:
            baseline = advs[:anchor_size].mean()
            advs[anchor_size:] -= baseline

        # Step 4: Normalize
        std = advs.std() + 1e-8
        advs = (advs - advs.mean()) / std

        return advs.tolist()

    def update(self, group_episodes: List[Dict],
               group_step_advantages: List[List[float]]) -> float:
        """Policy update using clipped PPO objective with per-step advantages."""
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0)
        total_steps = 0

        for episode, step_advs in zip(group_episodes, group_step_advantages):
            states = episode["states"]
            actions = episode["actions"]
            old_log_probs = episode["log_probs"]

            for state, action, old_lp, A_t in zip(states, actions, old_log_probs, step_advs):
                state_t = torch.FloatTensor(state)
                action_t = torch.tensor(action)
                A_scalar = float(A_t)

                probs = self.policy(state_t)
                dist = torch.distributions.Categorical(probs)
                new_lp = dist.log_prob(action_t)

                ratio = torch.exp(new_lp - old_lp)
                clipped = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
                policy_loss = -torch.min(ratio * A_scalar, clipped * A_scalar)
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
        """Collect group_size episodes and perform one policy update.

        Returns:
            dict with mean_energy and best_energy
        """
        self._ensure_initialized(env)
        group_episodes = [self.collect_episode(env) for _ in range(self.group_size)]
        group_step_advantages = [
            self.compute_step_advantages(ep) for ep in group_episodes
        ]
        loss = self.update(group_episodes, group_step_advantages)

        energies = [
            ep["final_energy"] for ep in group_episodes
            if ep["final_energy"] != float("inf")
        ]
        mean_energy = float(np.mean(energies)) if energies else float("inf")
        best_energy = float(min(energies)) if energies else float("inf")

        return {
            "mean_energy": mean_energy,
            "best_energy": best_energy,
            "loss": loss,
        }

    def learn(self, experience=None, total_timesteps: int = 10000,
              callback=None) -> Dict:
        """Train by running train_one_group() until total_timesteps are used."""
        env = self.env
        if env is None:
            return {"total_timesteps": total_timesteps}

        self._ensure_initialized(env)
        steps_done = 0

        while steps_done < total_timesteps:
            group_episodes = []
            group_steps = 0
            for _ in range(self.group_size):
                ep = self.collect_episode(env)
                group_episodes.append(ep)
                group_steps += len(ep["actions"])

            group_step_advantages = [
                self.compute_step_advantages(ep) for ep in group_episodes
            ]
            self.update(group_episodes, group_step_advantages)
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
                        "anchor_ratio": self.anchor_ratio,
                    },
                },
                path,
            )

    def load(self, path: str) -> None:
        data = torch.load(path, weights_only=False)
        cfg = data["config"]
        self._init_policy(cfg["obs_dim"], cfg["action_dim"])
        self.policy.load_state_dict(data["policy"])
