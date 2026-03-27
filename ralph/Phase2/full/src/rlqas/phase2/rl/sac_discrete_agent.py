"""
SACDiscreteAgent: Soft Actor-Critic for Discrete Action Spaces.

Reference: Christodoulou, P. (2019). "Soft Actor-Critic for Discrete Action Settings."
arXiv:1910.07207. Extended with stability improvements (2020-2021).

Why SAC-Discrete for RLQAS:
- Maximum entropy framework: entropy bonus encourages exploring diverse operator combinations
- Off-policy replay: sample-efficient — crucial when each energy eval calls a quantum solver
- Twin critics: reduces Q-value overestimation bias
- Automatic temperature tuning: adapts exploration/exploitation balance during training
- Handles discrete actions natively via softmax policy and exact entropy computation

Action space: Discrete (selects excitation operators from fixed set)
Reward structure: Sparse (energy improvements only when operator reduces energy)
"""

import os
from typing import Dict, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym

from rlqas.phase1.rl.base_agent import RLAgent


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    """Simple fixed-size replay buffer using numpy arrays."""

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.pos = 0
        self.size = 0
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self._obs[self.pos] = obs
        self._actions[self.pos] = action
        self._rewards[self.pos] = float(reward)
        self._next_obs[self.pos] = next_obs
        self._dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self._obs[idx], device=device),
            torch.as_tensor(self._actions[idx], device=device),
            torch.as_tensor(self._rewards[idx], device=device),
            torch.as_tensor(self._next_obs[idx], device=device),
            torch.as_tensor(self._dones[idx], device=device),
        )

    def __len__(self):
        return self.size


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class _DiscreteActor(nn.Module):
    """Policy network: state -> probability distribution over discrete actions."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        return probs


class _DiscreteCritic(nn.Module):
    """Q-network: state -> Q-value for each discrete action."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class SACDiscreteConfig:
    """Configuration for SACDiscreteAgent."""

    DEFAULT_CONFIG: Dict[str, Any] = {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,              # soft target-update rate
        "alpha": 0.2,              # initial entropy coefficient
        "auto_entropy_tuning": True,
        "batch_size": 64,
        "buffer_size": 10000,
        "learning_starts": 100,    # random steps before training begins
        "train_freq": 1,           # gradient update every N env steps
        "gradient_steps": 1,       # gradient updates per train_freq
        "hidden_sizes": (256, 256),
        "verbose": 0,
        "seed": 42,
        "use_gpu": True,
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SACDiscreteAgent(RLAgent):
    """Soft Actor-Critic for discrete action spaces.

    Uses maximum entropy RL with off-policy updates. Suited to RLQAS because:
    1. Entropy bonus encourages exploration of diverse operator combinations.
    2. Off-policy replay gives high sample efficiency (fewer expensive quantum evals).
    3. Twin critics reduce Q-value overestimation.
    4. Automatic temperature tuning adapts exploration-exploitation balance.

    Args:
        config: Agent configuration dictionary.
        env: Gymnasium environment with Discrete action space.
    """

    def __init__(self, config: Optional[Dict] = None, env=None):
        self._config_obj = SACDiscreteConfig(config)
        self.config = self._config_obj.config
        self.env = env

        # Networks — deferred until env is known
        self.device: str = "cpu"
        self.obs_dim: int = 0
        self.n_actions: int = 0
        self.actor: Optional[_DiscreteActor] = None
        self.critic1: Optional[_DiscreteCritic] = None
        self.critic2: Optional[_DiscreteCritic] = None
        self.critic1_target: Optional[_DiscreteCritic] = None
        self.critic2_target: Optional[_DiscreteCritic] = None
        self.actor_optim = None
        self.critic1_optim = None
        self.critic2_optim = None
        self.log_alpha: Optional[torch.Tensor] = None
        self.alpha_optim = None
        self.alpha: float = float(self.config.get("alpha", 0.2))
        self.replay_buffer: Optional[_ReplayBuffer] = None
        self.target_entropy: float = 0.0

        if env is not None:
            self._init_networks(env)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _get_device(self) -> str:
        if self.config.get("use_gpu", True) and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _init_networks(self, env) -> None:
        self.device = self._get_device()
        obs_space = env.observation_space
        act_space = env.action_space

        if isinstance(obs_space, gym.spaces.Box):
            obs_dim = int(np.prod(obs_space.shape))
        else:
            obs_dim = int(obs_space.n)
        n_actions = int(act_space.n)

        self.obs_dim = obs_dim
        self.n_actions = n_actions

        hidden = tuple(self.config.get("hidden_sizes", (256, 256)))
        lr = float(self.config.get("learning_rate", 3e-4))

        self.actor = _DiscreteActor(obs_dim, n_actions, hidden).to(self.device)
        self.critic1 = _DiscreteCritic(obs_dim, n_actions, hidden).to(self.device)
        self.critic2 = _DiscreteCritic(obs_dim, n_actions, hidden).to(self.device)
        self.critic1_target = _DiscreteCritic(obs_dim, n_actions, hidden).to(self.device)
        self.critic2_target = _DiscreteCritic(obs_dim, n_actions, hidden).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=lr)

        # Target entropy: ~98% of maximum entropy (log of uniform distribution)
        self.target_entropy = 0.98 * np.log(n_actions)
        if self.config.get("auto_entropy_tuning", True):
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()

        buf_size = int(self.config.get("buffer_size", 10000))
        self.replay_buffer = _ReplayBuffer(buf_size, obs_dim)

    # ------------------------------------------------------------------
    # Core interface methods
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray) -> Tuple[int, Dict]:
        """Select action given current state (stochastic during training)."""
        if self.actor is None:
            raise RuntimeError("SACDiscreteAgent not initialised with an environment.")
        obs_t = torch.as_tensor(
            state.flatten().astype(np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(obs_t)
        action = int(torch.multinomial(probs, 1).item())
        return action, {}

    def learn(
        self,
        experience: Optional[Dict] = None,
        total_timesteps: int = 10000,
    ) -> Dict:
        """Run SAC-Discrete training loop for total_timesteps env interactions.

        Args:
            experience: Unused (off-policy agent manages its own buffer).
            total_timesteps: Total number of environment steps to run.

        Returns:
            Training statistics dict.
        """
        if self.env is None or self.actor is None:
            raise RuntimeError("SACDiscreteAgent not initialised with an environment.")

        learning_starts = int(self.config.get("learning_starts", 100))
        train_freq = int(self.config.get("train_freq", 1))
        gradient_steps = int(self.config.get("gradient_steps", 1))
        batch_size = int(self.config.get("batch_size", 64))

        obs, _ = self.env.reset()
        episode_count = 0

        for t in range(total_timesteps):
            if t < learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _ = self.act(obs)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.add(
                obs.flatten().astype(np.float32),
                action,
                float(reward),
                next_obs.flatten().astype(np.float32),
                done,
            )
            obs = next_obs

            if done:
                obs, _ = self.env.reset()
                episode_count += 1

            if (t >= learning_starts
                    and t % train_freq == 0
                    and len(self.replay_buffer) >= batch_size):
                for _ in range(gradient_steps):
                    self._update(batch_size)

        return {
            "total_timesteps": total_timesteps,
            "episodes": episode_count,
            "algorithm": "sac_discrete",
        }

    def save(self, path: str) -> None:
        if self.actor is None:
            raise RuntimeError("SACDiscreteAgent not initialised.")
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic1": self.critic1.state_dict(),
                "critic2": self.critic2.state_dict(),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "n_actions": self.n_actions,
                "alpha": self.alpha,
            },
            path + ".pt",
        )

    def load(self, path: str) -> None:
        device = self._get_device()
        data = torch.load(path + ".pt", map_location=device)
        self.config = data["config"]
        self.obs_dim = data["obs_dim"]
        self.n_actions = data["n_actions"]
        self.alpha = data.get("alpha", 0.2)
        self.device = device
        hidden = tuple(self.config.get("hidden_sizes", (256, 256)))
        self.actor = _DiscreteActor(self.obs_dim, self.n_actions, hidden).to(device)
        self.critic1 = _DiscreteCritic(self.obs_dim, self.n_actions, hidden).to(device)
        self.critic2 = _DiscreteCritic(self.obs_dim, self.n_actions, hidden).to(device)
        self.actor.load_state_dict(data["actor"])
        self.critic1.load_state_dict(data["critic1"])
        self.critic2.load_state_dict(data["critic2"])

    def get_config(self) -> Dict:
        return dict(self.config)

    # ------------------------------------------------------------------
    # Internal update step
    # ------------------------------------------------------------------

    def _update(self, batch_size: int) -> None:
        """One gradient update step (critic + actor + temperature)."""
        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(
            batch_size, self.device
        )
        gamma = float(self.config.get("gamma", 0.99))
        tau = float(self.config.get("tau", 0.005))

        # ---- Critic targets ----
        with torch.no_grad():
            next_probs = self.actor(next_obs)                     # (B, A)
            next_log_probs = torch.log(next_probs + 1e-8)         # (B, A)
            q1_next = self.critic1_target(next_obs)               # (B, A)
            q2_next = self.critic2_target(next_obs)               # (B, A)
            min_q_next = torch.min(q1_next, q2_next)              # (B, A)
            # V(s') = E_a[Q(s',a) - alpha * log pi(a|s')]
            next_v = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=1)
            target_q = rewards + (1.0 - dones) * gamma * next_v  # (B,)

        # ---- Critic update ----
        q1_pred = self.critic1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = self.critic2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        c1_loss = F.mse_loss(q1_pred, target_q)
        c2_loss = F.mse_loss(q2_pred, target_q)

        self.critic1_optim.zero_grad()
        c1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        c2_loss.backward()
        self.critic2_optim.step()

        # ---- Actor update ----
        probs = self.actor(obs)                                    # (B, A)
        log_probs = torch.log(probs + 1e-8)                       # (B, A)
        with torch.no_grad():
            q1 = self.critic1(obs)
            q2 = self.critic2(obs)
            min_q = torch.min(q1, q2)
        # Maximise expected Q minus entropy cost
        actor_loss = (probs * (self.alpha * log_probs - min_q)).sum(dim=1).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # ---- Temperature (alpha) update ----
        if self.config.get("auto_entropy_tuning", True) and self.log_alpha is not None:
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()

        # ---- Soft target update ----
        for p, pt in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)
        for p, pt in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            pt.data.copy_(tau * p.data + (1.0 - tau) * pt.data)
