"""Double-DQN: Double Deep Q-Network with replay buffer."""
import os
import copy
import collections
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, List

from rlqas_chem.rl.base_agent import RLAgent


class _QNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleDQNAgent(RLAgent):
    """Double Deep Q-Network with experience replay.

    Decouples action selection (online network) from value evaluation
    (target network) to reduce overestimation bias. The replay buffer
    reuses past transitions for sample-efficient learning.

    Supports two instantiation conventions:
    - DoubleDQNAgent(obs_space, act_space, config)  — algorithm test style
    - DoubleDQNAgent(config=dict, env=env)          — AgentFactory style
    """

    def __init__(self, observation_space_or_config=None, action_space=None,
                 config=None, env=None):
        super().__init__()

        if action_space is not None:
            # Convention 1: DoubleDQNAgent(obs_space, act_space, config_dict)
            obs_space = observation_space_or_config
            act_space = action_space
            cfg = config if config is not None else {}
            self.env = None
        else:
            # Convention 2: DoubleDQNAgent(config=dict, env=env)
            cfg = (observation_space_or_config
                   if isinstance(observation_space_or_config, dict)
                   else (config or {}))
            self.env = env
            obs_space = env.observation_space if env is not None else None
            act_space = env.action_space if env is not None else None

        self.lr = cfg.get("lr", 1e-3)
        self.gamma = cfg.get("gamma", 0.99)
        self.epsilon = cfg.get("epsilon_start", 1.0)
        self.epsilon_end = cfg.get("epsilon_end", 0.05)
        self.epsilon_decay = cfg.get("epsilon_decay", 0.995)
        self.target_update_freq = cfg.get("target_update_freq", 100)
        self.batch_size = cfg.get("batch_size", 32)
        self._buffer_capacity = cfg.get("buffer_capacity", 10000)
        buffer_capacity = self._buffer_capacity

        self._obs_dim = None
        self._action_dim = None
        self.q_net = None
        self.target_q_net = None
        self.optimizer = None
        self.replay_buffer: collections.deque = collections.deque(
            maxlen=buffer_capacity
        )
        self.update_count = 0

        if obs_space is not None and act_space is not None:
            self._init_networks(obs_space.shape[0], act_space.n)

    def _init_networks(self, obs_dim: int, action_dim: int):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.q_net = _QNet(obs_dim, action_dim)
        self.target_q_net = copy.deepcopy(self.q_net)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

    def _ensure_initialized(self, env=None):
        if self.q_net is None:
            ref_env = env or self.env
            if ref_env is None:
                raise RuntimeError(
                    "DoubleDQNAgent: must provide env or initialize with obs/action spaces"
                )
            self._init_networks(
                ref_env.observation_space.shape[0],
                ref_env.action_space.n,
            )

    def act(self, obs: np.ndarray) -> Tuple[int, Dict]:
        """Deterministic (argmax) action selection.

        Returns:
            (action, {'epsilon': current_epsilon, 'q_values': q_values_list})
        """
        self._ensure_initialized()
        obs_t = torch.FloatTensor(np.array(obs)).flatten().unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        action = int(torch.argmax(q_values[0]).item())
        return action, {"epsilon": self.epsilon, "q_values": q_values[0].tolist()}

    def store(self, state, action: int, reward: float, next_state,
              done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.append((
            np.array(state, dtype=np.float32).flatten(),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32).flatten(),
            bool(done),
        ))

    def update_step(self) -> Optional[float]:
        """Perform one Double-DQN gradient step if buffer has enough samples."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions))
        rewards_t = torch.FloatTensor(np.array(rewards))
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(np.array(dones, dtype=np.float32))

        # Double-DQN: online net selects action, target net evaluates value
        with torch.no_grad():
            next_actions = torch.argmax(self.q_net(next_states_t), dim=1)
            next_q_vals = self.target_q_net(next_states_t)
            next_q = next_q_vals.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Periodically sync target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())

    def learn(self, experience=None, total_timesteps: int = 10000,
              callback=None, env=None) -> Dict:
        """Train step-by-step in environment using epsilon-greedy exploration.

        Args:
            total_timesteps: Number of environment steps to run.
            env: Environment to train in. Uses self.env if None.

        Raises:
            ValueError: If no environment is available.
        """
        if env is None:
            env = self.env
        if env is None:
            raise ValueError(
                "DoubleDQNAgent.learn() requires env — pass env to __init__ "
                "or provide it as argument."
            )

        self._ensure_initialized(env)
        obs, _ = env.reset()

        for step in range(total_timesteps):
            # Epsilon-greedy action selection
            if np.random.random() < self.epsilon:
                action = int(env.action_space.sample())
            else:
                action, _ = self.act(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.store(obs, action, reward, next_obs, done)
            self.update_step()

            obs = next_obs
            if done:
                obs, _ = env.reset()

            # Support SB3-style callbacks (optional)
            if callback is not None and hasattr(callback, "_on_step"):
                if not callback._on_step():
                    break

        return {"total_timesteps": total_timesteps, "epsilon": self.epsilon}

    def save(self, path: str) -> None:
        if self.q_net is not None:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(
                {
                    "q_net": self.q_net.state_dict(),
                    "target_q_net": self.target_q_net.state_dict(),
                    "config": {
                        "obs_dim": self._obs_dim,
                        "action_dim": self._action_dim,
                        "lr": self.lr,
                        "gamma": self.gamma,
                        "epsilon": self.epsilon,
                        "epsilon_end": self.epsilon_end,
                        "epsilon_decay": self.epsilon_decay,
                        "target_update_freq": self.target_update_freq,
                        "batch_size": self.batch_size,
                        "buffer_capacity": self._buffer_capacity,
                    },
                },
                path,
            )

    def load(self, path: str) -> None:
        data = torch.load(path, weights_only=False)
        cfg = data["config"]
        self._init_networks(cfg["obs_dim"], cfg["action_dim"])
        self.q_net.load_state_dict(data["q_net"])
        self.target_q_net.load_state_dict(data["target_q_net"])
        self.epsilon = cfg.get("epsilon", self.epsilon_end)
