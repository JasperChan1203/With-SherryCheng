"""
A2CAgent: Advantage Actor-Critic agent using Stable-Baselines3.

A2C is an on-policy actor-critic algorithm. Unlike PPO it does not clip
the policy update, making it faster per-update but potentially less stable.
A2C supports discrete action spaces and is suitable for quantum architecture
search where the agent selects excitation operators from a fixed set.
"""

import os
from typing import Dict, Tuple, Optional, Any
import numpy as np
import torch
from stable_baselines3 import A2C as SB3_A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym

from rlqas_chem.rl.base_agent import RLAgent


def get_device(use_gpu: bool = True) -> str:
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class A2CConfig:
    DEFAULT_CONFIG = {
        "learning_rate": 7e-4,
        "n_steps": 128,          # steps collected per update
        "gamma": 0.99,
        "gae_lambda": 1.0,       # 1.0 = MC returns (no GAE), as per original A2C
        "ent_coef": 0.05,        # entropy coefficient for exploration
        "vf_coef": 0.5,          # value function loss coefficient
        "max_grad_norm": 0.5,
        "policy_type": "MlpPolicy",
        "verbose": 0,
        "seed": 42,
        "use_gpu": True,
        "tensorboard_log": None,
        "n_envs": 1,
        "monitor_dir": None,
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)


class A2CAgent(RLAgent):
    """A2C agent for quantum architecture search.

    Wraps Stable-Baselines3 A2C and conforms to the RLAgent abstract interface.

    Args:
        config: Agent configuration dictionary.
        env: Gymnasium environment (UCCSearchEnv or compatible).
    """

    def __init__(self, config: Optional[Dict] = None, env=None):
        self._config = A2CConfig(config)
        self.config = self._config.config
        self.env = env
        self.model = None

        if env is not None:
            self._init_model(env)

    def _init_model(self, env) -> None:
        device = get_device(self.config.get("use_gpu", True))
        self.model = SB3_A2C(
            self.config.get("policy_type", "MlpPolicy"),
            env,
            learning_rate=self.config.get("learning_rate", 7e-4),
            n_steps=self.config.get("n_steps", 128),
            gamma=self.config.get("gamma", 0.99),
            gae_lambda=self.config.get("gae_lambda", 1.0),
            ent_coef=self.config.get("ent_coef", 0.05),
            vf_coef=self.config.get("vf_coef", 0.5),
            max_grad_norm=self.config.get("max_grad_norm", 0.5),
            verbose=self.config.get("verbose", 0),
            seed=self.config.get("seed", 42),
            device=device,
            tensorboard_log=self.config.get("tensorboard_log"),
        )

    def act(self, state: np.ndarray) -> Tuple[int, Dict]:
        if self.model is None:
            raise RuntimeError("A2CAgent not initialized with an environment.")
        action, _ = self.model.predict(state, deterministic=False)
        return int(action), {}

    def learn(
        self,
        experience: Optional[Dict] = None,
        total_timesteps: int = 10000,
    ) -> Dict:
        if self.model is None:
            raise RuntimeError("A2CAgent not initialized with an environment.")
        self.model.learn(total_timesteps=total_timesteps)
        return {"total_timesteps": total_timesteps, "algorithm": "a2c"}

    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("A2CAgent not initialized.")
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)

    def load(self, path: str) -> None:
        device = get_device(self.config.get("use_gpu", True))
        self.model = SB3_A2C.load(path, device=device)

    def get_config(self) -> Dict:
        return dict(self.config)
