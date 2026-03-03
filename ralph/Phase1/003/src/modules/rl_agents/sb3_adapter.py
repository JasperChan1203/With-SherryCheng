"""
SB3 adapter utilities for RLQAS.

This module provides adapter functions and utilities for integrating
Stable-Baselines3 with the RLQAS framework.
"""

import gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from typing import Callable, Optional, Union, List


def create_vectorized_env(
    env: Union[gym.Env, Callable[[], gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    monitor_dir: Optional[str] = None,
    wrapper_class: Optional[Callable] = None,
) -> VecEnv:
    """Create vectorized environment for SB3.

    Args:
        env: Gym environment or callable that creates a Gym environment.
        n_envs: Number of parallel environments.
        seed: Random seed for environment initialization.
        monitor_dir: Directory for monitoring (if None, no monitoring).
        wrapper_class: Optional wrapper class for environments.

    Returns:
        Vectorized environment compatible with SB3.
    """
    if isinstance(env, VecEnv):
        return env

    if callable(env):
        # env is a function that creates environments
        return make_vec_env(
            env,
            n_envs=n_envs,
            seed=seed,
            monitor_dir=monitor_dir,
            wrapper_class=wrapper_class,
        )
    else:
        # Single environment, wrap in DummyVecEnv
        return DummyVecEnv([lambda: env])


def set_sb3_seed(seed: int):
    """Set random seeds for SB3 and related libraries.

    Note: SB3 uses PyTorch, NumPy, and Python random. This function
    sets seeds for all of them to ensure reproducibility.

    Args:
        seed: Random seed.
    """
    import random
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_sb3_policy_class(policy_type: str):
    """Get SB3 policy class from policy type string.

    Args:
        policy_type: Policy type string ("MlpPolicy", "CnnPolicy", etc.)

    Returns:
        SB3 policy class.

    Raises:
        ValueError: If policy_type is not recognized.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import (
        ActorCriticPolicy,
        ActorCriticCnnPolicy,
        MultiInputActorCriticPolicy,
    )

    policy_map = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    if policy_type not in policy_map:
        raise ValueError(
            f"Unknown policy type: {policy_type}. "
            f"Available options: {list(policy_map.keys())}"
        )

    return policy_map[policy_type]


def load_sb3_model(path: str, **kwargs):
    """Load SB3 model from disk with proper error handling.

    Args:
        path: Path to saved model (.zip file).
        **kwargs: Additional arguments to pass to load().

    Returns:
        Loaded SB3 model.

    Raises:
        FileNotFoundError: If model file doesn't exist.
        ValueError: If model cannot be loaded.
    """
    import os
    from stable_baselines3 import PPO

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        model = PPO.load(path, **kwargs)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load SB3 model from {path}: {e}")