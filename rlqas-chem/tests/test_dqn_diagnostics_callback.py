# rlqas-chem/tests/test_dqn_diagnostics_callback.py
"""Tests for DQNDiagnosticsCallback."""

import json
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from rlqas_chem.rl.dqn_diagnostics_callback import DQNDiagnosticsCallback


class FakeModel:
    """Minimal SB3-like model mock."""
    def __init__(self):
        self.logger = MagicMock()
        self.logger.name_to_value = {
            "train/loss": 0.5,
            "train/exploration_rate": 0.8,
        }
        self.num_timesteps = 0


class FakeEnv:
    """Minimal env mock with global_best_energy."""
    def __init__(self, energy):
        self.global_best_energy = energy

    @property
    def unwrapped(self):
        return self


class FakeTrainingEnv:
    """Minimal DummyVecEnv mock."""
    def __init__(self, energies):
        self.envs = [FakeEnv(e) for e in energies]


def test_callback_records_samples_on_step():
    """DQNDiagnosticsCallback records a sample every sample_freq steps."""
    cb = DQNDiagnosticsCallback(output_path="/tmp/test_dqn_diag.json",
                                 sample_freq=3, verbose=0)
    cb.model = FakeModel()
    cb.training_env = FakeTrainingEnv([-7.88])

    # First 2 steps: no sample yet
    cb.model.num_timesteps = 1
    cb._on_step()
    cb.model.num_timesteps = 2
    cb._on_step()
    assert len(cb.samples) == 0

    # Third step triggers sample
    cb.model.num_timesteps = 3
    cb._on_step()
    assert len(cb.samples) == 1
    assert cb.samples[0]["q_loss"] == pytest.approx(0.5)
    assert cb.samples[0]["exploration_rate"] == pytest.approx(0.8)
    assert cb.samples[0]["global_best_energy"] == pytest.approx(-7.88)


def test_summary_energy_trend_pass():
    """summary() returns energy_trend_pass=True when energy improves."""
    cb = DQNDiagnosticsCallback(output_path="/tmp/test_dqn_diag.json", verbose=0)
    cb.samples = [
        {"step": i, "q_loss": 1.0 - i * 0.05, "exploration_rate": 1.0 - i * 0.05,
         "global_best_energy": -7.87 - i * 0.001}
        for i in range(10)
    ]
    s = cb.summary()
    assert s["energy_trend_pass"] is True
    assert s["exploration_decay_pass"] is True
    assert s["q_loss_trend_pass"] is True


def test_summary_energy_trend_fail():
    """summary() returns energy_trend_pass=False when energy does not improve."""
    cb = DQNDiagnosticsCallback(output_path="/tmp/test_dqn_diag.json", verbose=0)
    cb.samples = [
        {"step": i, "q_loss": 0.5, "exploration_rate": 0.1,
         "global_best_energy": -7.87}
        for i in range(10)
    ]
    s = cb.summary()
    assert s["energy_trend_pass"] is False


def test_callback_stores_none_when_logger_keys_missing():
    """_on_step stores None for q_loss/exploration_rate when keys absent from logger."""
    cb = DQNDiagnosticsCallback(output_path="/tmp/test_dqn_none.json",
                                 sample_freq=1, verbose=0)
    cb.model = FakeModel()
    cb.model.logger.name_to_value = {}  # no keys
    cb.training_env = FakeTrainingEnv([-7.88])

    cb.model.num_timesteps = 1
    cb._on_step()

    assert len(cb.samples) == 1
    assert cb.samples[0]["q_loss"] is None
    assert cb.samples[0]["exploration_rate"] is None


def test_on_training_start_resets_samples():
    """_on_training_start clears samples from any previous training run."""
    cb = DQNDiagnosticsCallback(output_path="/tmp/test_dqn_reset.json", verbose=0)
    cb.samples = [{"step": 1, "q_loss": 0.5, "exploration_rate": 0.8,
                   "global_best_energy": -7.88}]
    cb._sample_count = 1

    cb._on_training_start()

    assert cb.samples == []
    assert cb._sample_count == 0


def test_checkpoint_freq_triggers_intermediate_save():
    """checkpoint_freq=1 causes _save after every sample."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        cb = DQNDiagnosticsCallback(output_path=path, sample_freq=1,
                                     checkpoint_freq=1, verbose=0)
        cb.model = FakeModel()
        cb.training_env = FakeTrainingEnv([-7.88])

        # Trigger one sample
        cb.model.num_timesteps = 1
        cb._on_step()

        # File should be written after the first sample (checkpoint_freq=1)
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
    finally:
        os.unlink(path)


def test_save_writes_json():
    """_save() writes samples to the output path as valid JSON."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        cb = DQNDiagnosticsCallback(output_path=path, verbose=0)
        cb.samples = [{"step": 1, "q_loss": 0.4, "exploration_rate": 0.9,
                       "global_best_energy": -7.88}]
        cb._save()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["q_loss"] == pytest.approx(0.4)
    finally:
        os.unlink(path)


def test_dqn_agent_learn_accepts_callback():
    """DQNAgent.learn() must accept callback=None without raising TypeError."""
    from rlqas_chem.rl.dqn_agent import DQNAgent
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    agent = DQNAgent(config={"learning_starts": 10, "buffer_size": 100,
                              "batch_size": 10, "verbose": 0}, env=env)
    # Must not raise TypeError about unexpected keyword argument 'callback'
    agent.learn(total_timesteps=50, callback=None)
    env.close()
