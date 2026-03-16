"""
Unit tests for Sequential Testing Framework.

These tests verify SequentialRLTester, ComparisonUtilities, and MetricsCollector
functionality.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)

from rlqas.phase2.sequential_tester import (
    SequentialRLTester,
    ComparisonUtilities,
    MetricsCollector,
    create_metrics_collector,
    compare_energy_convergence,
    compare_training_efficiency,
    generate_summary_report,
)
from rlqas.phase2.rl import AgentFactory


class SimpleTestEnv(gym.Env):
    """Simple discrete environment for testing."""

    def __init__(self, seed=None):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        return self.np_random.uniform(-1.0, 1.0, size=5).astype(np.float32), {}

    def step(self, action):
        self._step_count += 1
        reward = 1.0 - (action * 0.1)  # Simple reward
        obs = self.np_random.uniform(-1.0, 1.0, size=5).astype(np.float32)
        done = self._step_count >= 50
        truncated = False
        return obs, reward, done, truncated, {}


def make_env(seed=None):
    """Factory function for creating test environments."""
    return SimpleTestEnv(seed=seed)


class TestSequentialRLTester:
    """Tests for SequentialRLTester class."""

    def test_initialization(self):
        """Test SequentialRLTester initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = SequentialRLTester(output_dir=tmpdir, verbose=0)
            assert tester.output_dir == tmpdir
            assert tester.verbose == 0
            assert tester.results == {}
            assert os.path.exists(tmpdir)

    def test_run_single_agent_ppo(self):
        """Test running a single PPO agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = SequentialRLTester(output_dir=tmpdir, verbose=0)
            result = tester.run_single_agent(
                agent_type="ppo",
                env_fn=make_env,
                agent_name="test_ppo",
                config={"verbose": 0, "n_steps": 64},
                total_timesteps=128,
                seed=42,
            )
            assert result["agent_type"] == "ppo"
            assert result["agent_name"] == "test_ppo"
            assert "final_metrics" in result
            assert "train_time" in result

    def test_run_single_agent_dqn(self):
        """Test running a single DQN agent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = SequentialRLTester(output_dir=tmpdir, verbose=0)
            result = tester.run_single_agent(
                agent_type="dqn",
                env_fn=make_env,
                agent_name="test_dqn",
                config={"verbose": 0, "buffer_size": 100},
                total_timesteps=128,
                seed=42,
            )
            assert result["agent_type"] == "dqn"
            assert result["agent_name"] == "test_dqn"
            assert "final_metrics" in result

    def test_run_sequential_test(self):
        """Test running sequential tests for multiple agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = SequentialRLTester(output_dir=tmpdir, verbose=0)
            agent_configs = [
                {"agent_type": "ppo", "name": "ppo_agent", "config": {"verbose": 0, "n_steps": 64}},
                {"agent_type": "dqn", "name": "dqn_agent", "config": {"verbose": 0, "buffer_size": 100}},
            ]
            results = tester.run_sequential_test(
                agent_configs=agent_configs,
                env_fn=make_env,
                test_name="test_sequential",
                total_timesteps=128,
                n_seeds=1,
            )
            assert "ppo_agent" in results
            assert "dqn_agent" in results
            assert results["ppo_agent"]["agent_type"] == "ppo"
            assert results["dqn_agent"]["agent_type"] == "dqn"

    def test_get_results(self):
        """Test getting results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = SequentialRLTester(output_dir=tmpdir, verbose=0)
            tester.run_single_agent(
                agent_type="ppo",
                env_fn=make_env,
                agent_name="test_agent",
                config={"verbose": 0, "n_steps": 64},
                total_timesteps=64,
            )

            # Get all results
            all_results = tester.get_results()
            assert "test_agent" in all_results

            # Get specific result
            specific = tester.get_results("test_agent")
            assert specific is not None
            assert specific["agent_type"] == "ppo"

    def test_compare_results(self):
        """Test comparing results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = SequentialRLTester(output_dir=tmpdir, verbose=0)
            tester.run_single_agent(
                agent_type="ppo",
                env_fn=make_env,
                agent_name="ppo",
                config={"verbose": 0, "n_steps": 64},
                total_timesteps=64,
            )
            comparison = tester.compare_results()
            assert "n_agents" in comparison
            assert comparison["n_agents"] == 1
            assert "ppo" in comparison["agents"]

    def test_save_comparison_report(self):
        """Test saving comparison report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tester = SequentialRLTester(output_dir=tmpdir, verbose=0)
            tester.run_single_agent(
                agent_type="ppo",
                env_fn=make_env,
                agent_name="ppo",
                config={"verbose": 0, "n_steps": 64},
                total_timesteps=64,
            )
            report_path = tester.save_comparison_report()
            assert os.path.exists(report_path)


class TestComparisonUtilities:
    """Tests for ComparisonUtilities class."""

    def test_initialization(self):
        """Test ComparisonUtilities initialization."""
        results = {"agent1": {"final_metrics": {"reward": 1.0}}}
        utils = ComparisonUtilities(results)
        assert utils.results == results

    def test_get_training_times(self):
        """Test getting training times."""
        results = {
            "agent1": {"train_time": 10.0, "avg_train_time": 10.0},
            "agent2": {"train_time": 20.0},
        }
        utils = ComparisonUtilities(results)
        times = utils.get_training_times()
        assert "agent1" in times
        assert "agent2" in times

    def test_get_final_metrics(self):
        """Test getting final metrics."""
        results = {
            "agent1": {"final_metrics": {"reward": 1.0, "loss": 0.5}},
            "agent2": {"final_metrics": {"reward": 2.0}},
        }
        utils = ComparisonUtilities(results)
        metrics = utils.get_final_metrics()
        assert "agent1" in metrics
        assert "agent2" in metrics

    def test_rank_by_metric(self):
        """Test ranking agents by metric."""
        results = {
            "agent1": {"final_metrics": {"reward": 1.0}},
            "agent2": {"final_metrics": {"reward": 2.0}},
            "agent3": {"final_metrics": {"reward": 3.0}},
        }
        utils = ComparisonUtilities(results)
        rankings = utils.rank_by_metric("reward", ascending=False)
        assert rankings[0][0] == "agent3"  # Highest first
        assert rankings[-1][0] == "agent1"  # Lowest last

    def test_generate_comparison_table(self):
        """Test generating comparison table."""
        results = {
            "agent1": {
                "agent_type": "ppo",
                "total_timesteps": 1000,
                "train_time": 10.0,
            },
        }
        utils = ComparisonUtilities(results)
        table = utils.generate_comparison_table()
        assert "agent1" in table
        assert "ppo" in table

    def test_find_best_agent(self):
        """Test finding best agent."""
        results = {
            "agent1": {"final_metrics": {"reward": 1.0}},
            "agent2": {"final_metrics": {"reward": 2.0}},
        }
        utils = ComparisonUtilities(results)
        best = utils.find_best_agent("reward", ascending=False)
        assert best == "agent2"


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_initialization(self):
        """Test MetricsCollector initialization."""
        collector = MetricsCollector()
        assert collector.metrics == {}
        assert collector.history == []

    def test_create_factory(self):
        """Test factory function."""
        collector = create_metrics_collector()
        assert isinstance(collector, MetricsCollector)

    def test_record_step(self):
        """Test recording training steps."""
        collector = MetricsCollector()
        collector.record_step(episode=0, reward=1.0, energy=-1.5, excitation_count=2)
        assert len(collector.history) == 1
        assert collector.history[0]["episode"] == 0
        assert collector.history[0]["reward"] == 1.0
        assert collector.history[0]["energy"] == -1.5

    def test_record_final_metrics(self):
        """Test recording final metrics."""
        collector = MetricsCollector()
        collector.record_final_metrics(
            final_energy=-1.5,
            final_reward=10.0,
            total_episodes=100,
            excitation_operators_used=5,
        )
        assert collector.metrics["final_energy"] == -1.5
        assert collector.metrics["final_reward"] == 10.0
        assert collector.metrics["excitation_operators_used"] == 5

    def test_get_aggregate_metrics(self):
        """Test getting aggregate metrics."""
        collector = MetricsCollector()
        for i in range(10):
            collector.record_step(episode=i, reward=float(i))
        aggregate = collector.get_aggregate_metrics()
        assert "mean_reward" in aggregate
        assert "std_reward" in aggregate
        assert aggregate["total_steps"] == 10

    def test_check_chemical_accuracy(self):
        """Test chemical accuracy checking."""
        collector = MetricsCollector()
        collector.record_final_metrics(
            final_energy=-1.5001,
            final_reward=10.0,
            total_episodes=100,
        )
        result = collector.check_chemical_accuracy(target_energy=-1.5, threshold=1.6e-3)
        assert result["achieved"] is True
        assert result["energy_error"] < 1.6e-3

    def test_check_chemical_accuracy_failure(self):
        """Test chemical accuracy check when threshold not met."""
        collector = MetricsCollector()
        collector.record_final_metrics(
            final_energy=-1.49,
            final_reward=10.0,
            total_episodes=100,
        )
        result = collector.check_chemical_accuracy(target_energy=-1.5, threshold=1.6e-3)
        assert result["achieved"] is False
        assert result["energy_error"] > 1.6e-3

    def test_get_convergence_info(self):
        """Test convergence info."""
        collector = MetricsCollector()
        # Add stable rewards (converged)
        for i in range(100):
            collector.record_step(episode=i, reward=1.0)
        conv_info = collector.get_convergence_info()
        assert conv_info["converged"] == True

    def test_to_dict(self):
        """Test converting to dictionary."""
        collector = MetricsCollector()
        collector.record_step(episode=0, reward=1.0)
        collector.record_final_metrics(
            final_energy=-1.5,
            final_reward=10.0,
            total_episodes=100,
        )
        result = collector.to_dict()
        assert "metrics" in result
        assert "aggregate" in result
        assert "history_length" in result


class TestComparisonFunctions:
    """Tests for comparison utility functions."""

    def test_compare_energy_convergence(self):
        """Test energy convergence comparison."""
        results = {
            "agent1": {"final_metrics": {"final_energy": -1.5001}},
            "agent2": {"final_metrics": {"final_energy": -1.49}},
        }
        comparison = compare_energy_convergence(
            results, target_energy=-1.5, chemical_accuracy=1.6e-3
        )
        assert "agents" in comparison
        assert comparison["best_agent"] == "agent1"

    def test_compare_training_efficiency(self):
        """Test training efficiency comparison."""
        results = {
            "agent1": {"total_timesteps": 1000, "avg_train_time": 10.0},
            "agent2": {"total_timesteps": 1000, "avg_train_time": 20.0},
        }
        efficiency = compare_training_efficiency(results)
        assert efficiency["most_efficient"] == "agent1"

    def test_generate_summary_report(self):
        """Test summary report generation."""
        results = {
            "agent1": {
                "agent_type": "ppo",
                "total_timesteps": 1000,
                "train_time": 10.0,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "report.txt")
            report = generate_summary_report(
                results, test_name="Test", output_path=report_path
            )
            assert "Test" in report
            assert "agent1" in report
            assert os.path.exists(report_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
