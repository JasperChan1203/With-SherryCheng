"""
Unit tests for Experiment Management System.

These tests verify ExperimentManager, ConfigLoader, and ResultsDatabase
functionality.
"""

import os
import sys
import tempfile
import pytest
import json

# Add paths for imports
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_DIR)
PHASE1_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, '..', '..', 'Phase1', '006', 'src'))
PHASE2_SRC = os.path.abspath(os.path.join(PROJECT_ROOT, 'src'))

sys.path.insert(0, PHASE1_SRC)
sys.path.insert(0, PHASE2_SRC)

from rlqas.phase2.experiment import (
    ExperimentManager,
    ConfigLoader,
    ExperimentConfig,
    load_config,
    save_config,
    create_template_config,
    ResultsDatabase,
)


class TestConfigLoader:
    """Tests for ConfigLoader class."""

    def test_initialization(self):
        """Test ConfigLoader initialization."""
        loader = ConfigLoader()
        assert loader._schema is not None
        assert "required" in loader._schema

    def test_validate_valid_config(self):
        """Test validating a valid configuration."""
        loader = ConfigLoader()
        config = {
            "name": "test_experiment",
            "type": "sequential_test",
            "verbose": 1,
        }
        errors = loader.validate(config)
        assert len(errors) == 0

    def test_validate_missing_required(self):
        """Test validation catches missing required fields."""
        loader = ConfigLoader()
        config = {"type": "sequential_test"}  # Missing name
        errors = loader.validate(config)
        assert any("name" in e for e in errors)

    def test_validate_invalid_type(self):
        """Test validation catches invalid experiment type."""
        # Should raise in __post_init__
        with pytest.raises(ValueError):
            ExperimentConfig(name="test", type="invalid_type")

    def test_parse_config(self):
        """Test parsing configuration."""
        loader = ConfigLoader()
        config = {
            "name": "test_exp",
            "type": "sequential_test",
            "description": "Test description",
            "verbose": 2,
        }
        parsed = loader.parse(config)
        assert parsed.name == "test_exp"
        assert parsed.type == "sequential_test"
        assert parsed.description == "Test description"
        assert parsed.verbose == 2

    def test_save_and_load_json(self):
        """Test saving and loading JSON configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config = {"name": "test", "type": "sequential_test"}

            save_config(config, config_path)
            loaded = load_config(config_path)

            assert loaded["name"] == "test"
            assert loaded["type"] == "sequential_test"

    def test_save_and_load_yaml(self):
        """Test saving and loading YAML configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.yaml")
            config = {"name": "test", "type": "hea_search"}

            save_config(config, config_path)
            loaded = load_config(config_path)

            assert loaded["name"] == "test"
            assert loaded["type"] == "hea_search"

    def test_merge_configs(self):
        """Test merging configurations."""
        loader = ConfigLoader()
        base = {"name": "base", "training": {"timesteps": 1000}}
        override = {"verbose": 2, "training": {"n_seeds": 5}}

        merged = loader.merge_configs(base, override)
        assert merged["name"] == "base"
        assert merged["verbose"] == 2
        assert merged["training"]["timesteps"] == 1000
        assert merged["training"]["n_seeds"] == 5

    def test_create_template_sequential(self):
        """Test creating sequential test template."""
        template = create_template_config(experiment_type="sequential_test")
        assert template["type"] == "sequential_test"
        assert "agents" in template
        assert len(template["agents"]) == 2

    def test_create_template_hea(self):
        """Test creating HEA search template."""
        template = create_template_config(experiment_type="hea_search")
        assert template["type"] == "hea_search"
        assert "hea" in template
        assert "n_qubits" in template["hea"]

    def test_create_template_custom(self):
        """Test creating custom template."""
        template = create_template_config(experiment_type="custom")
        assert template["type"] == "custom"


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ExperimentConfig(name="test", type="sequential_test")
        assert config.description == ""
        assert config.agents == []
        assert config.output_dir == "results/experiments"
        assert config.verbose == 1

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ExperimentConfig(
            name="custom_test",
            type="hea_search",
            description="My test",
            output_dir="custom_output",
            verbose=2,
        )
        assert config.description == "My test"
        assert config.output_dir == "custom_output"
        assert config.verbose == 2


class TestExperimentManager:
    """Tests for ExperimentManager class."""

    def test_initialization(self):
        """Test ExperimentManager initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir)
            assert manager.output_dir == tmpdir
            assert os.path.exists(tmpdir)

    def test_load_config_json(self):
        """Test loading JSON configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir)

            config_path = os.path.join(tmpdir, "config.json")
            config = {"name": "test", "type": "sequential_test"}
            with open(config_path, "w") as f:
                json.dump(config, f)

            loaded = manager.load_config(config_path)
            assert loaded["name"] == "test"
            assert loaded["type"] == "sequential_test"

    def test_load_config_yaml(self):
        """Test loading YAML configuration."""
        try:
            import yaml
            with tempfile.TemporaryDirectory() as tmpdir:
                manager = ExperimentManager(output_dir=tmpdir)

                config_path = os.path.join(tmpdir, "config.yaml")
                with open(config_path, "w") as f:
                    f.write("name: test\n")
                    f.write("type: hea_search\n")

                loaded = manager.load_config(config_path)
                assert loaded["name"] == "test"
                assert loaded["type"] == "hea_search"
        except ImportError:
            pytest.skip("PyYAML not installed")

    def test_load_config_not_found(self):
        """Test loading non-existent configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir)
            with pytest.raises(FileNotFoundError):
                manager.load_config("/nonexistent/path/config.json")

    def test_create_experiment(self):
        """Test creating an experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir)
            config = {"name": "test", "type": "sequential_test"}

            exp_id = manager.create_experiment(
                name="test_exp",
                experiment_type="sequential_test",
                config=config,
            )

            assert exp_id.startswith("test_exp_")
            assert exp_id in manager.experiments

    def test_run_experiment_sequential(self):
        """Test running sequential test experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir, log_level=40)  # ERROR level

            config = {
                "name": "test_seq",
                "type": "sequential_test",
                "agents": [
                    {
                        "agent_type": "dqn",
                        "name": "dqn_test",
                        "config": {"verbose": 0, "buffer_size": 100},
                    }
                ],
                "training": {"total_timesteps": 50},
            }

            results = manager.run_experiment(config=config)
            assert results["type"] == "sequential_test"

    def test_run_experiment_hea(self):
        """Test running HEA search experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir, log_level=40)

            config = {
                "name": "test_hea",
                "type": "hea_search",
                "hea": {
                    "n_qubits": 4,
                    "max_layers": 2,
                },
                "agent": {
                    "type": "dqn",
                    "config": {"verbose": 0, "buffer_size": 100},
                },
                "training": {"total_timesteps": 50},
            }

            results = manager.run_experiment(config=config)
            assert results["type"] == "hea_search"

    def test_run_experiment_custom(self):
        """Test running custom experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir)

            config = {
                "name": "test_custom",
                "type": "custom",
                "custom_field": "custom_value",
            }

            results = manager.run_experiment(config=config)
            assert results["type"] == "custom"

    def test_get_experiment_status(self):
        """Test getting experiment status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir)

            exp_id = manager.create_experiment(
                name="test",
                experiment_type="sequential_test",
                config={},
            )

            status = manager.get_experiment_status(exp_id)
            assert status["id"] == exp_id
            assert status["status"] == "created"

    def test_run_batch(self):
        """Test running batch experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir, log_level=40)

            configs = [
                {
                    "name": f"batch_{i}",
                    "type": "custom",
                }
                for i in range(3)
            ]

            results = manager.run_batch(configs)
            assert len(results) == 3

    def test_save_experiment_config(self):
        """Test saving experiment configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ExperimentManager(output_dir=tmpdir)

            exp_id = manager.create_experiment(
                name="test",
                experiment_type="sequential_test",
                config={"key": "value"},
            )

            config_path = manager.save_experiment_config(exp_id)
            assert os.path.exists(config_path)


class TestResultsDatabase:
    """Tests for ResultsDatabase class."""

    def test_initialization(self):
        """Test database initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)
            assert os.path.exists(db_path)

    def test_store_experiment(self):
        """Test storing experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            success = db.store_experiment(
                experiment_id="exp_001",
                name="test_exp",
                experiment_type="sequential_test",
                config={"key": "value"},
                status="created",
            )

            assert success is True

    def test_get_experiment(self):
        """Test retrieving experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            db.store_experiment(
                experiment_id="exp_001",
                name="test_exp",
                experiment_type="sequential_test",
                config={"key": "value"},
            )

            exp = db.get_experiment("exp_001")
            assert exp is not None
            assert exp["name"] == "test_exp"

    def test_get_experiments(self):
        """Test retrieving multiple experiments."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            db.store_experiment("exp_001", "test1", "sequential_test", {})
            db.store_experiment("exp_002", "test2", "hea_search", {})
            db.store_experiment("exp_003", "test3", "sequential_test", {})

            # Get all
            all_exps = db.get_experiments()
            assert len(all_exps) == 3

            # Filter by type
            seq_exps = db.get_experiments(experiment_type="sequential_test")
            assert len(seq_exps) == 2

    def test_update_status(self):
        """Test updating experiment status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            db.store_experiment("exp_001", "test", "sequential_test", {})

            db.update_experiment_status("exp_001", "running")
            exp = db.get_experiment("exp_001")
            assert exp["status"] == "running"

            db.update_experiment_status("exp_001", "completed", {"result": "success"})
            exp = db.get_experiment("exp_001")
            assert exp["status"] == "completed"

    def test_store_and_get_metrics(self):
        """Test storing and retrieving metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            db.store_experiment("exp_001", "test", "sequential_test", {})

            db.store_metric("exp_001", "final_energy", -1.5, "agent1")
            db.store_metric("exp_001", "final_energy", -1.4, "agent2")

            metrics = db.get_metrics("exp_001")
            assert len(metrics) == 2

    def test_get_comparison(self):
        """Test getting comparison metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            db.store_experiment("exp_001", "test1", "sequential_test", {})
            db.store_experiment("exp_002", "test2", "sequential_test", {})

            db.store_metric("exp_001", "reward", 10.0, "ppo")
            db.store_metric("exp_002", "reward", 8.0, "dqn")

            comparison = db.get_comparison("sequential_test", "reward")
            assert len(comparison) == 2
            assert comparison[0]["metric_value"] == 10.0

    def test_delete_experiment(self):
        """Test deleting experiment."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            db.store_experiment("exp_001", "test", "sequential_test", {})
            db.store_metric("exp_001", "reward", 10.0)

            db.delete_experiment("exp_001")

            exp = db.get_experiment("exp_001")
            assert exp is None

    def test_get_statistics(self):
        """Test getting database statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            db.store_experiment("exp_001", "test1", "sequential_test", {})
            db.store_experiment("exp_002", "test2", "hea_search", {})
            db.store_experiment("exp_003", "test3", "sequential_test", {})

            stats = db.get_statistics()
            assert stats["total_experiments"] == 3
            assert "sequential_test" in stats["by_type"]

    def test_export_results(self):
        """Test exporting results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            db = ResultsDatabase(db_path)

            db.store_experiment("exp_001", "test", "sequential_test", {"result": "data"})

            export_path = os.path.join(tmpdir, "export.json")
            db.export_results(output_path=export_path)

            assert os.path.exists(export_path)
            with open(export_path) as f:
                export_data = json.load(f)
            assert "experiments" in export_data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
