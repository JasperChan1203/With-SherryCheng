"""
Experiment Management System for RLQAS Phase 2.

This module implements the ExperimentManager class for managing
configuration-driven experiments with result storage and batch execution.
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from rlqas.phase2.sequential_tester import SequentialRLTester
from rlqas.phase2.hea_search import HEASearchController, HEAConfig


class ExperimentManager:
    """Manager for configuration-driven experiments.

    This class provides experiment lifecycle management including:
    - Loading and validating YAML/JSON configurations
    - Executing experiments from configuration files
    - Result collection and storage
    - Batch experiment execution
    - Standardized logging and checkpointing

    Args:
        output_dir: Directory for storing experiment results
        log_level: Logging level (default: INFO)
    """

    def __init__(self, output_dir: str = "results/experiments", log_level: int = logging.INFO):
        """Initialize experiment manager."""
        self.output_dir = output_dir
        self.experiments: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}
        self._current_experiment: Optional[str] = None

        # Set up logging
        self.logger = logging.getLogger("ExperimentManager")
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load experiment configuration from file.

        Supports both YAML and JSON formats.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        ext = os.path.splitext(config_path)[1].lower()

        with open(config_path, "r") as f:
            if ext in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif ext == ".json":
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration format: {ext}")

        # Validate configuration
        self._validate_config(config)

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate experiment configuration.

        Args:
            config: Configuration dictionary

        Raises:
            ValueError: If configuration is invalid
        """
        required_fields = ["name", "type"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required configuration field: {field}")

        valid_types = ["sequential_test", "hea_search", "hybrid_search", "custom"]
        if config["type"] not in valid_types:
            raise ValueError(
                f"Invalid experiment type: {config['type']}. "
                f"Must be one of {valid_types}"
            )

    def create_experiment(
        self,
        name: str,
        experiment_type: str,
        config: Dict[str, Any],
    ) -> str:
        """Create a new experiment.

        Args:
            name: Experiment name
            experiment_type: Type of experiment
            config: Experiment configuration

        Returns:
            Experiment ID
        """
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiments[experiment_id] = {
            "name": name,
            "type": experiment_type,
            "config": config,
            "created_at": datetime.now().isoformat(),
            "status": "created",
        }

        self.logger.info(f"Created experiment: {experiment_id}")
        return experiment_id

    def run_experiment(
        self,
        experiment_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run an experiment.

        Args:
            experiment_id: ID of existing experiment to run
            config: Configuration for new experiment (if experiment_id not provided)

        Returns:
            Experiment results dictionary

        Raises:
            ValueError: If neither experiment_id nor config is provided
            RuntimeError: If experiment type is not supported
        """
        if experiment_id is None and config is None:
            raise ValueError("Must provide either experiment_id or config")

        # Load or create experiment
        if experiment_id:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment not found: {experiment_id}")
            experiment = self.experiments[experiment_id]
            config = experiment["config"]
        else:
            name = config.get("name", "experiment")
            experiment_type = config.get("type", "custom")
            experiment_id = self.create_experiment(name, experiment_type, config)
            experiment = self.experiments[experiment_id]

        self._current_experiment = experiment_id
        experiment["status"] = "running"
        experiment["started_at"] = datetime.now().isoformat()

        self.logger.info(f"Running experiment: {experiment_id} ({experiment['type']})")

        # Execute based on type
        if experiment["type"] == "sequential_test":
            results = self._run_sequential_test(config)
        elif experiment["type"] == "hea_search":
            results = self._run_hea_search(config)
        elif experiment["type"] == "hybrid_search":
            results = self._run_hybrid_search(config)
        elif experiment["type"] == "custom":
            results = self._run_custom(config)
        else:
            raise RuntimeError(f"Unsupported experiment type: {experiment['type']}")

        # Store results
        self.results[experiment_id] = results
        experiment["status"] = "completed"
        experiment["completed_at"] = datetime.now().isoformat()
        experiment["results"] = results

        # Save results
        self._save_results(experiment_id, results)

        self.logger.info(f"Experiment completed: {experiment_id}")
        return results

    def _run_sequential_test(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run sequential test experiment.

        Args:
            config: Experiment configuration

        Returns:
            Results dictionary
        """
        from rlqas.phase2.sequential_tester import SequentialRLTester

        # Extract configuration
        agent_configs = config.get("agents", [])
        env_config = config.get("environment", {})
        training_config = config.get("training", {})

        # Create environment factory
        def make_env(seed=None):
            return self._create_environment(env_config, seed)

        # Create tester
        output_dir = config.get("output_dir", os.path.join(self.output_dir, "sequential"))
        tester = SequentialRLTester(output_dir=output_dir, verbose=config.get("verbose", 1))

        # Run test
        test_name = config.get("name", "sequential_test")
        results = tester.run_sequential_test(
            agent_configs=agent_configs,
            env_fn=make_env,
            test_name=test_name,
            total_timesteps=training_config.get("total_timesteps", 10000),
            n_seeds=training_config.get("n_seeds", 3),
        )

        return {
            "type": "sequential_test",
            "test_name": test_name,
            "results": results,
            "comparison": tester.compare_results(),
        }

    def _run_hea_search(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run HEA search experiment.

        Args:
            config: Experiment configuration

        Returns:
            Results dictionary
        """
        from rlqas.phase2.hea_search import HEASearchController, HEAConfig

        # Extract configuration
        hea_config = config.get("hea", {})
        agent_config = config.get("agent", {})
        training_config = config.get("training", {})

        # Create HEA config
        hea_cfg = HEAConfig(
            n_qubits=hea_config.get("n_qubits", 4),
            max_layers=hea_config.get("max_layers", 4),
            entanglement_patterns=hea_config.get("entanglement_patterns", ["linear", "circular"]),
            rotation_gates=hea_config.get("rotation_gates", ["rx", "ry", "rz"]),
            parameter_sharing=hea_config.get("parameter_sharing", "layer_wise"),
        )

        # Create controller
        output_dir = config.get("output_dir", os.path.join(self.output_dir, "hea"))
        controller = HEASearchController(
            n_qubits=hea_cfg.n_qubits,
            max_layers=hea_cfg.max_layers,
            entanglement_patterns=hea_cfg.entanglement_patterns,
            rotation_gates=hea_cfg.rotation_gates,
            output_dir=output_dir,
            verbose=config.get("verbose", 1),
        )

        # Run search
        results = controller.search(
            agent_type=agent_config.get("type", "ppo"),
            agent_config=agent_config.get("config", {}),
            total_timesteps=training_config.get("total_timesteps", 10000),
            target_energy=hea_config.get("target_energy"),
        )

        return {
            "type": "hea_search",
            "config": hea_cfg.to_dict(),
            "results": results,
        }

    def _run_hybrid_search(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run hybrid HEA+UCC architecture search experiment (Phase 3).

        Args:
            config: Experiment configuration dict.  Expected sub-sections:
                molecule, search, rl, simulation, evaluation, output.

        Returns:
            Results dictionary with best_energy, best_error, fusion_template, etc.
        """
        from rlqas.phase1.molecule.processor import process_molecule

        mol_cfg = config.get("molecule", {})
        formula = mol_cfg.get("formula", "H2")
        bond_length = mol_cfg.get("bond_length", 0.74)
        active_space = mol_cfg.get("active_space")
        if active_space is not None:
            active_space = tuple(active_space)
        basis_set = mol_cfg.get("basis_set", "sto-3g")
        transform = mol_cfg.get("transform", "jordan_wigner")

        molecule_data = process_molecule(
            formula,
            bond_length,
            "UCC",
            active_space=active_space,
            basis_set=basis_set,
            transform=transform,
        )

        from rlqas.phase3.hybrid_search.controller import HybridSearchController

        controller = HybridSearchController.from_config(molecule_data, config)

        rl_cfg = config.get("rl", {})
        n_episodes = rl_cfg.get("n_episodes", 500)
        result = controller.search(n_episodes=n_episodes)

        return {
            "type": "hybrid_search",
            "best_energy": float(result.best_energy) if result.best_energy is not None else None,
            "best_error": float(result.best_error) if result.best_error is not None else None,
            "fusion_template": list(result.fusion_template),
            "convergence_reached": bool(result.convergence_reached),
            "n_episodes": len(result.training_history),
        }

    def _run_custom(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run custom experiment.

        Args:
            config: Experiment configuration

        Returns:
            Results dictionary
        """
        # For custom experiments, just return config as results
        return {
            "type": "custom",
            "config": config,
            "message": "Custom experiment executed",
        }

    def _create_environment(self, env_config: Dict, seed: Optional[int] = None):
        """Create environment from configuration.

        Args:
            env_config: Environment configuration
            seed: Random seed

        Returns:
            Environment instance
        """
        # Default to simple test environment
        import gymnasium as gym
        from gymnasium import spaces
        import numpy as np

        class SimpleEnv(gym.Env):
            def __init__(self, seed=None):
                super().__init__()
                self.observation_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(5,), dtype=np.float32
                )
                self.action_space = spaces.Discrete(3)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return self.np_random.uniform(-1.0, 1.0, size=5).astype(np.float32), {}

            def step(self, action):
                reward = 1.0 - (action * 0.1)
                obs = self.np_random.uniform(-1.0, 1.0, size=5).astype(np.float32)
                done = False
                truncated = False
                return obs, reward, done, truncated, {}

        return SimpleEnv(seed=seed)

    def run_batch(
        self,
        configs: List[Dict[str, Any]],
        parallel: bool = False,
    ) -> Dict[str, Dict]:
        """Run multiple experiments in batch.

        Args:
            configs: List of experiment configurations
            parallel: Whether to run in parallel (not yet implemented)

        Returns:
            Dictionary mapping experiment IDs to results
        """
        batch_results = {}

        self.logger.info(f"Starting batch execution of {len(configs)} experiments")

        for i, config in enumerate(configs):
            self.logger.info(f"Running experiment {i+1}/{len(configs)}: {config.get('name', 'unnamed')}")
            results = self.run_experiment(config=config)
            experiment_id = list(self.experiments.keys())[-1]
            batch_results[experiment_id] = results

        self.logger.info(f"Batch execution completed: {len(batch_results)} experiments")
        return batch_results

    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get status of an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Status dictionary
        """
        if experiment_id not in self.experiments:
            return {"error": f"Experiment not found: {experiment_id}"}

        exp = self.experiments[experiment_id]
        return {
            "id": experiment_id,
            "name": exp["name"],
            "type": exp["type"],
            "status": exp["status"],
            "created_at": exp.get("created_at"),
            "started_at": exp.get("started_at"),
            "completed_at": exp.get("completed_at"),
        }

    def get_results(self, experiment_id: Optional[str] = None) -> Union[Dict, Optional[Dict]]:
        """Get experiment results.

        Args:
            experiment_id: Optional specific experiment ID

        Returns:
            Results dictionary or all results
        """
        if experiment_id:
            return self.results.get(experiment_id)
        return self.results

    def _save_results(self, experiment_id: str, results: Dict):
        """Save experiment results to disk.

        Args:
            experiment_id: Experiment ID
            results: Results dictionary
        """
        results_path = os.path.join(
            self.output_dir, f"{experiment_id}_results.json"
        )

        def convert_to_serializable(obj):
            import numpy as np
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            return obj

        serializable_results = convert_to_serializable(results)

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.debug(f"Results saved to {results_path}")

    def save_experiment_config(self, experiment_id: str) -> str:
        """Save experiment configuration to disk.

        Args:
            experiment_id: Experiment ID

        Returns:
            Path to saved configuration
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")

        config_path = os.path.join(
            self.output_dir, f"{experiment_id}_config.yaml"
        )

        experiment = self.experiments[experiment_id]
        config_data = {
            "experiment_id": experiment_id,
            "name": experiment["name"],
            "type": experiment["type"],
            "config": experiment["config"],
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        return config_path
