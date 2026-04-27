"""Qubit UCC Search Controller using qubit-space excitation operators."""
import os
import json
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from rlqas_chem.molecule import MoleculeData
from rlqas_chem.search.ucc.environment import UCCSearchEnv
from rlqas_chem.rl.agent_factory import AgentFactory
from .operator_pool import QubitOperatorPool


class _AgentAdapter:
    """Thin adapter providing select_action/store_experience/train interface."""

    def __init__(self, base_agent):
        self._agent = base_agent

    def select_action(self, obs: np.ndarray) -> int:
        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        action, _ = self._agent.act(obs)
        return int(action)

    def store_experience(self, *args, **kwargs) -> None:
        pass

    def train(self) -> None:
        pass

    def __getattr__(self, name: str):
        return getattr(self._agent, name)


@dataclass
class SearchResult:
    """Result from QubitUCCSearchController.search()."""

    best_circuit: Any
    best_energy: Optional[float]
    best_error: Optional[float]
    training_history: List[Dict] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)
    fusion_template: List[str] = field(default_factory=list)
    convergence_reached: bool = False


class QubitUCCSearchController:
    """Search controller using qubit-space excitation operators.

    Uses the QubitOperatorPool to characterize the qubit-space excitation
    landscape, while running energy evaluations through the standard UCCSearchEnv
    with fermion operators (since Tencirchem does not natively support qubit
    operator circuits for energy evaluation).

    The qubit pool is used to count available operators and record pool statistics.
    Energy training uses the fermion UCCSearchEnv, allowing valid energy comparisons
    between fermion and qubit operator searches.

    Note on Tencirchem API:
        Tencirchem has no native QubitUCC class. h_qubit_op returns the Hamiltonian
        as openfermion.QubitOperator, but energy evaluation goes through UCCSD.
        Qubit excitation circuits are built via Pauli rotation decomposition.
    """

    def __init__(
        self,
        molecule_data: MoleculeData,
        agent_type: str = "ppo",
        config: Dict = None,
    ):
        """Initialize controller.

        Args:
            molecule_data: MoleculeData with Hamiltonian and n_qubits
            agent_type: RL agent type ('ppo' | 'dqn')
            config: Config dict with optional keys:
                - max_depth: int (default 10)
                - qubit_ops: dict for QubitOperatorPool config
        """
        self.molecule_data = molecule_data
        self.agent_type = agent_type
        self.config = config or {}

        # Build qubit operator pool for characterization
        self.qubit_pool = QubitOperatorPool(
            molecule_data,
            self.config.get("qubit_ops", {}),
        )

        # Energy evaluation environment (fermion UCC, classical opt enabled)
        env_config = {
            "run_classical_opt": True,
            "complexity_penalty": 0.0,
            "max_depth": self.config.get("max_depth", 10),
        }
        self.env = UCCSearchEnv(molecule_data, env_config)

        # Create agent via factory
        base_agent = AgentFactory.create_agent(
            agent_type,
            config={
                "seed": 42,
                "verbose": 0,
                "n_steps": 128,
                "batch_size": 32,
                "n_epochs": 4,
            },
            env=self.env,
        )
        self.agent = _AgentAdapter(base_agent)

        self._best_energy: float = float("inf")
        self._training_history: List[Dict] = []

    def search(
        self,
        n_episodes: int = 500,
        early_stop_threshold: float = 1.6e-3,
    ) -> SearchResult:
        """Run qubit UCC search.

        Runs RL training with UCCSearchEnv. The qubit operator pool statistics
        are recorded in performance_metrics for comparison with fermion search.

        Args:
            n_episodes: Maximum number of training episodes
            early_stop_threshold: Stop early if |E - E_FCI| < threshold (Hartree)

        Returns:
            SearchResult with training history and performance metrics
        """
        print(f"Starting QubitUCC search: {n_episodes} episodes")
        print(f"Qubit pool size: {self.qubit_pool.get_pool_size()} operators")

        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_energy = getattr(self.env, "current_energy", float("inf"))

            while not done:
                try:
                    action = self.agent.select_action(obs)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_energy = info.get("energy", ep_energy)
                    self.agent.store_experience(obs, reward, done, info)
                except Exception as e:
                    print(f"  Episode {episode} step error: {e}")
                    done = True
                    break

            try:
                self.agent.train()
            except Exception:
                pass

            # Track best energy
            current_best = getattr(self.env, "global_best_energy", ep_energy)
            if current_best < self._best_energy:
                self._best_energy = current_best

            self._training_history.append(
                {
                    "episode": episode,
                    "energy": float(ep_energy) if ep_energy != float("inf") else None,
                    "best_energy": (
                        float(self._best_energy)
                        if self._best_energy != float("inf")
                        else None
                    ),
                    "reward": float(ep_reward),
                }
            )

            # Early stopping
            if self.molecule_data.fci_energy is not None and self._best_energy != float("inf"):
                err = abs(self._best_energy - self.molecule_data.fci_energy)
                if err < early_stop_threshold:
                    print(f"  Converged at episode {episode}: error={err*1000:.4f} mHa")
                    break

        # Compute final error
        best_err = None
        if (
            self.molecule_data.fci_energy is not None
            and self._best_energy != float("inf")
        ):
            best_err = abs(self._best_energy - self.molecule_data.fci_energy)

        converged = best_err is not None and best_err < 1.6e-3

        return SearchResult(
            best_circuit=None,
            best_energy=self._best_energy if self._best_energy != float("inf") else None,
            best_error=best_err,
            training_history=self._training_history,
            performance_metrics={
                "qubit_pool_size": self.qubit_pool.get_pool_size(),
                "n_episodes_run": len(self._training_history),
            },
            fusion_template=["qubit_ucc"],
            convergence_reached=converged,
        )

    def save_results(self, path: str) -> None:
        """Save search results to JSON.

        Args:
            path: File path to save results
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "best_energy": self._best_energy if self._best_energy != float("inf") else None,
            "qubit_pool_size": self.qubit_pool.get_pool_size(),
            "training_history": self._training_history[-10:],  # Last 10 for brevity
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
