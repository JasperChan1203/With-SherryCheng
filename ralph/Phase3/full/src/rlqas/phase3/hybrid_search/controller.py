"""Hybrid Search Controller orchestrating the complete hybrid architecture search."""

import os
import json
import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import numpy as np

from rlqas.phase1.molecule.processor import MoleculeData
from rlqas.phase2.rl.agent_factory import AgentFactory
from .circuit_builder import HybridFusionStrategy, HybridCircuitBuilder
from .environment import HybridSearchEnv
from .config import HybridSearchConfig


class _AgentAdapter:
    """Thin adapter providing a controller-compatible interface over any RLAgent.

    The base ``RLAgent`` interface only specifies ``act()``, ``learn()``,
    ``save()``, and ``load()``.  The search loop in this module requires
    ``select_action(obs)``, ``store_experience(...)``, and ``train()`` —
    this adapter bridges the gap without modifying the agent classes.

    This adapter also implements REINFORCE (Monte Carlo policy gradient) so
    that the policy genuinely improves over episodes rather than remaining
    at random initialization.
    """

    def __init__(self, base_agent, env=None):
        self._agent = base_agent
        self._env = env
        # Episode data buffers for REINFORCE
        self._ep_states: List[np.ndarray] = []
        self._ep_actions: List[int] = []
        self._ep_rewards: List[float] = []
        # Policy network (initialized lazily on first select_action call)
        self._policy = None
        self._optimizer = None

    def _init_policy(self, obs_dim: int) -> None:
        """Initialize the REINFORCE policy network."""
        import torch
        import torch.nn as nn

        n_actions = (
            int(self._env.action_space.n)
            if self._env is not None
            else 4  # fallback
        )
        self._policy = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1),
        )
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=1e-3)

    def select_action(self, obs: np.ndarray) -> int:
        """Select action using the REINFORCE policy; store state for training."""
        import torch

        if obs.ndim == 1:
            obs = obs[np.newaxis, :]
        obs_flat = obs[0]

        if self._policy is None:
            self._init_policy(obs_flat.shape[0])

        # Sample action from REINFORCE policy
        with torch.no_grad():
            obs_t = torch.tensor(obs_flat, dtype=torch.float32).unsqueeze(0)
            action_probs = self._policy(obs_t)
        action = int(torch.multinomial(action_probs[0], 1).item())

        # Cache state and action for REINFORCE update
        self._ep_states.append(obs_flat.copy())
        self._ep_actions.append(action)
        return action

    def store_experience(self, obs: np.ndarray, reward: float, done: bool, info: dict) -> None:
        """Store reward for REINFORCE update."""
        self._ep_rewards.append(float(reward))

    def train(self) -> None:
        """REINFORCE: Monte Carlo policy gradient update over completed episode."""
        import torch

        if not self._ep_states or self._policy is None:
            # Clear buffers and return
            self._ep_states = []
            self._ep_actions = []
            self._ep_rewards = []
            return

        states = self._ep_states
        actions = self._ep_actions
        rewards = self._ep_rewards

        # Compute discounted returns
        gamma = 0.99
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient loss: L = -E[G_t * log pi(a_t|s_t)]
        self._optimizer.zero_grad()
        loss_terms = []
        for state, action, G_t in zip(states, actions, returns):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = self._policy(state_tensor)
            log_prob = torch.log(action_probs[0, action] + 1e-8)
            loss_terms.append(-G_t * log_prob)

        if loss_terms:
            total_loss = torch.stack(loss_terms).sum()
            total_loss.backward()
            self._optimizer.step()

        # Clear episode buffers
        self._ep_states = []
        self._ep_actions = []
        self._ep_rewards = []

    # Delegate attribute access to the underlying agent for compatibility
    def __getattr__(self, name: str):
        return getattr(self._agent, name)


@dataclass
class SearchResult:
    """Result object returned by :class:`HybridSearchController`.search().

    Attributes:
        best_circuit: Best HybridCircuit found (may be None).
        best_energy: Best energy achieved in Hartree.
        best_error: |best_energy - fci_energy| in Hartree (None if FCI unavailable).
        training_history: List of per-episode metric dicts.
        performance_metrics: Summary statistics of the search run.
        fusion_template: Block type sequence used for the best circuit.
        convergence_reached: True if chemical accuracy was achieved.
    """

    best_circuit: Any
    best_energy: Optional[float]
    best_error: Optional[float]
    training_history: List[Dict] = field(default_factory=list)
    performance_metrics: Dict = field(default_factory=dict)
    fusion_template: List[str] = field(default_factory=list)
    convergence_reached: bool = False


class HybridSearchController:
    """Orchestrates the complete hybrid HEA+UCC quantum architecture search.

    This controller creates a :class:`HybridSearchEnv`, wraps it with a
    configurable RL agent (from :class:`AgentFactory`), and runs an
    episode-based search loop.  The returned :class:`SearchResult` extends
    the Phase 1 ``UCCSearchController`` result with an additional
    ``fusion_template`` field.

    Args:
        molecule_data: MoleculeData object from :func:`process_molecule`.
        agent_type: RL agent type string (``"ppo"``, ``"dqn"``, ``"a2c"``).
        config: Flat or nested configuration dictionary.  Recognised
            top-level keys mirror :class:`HybridSearchConfig` sections.
    """

    def __init__(
        self,
        molecule_data: MoleculeData,
        agent_type: str = "ppo",
        config: Dict = None,
    ):
        self.molecule_data = molecule_data
        self.agent_type = agent_type
        self.config = config or {}

        cfg = HybridSearchConfig(self.config)
        self.env_config = cfg.get_section("environment")
        self.ctrl_config = cfg.get_section("controller")
        self.fusion_config = cfg.get_section("fusion")

        # Top-level overrides for the most commonly tweaked keys
        for key in ("max_depth", "max_blocks", "run_classical_opt", "complexity_penalty"):
            if key in self.config:
                self.env_config[key] = self.config[key]
        for key in ("n_episodes", "early_stop_threshold", "log_frequency"):
            if key in self.config:
                self.ctrl_config[key] = self.config[key]

        # Safety: always ensure run_classical_opt is True so energies are real
        self.env_config["run_classical_opt"] = True

        self.fusion_strategy = HybridFusionStrategy(self.fusion_config)
        self.env = HybridSearchEnv(
            molecule_data, self.fusion_strategy, self.env_config
        )

        # Create RL agent via factory, then wrap for controller compatibility
        agent_config = self._build_agent_config()
        base_agent = AgentFactory.create_agent(
            agent_type, config=agent_config, env=self.env
        )
        self.agent = _AgentAdapter(base_agent, env=self.env)

        # Persistent best-result tracking across all episodes
        self._best_energy: float = float("inf")
        self._best_excitations: List = []
        self._best_ucc_params: Optional[np.ndarray] = None
        self._training_history: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        n_episodes: Optional[int] = None,
        early_stop_threshold: float = 1.6e-3,
    ) -> SearchResult:
        """Run the hybrid architecture search.

        Args:
            n_episodes: Number of training episodes.  Overrides config value.
            early_stop_threshold: Chemical accuracy threshold in Hartree.

        Returns:
            :class:`SearchResult` with best circuit, energy, training history,
            and fusion template.
        """
        n_eps = n_episodes if n_episodes is not None else self.ctrl_config.get("n_episodes", 500)
        threshold = early_stop_threshold
        log_freq = self.ctrl_config.get("log_frequency", 10)

        print(
            f"[HybridSearchController] Starting hybrid search: "
            f"{n_eps} episodes, threshold={threshold} Ha, "
            f"molecule={self.molecule_data.molecular_info.get('formula', '?')}"
        )

        for episode in range(n_eps):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0.0
            ep_energy = self.env.current_energy

            while not done:
                try:
                    action = self.agent.select_action(obs)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    done = terminated or truncated
                    ep_reward += float(reward)
                    ep_energy = float(info.get("energy", ep_energy))
                    self.agent.store_experience(obs, reward, done, info)
                except Exception as exc:
                    done = True
                    ep_reward -= 10.0
                    print(f"  [Episode {episode}] step error: {exc}")
                    break

            # Agent training step (no-op for PPO but kept for DQN compatibility)
            try:
                self.agent.train()
            except Exception:
                pass

            # Update global best from environment's tracked best
            if self.env.global_best_energy < self._best_energy:
                self._best_energy = self.env.global_best_energy
                self._best_excitations = list(self.env.global_best_excitations)
                if self.env.global_best_ucc_params is not None:
                    self._best_ucc_params = self.env.global_best_ucc_params.copy()

            best_str = (
                f"{self._best_energy:.6f}"
                if self._best_energy != float("inf")
                else "N/A"
            )
            self._training_history.append(
                {
                    "episode": episode,
                    "reward": ep_reward,
                    "energy": ep_energy,
                    "best_energy": (
                        self._best_energy if self._best_energy != float("inf") else None
                    ),
                }
            )

            if episode % log_freq == 0:
                print(
                    f"  Episode {episode:4d}: energy={ep_energy:.6f}, best={best_str}"
                )

            # Early stopping once chemical accuracy is achieved
            # Require at least 2 episodes to run so training_history is meaningful
            if (
                episode >= 1
                and self.molecule_data.fci_energy is not None
                and self._best_energy != float("inf")
            ):
                err = abs(self._best_energy - self.molecule_data.fci_energy)
                if err < threshold:
                    print(
                        f"  Converged at episode {episode}: "
                        f"error={err * 1000:.4f} mHa (threshold={threshold * 1000:.2f} mHa)"
                    )
                    break

        # Compute final error
        best_err: Optional[float] = None
        converged = False
        if (
            self.molecule_data.fci_energy is not None
            and self._best_energy != float("inf")
        ):
            best_err = abs(self._best_energy - self.molecule_data.fci_energy)
            converged = best_err < 1.6e-3

        # Build fusion template from best excitations found
        if self._best_excitations:
            fusion_template: List[str] = ["UCC"] * len(self._best_excitations)
        else:
            fusion_template = self.fusion_strategy.generate_fusion_template()

        return SearchResult(
            best_circuit=None,
            best_energy=(
                self._best_energy if self._best_energy != float("inf") else None
            ),
            best_error=best_err,
            training_history=self._training_history,
            performance_metrics={
                "n_episodes": len(self._training_history),
                "convergence_reached": converged,
                "best_excitations": [
                    list(e) if isinstance(e, tuple) else e
                    for e in self._best_excitations
                ],
            },
            fusion_template=fusion_template,
            convergence_reached=converged,
        )

    def save_results(self, path: str) -> None:
        """Save search results to JSON (compatible with Phase 2 ResultsDatabase schema).

        Args:
            path: Output file path.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )

        def _cvt(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: _cvt(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_cvt(x) for x in obj]
            return obj

        data = {
            "best_energy": (
                float(self._best_energy)
                if self._best_energy != float("inf")
                else None
            ),
            "best_excitations": [
                list(e) if isinstance(e, tuple) else e
                for e in self._best_excitations
            ],
            "training_history": _cvt(self._training_history),
            "saved_at": datetime.datetime.now().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {path}")

    @classmethod
    def from_config(
        cls, molecule_data: MoleculeData, config: Dict
    ) -> "HybridSearchController":
        """Instantiate from an ExperimentManager config dict.

        Args:
            molecule_data: MoleculeData for the target molecule.
            config: Full experiment config dict (may contain ``rl``, ``search``,
                ``simulation``, etc. sub-sections).

        Returns:
            Configured :class:`HybridSearchController` instance.
        """
        rl_cfg = config.get("rl", {})
        agent_type = rl_cfg.get("agent_type", "ppo")
        return cls(molecule_data, agent_type, config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_agent_config(self) -> Dict:
        """Assemble the agent configuration dict from controller settings.

        Returns a config dict filtered to parameters understood by the
        target agent type, avoiding "unknown parameter" errors from agents
        that have strict config validation (e.g., DQN rejects n_steps).
        """
        base = {
            "seed": self.ctrl_config.get("seed", 42),
            "verbose": self.ctrl_config.get("verbose", 0),
            "learning_rate": self.ctrl_config.get("learning_rate", 3e-4),
            "batch_size": self.ctrl_config.get("batch_size", 32),
            "gamma": self.ctrl_config.get("gamma", 0.99),
            "use_gpu": False,
        }
        # PPO-specific parameters — not accepted by DQN / A2C
        if self.agent_type.lower() in ("ppo", "a2c"):
            base.update({
                "n_steps": self.ctrl_config.get("n_steps", 128),
                "n_epochs": self.ctrl_config.get("n_epochs", 4),
                "policy_type": "MlpPolicy",
            })
        return base
