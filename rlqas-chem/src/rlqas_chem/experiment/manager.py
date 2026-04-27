"""Experiment manager for rlqas-chem."""
import json
from typing import Dict, Any, Optional

from rlqas_chem.api import search


class Experiment:
    """Manages an RLQAS experiment with molecule, search, and RL configs."""

    def __init__(self, molecule_config: Dict[str, Any],
                 search_config: Dict[str, Any],
                 rl_config: Dict[str, Any]):
        self.molecule_config = molecule_config
        self.search_config = search_config
        self.rl_config = rl_config
        self._result: Optional[Dict] = None

    def run(self) -> Dict[str, Any]:
        """Run the experiment and return results dict."""
        self._result = search(
            molecule=self.molecule_config["formula"],
            bond_length=self.molecule_config["bond_length"],
            ansatz_type=self.search_config.get("ansatz_type", "UCC"),
            agent_type=self.rl_config.get("agent_type", "ppo"),
            n_episodes=self.rl_config.get("n_episodes", 500),
            active_space=self.molecule_config.get("active_space"),
            basis_set=self.molecule_config.get("basis_set", "sto-3g"),
            transform=self.molecule_config.get("transform", "jordan_wigner"),
            config=self.search_config,
        )
        return self._result

    def save(self, path: str):
        """Save results to path as JSON."""
        if self._result is None:
            raise RuntimeError("Call run() before save()")
        with open(path, "w") as f:
            json.dump(self._result, f, indent=2)

    def load(self, path: str) -> Dict[str, Any]:
        """Load results from path."""
        with open(path) as f:
            self._result = json.load(f)
        return self._result
