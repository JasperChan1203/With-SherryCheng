"""Top-level RLQAS API: search() and Experiment."""
from __future__ import annotations
import json
from typing import Optional, Tuple, Dict, Any

from rlqas_chem.molecule.processor import process_molecule
from rlqas_chem.search.ucc.controller import UCCSearchController
from rlqas_chem.search.hybrid.controller import HybridSearchController

_VALID_ANSATZ = ("UCC", "HEA", "HYBRID")
_VALID_AGENTS = ("ppo", "dqn", "a2c", "sac_discrete", "grpo")
_VALID_OPERATOR_POOLS = ("fop", "qop")

# UCCSearchController supports PPO and GRPO; all others route to HybridSearchController
_UCC_AGENTS = ("ppo", "grpo")

_BASE_CONFIG = {
    "run_classical_opt": True,      # MUST stay True — disabling breaks energy evaluation
    "complexity_penalty": 0.0,      # MUST stay 0.0 — non-zero is 62x too large
    "param_init_strategy": "zeros",
}


def search(
    molecule: str,
    bond_length: float,
    ansatz_type: str = "UCC",
    agent_type: str = "ppo",
    n_episodes: int = 500,
    active_space: Optional[Tuple[int, int]] = None,
    basis_set: str = "sto-3g",
    transform: str = "jordan_wigner",
    early_stop_threshold: float = 1.6e-3,
    config: Optional[Dict[str, Any]] = None,
    operator_pool: str = "fop",
) -> Dict[str, Any]:
    """Run RLQAS architecture search.

    Args:
        molecule: Molecular formula, e.g. "LiH", "BeH2", "H4"
        bond_length: Bond length in Angstroms
        ansatz_type: "UCC", "HEA", or "HYBRID"
        agent_type: "ppo", "dqn", "a2c", "sac_discrete", or "grpo"
        n_episodes: Number of RL training episodes
        active_space: (n_electrons, n_orbitals); None = use default
        basis_set: Basis set (default "sto-3g")
        transform: Fermion-qubit transform (default "jordan_wigner")
        early_stop_threshold: Stop when error < this value (Ha)
        config: Optional dict merged into controller config
        operator_pool: "fop" (fermion operator pool, default) or "qop" (qubit operator pool)

    Returns:
        dict with keys: best_energy, fci_energy, energy_error_mha,
        chemical_accuracy, n_operators, fusion_template, molecule,
        bond_length, ansatz_type, agent_type, n_episodes_run, n_qubits
    """
    if ansatz_type not in _VALID_ANSATZ:
        raise ValueError(
            f"Invalid ansatz_type '{ansatz_type}'. Valid options: {_VALID_ANSATZ}"
        )
    if agent_type not in _VALID_AGENTS:
        raise ValueError(
            f"Invalid agent_type '{agent_type}'. Valid options: {_VALID_AGENTS}"
        )
    if operator_pool not in _VALID_OPERATOR_POOLS:
        raise ValueError(
            f"Invalid operator_pool '{operator_pool}'. Valid options: {_VALID_OPERATOR_POOLS}"
        )

    # process_molecule accepts "UCC", "HEA", "HYBRID", "MIXED"
    mol_ansatz = ansatz_type if ansatz_type in ("UCC", "HEA", "HYBRID") else "HEA"
    mol = process_molecule(
        molecule, bond_length, mol_ansatz,
        active_space=active_space,
        basis_set=basis_set,
        transform=transform,
    )

    ctrl_config = {
        **_BASE_CONFIG,
        "controller": {
            "n_episodes": n_episodes,
            "early_stop_threshold": early_stop_threshold,
        },
    }
    if config:
        _deep_merge(ctrl_config, config)

    if ansatz_type == "UCC" and operator_pool == "qop":
        # QOP: qubit operator pool — route to QubitUCCSearchController
        from rlqas_chem.search.qop import QubitUCCSearchController
        ctrl = QubitUCCSearchController(mol, agent_type=agent_type, config=ctrl_config)
        result = ctrl.search(n_episodes=n_episodes,
                             early_stop_threshold=early_stop_threshold)
        best_energy = float(_extract(result, "best_energy") or float('inf'))
        n_operators = _extract(result, "performance_metrics", default={}).get("qubit_pool_size")
        fusion_template = None

    elif ansatz_type == "UCC" and agent_type in _UCC_AGENTS:
        # UCC with PPO: use UCCSearchController
        ctrl = UCCSearchController(mol, agent_type=agent_type, config=ctrl_config)
        result = ctrl.search(n_episodes=n_episodes,
                             early_stop_threshold=early_stop_threshold)
        best_energy = float(_extract(result, "best_energy"))
        n_ops = _extract(result, "best_excitations")
        n_operators = len(n_ops) if n_ops else None
        fusion_template = None

    else:
        # HEA, HYBRID, or UCC+non-PPO: use HybridSearchController
        ctrl = HybridSearchController(mol, agent_type=agent_type, config=ctrl_config)
        result = ctrl.search(n_episodes=n_episodes,
                             early_stop_threshold=early_stop_threshold)
        best_energy = float(_extract(result, "best_energy"))
        ft = _extract(result, "fusion_template", default=None)
        fusion_template = list(ft) if ft else None
        n_operators = None

    fci_energy = float(mol.fci_energy)
    error_mha = float(abs(best_energy - fci_energy) * 1000)

    return {
        "best_energy": best_energy,
        "fci_energy": fci_energy,
        "energy_error_mha": error_mha,
        "chemical_accuracy": bool(error_mha < 1.6),
        "n_operators": n_operators,
        "fusion_template": fusion_template,
        "molecule": molecule,
        "bond_length": bond_length,
        "ansatz_type": ansatz_type,
        "agent_type": agent_type,
        "n_episodes_run": n_episodes,
        "n_qubits": mol.n_qubits,
    }


def _deep_merge(base: dict, override: dict):
    """Deep-merge override into base in-place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _extract(result, key, default=None):
    """Extract key from SearchResult (dict or object)."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)
