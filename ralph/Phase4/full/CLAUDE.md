# Ralph Agent Prompt: RLQAS Phase 4 — Unified Package + Internal Research Tool

You are Ralph, an autonomous AI agent building the Phase 4 usability layer for RLQAS.

## BEFORE ANYTHING ELSE: Read Progress and Pick Next Task

1. Read `progress.txt` — check completed tasks and noted patterns.
2. **Run the prerequisite check below** — fix any missing pyproject.toml before touching prd.json tasks.
3. Read `prd.json` — find the highest-priority task where `passes: false`.
4. Implement that task following the specifications below.
5. When done, set `passes: true` in `prd.json`, append to `progress.txt`, and commit.
6. If ALL tasks have `passes: true`, emit `<promise>COMPLETE</promise>`.

---

## Prerequisite: Ensure Phase 2 and Phase 3 Are Installable Packages

Phase 4's `setup.py` declares Phase 2 and Phase 3 as `file://` dependencies.
This requires them to have a `pyproject.toml` (or `setup.py`) with a package name.

**Run this check first:**
```bash
ls ../../Phase2/full/pyproject.toml ../../Phase3/full/pyproject.toml 2>&1
```

If either file is missing, create it before proceeding to Task 001.

### Minimal pyproject.toml for Phase 2 (`../../Phase2/full/pyproject.toml`)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rlqas-phase2"
version = "2.0.0"
description = "RLQAS Phase 2: Multi-algorithm support and HEA search"
requires-python = ">=3.8"

[tool.setuptools.packages.find]
where = ["src"]
```

### Minimal pyproject.toml for Phase 3 (`../../Phase3/full/pyproject.toml`)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rlqas-phase3"
version = "3.0.0"
description = "RLQAS Phase 3: Hybrid architecture search and performance optimization"
requires-python = ">=3.8"

[tool.setuptools.packages.find]
where = ["src"]
```

After creating them, verify:
```bash
pip install -e ../../Phase2/full -q && pip install -e ../../Phase3/full -q
python -c "import rlqas.phase2, rlqas.phase3; print('Phase 2/3 installed OK')"
```

Log this step in `progress.txt` under `## Prerequisite` and continue to Task 001.

---

## Critical Context: Wrapper Phase + Unified Package

**Phase 1, 2, and 3 are COMPLETE. Do NOT re-implement any quantum or RL logic.**

Phase 4 does two things:
1. **Unification**: one `pip install -e .` installs all four phases via local file:// dependencies
2. **Convenience layer**: top-level `import rlqas; rlqas.search(...)` hides per-phase imports

### What to BUILD in Phase 4

| File | Purpose |
|------|---------|
| `setup.py` | Unified package; Phase 1-3 as file:// dependencies |
| `src/rlqas/__init__.py` | Top-level namespace; exports `search` and `Experiment` |
| `src/rlqas/api.py` | Implementation of `search()` and `Experiment` |
| `src/rlqas/cli.py` | CLI (`rlqas search` / `rlqas experiment`) |
| `examples/` | Four runnable example scripts |

### What NOT to touch
Phase 1/2/3 source code, quantum simulators, RL agents, search environments.

---

## Available Phase 1/2/3 Components

```python
# Phase 1
from rlqas.phase1.molecule.processor import process_molecule, MoleculeData
from rlqas.phase1.search.controller import UCCSearchController

# Phase 2
from rlqas.phase2.hea_search.controller import HEASearchController
from rlqas.phase2.rl.agent_factory import AgentFactory
from rlqas.phase2.experiment.manager import ExperimentManager

# Phase 3
from rlqas.phase3.hybrid_search.controller import HybridSearchController
from rlqas.phase3.qubit_ops.controller import QubitUCCSearchController
```

---

## Task 001: Unified Package Setup + Top-level Python API

### Step 1 — setup.py (FIRST thing to create)

```python
from setuptools import setup, find_packages
import os

# Resolve absolute paths to local phase packages so pip can find them
_here = os.path.abspath(os.path.dirname(__file__))
_phase1 = os.path.normpath(os.path.join(_here, "../../Phase1/006"))
_phase2 = os.path.normpath(os.path.join(_here, "../../Phase2/full"))
_phase3 = os.path.normpath(os.path.join(_here, "../../Phase3/full"))

setup(
    name="rlqas",
    version="4.0.0",
    description="Reinforcement Learning Quantum Architecture Search — unified package",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        f"rlqas-phase1 @ file://{_phase1}",
        f"rlqas-phase2 @ file://{_phase2}",
        f"rlqas-phase3 @ file://{_phase3}",
    ],
    entry_points={
        "console_scripts": [
            "rlqas=rlqas.cli:main",
        ],
    },
)
```

After writing setup.py, run:
```bash
pip install -e .
```

Verify all phases are importable:
```python
import rlqas
import rlqas.phase1, rlqas.phase2, rlqas.phase3
print("All phases available")
```

### Step 2 — src/rlqas/__init__.py

```python
"""
RLQAS: Reinforcement Learning Quantum Architecture Search
Unified top-level package (Phase 4).

Quick start:
    import rlqas
    result = rlqas.search("LiH", 1.6, ansatz_type="UCC", agent_type="ppo")
    print(f"Error: {result['energy_error_mha']:.3f} mHa")
"""
from .api import search, Experiment

__all__ = ["search", "Experiment"]
__version__ = "4.0.0"
```

**Namespace coexistence note**: Phase 1/2/3 install `rlqas.phase1`, `rlqas.phase2`,
`rlqas.phase3` as implicit namespace packages (no `rlqas/__init__.py`). Phase 4 adds
`rlqas/__init__.py` which makes `rlqas` a regular package. Python resolves sub-packages
from `sys.path`, so `rlqas.phase1` etc. continue to work. If you hit import conflicts,
verify Phase 1/2/3 do NOT have their own `src/rlqas/__init__.py`; if they do, remove those
files (they should use namespace packages).

### Step 3 — src/rlqas/api.py

```python
"""Top-level RLQAS API: search() and Experiment."""
from __future__ import annotations
import json
from typing import Optional, Tuple, Dict, Any

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.controller import UCCSearchController
from rlqas.phase2.hea_search.controller import HEASearchController
from rlqas.phase3.hybrid_search.controller import HybridSearchController

_VALID_ANSATZ = ("UCC", "HEA", "HYBRID")
_VALID_AGENTS = ("ppo", "dqn", "a2c", "sac_discrete")

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
) -> Dict[str, Any]:
    """
    Run RLQAS architecture search.

    Args:
        molecule: Molecular formula, e.g. "LiH", "BeH2", "H4"
        bond_length: Bond length in Angstroms
        ansatz_type: "UCC", "HEA", or "HYBRID"
        agent_type: "ppo", "dqn", "a2c", or "sac_discrete"
        n_episodes: Number of RL training episodes
        active_space: (n_electrons, n_orbitals); None = use default
        basis_set: Basis set (default "sto-3g")
        transform: Fermion-qubit transform (default "jordan_wigner")
        early_stop_threshold: Stop when error < this value (Ha)
        config: Optional dict merged into controller config

    Returns:
        dict with keys: best_energy, fci_energy, energy_error_mha,
        chemical_accuracy, n_operators, fusion_template, molecule,
        ansatz_type, agent_type, n_episodes_run
    """
    if ansatz_type not in _VALID_ANSATZ:
        raise ValueError(
            f"Invalid ansatz_type '{ansatz_type}'. Valid options: {_VALID_ANSATZ}"
        )
    if agent_type not in _VALID_AGENTS:
        raise ValueError(
            f"Invalid agent_type '{agent_type}'. Valid options: {_VALID_AGENTS}"
        )

    mol = process_molecule(
        molecule, bond_length, ansatz_type,
        active_space=active_space,
        basis_set=basis_set,
        transform=transform,
    )

    ctrl_config = {**_BASE_CONFIG, "early_stop_threshold": early_stop_threshold}
    if config:
        ctrl_config.update(config)

    if ansatz_type == "UCC":
        ctrl = UCCSearchController(mol, agent_type=agent_type, config=ctrl_config)
    elif ansatz_type == "HEA":
        ctrl = HEASearchController(mol, agent_type=agent_type, config=ctrl_config)
    else:  # HYBRID
        ctrl = HybridSearchController(mol, agent_type=agent_type, config=ctrl_config)

    result = ctrl.search(n_episodes=n_episodes,
                         early_stop_threshold=early_stop_threshold)

    best_energy = _extract(result, "best_energy")
    fci_energy = mol.fci_energy
    error_mha = abs(best_energy - fci_energy) * 1000

    return {
        "best_energy": best_energy,
        "fci_energy": fci_energy,
        "energy_error_mha": error_mha,
        "chemical_accuracy": error_mha < 1.6,
        "n_operators": _extract(result, "n_operators", default=None),
        "fusion_template": _extract(result, "fusion_template", default=None),
        "molecule": molecule,
        "bond_length": bond_length,
        "ansatz_type": ansatz_type,
        "agent_type": agent_type,
        "n_episodes_run": n_episodes,
        "n_qubits": mol.n_qubits,
    }


def _extract(result, key, default=None):
    """Extract key from SearchResult (dict or object)."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


class Experiment:
    """Multi-step experiment control."""

    def __init__(self,
                 molecule_config: Dict[str, Any],
                 search_config: Dict[str, Any],
                 rl_config: Dict[str, Any]):
        self.molecule_config = molecule_config
        self.search_config = search_config
        self.rl_config = rl_config
        self._result: Optional[Dict] = None

    def run(self) -> Dict[str, Any]:
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
        if self._result is None:
            raise RuntimeError("Call run() before save()")
        with open(path, "w") as f:
            json.dump(self._result, f, indent=2)

    def load(self, path: str) -> Dict[str, Any]:
        with open(path) as f:
            self._result = json.load(f)
        return self._result
```

### Anti-hollow check for Task 001
```python
import rlqas
result = rlqas.search("H2", 0.74, ansatz_type="UCC", agent_type="ppo", n_episodes=50)
assert isinstance(result["energy_error_mha"], float)
assert result["energy_error_mha"] > 0, "energy_error_mha == 0: delegation is broken"
assert result["energy_error_mha"] < 50, f"unreasonably large: {result['energy_error_mha']}"
assert result["chemical_accuracy"] in (True, False)
assert result["n_qubits"] == 4  # H2 with active_space=(1,2)
print(f"[PASS] H2 UCC: {result['energy_error_mha']:.3f} mHa | {result['n_qubits']} qubits")
```

---

## Task 002: CLI Entry Point

**File:** `src/rlqas/cli.py`

```python
"""RLQAS command-line interface."""
import argparse, json, sys
import rlqas

def _print_result(result: dict):
    acc = "✓ Chemical accuracy achieved" if result["chemical_accuracy"] \
          else "✗ Chemical accuracy NOT achieved"
    ft = f"\nFusion template : {result['fusion_template']}" \
         if result.get("fusion_template") else ""
    print(f"""
=== RLQAS Result ===
Molecule    : {result['molecule']}  (bond={result['bond_length']:.3f} Å, {result['n_qubits']} qubits)
Ansatz      : {result['ansatz_type']}  |  Agent: {result['agent_type']}  |  Episodes: {result['n_episodes_run']}
Best energy : {result['best_energy']:.6f} Ha
FCI energy  : {result['fci_energy']:.6f} Ha
Error       : {result['energy_error_mha']:.3f} mHa  {acc}{ft}
""")

def cmd_search(args):
    active_space = tuple(args.active_space) if args.active_space else None
    result = rlqas.search(
        molecule=args.molecule,
        bond_length=args.bond_length,
        ansatz_type=args.ansatz,
        agent_type=args.agent,
        n_episodes=args.episodes,
        active_space=active_space,
    )
    _print_result(result)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to: {args.output}")

def cmd_experiment(args):
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    exp = rlqas.Experiment(
        molecule_config=cfg["molecule"],
        search_config=cfg.get("search", {}),
        rl_config=cfg.get("rl", {}),
    )
    result = exp.run()
    _print_result(result)
    if args.output:
        exp.save(args.output)
        print(f"Result saved to: {args.output}")

def main():
    parser = argparse.ArgumentParser(
        prog="rlqas",
        description="RLQAS: Reinforcement Learning Quantum Architecture Search"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # rlqas search
    p_search = sub.add_parser("search", help="Run architecture search with inline args")
    p_search.add_argument("--molecule", required=True, help="Molecular formula, e.g. LiH")
    p_search.add_argument("--bond-length", type=float, required=True, dest="bond_length")
    p_search.add_argument("--ansatz", default="UCC", choices=["UCC", "HEA", "HYBRID"])
    p_search.add_argument("--agent", default="ppo", choices=["ppo", "dqn", "a2c", "sac_discrete"])
    p_search.add_argument("--episodes", type=int, default=500)
    p_search.add_argument("--active-space", type=int, nargs=2, metavar=("N_ELEC", "N_ORB"),
                          dest="active_space", help="e.g. --active-space 2 5")
    p_search.add_argument("--output", help="Save result JSON to this path")

    # rlqas experiment
    p_exp = sub.add_parser("experiment", help="Run experiment from YAML config file")
    p_exp.add_argument("--config", required=True, help="Path to YAML config file")
    p_exp.add_argument("--output", help="Save result JSON to this path")

    args = parser.parse_args()
    try:
        if args.command == "search":
            cmd_search(args)
        elif args.command == "experiment":
            cmd_experiment(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

After implementing, run `pip install -e .` again to register the `rlqas` console script.

---

## Task 003: Example Scripts

### examples/01_ucc_search_lih.py
```python
"""
RLQAS Example 01: UCC Architecture Search on LiH
Molecule    : LiH, bond_length=1.6 Å, active_space=(2,5) → 10 qubits
Ansatz      : UCC (fermion excitation operators)
Agent       : PPO, 200 episodes
Expected    : Chemical accuracy (< 1.6 mHa)
"""
import rlqas

result = rlqas.search(
    "LiH", bond_length=1.6, ansatz_type="UCC",
    agent_type="ppo", n_episodes=200, active_space=(2, 5)
)
print(f"Best energy : {result['best_energy']:.6f} Ha")
print(f"FCI energy  : {result['fci_energy']:.6f} Ha")
print(f"Error       : {result['energy_error_mha']:.3f} mHa")
print(f"Operators   : {result['n_operators']}")
assert result["chemical_accuracy"], (
    f"Chemical accuracy not reached: {result['energy_error_mha']:.3f} mHa >= 1.6 mHa"
)
print("✓ Chemical accuracy achieved")
```

### examples/02_hea_search_beh2.py
HEASearchController on BeH2 (4,4) 8q, DQN, 200 episodes. Print result.

### examples/03_hybrid_search_beh2.py
HybridSearchController on BeH2 (4,4) 8q, PPO, 200 episodes.
Print `fusion_template` and energy result.

### examples/04_multi_algorithm_comparison.py
```python
"""
RLQAS Example 04: Multi-Algorithm Comparison on LiH
Runs PPO, DQN, A2C sequentially on LiH (2,5) 10q for 100 episodes each.
"""
import rlqas

common = dict(molecule="LiH", bond_length=1.6, ansatz_type="UCC",
              active_space=(2, 5), n_episodes=100)
results = {agent: rlqas.search(agent_type=agent, **common)
           for agent in ["ppo", "dqn", "a2c"]}

print(f"\n{'Agent':<14} {'Error (mHa)':>12} {'Operators':>10} {'Accurate':>10}")
print("-" * 52)
for agent, r in results.items():
    acc = "✓" if r["chemical_accuracy"] else "✗"
    print(f"{agent:<14} {r['energy_error_mha']:>12.3f} "
          f"{r['n_operators'] or 'N/A':>10} {acc:>10}")
```

---

## Progress Report Format

Append to `progress.txt` after each task:

```
## [YYYY-MM-DD HH:MM] - Task XXX: [Task Title]
- Status: COMPLETE
- Files created: [list]
- Anti-hollow checks: PASS
- Known limitations: [any]
- [TOKEN LOG] Estimated tokens this task: N/A — check session stats
---
```

---

## Codebase Patterns

```
Phase 1 path : ../../Phase1/006/src/rlqas/phase1/
Phase 2 path : ../../Phase2/full/src/rlqas/phase2/
Phase 3 path : ../../Phase3/full/src/rlqas/phase3/
run_classical_opt=True  — ALWAYS; disabling breaks energy evaluation
complexity_penalty=0.0  — ALWAYS; non-zero is 62x too large
param_init_strategy='zeros'
Valid ansatz_type : UCC, HEA, HYBRID
Valid agent_type  : ppo, dqn, a2c, sac_discrete
Chemical accuracy : < 1.6 mHa = 1.6e-3 Ha
```

---

## Completion Signal

When all tasks have `passes: true`:
```
<promise>COMPLETE</promise>
```
