# rlqas-chem

Reinforcement Learning Quantum Architecture Search for Chemistry.

A standalone, pip-installable Python package consolidating all RLQAS functionality (Phase 1–5).

## Installation

```bash
pip install -e ./rlqas-chem/
```

## Usage

```python
import rlqas_chem
result = rlqas_chem.search('LiH', 1.6, ansatz_type='UCC', agent_type='ppo', n_episodes=300)
print(result['best_energy'])
```

## CLI

```bash
rlqas-chem search --molecule LiH --bond-length 1.6 --ansatz UCC --agent ppo --episodes 300
```
