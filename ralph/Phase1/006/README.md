# RLQAS Phase 1: Integrated Package

**Reinforcement Learning for Quantum Architecture Search - Phase 1**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This package integrates all Phase 1 modules of the RLQAS project into a unified, production-ready Python package. It provides:

- **Molecular processing** for quantum chemistry calculations
- **Quantum circuit simulation** with multiple backends
- **Reinforcement learning agents** for architecture search
- **UCC search environment** for quantum circuit design
- **Validation tools** for performance and accuracy assessment

## Installation

### From source
```bash
git clone <repository-url>
cd rlqas-phase1
pip install -e .
```

### Dependencies
All dependencies are listed in `pyproject.toml` and will be installed automatically.

## Quick Start

### Basic Usage
```python
import rlqas.phase1 as rlqas

# Process a molecule
from rlqas.molecule import process_molecule, MoleculeData
molecule_data = process_molecule("LiH", 1.6, "UCC", active_space=(2,3))

# Run UCC search
from rlqas.search import UCCSearchController
controller = UCCSearchController(molecule_data)
results = controller.search(n_episodes=500)

print(f"Best energy: {results['best_energy']} Hartree")
```

### Running Validation
```bash
# Run LiH validation test
python -m rlqas.validation.validator --config fast

# Run full validation
python -m rlqas.validation.validator --config full
```

## Features

### Molecular Processing
- Process H₂, LiH, BeH₂ molecules
- Support for active space selection
- Multiple fermion-to-qubit transformations (parity, Jordan-Wigner, Bravyi-Kitaev)
- Hartree-Fock and Full CI energy calculations

### Quantum Simulation
- Abstract simulator interface
- Tencirchem CI vector engine backend
- Statevector and MPS simulation support
- Memory estimation and automatic fallback

### Reinforcement Learning
- PPO agent implementation
- Gymnasium-compatible environments
- Configurable hyperparameters
- GPU acceleration support

### UCC Architecture Search
- Gymnasium environment for circuit search
- Configurable reward functions
- Circuit complexity management
- Early stopping and checkpointing

### Validation and Benchmarking
- Chemical accuracy validation (<1.6 mHa target)
- Performance benchmarking (8-qubit <500ms target)
- Comprehensive metrics collection
- Report generation

## Documentation

- **API Reference**: See `API.md`
- **Examples**: See `examples/` directory
- **Validation Procedure**: See `results/final_validation/validation_report.md`

## Project Structure

```
src/rlqas/phase1/
├── molecule/          # Molecular processing
├── simulator/         # Quantum simulation
├── rl/               # Reinforcement learning
├── search/           # UCC architecture search
├── validation/       # Validation and testing
└── utils/           # Shared utilities
```

## Performance Targets

- **Chemical Accuracy**: <1.6 mHa error for LiH with active_space=(2,3)
- **Simulation Speed**: <500ms for 8-qubit energy evaluation
- **Validation Time**: <2 hours for full LiH validation
- **Memory Efficiency**: Accurate memory estimation for large systems

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use RLQAS in your research, please cite:

```bibtex
@software{rlqas2026,
  title = {RLQAS: Reinforcement Learning for Quantum Architecture Search},
  author = {RLQAS Research Team},
  year = {2026},
  url = {https://github.com/your-org/rlqas}
}
```

## Acknowledgements

- Built upon Tencirchem, OpenFermion, PySCF, and Stable-Baselines3
- Inspired by quantum chemistry and reinforcement learning research
- Developed with support from the research community

---

**Note**: This is the integrated Phase 1 package. For original Task implementations, see the `../001` to `../005` directories.