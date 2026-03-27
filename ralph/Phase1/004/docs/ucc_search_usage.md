# UCC Search Module Usage Guide

## Overview
The UCC Search Module implements a reinforcement learning environment and controller for searching optimal UCC (Unitary Coupled Cluster) quantum circuit architectures. This module integrates with Phase 1 Tasks 001 (Molecule Processing), 002 (Quantum Simulator), and 003 (PPO RL Agent).

## Installation
Ensure you have installed the dependencies:
```bash
pip install tencirchem-ng>=2024.10 openfermion>=1.5 pyscf>=2.0.0 gym>=0.21.0 stable-baselines3>=2.0.0 torch>=1.9.0
```

Add the sibling task directories to your Python path:
```python
import sys
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")
```

## Basic Usage

### 1. Process a Molecule (Task 001)
```python
from src.modules.molecule_processor import process_molecule
molecule_data = process_molecule("H2", 0.74, "UCC")
```

### 2. Create UCC Search Environment
```python
from src.modules.ucc_search import UCCSearchEnv
env = UCCSearchEnv(molecule_data, config={"environment": {"max_depth": 10}})
```

### 3. Create and Run Search Controller
```python
from src.modules.ucc_search import UCCSearchController
controller = UCCSearchController(molecule_data, agent_type='ppo')
results = controller.search(n_episodes=1000, early_stop_threshold=1.6e-3)
```

### 4. Examine Results
```python
print(f"Best energy: {results['best_energy']}")
print(f"Best circuit depth: {len(results['best_excitations'])}")
print(f"Convergence reached: {results['convergence_reached']}")
```

## Configuration
The module uses `UCCSearchConfig` for centralized configuration management. Default values follow RLQAS specification section 3.4.

### Example Custom Configuration
```python
config = {
    "environment": {
        "max_depth": 5,
        "max_excitations": 10,
        "param_init_strategy": "zeros"
    },
    "reward_function": {
        "energy_weight": 1.0,
        "complexity_penalty": 0.01,
        "baseline_type": "hartree_fock"
    },
    "controller": {
        "agent_type": "ppo",
        "n_episodes": 500,
        "early_stop_threshold": 1.6e-3,
        "checkpoint_frequency": 50
    }
}
```

## Key Components

### UCCSearchEnv
- Gym-compatible environment for UCC architecture search
- Actions: discrete selection of excitation operators
- Observations: energy, circuit parameters, architecture encoding
- Rewards: energy improvement with complexity penalty

### UCCCircuitBuilder
- Builds parameterized quantum circuits from excitation sequences
- Supports single and double excitations
- Uses tencirchem for circuit construction and energy evaluation

### UCCRewardFunction
- Computes rewards based on energy improvement
- Configurable baseline types: hartree_fock, current_best, rolling_average
- Includes complexity penalty and optional shaping rewards

### UCCSearchController
- Manages complete search process
- Integrates environment, RL agent, and simulator
- Supports early stopping, checkpointing, and results logging

## Advanced Usage

### Custom RL Agents
The controller supports different agent types (currently only 'ppo'). To use a custom agent, subclass `UCCPPOAgent` and override the agent creation logic in `UCCSearchController`.

### Parallel Execution
For larger molecules, consider using vectorized environments or distributed training by modifying the controller configuration.

### Checkpointing and Resumption
```python
# Save results
controller.save_results("search_results.json")

# Load and resume
controller2 = UCCSearchController(molecule_data)
controller2.load_results("search_results.json")
# Continue search
results = controller2.search(n_episodes=500)
```

## Troubleshooting

### Import Errors
Ensure Task 001, 002, and 003 modules are accessible. Check that the directories exist and contain the required Python modules.

### Simulation Errors
If energy evaluation fails, verify that the molecule data contains a valid Hamiltonian and that tencirchem can process the molecule.

### RL Training Instability
Adjust reward scaling, learning rate, or other hyperparameters in the controller configuration.

## Performance Tips
- Start with small molecules (H2) for validation
- Use `param_init_strategy: "zeros"` for deterministic testing
- Set `train_frequency: 0` for quick integration tests
- Monitor energy progression and reward signals for debugging

## Integration with Task 005
This module is designed to be used directly by Task 005 for LiH molecule validation. Ensure consistent configuration and random seeds for reproducible results.

---

For detailed API documentation, refer to the docstrings in each module file.