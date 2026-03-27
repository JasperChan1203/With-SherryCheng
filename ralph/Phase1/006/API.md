# RLQAS Phase 1 API Documentation

This document describes the public interfaces of the RLQAS Phase 1 integrated package.

## Package Structure

```
rlqas.phase1
├── molecule           # Molecular processing
├── simulator          # Quantum simulation
├── rl                 # Reinforcement learning agents
├── search             # UCC architecture search
├── validation         # Validation and testing utilities
└── utils              # Shared utilities
```

## Molecule Module

### `process_molecule`
```python
def process_molecule(
    molecule: str,
    bond_length: float,
    ansatz_type: str,
    active_space: Optional[Tuple[int, int]] = None,
    basis_set: str = "sto-3g",
    transform: str = "jordan_wigner"
) -> MoleculeData
```
Process a molecule and generate quantum computation inputs.

**Parameters:**
- `molecule`: Molecular formula ('H2', 'LiH', 'BeH2')
- `bond_length`: Bond length in Ångstroms
- `ansatz_type`: Ansatz type ('UCC', 'HEA', 'MIXED')
- `active_space`: Optional (n_electrons, n_orbitals) active space
- `basis_set`: Basis set string (default 'sto-3g')
- `transform`: Fermion-to-qubit transformation ('parity', 'jordan_wigner', 'bravyi_kitaev')

**Returns:** `MoleculeData` object containing Hamiltonian, qubit count, reference state, FCI energy, and molecular information.

### `MoleculeData` Dataclass
Container for molecule processing results.

**Attributes:**
- `hamiltonian`: `QubitOperator` - Qubit Hamiltonian
- `n_qubits`: `int` - Number of qubits
- `reference_state`: `np.ndarray` - Reference state (Hartree-Fock)
- `fci_energy`: `float` - Exact FCI energy
- `molecular_info`: `Dict` - Original molecular information
- `ucc_object`: `Any` - Tencirchem UCC object (for consistency)
- `ucc_sd_object`: `Any` - Tencirchem UCCSD object (for circuit building)

## Simulator Module

### `QuantumSimulator` Abstract Base Class
Defines the interface for quantum circuit simulation.

**Methods:**
- `compute_energy(circuit, hamiltonian, initial_state=None) -> float`
- `get_max_qubits() -> int`
- `estimate_memory(n_qubits) -> float`

### `TencirchemCISimulator`
Implementation using Tencirchem's CI vector engine.

**Configuration parameters:**
- `engine`: 'ci_vector', 'statevector', 'mps', 'custom'
- `precision`: Energy convergence tolerance
- `max_memory_gb`: Maximum memory allocation before fallback
- `fallback_method`: Fallback method when CI vector exceeds memory
- `use_gpu`: Enable GPU acceleration if available

### `SimulatorFactory`
Factory for creating appropriate quantum simulators based on system scale.

**Methods:**
- `create_simulator(n_qubits, config=None) -> QuantumSimulator`

## RL Module

### `RLAgent` Abstract Base Class
Defines the interface for reinforcement learning agents.

**Methods:**
- `train(env, total_timesteps)`
- `predict(observation) -> action`
- `save(path)`
- `load(path)`

### `PPOAgent`
Proximal Policy Optimization agent implementation.

**Configuration parameters:**
- `learning_rate`: Learning rate for policy optimizer
- `n_steps`: Number of steps per update
- `batch_size`: Minibatch size
- `n_epochs`: Number of optimization epochs per update
- `gamma`: Discount factor
- `ent_coef`: Entropy coefficient for exploration

### `AgentConfig`
Configuration dataclass for RL agents.

## Search Module

### `UCCSearchEnv`
Gymnasium environment for UCC architecture search.

**Observation space:** `Box` representing Hamiltonian terms, current energy, etc.
**Action space:** `Discrete(3)` (add single, add double, terminate)

**Methods:**
- `reset() -> (observation, info)`
- `step(action) -> (obs, reward, terminated, truncated, info)`

### `UCCSearchController`
High-level controller for UCC search workflow.

**Methods:**
- `search(n_episodes, early_stop_threshold=None) -> Dict`
  Runs UCC search and returns results dictionary.

### `CircuitBuilder`
Builds parameterized quantum circuits from excitation lists.

### `RewardFunction`
Computes rewards based on energy improvement and circuit complexity.

## Validation Module

### `run_lih_validation`
```python
def run_lih_validation(
    active_space: Tuple[int, int] = (2, 3),
    n_episodes: int = 500,
    early_stop_threshold: float = 1.6e-3,
    config: Optional[Dict] = None
) -> Dict
```
Run chemical accuracy validation for LiH molecule.

**Returns:** Dictionary with validation results including best energy, FCI energy, error, and metrics.

### `MetricsCollector`
Collects and aggregates performance metrics during validation.

### `ReportGenerator`
Generates validation reports in Markdown format.

## Utilities Module

### `transforms`
- `compute_reference_state(hamiltonian, n_qubits, transform, n_electrons) -> np.ndarray`
- `get_hartree_fock_bitstring(n_qubits, n_electrons, transform) -> int`

### `chemistry`
Chemistry-related utilities and constants.

### `logger`
Logging utilities with configurable log levels.

## Usage Examples

### Basic Workflow
```python
import rlqas.phase1 as rlqas
from rlqas.phase1.molecule import process_molecule
from rlqas.phase1.search import UCCSearchController

# Process molecule
data = process_molecule("LiH", 1.6, "UCC", active_space=(2,3))

# Create controller and run search
controller = UCCSearchController(data, agent_type='ppo')
results = controller.search(n_episodes=500)

print(f"Best energy: {results['best_energy']} Hartree")
print(f"Error: {(results['best_energy'] - data.fci_energy)*1000:.2f} mHa")
```

### Performance Benchmarking
```python
from examples.benchmark_8qubit import benchmark_8qubit
results = benchmark_8qubit(n_trials=5)
```

### Validation
```python
from rlqas.phase1.validation import run_lih_validation
results = run_lih_validation(active_space=(2,3), n_episodes=1000)
```

## Dependencies

See `requirements.txt` and `pyproject.toml` for complete dependency list.

## Notes

- Default fermion-to-qubit transformation is Jordan-Wigner
- Chemical accuracy validation uses Jordan-Wigner transformation due to parity transformation incompatibility with circuit builder
- All Gymnasium environments return 5-tuple (obs, reward, terminated, truncated, info)
- Memory estimation provides conservative upper bounds for chemical systems

---
*Generated: 2026-03-02*
*Package version: 1.0.0*