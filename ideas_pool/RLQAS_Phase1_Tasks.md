# RLQAS Phase 1 Task Breakdown

## Overview
**Phase 1 Goal**: Implement core UCC search functionality with validation on LiH molecule
**Total Tasks**: 5 (RLQAS_Phase1_001 to 005)
**Expected Deliverables**:
- Runnable UCC search prototype system
- LiH test results (verifying chemical accuracy achievability)
- Basic evaluation and debugging tools

## Task Dependencies
```
RLQAS_Phase1_001 (Molecule Processing)
     ↓
RLQAS_Phase1_002 (Quantum Simulator)
     ↓
RLQAS_Phase1_003 (PPO RL Agent)
     ↓
RLQAS_Phase1_004 (UCC Search Module)
     ↓
RLQAS_Phase1_005 (LiH Validation Test)
```

---

## RLQAS_Phase1_001: Molecule Processing Module

### Task Metadata
- **ID**: RLQAS_Phase1_001
- **Priority**: P0 (Foundation for all other tasks)
- **Dependencies**: None
- **Estimated Complexity**: Medium
- **Related Spec Section**: 3.1

### Functional Description
Implement the molecule processing module that converts molecular information into quantum computation inputs using Tencirchem-ng 2024.10.

### Specific Requirements
1. Implement the `process_molecule()` function as defined in specification section 3.1
2. Create the `MoleculeData` dataclass with all required fields
3. Integrate with Tencirchem-ng 2024.10 for molecular integrals and Hamiltonian generation
4. Support basic test molecules: H₂, LiH, BeH₂
5. Implement proper error handling for invalid inputs

### Implementation Details
**File Structure**:
```
src/modules/molecule_processor.py
    - class MoleculeProcessor
    - function process_molecule()
    - dataclass MoleculeData

tests/test_molecule_processor.py
    - test_h2_processing()
    - test_lih_processing()
    - test_invalid_inputs()
```

**Core Interfaces**:
```python
@dataclass
class MoleculeData:
    hamiltonian: QubitOperator      # Qubit Hamiltonian
    n_qubits: int                   # Number of qubits
    reference_state: np.ndarray     # Reference state (Hartree-Fock)
    fci_energy: float               # Exact FCI energy
    molecular_info: Dict            # Original molecular information

def process_molecule(
    molecule: str,           # Molecular formula, e.g., 'LiH', 'BeH2', 'H4'
    bond_length: float,      # Bond length (Å)
    ansatz_type: str,        # 'UCC', 'HEA', 'MIXED'
    active_space: Optional[Tuple[int, int]] = None,  # (Number of active electrons, number of active orbitals)
    basis_set: str = "sto-3g",  # Basis set
    transform: str = "parity"   # Fermion-to-qubit transformation
) -> MoleculeData
```

### Test Requirements
- **Unit Tests**: Test H₂ (2 qubits) basic functionality
- **Integration Tests**: Test LiH (4 qubits) core functionality
- **Error Cases**: Test invalid molecule names, bond lengths, etc.
- **Coverage**: >80% code coverage

### Acceptance Criteria
- [ ] `process_molecule()` correctly processes H₂ molecule (2 qubits)
- [ ] `process_molecule()` correctly processes LiH molecule (4 qubits)
- [ ] `MoleculeData` dataclass contains all required fields
- [ ] Integration with Tencirchem-ng 2024.10 works without errors
- [ ] All unit tests pass with >80% coverage
- [ ] Code follows PEP8 standards

### Output Validation
The module should produce valid `MoleculeData` objects that can be used by subsequent modules (simulator, search environment).

---

## RLQAS_Phase1_002: Quantum Simulator Module (Tencirchem CI Vector)

### Task Metadata
- **ID**: RLQAS_Phase1_002
- **Priority**: P0
- **Dependencies**: RLQAS_Phase1_001 (MoleculeData output)
- **Estimated Complexity**: High
- **Related Spec Section**: 3.2

### Functional Description
Implement the quantum simulator module using Tencirchem-ng 2024.10 CI vector engine. Create configurable simulation interfaces for systems with 4-20+ qubits.

### Specific Requirements
1. Implement the `QuantumSimulator` abstract base class with required methods
2. Create `TencirchemCISimulator` class implementing the CI vector engine
3. Implement `SimulatorFactory` for creating simulators based on system scale
4. Support configuration-driven simulator selection
5. Implement performance estimation methods

### Implementation Details
**File Structure**:
```
src/modules/quantum_simulator.py
    - class QuantumSimulator(ABC)
    - class TencirchemCISimulator(QuantumSimulator)
    - class SimulatorFactory

tests/test_quantum_simulator.py
    - test_simulator_interface()
    - test_ci_vector_performance()
    - test_simulator_factory()
```

**Core Interfaces**:
```python
class QuantumSimulator(ABC):
    @abstractmethod
    def compute_energy(
        self,
        circuit: QuantumCircuit,
        hamiltonian: QubitOperator,
        initial_state: Optional[np.ndarray] = None
    ) -> float:
        pass

    @abstractmethod
    def get_max_qubits(self) -> int:
        pass

    @abstractmethod
    def estimate_memory(self, n_qubits: int) -> float:
        pass

class SimulatorFactory:
    @staticmethod
    def create_simulator(
        n_qubits: int,
        config: Dict = None
    ) -> QuantumSimulator:
        # Implementation as per spec section 3.2
```

### Test Requirements
- **Performance Tests**: Measure single energy evaluation time (<100ms for 8 qubits)
- **Memory Tests**: Validate memory estimation accuracy
- **Integration Tests**: Test with actual circuits from molecule processor
- **Coverage**: >75% code coverage

### Acceptance Criteria
- [ ] `QuantumSimulator` abstract class correctly defined
- [ ] `TencirchemCISimulator` implements all abstract methods
- [ ] `SimulatorFactory` can create appropriate simulators based on qubit count
- [ ] Single energy evaluation for 8-qubit circuit completes in <100ms
- [ ] Memory estimation method provides reasonable estimates
- [ ] All tests pass with >75% coverage

### Configuration Example
```python
config = {
    "engine": "ci_vector",
    "precision": 1e-8,
    "use_symmetry": True,
    "max_memory_gb": 32,
    "fallback_method": "statevector"
}
```

---

## RLQAS_Phase1_003: PPO RL Agent (Basic Implementation)

### Task Metadata
- **ID**: RLQAS_Phase1_003
- **Priority**: P0
- **Dependencies**: None (but will be used by RLQAS_Phase1_004)
- **Estimated Complexity**: High
- **Related Spec Section**: 3.3

### Functional Description
Implement a basic Proximal Policy Optimization (PPO) agent following the RL agent interface. Focus on algorithm correctness rather than advanced features.

### Specific Requirements
1. Implement the `RLAgent` abstract base class
2. Create `PPOAgent` class implementing the PPO algorithm
3. Implement standard RL agent interface methods: `act()`, `learn()`, `save()`, `load()`
4. Include basic neural network architecture for policy and value functions
5. Provide reasonable default hyperparameters

### Implementation Details
**File Structure**:
```
src/modules/rl_agents/
    - base_agent.py (RLAgent abstract class)
    - ppo_agent.py (PPOAgent implementation)
    - networks.py (Neural network definitions)

tests/test_rl_agents.py
    - test_agent_interface()
    - test_ppo_learning()
    - test_save_load()
```

**Core Interfaces**:
```python
class RLAgent(ABC):
    @abstractmethod
    def act(self, state: np.ndarray) -> Tuple[int, Dict]:
        pass

    @abstractmethod
    def learn(self, experience: Dict) -> Dict:
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

class PPOAgent(RLAgent):
    def __init__(self, config: Dict):
        # Default config as per spec
        default_config = {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10
        }
```

### Test Requirements
- **Unit Tests**: Test agent interface compliance
- **Learning Tests**: Test on simple environments (CartPole or custom toy environment)
- **Integration Tests**: Test save/load functionality
- **Coverage**: >70% code coverage

### Acceptance Criteria
- [ ] `RLAgent` abstract class correctly defined
- [ ] `PPOAgent` implements all abstract methods
- [ ] Agent can learn simple tasks (e.g., CartPole environment)
- [ ] Save/load functionality works correctly
- [ ] Default hyperparameters produce stable learning
- [ ] All tests pass with >70% coverage

### Note
This implementation prioritizes correctness over performance. Advanced optimizations can be added in later phases.

---

## RLQAS_Phase1_004: UCC Search Module (Core)

### Task Metadata
- **ID**: RLQAS_Phase1_004
- **Priority**: P0 (Phase 1 Core)
- **Dependencies**: RLQAS_Phase1_001, RLQAS_Phase1_002, RLQAS_Phase1_003
- **Estimated Complexity**: Very High
- **Related Spec Section**: 3.4

### Functional Description
Implement the UCC-specific search environment and controller. This is the core module that ties together molecule processing, simulation, and RL for UCC architecture search.

### Specific Requirements
1. Implement `UCCSearchEnv` class (gym.Env compatible)
2. Create `UCCCircuitBuilder` for constructing UCC circuits
3. Implement `UCCRewardFunction` for computing rewards
4. Create `UCCSearchController` to manage the search process
5. Support basic UCC excitation operators (single and double excitations)

### Implementation Details
**File Structure**:
```
src/modules/ucc_search/
    - environment.py (UCCSearchEnv)
    - circuit_builder.py (UCCCircuitBuilder)
    - reward_function.py (UCCRewardFunction)
    - controller.py (UCCSearchController)

tests/test_ucc_search.py
    - test_environment_interface()
    - test_circuit_builder()
    - test_reward_function()
    - test_search_controller()
```

**Core Interfaces**:
```python
class UCCSearchEnv(gym.Env):
    def __init__(self, molecule_data: MoleculeData, config: Dict = None):
        # As per spec section 3.4

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        pass

    def reset(self) -> np.ndarray:
        pass

class UCCSearchController:
    def __init__(self, molecule_data: MoleculeData, agent_type: str = "ppo", config: Dict = None):
        pass

    def search(self, n_episodes: int = 1000, early_stop_threshold: float = 1.6e-3) -> SearchResult:
        pass
```

### Test Requirements
- **Unit Tests**: Test each component individually
- **Integration Tests**: Test full search pipeline with simple molecules
- **Performance Tests**: Measure search iteration time
- **Coverage**: >70% code coverage

### Acceptance Criteria
- [ ] `UCCSearchEnv` implements gym.Env interface correctly
- [ ] `UCCCircuitBuilder` can build valid UCC circuits
- [ ] `UCCRewardFunction` computes meaningful rewards
- [ ] `UCCSearchController` can run search episodes
- [ ] Full pipeline works end-to-end with H₂ molecule
- [ ] All tests pass with >70% coverage

### Configuration Example
```python
env_config = {
    "max_depth": 10,
    "max_excitations": 20,
    "use_sqeb": True,
    "param_init_strategy": "random"
}
```

---

## RLQAS_Phase1_005: LiH Validation Test

### Task Metadata
- **ID**: RLQAS_Phase1_005
- **Priority**: P0 (Validation)
- **Dependencies**: RLQAS_Phase1_001, RLQAS_Phase1_002, RLQAS_Phase1_003, RLQAS_Phase1_004
- **Estimated Complexity**: Medium
- **Related Spec Section**: 5.1, 6.1

### Functional Description
Create comprehensive validation tests for the complete RLQAS system using LiH molecule. This task validates that all Phase 1 components work together correctly.

### Specific Requirements
1. Create end-to-end test script for LiH UCC search
2. Implement performance metrics collection
3. Create validation criteria based on chemical accuracy
4. Generate test report with results
5. Implement basic evaluation tools

### Implementation Details
**File Structure**:
```
scripts/
    - validate_lih.py (Main validation script)
    - run_phase1_test.py (Complete Phase 1 test)

tests/integration/
    - test_lih_validation.py
    - test_phase1_integration.py

results/
    - lih_test_results/ (Output directory)
        - metrics.json
        - circuit_visualizations/
        - training_logs/
```

**Core Test Script**:
```python
def run_lih_validation():
    # 1. Process LiH molecule
    molecule_data = process_molecule(
        molecule="LiH",
        bond_length=1.6,
        ansatz_type="UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="parity"
    )

    # 2. Create simulator
    simulator = SimulatorFactory.create_simulator(
        molecule_data.n_qubits,
        {"max_memory_gb": 32}
    )

    # 3. Create and run UCC search
    controller = UCCSearchController(
        molecule_data,
        agent_type="ppo",
        config={"max_depth": 12, "max_excitations": 15}
    )

    result = controller.search(n_episodes=500, early_stop_threshold=1.6e-3)

    # 4. Evaluate results
    metrics = evaluate_results(result, molecule_data.fci_energy)

    # 5. Generate report
    generate_validation_report(metrics)
```

### Test Requirements
- **End-to-End Test**: Complete system test with LiH
- **Performance Validation**: Verify chemical accuracy achievement
- **Integration Test**: All modules work together correctly
- **Report Generation**: Clear test results output

### Acceptance Criteria
- [ ] End-to-end test runs without errors
- [ ] System achieves chemical accuracy (<1.6 mHa error) on LiH within reasonable time
- [ ] Test completes in <2 hours (performance goal)
- [ ] Comprehensive test report generated
- [ ] All integration tests pass
- [ ] Results demonstrate working UCC search prototype

### Success Metrics
1. **Functional Success**: System produces valid UCC circuits for LiH
2. **Accuracy Success**: Achieves chemical accuracy (<1.6 mHa error)
3. **Performance Success**: Single experiment <2 hours
4. **Integration Success**: All modules work together seamlessly

---

## Phase 1 Complete Deliverables Checklist

After completing all 5 tasks, the following should be verified:

### Core System
- [ ] Runnable UCC search prototype system
- [ ] LiH validation test passing
- [ ] Basic evaluation and debugging tools

### Code Quality
- [ ] Overall code coverage >70%
- [ ] PEP8 compliance throughout
- [ ] Clear documentation and docstrings

### Documentation
- [ ] API documentation for all Phase 1 modules
- [ ] Quick start guide for running LiH test
- [ ] Troubleshooting guide for common issues

### Next Steps Ready
- [ ] Foundation laid for Phase 2 (multi-algorithm support)
- [ ] Architecture extensible for HEA and hybrid search
- [ ] Performance baseline established for optimization

---

## Environment Setup Requirements

Before starting Phase 1 implementation, ensure the following environment is available:

```bash
# Python version
python>=3.8

# Core dependencies
pip install numpy>=1.21 scipy>=1.7 pandas>=1.3

# Quantum computing dependencies
pip install tencirchem-ng>=2024.10 openfermion>=1.5

# RL dependencies
pip install torch>=1.9 gym>=0.21

# Development tools
pip install pytest>=7.0 pytest-cov>=4.0 black>=22.0
```

## Task Assignment Recommendation

Assign tasks in dependency order:
1. Start with RLQAS_Phase1_001 and RLQAS_Phase1_003 (can proceed in parallel)
2. Once 001 is complete, start RLQAS_Phase1_002
3. Once 001, 002, 003 are complete, start RLQAS_Phase1_004
4. Finally, execute RLQAS_Phase1_005 for validation

This parallel approach minimizes idle time while respecting dependencies.
