# RLQAS Phase 1 Task 004 - Knowledge Base

## Overview
This document captures patterns, learnings, and insights gained during the implementation of RLQAS Phase 1 Task 004 - UCC Search Module.

## Core Concepts

### UCC (Unitary Coupled Cluster) Search
- **Purpose**: Search for optimal UCC circuit architectures for quantum chemistry simulation
- **Components**: Environment, circuit builder, reward function, controller
- **Integration**: Combines molecule processing, quantum simulation, and reinforcement learning

### Key Implementation Patterns

#### Gym Environment Design
- **Observation Space**: Should include circuit state, energy, and other relevant information
- **Action Space**: Discrete actions corresponding to excitation operators
- **Step Method**: Must handle circuit building, energy evaluation, reward computation
- **Termination Conditions**: Max depth, convergence threshold, invalid actions

#### Circuit Building
- **Excitation Operators**: Support single (i,j) and double (i,j,k,l) excitations
- **Parameter Management**: Initialize, update, and optimize circuit parameters
- **Circuit Validation**: Ensure circuits are valid and compatible with simulator

#### Reward Design
- **Energy Improvement**: Primary reward signal
- **Complexity Penalty**: Penalize circuit depth/number of excitations
- **Reward Shaping**: Optional additional shaping rewards
- **Baseline Types**: Support for hartree_fock, current_best, and rolling_average baselines
- **Configurable Parameters**: energy_weight, complexity_penalty, baseline_type, shaping_rewards
- **First Evaluation Handling**: Return 0.0 reward for first evaluation to establish baseline
- **Shaping Implementation**: Consecutive improvement bonuses and small penalties for energy increases
- **Edge Cases**: Handle zero improvement, negative improvement, and first evaluation

#### Search Controller
- **Component Integration**: Coordinate environment, agent, and simulator
- **Episode Management**: Handle training loops and early stopping
- **Result Logging**: Save best circuits, energies, training history

## Technical Challenges and Solutions

### Dependency Management
**Challenge**: Importing modules from Tasks 001, 002, 003
**Solution**: Use sys.path.append() to add sibling directories to Python path

```python
import sys
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")
```

### Gymnasium Compatibility
**Challenge**: Gym API changes in newer versions
**Solution**: Use compatibility wrappers or target specific gym version

### Quantum Circuit Representation
**Challenge**: Representing UCC circuits in a format compatible with simulator
**Solution**: Use Tencirchem circuit representation or convert to simulator format

## Testing Strategies

### Unit Testing
- Test each component in isolation
- Mock dependencies where appropriate
- Focus on interface compliance

### Integration Testing
- Test component interactions
- Use mock data initially, then real molecule data
- Verify end-to-end pipeline

### Performance Testing
- Measure single step time
- Monitor memory usage
- Profile critical sections

## Configuration Management

### Default Parameters
Based on RLQAS specification section 3.4:
- Environment: max_depth=10, max_excitations=20, use_sqeb=True
- Reward: energy_weight=1.0, complexity_penalty=0.01
- Controller: agent_type="ppo", n_episodes=1000

### Validation Rules
- Validate configuration on initialization
- Provide sensible defaults for missing parameters
- Support configuration overrides

## Reproducibility

### Random Seeds
- Set seeds for numpy, torch, random modules
- Document seed usage in tests
- Ensure deterministic behavior

### Result Logging
- Log hyperparameters and configuration
- Save training history
- Enable experiment reproduction

## Integration Patterns

### With Task 001 (Molecule Processing)
```python
from src.modules.molecule_processor import MoleculeData, process_molecule
molecule_data = process_molecule("H2", 0.74, "UCC")
```

### With Task 002 (Quantum Simulator)
```python
from src.modules.quantum_simulator import SimulatorFactory
simulator = SimulatorFactory.create_simulator(molecule_data.n_qubits)
```

### With Task 003 (RL Agent)
```python
from src.modules.rl_agents import PPOAgent
agent = PPOAgent(config={"use_gpu": False, "seed": 42})
```

## File Structure Template
```
src/modules/ucc_search/
├── __init__.py
├── environment.py      # UCCSearchEnv
├── circuit_builder.py  # UCCCircuitBuilder
├── reward_function.py  # UCCRewardFunction
├── controller.py       # UCCSearchController
└── config.py          # Configuration management

tests/
├── test_environment.py
├── test_circuit_builder.py
├── test_reward_function.py
├── test_controller.py
└── test_integration.py
```

## Common Pitfalls and Solutions

### 1. Circular Imports
**Problem**: Components importing each other creating circular dependencies
**Solution**: Use forward references, reorganize imports, or create common interface module

### 2. Memory Leaks
**Problem**: Simulator or circuit objects not properly cleaned up
**Solution**: Implement proper cleanup in close() methods, use context managers

### 3. Training Instability
**Problem**: RL agent training unstable due to reward scaling
**Solution**: Normalize rewards, clip values, adjust reward shaping

### 4. Integration Failures
**Problem**: Modules fail to work together due to interface mismatches
**Solution**: Define clear interfaces, write integration tests early, use adapter patterns

## Performance Considerations

### Circuit Building
- Cache frequently built circuits
- Pre-compute excitation operator lists
- Use efficient data structures

### Energy Evaluation
- Batch evaluations where possible
- Use simulator caching features
- Consider approximate methods for large circuits

### RL Training
- Use vectorized environments if possible
- Adjust batch sizes for memory constraints
- Monitor training stability

## Success Metrics

### Technical Metrics
- Code coverage >70%
- All tests passing
- Integration with all dependent modules
- H2 molecule test demonstrates energy improvement

### Functional Metrics
- Environment correctly implements gym.Env
- Circuit builder creates valid circuits
- Reward function computes appropriate rewards
- Controller manages complete search process

### Integration Metrics
- Successful import of Tasks 001, 002, 003 modules
- End-to-end H2 test completes without errors
- Ready for Task 005 LiH validation

### Implementation Insights

#### Configuration Mismatch Between Controller and PPOAgent
**Problem**: The controller passed "policy" parameter but PPOAgent expects "policy_type".
**Solution**: Updated controller's agent_config mapping to use correct parameter names from Task 003's AgentConfig.DEFAULT_CONFIG.

#### Global Best Energy Tracking
**Problem**: Environment's global_best_energy remained None, preventing controller from tracking best overall energy.
**Solution**: Initialize global_best_energy, global_best_excitations, and global_best_params in reset() method to Hartree-Fock reference values.

#### JSON Serialization of NumPy Types
**Problem**: Saving results failed due to numpy bool_ not being JSON serializable.
**Solution**: Added recursive conversion function in save_results() to convert numpy types to Python native types.

#### Integration Testing Strategy
**Approach**: Created layered tests: unit tests with mocks, integration tests with mocked dependencies, and end-to-end test with real H2 molecule data. This ensures each component works independently and together.

---
*This knowledge base will be updated as implementation progresses.*