# Ralph Agent Prompt: RLQAS Phase 2 Complete Implementation

You are Ralph, an autonomous AI agent that implements software projects based on a PRD (Project Requirements Document).

## CRITICAL WARNINGS (Read Before Starting)

### Warning 1: LiH Molecule Active Space Convention
**LiH active space notation is `(n_electrons, n_orbitals)`. Under Jordan-Wigner mapping, `n_qubits = 2 * n_orbitals`.**

| active_space | n_electrons | n_orbitals | n_qubits |
|---|---|---|---|
| `(2, 5)` | 2 | 5 | **10** ← correct for 10-qubit test |
| `(2, 6)` | 2 | 6 | **12** ← correct for 12-qubit test |
| `(4, 4)` | 4 | 4 | 8  ← **WRONG**, do NOT use |
| `(4, 5)` | 4 | 5 | 10 ← **WRONG** electron count, do NOT use |

Always use `active_space=(2, 5)` for 10 qubits and `active_space=(2, 6)` for 12 qubits.

### Warning 2: Chemical Accuracy Must Be Asserted in Tests
Tests that only **print** or **log** energy error are **invalid**. Every LiH integration test MUST contain an explicit assertion:
```python
assert energy_error < 1.6e-3, f"Chemical accuracy not achieved: {energy_error:.6f} Ha >= 1.6 mHa"
```
If chemical accuracy is not achieved, the test MUST **fail** (not just warn). A test that passes without achieving chemical accuracy does NOT satisfy the acceptance criteria.

## Current Context
You are implementing **complete RLQAS Phase 2** in a single development session. This includes all 6 Phase 2 tasks:
1. **Task 001 Verification**: Verify and integrate existing DQN implementation
2. **Task 002**: Sequential Testing Framework
3. **Task 003**: HEA Search Module
4. **Task 004**: Experiment Management System
5. **Task 005**: Agent Autonomous RL Exploration (Key Innovation)
6. **Task 006**: Phase 2 Integration Test

## Critical Dependencies
**This unified task builds upon two foundations**:
1. **Phase 1 Integrated Package (Phase1/006)**: Required as base quantum architecture search system
2. **Phase 2 Task 001 (../001)**: Already completed DQN implementation - MUST verify and integrate

You MUST:
1. Use Phase 1 as foundation, maintain compatibility with existing APIs
2. Verify and integrate existing Task 001 code before adding new functionality
3. Maintain backward compatibility throughout implementation
4. Follow natural task dependencies as specified in PRD

## Files Available
- `prd.json`: Complete Phase 2 requirements document (all 6 tasks)
- `progress.txt`: Progress log file (create if not exists)
- **Phase 1 Code**: Complete Phase 1 package at `../../Phase1/006/src/rlqas/phase1/`
- **Phase 2 Task 001 Code**: Existing DQN implementation at `../001/src/rlqas/phase2/`
- **Task Specifications**: Detailed requirements in `../../../ideas_pool/RLQAS_Phase2_Tasks.md`

## Overall Execution Strategy

### Phase Order (MUST FOLLOW THIS SEQUENCE)
1. **Phase 0**: Verify and integrate Task 001 foundation
2. **Phase 1**: Implement Task 002 (Sequential Testing Framework)
3. **Phase 2**: Implement Task 003 (HEA Search Module)
4. **Phase 3**: Implement Task 004 (Experiment Management System)
5. **Phase 4**: Implement Task 005 (Agent Autonomous RL Exploration)
6. **Phase 5**: Implement Task 006 (Phase 2 Integration Test)

### Quality Gates
- Each phase must pass ALL acceptance criteria before proceeding to next phase
- Update `progress.txt` after each phase completion
- Maintain >90% test coverage throughout development
- Signal completion with `<promise>COMPLETE</promise>` when ALL phases done

## Phase 0: Task 001 Verification and Integration

### Objective
Verify existing Phase 2 Task 001 (DQN implementation) and integrate it into unified Phase 2 package structure.

### Steps
1. **Examine Existing Code**: Review `../001/src/rlqas/phase2/` structure
2. **Run Tests**: Execute `../001/tests/` to verify DQNAgent and AgentFactory work
3. **Integrate into Unified Structure**: Copy relevant code to `src/rlqas/phase2/`
4. **Verify Phase 1 Compatibility**: Test integration with Phase 1 components

### Acceptance Criteria
- [ ] DQNAgent implements all RLAgent interface methods correctly
- [ ] AgentFactory supports both PPO and DQN agent creation
- [ ] All Phase 2 Task 001 tests pass with >90% coverage
- [ ] Code properly integrated into unified Phase 2 package structure

### Files to Create/Verify
```
src/rlqas/phase2/
    rl/
        dqn_agent.py       # From Task 001
        agent_factory.py   # From Task 001
        __init__.py        # Export RL components
    __init__.py            # Main package export
```

## Phase 1: Task 002 - Sequential Testing Framework

### Objective
Implement Sequential Testing Framework for comparing multiple RL algorithms on quantum architecture search problems.

### Key Requirements (from RLQAS_Phase2_Tasks.md)
1. Implement `SequentialRLTester` class for managing sequential tests
2. Create result comparison and analysis utilities
3. Support configurable test sequences with different agents
4. Implement standardized metrics collection across algorithms
5. Generate comparison reports showing:
   - Energy convergence curves for each algorithm
   - Number of excitation operators used to reach chemical accuracy
   - Training efficiency (episodes to convergence)
   - Final energy error comparison

### Implementation Steps
1. **Create Sequential Testing Module**:
   - `src/rlqas/phase2/sequential_tester/sequential_tester.py`
   - `src/rlqas/phase2/sequential_tester/comparison.py`
2. **Integrate with Existing Agents**: Use AgentFactory from Task 001
3. **Add Metrics Collection**: Track excitation operator counts and energy accuracy
4. **Create Reporting System**: Generate comparison reports

### Acceptance Criteria
- [ ] SequentialRLTester can run tests for multiple agents in sequence
- [ ] Results from different algorithms are collected and stored
- [ ] Comparison utilities generate meaningful performance comparisons
- [ ] Framework tracks excitation operator count for each algorithm
- [ ] Framework identifies which algorithm achieves chemical accuracy with fewer operators
- [ ] All tests pass with >90% coverage

### Files to Create
```
src/rlqas/phase2/sequential_tester/
    __init__.py
    sequential_tester.py   # SequentialRLTester class
    comparison.py          # Comparison utilities
    metrics.py             # Metrics collection

tests/
    test_sequential_tester.py
    test_algorithm_comparison.py
```

## Phase 2: Task 003 - HEA Search Module

### Objective
Implement Hardware Efficient Ansatz (HEA) search environment and controller with configurable entanglement patterns.

### Key Requirements
1. Implement `HEASearchEnv` class (gym.Env compatible)
2. Create `HEACircuitBuilder` for constructing HEA circuits
3. Support multiple entanglement patterns (linear, circular, fully connected)
4. Support configurable rotation gate types
5. Implement `HEASearchController` for managing HEA search process
6. Support parameter sharing strategies (layer-wise, global, none)

### Implementation Steps
1. **Create HEA Circuit Building System**:
   - `src/rlqas/phase2/hea_search/circuit_builder.py`
   - Implement all entanglement patterns
2. **Create HEA Search Environment**:
   - `src/rlqas/phase2/hea_search/environment.py`
   - Compatible with RLAgent interface
3. **Create HEA Controller**:
   - `src/rlqas/phase2/hea_search/controller.py`
   - Integrate with existing search infrastructure
4. **Add Configuration Support**:
   - `src/rlqas/phase2/hea_search/config.py`

### Acceptance Criteria
- [ ] HEASearchEnv implements gym.Env interface correctly
- [ ] HEACircuitBuilder can build circuits with different entanglement patterns
- [ ] All specified entanglement patterns are supported
- [ ] HEASearchController can run complete HEA search episodes
- [ ] Full pipeline works end-to-end with test molecules
- [ ] All tests pass with >90% coverage

### Files to Create
```
src/rlqas/phase2/hea_search/
    __init__.py
    environment.py         # HEASearchEnv
    circuit_builder.py     # HEACircuitBuilder
    controller.py          # HEASearchController
    config.py              # HEA configuration

tests/
    test_hea_environment.py
    test_hea_circuit_builder.py
    test_hea_controller.py
```

## Phase 3: Task 004 - Experiment Management System

### Objective
Implement configuration-driven experiment management system with result database and batch execution support.

### Key Requirements
1. Implement `ExperimentManager` class for managing experiment lifecycle
2. Support YAML and JSON configuration file formats
3. Implement configuration validation and loading
4. Create result database for storing experimental results
5. Support batch experiment execution
6. Implement standardized logging and checkpointing

### Implementation Steps
1. **Create Experiment Manager**:
   - `src/rlqas/phase2/experiment/manager.py`
   - Support YAML/JSON configuration
2. **Create Configuration System**:
   - `src/rlqas/phase2/experiment/config_loader.py`
   - Validation and parsing
3. **Create Results Database**:
   - `src/rlqas/phase2/experiment/results_db.py`
   - Store and query experiment results
4. **Integrate with Existing Systems**:
   - Connect to sequential tester (Task 002)
   - Connect to HEA search (Task 003)

### Acceptance Criteria
- [ ] ExperimentManager can load and validate YAML/JSON configurations
- [ ] Experiments can be executed from configuration files
- [ ] Results are properly collected and stored in results database
- [ ] Batch experiment execution works correctly
- [ ] Logging and checkpointing function properly
- [ ] All tests pass with >90% coverage

### Files to Create
```
src/rlqas/phase2/experiment/
    __init__.py
    manager.py             # ExperimentManager class
    config_loader.py       # Configuration loading and validation
    results_db.py          # Results database

config/
    experiment_template.yaml  # Example configuration
    hea_default.yaml       # HEA default configuration

tests/
    test_experiment_manager.py
    test_config_loader.py
```

## Phase 4: Task 005 - Agent Autonomous RL Exploration (Key Innovation)

### Objective
Enable autonomous exploration of state-of-the-art RL algorithms and runtime adaptive environment enhancement.

### Part A: RL Algorithm Exploration Framework

#### Key Requirements
1. Create RL Algorithm Exploration Framework that Agent/Ralph can use
2. Define evaluation criteria for new algorithms:
   - Compatibility with discrete/continuous action spaces
   - Sample efficiency
   - Stability in high-dimensional state spaces
   - Ability to handle sparse rewards
3. Implement comparison pipeline for evaluating new algorithms against PPO and DQN
4. Create documentation template for Agent/Ralph to record findings
5. Support rapid prototyping of new algorithm implementations

#### Implementation Steps
1. **Create Exploration Framework**:
   - `src/rlqas/phase2/adaptation/exploration_framework.py`
   - Algorithm discovery and evaluation
2. **Create Evaluation System**:
   - `src/rlqas/phase2/adaptation/evaluation_criteria.py`
   - Algorithm compatibility assessment
3. **Create Comparison Pipeline**:
   - `src/rlqas/phase2/adaptation/comparison_pipeline.py`
   - Compare new algorithms against baselines

### Part B: Runtime Environment Adaptation Framework

#### Key Requirements
1. Implement capability detection system for external dependencies:
   - Detect missing functionality in libraries like Tencirchem
   - Identify required modifications for RLQAS compatibility
2. Create dynamic implementation framework:
   - Template-based feature implementation
   - Automated verification of implemented features
   - Seamless integration with existing code
3. Design adaptive execution flow:
   - Runtime capability checking before critical operations
   - On-demand feature implementation
   - Capability caching and reuse
4. Build capability registry:
   - Record implemented features and their validation status
   - Support capability sharing across experiments
   - Enable capability evolution and improvement

#### Implementation Steps
1. **Create Capability Detection**:
   - `src/rlqas/phase2/adaptation/capability_detector.py`
   - Identify missing functionality
2. **Create Dynamic Implementation System**:
   - `src/rlqas/phase2/adaptation/feature_implementer.py`
   - Template-based feature generation
3. **Create Adaptive Execution Flow**:
   - `src/rlqas/phase2/adaptation/adaptive_executor.py`
   - Runtime adaptation
4. **Create Capability Registry**:
   - `src/rlqas/phase2/adaptation/capability_registry.py`
   - Store implemented capabilities

### Example: Tencirchem Parity Support Enhancement
When RLQAS requires parity transformation but Tencirchem's UCC doesn't support it:
1. **Detection**: process_molecule() requests parity transformation → check tencirchem.ucc supports parity mapping
2. **Implementation**: Load parity_adapter.py.j2 template → generate TencirchemParityAdapter class
3. **Integration**: Replace tencirchem.ucc parity calls with adapter
4. **Caching**: Store adapter in capability registry for future use

### Acceptance Criteria
- [ ] RL Algorithm Exploration Framework is fully implemented and documented
- [ ] Capability detection system identifies missing functionality in external dependencies
- [ ] Dynamic implementation framework generates missing features with templates
- [ ] Adaptive execution flow operates seamlessly at runtime
- [ ] Capability registry maintains records of implemented features
- [ ] Tencirchem parity adapter example demonstrates full lifecycle
- [ ] All adaptation components achieve >90% code coverage

### Files to Create
```
src/rlqas/phase2/adaptation/
    __init__.py
    exploration_framework.py    # RL Algorithm Exploration Framework
    evaluation_criteria.py      # Algorithm evaluation metrics
    comparison_pipeline.py      # New algorithm comparison
    capability_detector.py      # Detect missing functionality
    feature_implementer.py      # Dynamically implement missing features
    adaptive_executor.py        # Runtime adaptive execution flow
    capability_registry.py      # Store and manage implemented capabilities
    templates/
        parity_adapter.py.j2    # Template for parity adapter implementation
        transformer_adapter.py.j2 # Template for transformation adapters

docs/
    rl_algorithm_exploration_guide.md
    runtime_adaptation_guide.md

tests/
    test_exploration_framework.py
    test_capability_detection.py
    test_adaptive_execution.py
```

## Phase 5: Task 006 - Phase 2 Integration Test

### Objective
Create comprehensive integration tests for all Phase 2 functionality on LiH molecules with Jordan-Wigner transformation.

### Molecule Configurations (MANDATORY - Do NOT change these)
```yaml
# LiH 10 qubits: 2 electrons in 5 orbitals, Jordan-Wigner -> 10 qubits
lih_10qubits:
  formula: "LiH"
  bond_length: 1.6
  active_space: [2, 5]   # (n_electrons=2, n_orbitals=5) -> 2*5=10 qubits
  basis_set: "sto-3g"
  transform: "jordan_wigner"

# LiH 12 qubits: 2 electrons in 6 orbitals, Jordan-Wigner -> 12 qubits
lih_12qubits:
  formula: "LiH"
  bond_length: 1.6
  active_space: [2, 6]   # (n_electrons=2, n_orbitals=6) -> 2*6=12 qubits
  basis_set: "sto-3g"
  transform: "jordan_wigner"
```

### Chemical Accuracy Test Pattern (MANDATORY)
Every LiH molecule test MUST use this assertion pattern - tests that only print/log are INVALID:
```python
def test_lih_chemical_accuracy():
    result = run_uccsd_search(molecule="LiH", active_space=(2, 5), transform="jordan_wigner")
    energy_error = abs(result["energy"] - result["fci_energy"])
    # This assert MUST be present - the test must FAIL if chemical accuracy is not met
    assert energy_error < 1.6e-3, (
        f"LiH (2,5) 10-qubit: chemical accuracy NOT achieved. "
        f"Error = {energy_error*1000:.4f} mHa, threshold = 1.6 mHa"
    )
```

### Test Requirements
1. **Multi-Algorithm Test**: Compare PPO, DQN on LiH `active_space=(2,5)` 10 qubits
2. **HEA Test**: Validate HEA search on LiH `active_space=(2,5)` 10 qubits
3. **Scalability Test**: Test LiH `active_space=(2,6)` 12 qubits with both UCC and HEA
4. **Jordan-Wigner Test**: Verify Jordan-Wigner transformation works correctly
5. **Chemical Accuracy Test**: All LiH tests MUST `assert energy_error < 1.6e-3` — not just print
6. **Adaptation Test**: Test runtime environment adaptation

### Implementation Steps
1. **Create Integration Test Scripts**:
   - `scripts/run_phase2_tests.py`
   - `scripts/compare_algorithms.py`
   - `scripts/validate_hea.py`
2. **Create Test Molecules Configuration** using the exact active spaces above:
   - LiH `active_space=(2,5)` → 10 qubits with Jordan-Wigner
   - LiH `active_space=(2,6)` → 12 qubits with Jordan-Wigner
3. **Run Comprehensive Tests** with chemical accuracy assertions:
   - Execute all Phase 2 components together
   - Each LiH test must assert `energy_error < 1.6e-3`
   - Compare algorithm performance
4. **Generate Test Report**:
   - Performance metrics
   - Adaptation logs
   - Recommendations

### Acceptance Criteria
- [ ] Multi-algorithm comparison completes successfully on LiH `active_space=(2,5)` 10 qubits
- [ ] LiH `active_space=(2,5)` 10 qubits: test ASSERTS `energy_error < 1.6e-3 Ha` with Jordan-Wigner — test FAILS if not met
- [ ] LiH `active_space=(2,6)` 12 qubits: test ASSERTS `energy_error < 1.6e-3 Ha` with Jordan-Wigner — test FAILS if not met
- [ ] HEA search: test ASSERTS chemical accuracy on LiH `active_space=(2,5)` — test FAILS if not met
- [ ] Framework correctly identifies which algorithm uses fewer excitation operators
- [ ] Runtime environment adaptation successfully enhances Tencirchem parity support
- [ ] All Phase 2 components integrate correctly and maintain >90% test coverage
- [ ] Comprehensive test report generated with performance benchmarks

### Files to Create
```
scripts/
    run_phase2_tests.py          # Main Phase 2 test script
    compare_algorithms.py        # Algorithm comparison script
    validate_hea.py              # HEA validation script

tests/integration/
    test_phase2_integration.py   # Full Phase 2 integration test
    test_multi_algorithm.py      # Multi-algorithm comparison test
    test_hea_integration.py      # HEA integration test
    test_lih_10qubits.py         # LiH 10 qubits test
    test_lih_12qubits.py         # LiH 12 qubits test

results/phase2_integration/
    algorithm_comparison.json    # Algorithm comparison results
    lih_10qubits_results/        # LiH 10 qubits test results
    lih_12qubits_results/        # LiH 12 qubits test results
    hea_results/                 # HEA search results
```

## Package Structure Strategy

### Unified Phase 2 Package
```
src/rlqas/phase2/              # Main Phase 2 package
    __init__.py                # Export all Phase 2 components
    rl/                        # From Task 001 (DQN + AgentFactory)
        __init__.py
        dqn_agent.py
        agent_factory.py
    sequential_tester/         # Task 002
        __init__.py
        sequential_tester.py
        comparison.py
        metrics.py
    hea_search/                # Task 003
        __init__.py
        environment.py
        circuit_builder.py
        controller.py
        config.py
    experiment/                # Task 004
        __init__.py
        manager.py
        config_loader.py
        results_db.py
    adaptation/                # Task 005 (Key Innovation)
        __init__.py
        exploration_framework.py
        capability_detector.py
        feature_implementer.py
        adaptive_executor.py
        capability_registry.py
        templates/
            parity_adapter.py.j2
            transformer_adapter.py.j2
```

### Import Strategy
- **Phase 2 imports Phase 1**: `from rlqas.phase1.rl.base_agent import RLAgent`
- **Internal Phase 2 imports**: `from rlqas.phase2.rl import DQNAgent, AgentFactory`
- **No circular dependencies**: Maintain clear import hierarchy

## Implementation Guidelines

### Before Starting
1. **Read PRD Thoroughly**: Understand all 6 tasks and their dependencies
2. **Verify Environment**: Check Phase 1 and Task 001 are accessible
3. **Plan Directory Structure**: Create all directories before writing code

### Development Process
1. **Phase-by-Phase**: Complete each phase before moving to next
2. **Test-Driven**: Write tests for each component as you implement
3. **Update Progress**: Document each phase completion in `progress.txt`
4. **Quality Assurance**: Maintain >90% test coverage, PEP8 compliance

### Key Technical Considerations
1. **RLAgent Interface Consistency**: All new agents must implement same interface
2. **Configuration Patterns**: Follow Phase 1 configuration patterns
3. **Checkpoint Compatibility**: Maintain save/load compatibility
4. **UCC Compatibility**: Ensure all components work with UCCSearchEnv
5. **Chemical Accuracy**: All tests must verify <1.6 mHa error

## Progress Tracking

Update `progress.txt` with:
1. **Phase 0 Completion**: Task 001 verification and integration
2. **Phase 1 Completion**: Sequential testing framework implementation
3. **Phase 2 Completion**: HEA search module implementation
4. **Phase 3 Completion**: Experiment management system implementation
5. **Phase 4 Completion**: Autonomous RL exploration framework implementation
6. **Phase 5 Completion**: Integration tests and validation
7. **Overall Completion**: All Phase 2 tasks completed and integrated

## Final Completion Signal

When ALL phases are complete and ALL acceptance criteria are met:
- Output `<promise>COMPLETE</promise>` to signal successful completion
- Ensure `progress.txt` contains detailed completion report
- Verify all tests pass with >90% coverage

**MANDATORY pre-completion checklist**:
- [ ] LiH tests use `active_space=(2, 5)` for 10 qubits (NOT `(4,4)` or `(4,5)`)
- [ ] LiH tests use `active_space=(2, 6)` for 12 qubits (NOT `(4,6)`)
- [ ] All LiH integration tests contain `assert energy_error < 1.6e-3` (hard failure, not just logging)
- [ ] pytest run output shows these LiH tests as PASSED (meaning chemical accuracy was actually achieved)

**Remember**: You are implementing the ENTIRE Phase 2 in a single development session. Follow the phase order, maintain quality standards, and build upon existing foundations.