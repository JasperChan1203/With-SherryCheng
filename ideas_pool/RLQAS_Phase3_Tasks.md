# RLQAS Phase 3 Task Breakdown

## Overview
**Phase 3 Goal**: Implement hybrid architecture search (HEA-UCC fusion), performance optimization (batch evaluation, CI vector tuning, memory management), circuit encoding module, and qubit operator extension for UCC search. Validate on medium-to-large systems (BeH₂, H₄, H₆).
**Total Tasks**: 7 (RLQAS_Phase3_001 to 007)
**Expected Deliverables**:
- Hybrid HEA-UCC architecture search system
- Batch evaluation with measurable performance improvement over Phase 1/2 baselines
- Configurable circuit encoding module (matrix-based primary, others optional)
- Validated performance on BeH₂ (10–14 qubits), H₄ (8 qubits), H₆ (12 qubits)
- Qubit operator search domain extending the fermion operator baseline (Task 007)

## Prerequisites
- **Phase 1 Complete Package** (`src/rlqas/phase1/`): Molecule processing, CI vector simulator, PPO agent, UCC search
- **Phase 2 Complete Package** (`src/rlqas/phase2/`): DQN/A2C/SAC agents, AgentFactory, SequentialRLTester, HEASearchEnv, ExperimentManager, AdaptationFramework

## Task Dependencies
```
RLQAS_Phase3_001 (Hybrid Circuit Builder)
     ↓
RLQAS_Phase3_002 (Hybrid Search Environment)
     ↓
RLQAS_Phase3_003 (Hybrid Search Controller)
     |
     +——————————————————————————+——————————————————————+
     |                          |                      |
RLQAS_Phase3_004           RLQAS_Phase3_005    RLQAS_Phase3_007
(Batch Evaluation &         (Circuit Encoding   (Qubit Operator
 Performance Optimization)   Module)             Extension)
     |                          |                      |
     +——————————————————————————+——————————————————————+
                              ↓
              RLQAS_Phase3_006 (Phase 3 Integration Test)
```

---

## Phase 3 Execution Permissions

### Environment Modification Permission (Inherited from Phase 2)

Ralph is **explicitly permitted** to modify the runtime environment during Phase 3 execution. This includes, but is not limited to:

1. **Installing Python packages**: `pip install <package>` to pull in missing dependencies (e.g. `sb3_contrib`, `openfermion`, `cirq`, etc.)
2. **Patching library source code**: If a third-party library (e.g. Tencirchem, OpenFermion) lacks a required feature or has a bug affecting RLQAS, Ralph may directly edit the installed library's `.py` files to add or fix the functionality. Such patches **must** be documented in `progress.txt` under a "Library Patches" heading, including: which file was patched, what was changed, and why.
3. **Creating wrapper/adapter modules**: Ralph may create `*_adapter.py` shim files that monkey-patch or wrap library APIs without modifying the library itself.
4. **Downloading reference data**: Ralph may fetch molecule geometry files, basis set data, or benchmark datasets from public sources if needed.

**Constraint**: All environment changes must be idempotent (safe to run twice). Document every change in `progress.txt`.

### Token Consumption Tracking

Ralph **must** record accumulated token consumption at the end of each completed Phase/task in `progress.txt`. Use the following format:

```
[TOKEN LOG] Phase 3 Task 001 complete
  Session tokens (this task): input=XXXXX, output=XXXXX
  Session cumulative total:   input=XXXXX, output=XXXXX
```

**How to implement**: The Claude Code environment exposes usage statistics through the session. At the end of each task, Ralph should emit a structured token log line. If running via the Anthropic API directly, capture `response.usage.input_tokens` and `response.usage.output_tokens` from each API call and accumulate them. If running via Claude Code CLI, use the session's built-in usage tracking.

The token log serves as an audit trail for cost estimation and helps identify unexpectedly expensive tasks early.

---

## CRITICAL WARNINGS (Read Before Starting)

### Warning 1: Active Space Conventions for Phase 3 Molecules

Under Jordan-Wigner mapping, `n_qubits = 2 * n_orbitals`.

| Molecule | active_space | n_electrons | n_orbitals | n_qubits |
|----------|-------------|-------------|------------|----------|
| H₄  | `(4, 4)` | 4 | 4 | **8** |
| BeH₂ | `(4, 4)` | 4 | 4 | **8** (minimal) |
| BeH₂ | `(4, 5)` | 4 | 5 | **10** |
| BeH₂ | `(6, 6)` | 6 | 6 | **12** |
| BeH₂ | `(8, 7)` | 8 | 7 | **14** |
| H₆  | `(6, 6)` | 6 | 6 | **12** |

Always verify `n_qubits = molecule_data.n_qubits` after `process_molecule()` — never hardcode qubit counts.

### Warning 2: Chemical Accuracy Assertions are Mandatory
Tests that only **print** or **log** energy errors are **invalid**. Every integration test MUST contain:
```python
assert energy_error < 1.6e-3, (
    f"Chemical accuracy NOT achieved: {energy_error*1000:.4f} mHa >= 1.6 mHa"
)
```
A test that passes without asserting chemical accuracy does NOT satisfy acceptance criteria.

### Warning 3: Performance Baselines Must Be Measured
Performance optimization claims must be backed by actual timing benchmarks comparing Phase 3 batch evaluation against Phase 1/2 single-evaluation loops. Do NOT claim speedup without measurement data stored in results files.

---

## RLQAS_Phase3_001: Hybrid Circuit Builder

### Task Metadata
- **ID**: RLQAS_Phase3_001
- **Priority**: P0 (Foundation for hybrid search)
- **Dependencies**: Phase 1 UCC circuit builder, Phase 2 HEA circuit builder
- **Estimated Complexity**: High
- **Related Spec Section**: 3.6

### Functional Description
Implement the `HybridFusionStrategy` and `HybridCircuitBuilder` that can fuse HEA and UCC sub-circuits according to configurable fusion templates. This is the foundational building block for the hybrid architecture search.

### Specific Requirements
1. Implement `HybridFusionStrategy` class with three fusion modes: `sequential`, `parallel`, `conditional`
2. Implement `HybridCircuitBuilder` that wraps both `UCCCircuitBuilder` (Phase 1) and `HEACircuitBuilder` (Phase 2)
3. Support fusion template specification as ordered list: e.g., `["HEA", "UCC", "HEA"]`
4. Support configurable block sizes: `min_ucc_components`, `max_ucc_components`, `hea_layers_per_block`
5. Validate that fused circuits remain valid quantum circuits (qubit counts match, gate types compatible)
6. Implement circuit serialization for fused circuits (save/load fusion configurations)

### Implementation Details
**File Structure**:
```
src/rlqas/phase3/hybrid_search/
    circuit_builder.py      # HybridCircuitBuilder + HybridFusionStrategy
    __init__.py

tests/
    test_hybrid_circuit_builder.py
```

**Core Interfaces**:
```python
class HybridFusionStrategy:
    def __init__(self, config: Dict = None):
        """
        config example:
        {
            "fusion_mode": "sequential",  # "sequential" | "parallel" | "conditional"
            "min_ucc_components": 1,
            "max_ucc_components": 5,
            "hea_layers_per_block": 2
        }
        """

    def generate_fusion_template(self) -> List[str]:
        """
        Returns ordered list of block types.
        Examples:
          ["HEA", "UCC", "HEA"]          # sequential
          ["HEA_UCC", "HEA_UCC"]         # parallel fusion
          ["HEA", "UCC"] (conditional)   # conditional on energy improvement
        """

    def fuse_circuits(
        self,
        hea_circuit: QuantumCircuit,
        ucc_circuit: QuantumCircuit
    ) -> QuantumCircuit:
        """Compose HEA and UCC sub-circuits according to fusion template"""


class HybridCircuitBuilder:
    def __init__(
        self,
        molecule_data: MoleculeData,
        fusion_strategy: HybridFusionStrategy,
        config: Dict = None
    ):

    def build_block(
        self,
        block_type: str,           # "HEA" | "UCC" | "HEA_UCC"
        block_spec: Dict           # block-specific parameters
    ) -> QuantumCircuit:

    def build_hybrid_circuit(
        self,
        template: List[str],       # e.g. ["HEA", "UCC", "HEA"]
        block_specs: List[Dict]    # one spec dict per template entry
    ) -> QuantumCircuit:

    def save_fusion_config(self, path: str, template: List[str], block_specs: List[Dict]):
        """Save fusion configuration to JSON for reproducibility"""

    def load_fusion_config(self, path: str) -> Tuple[List[str], List[Dict]]:
        """Load saved fusion configuration"""
```

### Test Requirements
- **Unit Tests**: Test each fusion mode independently with mock circuits
- **Integration Tests**: Test `build_hybrid_circuit()` with actual H₂ molecule (2 qubits)
- **Validation Tests**: Verify fused circuits are valid (correct qubit count, gate structure)
- **Round-trip Tests**: Save then load fusion configs, verify reproducibility
- **Coverage**: >85% code coverage

### Acceptance Criteria
- [ ] `HybridFusionStrategy` implements all three fusion modes (`sequential`, `parallel`, `conditional`)
- [ ] `generate_fusion_template()` respects `min_ucc_components` and `max_ucc_components` bounds
- [ ] `HybridCircuitBuilder.build_hybrid_circuit()` produces valid circuits for H₂ and LiH
- [ ] Fusion config save/load round-trip produces identical circuits
- [ ] All unit and integration tests pass with >85% coverage
- [ ] Code follows PEP8; type hints on all public methods

---

## RLQAS_Phase3_002: Hybrid Search Environment

### Task Metadata
- **ID**: RLQAS_Phase3_002
- **Priority**: P0
- **Dependencies**: RLQAS_Phase3_001, Phase 1 `UCCSearchEnv`, Phase 2 `HEASearchEnv`
- **Estimated Complexity**: High
- **Related Spec Section**: 3.6

### Functional Description
Implement `HybridSearchEnv` as a `gym.Env`-compatible environment that allows an RL agent to iteratively build a hybrid HEA-UCC circuit. The action space covers both HEA-layer additions and UCC-excitation additions; the state space encodes the current hybrid circuit structure plus molecular features.

### Specific Requirements
1. Implement `HybridSearchEnv(gym.Env)` combining action spaces from `UCCSearchEnv` and `HEASearchEnv`
2. Support **composite action space**: agent selects (a) block type (HEA/UCC), then (b) block-specific sub-action
3. Encode circuit state as matrix representation (primary) — rows = qubits, columns = time steps, cell values = gate types
4. Include an `encoding_method` config field so alternative encodings (sparse, one-hot) can be swapped in for experimentation
5. Implement `HybridRewardFunction` that balances energy accuracy, circuit depth, gate count, and architecture complexity penalty
6. Support early stopping when chemical accuracy is achieved

### Implementation Details
**File Structure**:
```
src/rlqas/phase3/hybrid_search/
    environment.py          # HybridSearchEnv + HybridRewardFunction
    __init__.py             # update to export environment

tests/
    test_hybrid_environment.py
```

**Core Interfaces**:
```python
class HybridSearchEnv(gym.Env):
    def __init__(
        self,
        molecule_data: MoleculeData,
        fusion_strategy: HybridFusionStrategy,
        config: Dict = None
    ):
        """
        config example:
        {
            "max_depth": 15,
            "max_blocks": 6,               # max number of HEA/UCC blocks
            "encoding_method": "matrix",   # "matrix" | "sparse" | "one_hot"
            "use_sqeb": True,
            "rotation_gates": ["rx", "ry", "rz"],
            "entanglement_patterns": ["linear", "circular"]
        }
        """

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action encodes: block_type (HEA or UCC) + block-specific index
        Returns: (next_state, reward, done, info)
        info contains: {"energy": float, "circuit_depth": int, "n_blocks": int}
        """

    def reset(self) -> np.ndarray:
        """Reset to empty circuit, return initial state vector"""

    def get_circuit(self) -> QuantumCircuit:
        """Return current circuit for external evaluation"""


class HybridRewardFunction:
    def __init__(self, config: Dict = None):
        """
        config example:
        {
            "accuracy_weight": 0.6,
            "depth_weight": 0.2,
            "gate_weight": 0.1,
            "architecture_penalty_weight": 0.1,  # penalize excessive block count
            "use_intermediate_rewards": True
        }
        """

    def compute_reward(
        self,
        circuit: QuantumCircuit,
        energy: float,
        fci_energy: float,
        step_info: Dict
    ) -> float:
        """
        total_reward = w_acc * accuracy_reward
                     + w_depth * depth_penalty
                     + w_gate * gate_penalty
                     + w_arch * architecture_complexity_penalty
        """
```

### Test Requirements
- **Interface Tests**: Verify `gym.Env` compliance (observation_space, action_space, step, reset)
- **State Encoding Tests**: Verify matrix encoding dimensions match `(n_qubits, max_depth)` for different molecules
- **Reward Tests**: Verify reward values are finite and in reasonable range for LiH
- **Episode Tests**: Run 10 complete episodes on H₂; verify no crashes, reward trends positive
- **Coverage**: >85% code coverage

### Acceptance Criteria
- [ ] `HybridSearchEnv` passes `gym.utils.env_checker` (or equivalent interface validation)
- [ ] Matrix encoding produces state vector of consistent dimension across episodes
- [ ] `encoding_method` config field successfully switches between matrix and sparse representations
- [ ] `HybridRewardFunction` produces finite, non-NaN rewards for valid circuits
- [ ] 10 complete random-policy episodes on H₂ complete without errors
- [ ] All tests pass with >85% coverage

---

## RLQAS_Phase3_003: Hybrid Search Controller

### Task Metadata
- **ID**: RLQAS_Phase3_003
- **Priority**: P0
- **Dependencies**: RLQAS_Phase3_001, RLQAS_Phase3_002, Phase 2 `AgentFactory`
- **Estimated Complexity**: Medium
- **Related Spec Section**: 3.6

### Functional Description
Implement `HybridSearchController` that orchestrates the complete hybrid architecture search. Mirrors the design of Phase 1 `UCCSearchController` and Phase 2 `HEASearchController` for API consistency. Integrates with Phase 2 `ExperimentManager` and `SequentialRLTester`.

### Specific Requirements
1. Implement `HybridSearchController` with the same interface pattern as `UCCSearchController`
2. Accept any `RLAgent` via `AgentFactory` (support PPO, DQN, A2C, SAC)
3. Implement search loop with configurable early stopping (chemical accuracy threshold)
4. Track and record best circuit found across all episodes
5. Integrate with Phase 2 `ExperimentManager` configuration schema (add `"HYBRID"` as a valid `ansatz_type`)
6. Implement result serialization compatible with Phase 2 `ResultsDatabase`

### Implementation Details
**File Structure**:
```
src/rlqas/phase3/hybrid_search/
    controller.py           # HybridSearchController
    config.py               # HybridSearchConfig dataclass
    __init__.py             # update exports

tests/
    test_hybrid_controller.py
```

**Core Interfaces**:
```python
class HybridSearchController:
    def __init__(
        self,
        molecule_data: MoleculeData,
        agent_type: str = "ppo",
        config: Dict = None
    ):
        self.env = HybridSearchEnv(molecule_data, ...)
        self.agent = AgentFactory.create_agent(
            agent_type,
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            config.get("agent_config", {}) if config else {}
        )

    def search(
        self,
        n_episodes: int = 1000,
        early_stop_threshold: float = 1.6e-3
    ) -> SearchResult:
        """
        Returns SearchResult containing:
          best_circuit: QuantumCircuit
          best_energy: float
          best_error: float
          training_history: List[Dict]  # per-episode metrics
          performance_metrics: Dict
          fusion_template: List[str]    # template of best circuit
        """

    def save_results(self, path: str):
        """Save search results to JSON; compatible with ResultsDatabase schema"""

    @classmethod
    def from_config(cls, molecule_data: MoleculeData, config: Dict) -> "HybridSearchController":
        """Instantiate from ExperimentManager config dict"""
```

**ExperimentManager integration** — extend Phase 2 `manager.py` to handle `ansatz_type == "HYBRID"`:
```python
# In Phase 2 experiment/manager.py run_experiment():
elif self.config["search"]["ansatz_type"] == "HYBRID":
    search_module = HybridSearchController(
        molecule_data,
        self.config["rl"]["agent_type"],
        self.config
    )
```

### Test Requirements
- **Unit Tests**: Test search loop with mock environment and agent
- **Integration Tests**: Run 50-episode search on LiH (4 qubits) with PPO; verify `SearchResult` fields are populated
- **API Consistency Tests**: Verify `HybridSearchController.search()` returns same `SearchResult` type as `UCCSearchController`
- **Serialization Tests**: Save then reload results; verify identical data
- **Coverage**: >85% code coverage

### Acceptance Criteria
- [ ] `HybridSearchController.search()` runs without error on LiH (4 qubits) for 50 episodes
- [ ] `SearchResult` contains `fusion_template` field identifying block sequence of best circuit
- [ ] `from_config()` correctly instantiates controller from ExperimentManager config dict
- [ ] Phase 2 `ExperimentManager` correctly dispatches to `HybridSearchController` for `ansatz_type="HYBRID"`
- [ ] Result JSON is loadable by Phase 2 `ResultsDatabase`
- [ ] All tests pass with >85% coverage

---

## RLQAS_Phase3_004: Batch Evaluation & Performance Optimization

### Task Metadata
- **ID**: RLQAS_Phase3_004
- **Priority**: P1
- **Dependencies**: Phase 1 `TencirchemCISimulator`, RLQAS_Phase3_003 (for integration)
- **Estimated Complexity**: High
- **Related Spec Section**: 3.2 (Performance Requirements), 8.1 (Performance Optimization)

### Functional Description
Implement a `BatchEvaluator` that evaluates multiple quantum circuits in a single call, replacing the per-step `compute_energy()` call pattern used in Phase 1/2 training loops. Also implement systematic CI vector performance benchmarking and memory usage optimization utilities.

### Specific Requirements

#### Part A: Batch Evaluator
1. Implement `BatchEvaluator` that accepts a list of circuits and returns a list of energies
2. Support configurable batch size (default 16, tunable based on available memory)
3. Implement internal queue-based batching: during RL training, collect `batch_size` pending evaluations before dispatching to CI vector engine
4. Provide `BatchEvaluatorConfig` dataclass with all tunable parameters
5. Maintain drop-in compatibility: wrap existing `QuantumSimulator.compute_energy()` so search environments can switch to batch mode with a config flag

#### Part B: CI Vector Performance Benchmarking
1. Implement `CIVectorBenchmark` that measures actual energy evaluation times across qubit counts (4, 6, 8, 10, 12, 14 qubits)
2. Record benchmark results to JSON for use by `SimulatorFactory` auto-selection logic
3. Update `SimulatorFactory` to read benchmark results and choose optimal strategy per qubit count
4. Implement checkpoint system: if a CI vector job exceeds time limit, save partial results and resume

#### Part C: Memory Optimization
1. Implement `MemoryManager` that monitors current memory usage during training
2. Support automatic reduction of `batch_size` or circuit complexity when memory threshold is approached
3. Provide utilities for releasing intermediate computation graphs after each batch

### Implementation Details
**File Structure**:
```
src/rlqas/phase3/performance/
    __init__.py
    batch_evaluator.py      # BatchEvaluator + BatchEvaluatorConfig
    benchmarking.py         # CIVectorBenchmark
    memory_manager.py       # MemoryManager
    checkpoint.py           # Checkpoint save/load for long-running jobs

tests/
    test_batch_evaluator.py
    test_benchmarking.py
    test_memory_manager.py

benchmarks/
    ci_vector_benchmark_results.json   # Pre-recorded results (generated by benchmarking.py)
```

**Core Interfaces**:
```python
@dataclass
class BatchEvaluatorConfig:
    batch_size: int = 16
    max_memory_gb: float = 32.0
    timeout_per_eval_ms: float = 200.0
    use_async: bool = False            # future extension; False in Phase 3

class BatchEvaluator:
    def __init__(
        self,
        simulator: QuantumSimulator,
        config: BatchEvaluatorConfig = None
    ):

    def evaluate_batch(
        self,
        circuits: List[QuantumCircuit],
        hamiltonian: QubitOperator,
        initial_states: Optional[List[np.ndarray]] = None
    ) -> List[float]:
        """Evaluate a list of circuits; returns list of energies in same order"""

    def evaluate_single(
        self,
        circuit: QuantumCircuit,
        hamiltonian: QubitOperator,
        initial_state: Optional[np.ndarray] = None
    ) -> float:
        """Drop-in replacement for QuantumSimulator.compute_energy()"""


class CIVectorBenchmark:
    def run_benchmark(
        self,
        qubit_counts: List[int],
        n_trials: int = 5
    ) -> Dict[int, Dict]:
        """
        Returns:
        {
          8: {"mean_ms": 45.2, "std_ms": 3.1, "max_ms": 52.0},
          10: {"mean_ms": 180.5, ...},
          ...
        }
        """

    def save_results(self, path: str):
        """Save benchmark JSON for SimulatorFactory to read"""

    def load_results(self, path: str) -> Dict:
        """Load previously recorded benchmark results"""
```

### Performance Targets
- **Batch evaluation speedup**: ≥1.5× throughput vs. sequential evaluation (measured on 8-qubit circuits, batch_size=16)
- **Single energy evaluation**: <100ms for 8-qubit circuits (carried over from Phase 1 spec)
- **Memory usage**: stays within `max_memory_gb` limit during any training run

### Test Requirements
- **Throughput Tests**: Measure batch vs. sequential throughput on 8-qubit circuits; assert ≥1.5× speedup
- **Correctness Tests**: Verify batch results match single-call results within `1e-8` tolerance
- **Memory Tests**: Verify `MemoryManager` triggers batch size reduction before OOM
- **Benchmark Tests**: Run `CIVectorBenchmark` on 4–8 qubits; verify JSON output format
- **Coverage**: >80% code coverage

### Acceptance Criteria
- [ ] `BatchEvaluator.evaluate_batch()` produces energies matching individual `compute_energy()` calls within `1e-8`
- [ ] Batch evaluation achieves ≥1.5× throughput vs. sequential on 8-qubit circuits (documented in test output)
- [ ] `CIVectorBenchmark` runs successfully for 4–12 qubits; results saved to JSON
- [ ] `SimulatorFactory` reads benchmark JSON and selects appropriate engine
- [ ] `MemoryManager` prevents OOM by adapting batch size (tested with memory limit mock)
- [ ] Checkpoint save/load preserves training state across resume
- [ ] All tests pass with >80% coverage

---

## RLQAS_Phase3_005: Circuit Encoding Module

### Task Metadata
- **ID**: RLQAS_Phase3_005
- **Priority**: P1
- **Dependencies**: Phase 1 `UCCSearchEnv`, Phase 2 `HEASearchEnv`, RLQAS_Phase3_002
- **Estimated Complexity**: Medium
- **Related Spec Section**: 3.2 (Encoding Method Support), 3.4 (State space encoding)

### Functional Description
Implement a standalone `CircuitEncoder` module that converts a `QuantumCircuit` object into a fixed-size state vector for RL consumption. The primary encoding is matrix-based; additional encodings (sparse, one-hot) are provided for experimental comparison. All search environments (UCC, HEA, Hybrid) should delegate state construction to this module.

### Specific Requirements
1. Implement `CircuitEncoder` abstract base class and three concrete implementations:
   - `MatrixEncoder` (primary): gate-type matrix of shape `(n_qubits, max_depth)`
   - `SparseEncoder`: COO sparse representation flattened to fixed-size dense vector
   - `OneHotEncoder`: one-hot gate type encoding per gate position
2. Implement `EncoderFactory` for creating encoders by name
3. Add `encoding_method` configuration to `UCCSearchEnv`, `HEASearchEnv`, and `HybridSearchEnv` so each can delegate to `CircuitEncoder`
4. Provide a `EncodingBenchmark` utility that measures encoding time and resulting vector size for each method
5. Document encoding format so RL agents can interpret state dimensions

### Implementation Details
**File Structure**:
```
src/rlqas/phase3/encoding/
    __init__.py
    base_encoder.py         # CircuitEncoder ABC
    matrix_encoder.py       # MatrixEncoder (primary)
    sparse_encoder.py       # SparseEncoder
    one_hot_encoder.py      # OneHotEncoder
    encoder_factory.py      # EncoderFactory
    benchmark.py            # EncodingBenchmark

tests/
    test_matrix_encoder.py
    test_sparse_encoder.py
    test_encoder_factory.py
    test_encoding_benchmark.py
```

**Core Interfaces**:
```python
class CircuitEncoder(ABC):
    @abstractmethod
    def encode(self, circuit: QuantumCircuit, n_qubits: int, max_depth: int) -> np.ndarray:
        """
        Returns fixed-size 1D numpy array representing the circuit.
        Output dimension must be deterministic given (n_qubits, max_depth).
        """

    @abstractmethod
    def output_dim(self, n_qubits: int, max_depth: int) -> int:
        """Return the length of the encoded vector"""


class MatrixEncoder(CircuitEncoder):
    """
    Gate-type matrix encoding.
    Cell (q, t) = integer gate type index for gate on qubit q at time step t.
    0 = identity (empty slot).
    Flattened row-major to 1D vector of length n_qubits * max_depth.
    """

class SparseEncoder(CircuitEncoder):
    """
    Encode only non-identity gates as (qubit_idx, time_step, gate_type) triples.
    Pad to fixed length max_gates * 3 for fixed output dimension.
    """

class EncoderFactory:
    @staticmethod
    def create(
        encoding_method: str,    # "matrix" | "sparse" | "one_hot"
        config: Dict = None
    ) -> CircuitEncoder:

class EncodingBenchmark:
    def run(
        self,
        circuits: List[QuantumCircuit],
        n_qubits: int,
        max_depth: int
    ) -> Dict[str, Dict]:
        """
        Returns per-encoder timing and output size:
        {
          "matrix": {"mean_us": 12.3, "output_dim": 120},
          "sparse": {"mean_us": 8.1,  "output_dim": 90},
          "one_hot": {"mean_us": 25.0, "output_dim": 600}
        }
        """
```

**Integration with existing environments** — after this task, update Phase 1 `UCCSearchEnv` and Phase 2 `HEASearchEnv`:
```python
# In UCCSearchEnv.__init__():
encoding_method = config.get("encoding_method", "matrix")
self.encoder = EncoderFactory.create(encoding_method)

# In UCCSearchEnv._get_state():
circuit_vec = self.encoder.encode(self.current_circuit, self.n_qubits, self.max_depth)
state = np.concatenate([circuit_vec, energy_features, resource_features, mol_features])
```

### Test Requirements
- **Correctness Tests**: Verify `MatrixEncoder` output matches hand-crafted expected matrix for simple 2-qubit circuits
- **Dimension Tests**: Verify `output_dim()` matches actual `encode()` output length for all encoders
- **Round-trip Tests**: Verify two identical circuits always produce identical encodings (determinism)
- **Benchmark Tests**: Run `EncodingBenchmark` on 10 random circuits; verify results dict format
- **Integration Tests**: Verify `UCCSearchEnv` with `encoding_method="sparse"` still runs 5 episodes without error
- **Coverage**: >85% code coverage

### Acceptance Criteria
- [ ] `MatrixEncoder`, `SparseEncoder`, `OneHotEncoder` all implement `CircuitEncoder` interface
- [ ] `output_dim()` always matches actual `encode()` output length
- [ ] Encoding is deterministic: same circuit always produces same vector
- [ ] `EncoderFactory` creates all three encoder types by name
- [ ] `UCCSearchEnv`, `HEASearchEnv`, `HybridSearchEnv` all accept `encoding_method` config and delegate to `CircuitEncoder`
- [ ] `EncodingBenchmark` records timing and output sizes for all three encoders
- [ ] All tests pass with >85% coverage

---

## RLQAS_Phase3_006: Phase 3 Integration Test

### Task Metadata
- **ID**: RLQAS_Phase3_006
- **Priority**: P0 (Validation)
- **Dependencies**: RLQAS_Phase3_001 through 005
- **Estimated Complexity**: Medium
- **Related Spec Section**: 5.1, 5.2, 5.3, 6.1

### Functional Description
Create comprehensive integration tests validating all Phase 3 components on medium-to-large molecular systems. Covers hybrid architecture search, batch evaluation performance, and circuit encoding on BeH₂ (8–14 qubits), H₄ (8 qubits), and H₆ (12 qubits).

### Molecule Configurations (MANDATORY — Do NOT change these)
```python
# BeH₂ scalability tests (Jordan-Wigner)
beh2_8qubits = {
    "formula": "BeH2", "bond_length": 1.3,
    "active_space": (4, 4),           # 4 electrons, 4 orbitals -> 8 qubits
    "basis_set": "sto-3g", "transform": "jordan_wigner"
}
beh2_10qubits = {
    "formula": "BeH2", "bond_length": 1.3,
    "active_space": (4, 5),           # 4 electrons, 5 orbitals -> 10 qubits
    "basis_set": "sto-3g", "transform": "jordan_wigner"
}
beh2_12qubits = {
    "formula": "BeH2", "bond_length": 1.3,
    "active_space": (6, 6),           # 6 electrons, 6 orbitals -> 12 qubits
    "basis_set": "sto-3g", "transform": "jordan_wigner"
}
beh2_14qubits = {
    "formula": "BeH2", "bond_length": 1.3,
    "active_space": (8, 7),           # 8 electrons, 7 orbitals -> 14 qubits
    "basis_set": "sto-3g", "transform": "jordan_wigner"
}

# H₄ hydrogen chain (Jordan-Wigner)
h4_8qubits = {
    "formula": "H4", "geometry": "linear", "bond_length": 0.74,
    "active_space": (4, 4),           # 4 electrons, 4 orbitals -> 8 qubits
    "basis_set": "sto-3g", "transform": "jordan_wigner"
}

# H₆ hydrogen chain (Jordan-Wigner) — full space, no active_space restriction
# H₆ with STO-3G has 6 electrons in 6 orbitals naturally; (6,6) IS the full space.
# Do NOT pass active_space to process_molecule — use the full orbital space.
h6_12qubits = {
    "formula": "H6", "geometry": "linear", "bond_length": 0.74,
    "basis_set": "sto-3g", "transform": "jordan_wigner"
    # no active_space key — full space
}
```

### Chemical Accuracy Assertion Pattern (MANDATORY for all molecule tests)
```python
def test_beh2_8qubits_chemical_accuracy():
    result = run_hybrid_search(molecule_config=beh2_8qubits, n_episodes=500)
    energy_error = abs(result["energy"] - result["fci_energy"])
    assert energy_error < 1.6e-3, (
        f"BeH₂ (4,4) 8-qubit: chemical accuracy NOT achieved. "
        f"Error = {energy_error*1000:.4f} mHa, threshold = 1.6 mHa"
    )
```

### Test Suite Overview

#### Test 1: Hybrid Architecture Search on LiH (Smoke Test)
- Run `HybridSearchController` with PPO on LiH (4 qubits) for 100 episodes
- Verify `SearchResult.fusion_template` is populated
- Assert `energy_error < 1.6e-3`

#### Test 2: Multi-Algorithm Hybrid Search on BeH₂ 8 qubits
- Use `SequentialRLTester` with PPO + DQN on BeH₂ `active_space=(4,4)` → 8 qubits
- Assert chemical accuracy for both algorithms
- Record which algorithm achieves chemical accuracy with fewer excitation operators

#### Test 3: Batch Evaluation Performance
- Compare batch vs. sequential evaluation on BeH₂ 8-qubit circuits
- Assert batch throughput ≥ 1.5× sequential throughput (measured, not estimated)
- Record timing results to `results/phase3_integration/batch_benchmark.json`

#### Test 4: Circuit Encoding Comparison
- Run 50 episodes of `HybridSearchEnv` on LiH with each encoding method (matrix, sparse, one_hot)
- Assert all three reach same energy within `1e-6` tolerance after same random seed
- Record encoding timings to `results/phase3_integration/encoding_benchmark.json`

#### Test 5: Scalability — BeH₂ 10 and 12 qubits
- Run UCC search (Phase 1 controller) on BeH₂ `active_space=(4,5)` → 10 qubits, assert `energy_error < 1.6e-3`
- Run Hybrid search on BeH₂ `active_space=(6,6)` → 12 qubits, assert `energy_error < 1.6e-3`

#### Test 6: Hydrogen Chain Correlation Effects
- Run UCC search on H₄ `active_space=(4,4)` → 8 qubits; assert `energy_error < 1.6e-3`
- Run Hybrid search on H₆ **full space** (no `active_space` restriction; STO-3G naturally gives 6 electrons, 6 orbitals → 12 qubits); assert `energy_error < 1.6e-3`

#### Test 7: ExperimentManager End-to-End with HYBRID ansatz
- Run `ExperimentManager` with a YAML config specifying `ansatz_type: "HYBRID"` on BeH₂ 8 qubits
- Assert `ExperimentResult` is returned with populated `analysis` field

### Implementation Details
**File Structure**:
```
scripts/
    run_phase3_tests.py          # Main Phase 3 test runner
    compare_hybrid_vs_ucc.py     # Hybrid vs UCC performance comparison
    run_scalability_tests.py     # BeH₂ 10–14 qubit scalability

tests/integration/
    test_phase3_integration.py   # Full Phase 3 integration suite
    test_hybrid_search.py        # Hybrid search component tests
    test_batch_performance.py    # Batch evaluation performance tests
    test_encoding_comparison.py  # Encoding method comparison tests
    test_beh2_scalability.py     # BeH₂ 8–14 qubit tests
    test_hydrogen_chain.py       # H₄, H₆ correlation tests

config/
    phase3_beh2_experiment.yaml  # ExperimentManager config for BeH₂ HYBRID
    phase3_h4_experiment.yaml    # ExperimentManager config for H₄

results/phase3_integration/
    algorithm_comparison.json    # PPO vs DQN hybrid results
    batch_benchmark.json         # Batch vs sequential timing
    encoding_benchmark.json      # Encoding method comparison
    beh2_scalability/            # BeH₂ 8–14 qubit results
    hydrogen_chain/              # H₄, H₆ results
```

**Example ExperimentManager YAML for Phase 3**:
```yaml
experiment:
  name: "phase3_beh2_hybrid"
  description: "BeH₂ hybrid architecture search - Phase 3 integration"

molecule:
  formula: "BeH2"
  bond_length: 1.3
  active_space: [4, 4]
  basis_set: "sto-3g"
  transform: "jordan_wigner"

search:
  ansatz_type: "HYBRID"
  max_depth: 15
  max_blocks: 6
  fusion_mode: "sequential"
  encoding_method: "matrix"

rl:
  agent_type: "ppo"
  n_episodes: 500

simulation:
  engine: "ci_vector"
  precision: 1e-8
  max_memory_gb: 32
  use_batch_evaluator: true
  batch_size: 16

evaluation:
  n_repeats: 3
  metrics_to_collect:
    - "energy_error"
    - "circuit_depth"
    - "n_blocks"
    - "fusion_template"
    - "training_time"

output:
  directory: "./results/phase3_integration/beh2_hybrid"
  save_circuits: true
  save_training_logs: true
```

### Acceptance Criteria
- [ ] LiH (4 qubits) smoke test: `HybridSearchController` runs 100 episodes; asserts `energy_error < 1.6e-3`
- [ ] BeH₂ 8-qubit: PPO and DQN both achieve `energy_error < 1.6e-3`; which uses fewer operators is recorded
- [ ] Batch evaluation: ≥1.5× throughput measured and asserted on BeH₂ 8-qubit circuits
- [ ] Circuit encoding: all three methods produce identical energies (within `1e-6`) under same random seed
- [ ] BeH₂ 10-qubit: asserts `energy_error < 1.6e-3` with Jordan-Wigner
- [ ] BeH₂ 12-qubit: asserts `energy_error < 1.6e-3` with Jordan-Wigner
- [ ] H₄ 8-qubit: asserts `energy_error < 1.6e-3`
- [ ] H₆ 12-qubit: asserts `energy_error < 1.6e-3` (full space, no active_space restriction)
- [ ] `ExperimentManager` end-to-end with `ansatz_type="HYBRID"` YAML config completes without error
- [ ] All integration tests in `tests/integration/` pass; benchmark JSON files generated in `results/phase3_integration/`
- [ ] `progress.txt` updated with Phase 3 completion summary

---

## RLQAS_Phase3_007: Qubit Operator Extension for UCC Search

### Task Metadata
- **ID**: RLQAS_Phase3_007
- **Priority**: P1
- **Dependencies**: RLQAS_Phase3_001, RLQAS_Phase3_002, RLQAS_Phase3_003
- **Estimated Complexity**: Medium
- **Related Spec Section**: 3.6 (UCC search extensibility)

### Background

Tencirchem's UCC class supports two types of excitation operators:
- **Fermion operators** (default): Excitation operators defined in Fock space, e.g. `a†_p a_q`. These are the operators used in Phase 1/2 search.
- **Qubit operators**: Pauli-string excitations defined directly in qubit space, e.g. `X_0 Y_1 - Y_0 X_1`. These can provide a different expressivity profile and may achieve lower circuit depth for certain molecules.

The current search domain in `UCCSearchEnv` and `HybridSearchEnv` exclusively uses fermion operators. This task extends the framework to optionally use qubit operators as the action space, enabling head-to-head comparison.

### Functional Description

Implement a `QubitOperatorPool` class that generates a qubit-space operator pool for a given molecule, and integrate it as an alternative action space in `UCCSearchEnv` and `HybridSearchEnv`. Add a `QubitUCCSearchController` that mirrors `UCCSearchController` but operates over the qubit operator pool.

### Specific Requirements

1. **Investigate Tencirchem qubit operator API**: Inspect `tencirchem.ucc` to identify the correct API for specifying qubit-space operators (look for `QubitUCC`, qubit excitation lists, or Pauli string inputs). Document findings in `progress.txt` under "Qubit Operator API Investigation". If Tencirchem does not natively support qubit operators, implement an adapter.

2. **Implement `QubitOperatorPool`**: Generate the full qubit operator pool for a given `MoleculeData`. The pool should contain all single and double Pauli-string excitations compatible with the molecular Hamiltonian.

3. **Extend `UCCSearchEnv`**: Add `operator_type` config field (`"fermion"` | `"qubit"`, default `"fermion"`). When `"qubit"`, the action space indexes into `QubitOperatorPool` instead of the fermion excitation pool.

4. **Extend `HybridSearchEnv`**: Similarly add `operator_type` config field so the UCC blocks in hybrid circuits can also use qubit operators.

5. **Implement `QubitUCCSearchController`**: Mirrors `UCCSearchController`; uses `operator_type="qubit"` environment config. Accepts any `RLAgent` from `AgentFactory`.

6. **Comparison benchmark**: Run both fermion and qubit operator search on LiH 10-qubit with PPO (300 episodes each). Record which achieves chemical accuracy with fewer operators. Save results to `results/phase3_integration/qubit_vs_fermion_lih_10q.json`.

### Implementation Details

**File Structure**:
```
src/rlqas/phase3/qubit_ops/
    __init__.py
    operator_pool.py        # QubitOperatorPool
    controller.py           # QubitUCCSearchController
    adapter.py              # TencirchemQubitAdapter (if needed)

tests/
    test_qubit_operator_pool.py
    test_qubit_ucc_search.py
```

**Core Interfaces**:
```python
class QubitOperatorPool:
    def __init__(self, molecule_data: MoleculeData, config: Dict = None):
        """
        Builds qubit-space excitation operator pool.
        config example:
        {
            "excitation_level": "sd",       # "s" | "d" | "sd" (singles+doubles)
            "symmetry_filter": True,         # filter by particle-number symmetry
            "max_operators": 100             # cap pool size
        }
        """

    def get_pool(self) -> List[Any]:
        """Return list of qubit operators (Pauli strings or Tencirchem-compatible objects)"""

    def get_pool_size(self) -> int:
        """Return number of operators in pool"""

    def operator_to_circuit(
        self,
        op_index: int,
        n_qubits: int
    ) -> QuantumCircuit:
        """Convert a pool operator to a QuantumCircuit block"""


class QubitUCCSearchController:
    def __init__(
        self,
        molecule_data: MoleculeData,
        agent_type: str = "ppo",
        config: Dict = None
    ):
        """
        Identical interface to UCCSearchController, but
        uses operator_type="qubit" in environment config.
        """

    def search(
        self,
        n_episodes: int = 500,
        early_stop_threshold: float = 1.6e-3
    ) -> SearchResult:
        """Same return type as UCCSearchController.search()"""
```

**Integration with existing environments**:
```python
# In UCCSearchEnv.__init__():
operator_type = config.get("operator_type", "fermion")
if operator_type == "qubit":
    self.operator_pool = QubitOperatorPool(molecule_data, config)
else:
    self.operator_pool = FermionOperatorPool(molecule_data, config)  # existing

self.action_space = spaces.Discrete(self.operator_pool.get_pool_size())
```

### Test Requirements
- **API Investigation Tests**: Verify `QubitOperatorPool` can be instantiated for H₂ and LiH without error
- **Pool Size Tests**: Verify pool size is non-zero and bounded by `max_operators`
- **Circuit Construction Tests**: Verify each operator converts to a valid `QuantumCircuit` with correct qubit count
- **Environment Tests**: Verify `UCCSearchEnv(operator_type="qubit")` passes `gym.Env` compliance check
- **Energy Tests**: Verify qubit operator environment returns physically meaningful energies (below Hartree-Fock) for LiH
- **Comparison Tests**: Run fermion vs qubit search on LiH 10q; record results (no chemical accuracy assertion required — comparison is the goal)
- **Coverage**: >80% code coverage

### Acceptance Criteria
- [ ] `QubitOperatorPool` generates a non-empty qubit operator pool for H₂ (≥1 operator) and LiH (≥5 operators)
- [ ] `UCCSearchEnv` with `operator_type="qubit"` runs 10 complete episodes on LiH 10q without error
- [ ] `QubitUCCSearchController.search()` returns a `SearchResult` with populated `best_energy` field
- [ ] Comparison JSON `qubit_vs_fermion_lih_10q.json` saved with both results
- [ ] If qubit operators achieve chemical accuracy: assert `energy_error < 1.6e-3`; otherwise document the gap
- [ ] `progress.txt` contains "Qubit Operator API Investigation" section documenting how Tencirchem qubit operators work
- [ ] All tests pass with >80% coverage

---

## Phase 3 Complete Deliverables Checklist

### Core System
- [ ] `HybridCircuitBuilder` + `HybridFusionStrategy` (Task 001)
- [ ] `HybridSearchEnv` + `HybridRewardFunction` (Task 002)
- [ ] `HybridSearchController` integrated with Phase 2 `ExperimentManager` (Task 003)
- [ ] `BatchEvaluator` + `CIVectorBenchmark` + `MemoryManager` (Task 004)
- [ ] `CircuitEncoder` hierarchy + `EncoderFactory` (Task 005)
- [ ] Integration tests passing on BeH₂, H₄, H₆ (Task 006)
- [ ] `QubitOperatorPool` + `QubitUCCSearchController` + fermion-vs-qubit comparison (Task 007)

### Package Structure
```
src/rlqas/phase3/
    __init__.py
    hybrid_search/
        __init__.py
        circuit_builder.py      # HybridCircuitBuilder, HybridFusionStrategy
        environment.py          # HybridSearchEnv, HybridRewardFunction
        controller.py           # HybridSearchController
        config.py               # HybridSearchConfig
    performance/
        __init__.py
        batch_evaluator.py      # BatchEvaluator, BatchEvaluatorConfig
        benchmarking.py         # CIVectorBenchmark
        memory_manager.py       # MemoryManager
        checkpoint.py           # Checkpoint utilities
    encoding/
        __init__.py
        base_encoder.py         # CircuitEncoder ABC
        matrix_encoder.py       # MatrixEncoder
        sparse_encoder.py       # SparseEncoder
        one_hot_encoder.py      # OneHotEncoder
        encoder_factory.py      # EncoderFactory
        benchmark.py            # EncodingBenchmark
    qubit_ops/
        __init__.py
        operator_pool.py        # QubitOperatorPool
        controller.py           # QubitUCCSearchController
        adapter.py              # TencirchemQubitAdapter (if needed)
```

### Code Quality
- [ ] Overall code coverage >85% across all Phase 3 modules
- [ ] PEP8 compliance; type hints on all public interfaces
- [ ] Docstrings on all classes and public methods

### Performance Evidence
- [ ] `benchmarks/ci_vector_benchmark_results.json` recorded
- [ ] `results/phase3_integration/batch_benchmark.json` showing ≥1.5× speedup
- [ ] `results/phase3_integration/encoding_benchmark.json` showing timing per encoder
- [ ] `results/phase3_integration/qubit_vs_fermion_lih_10q.json` showing fermion vs qubit comparison

### Token Tracking Evidence
- [ ] `progress.txt` contains `[TOKEN LOG]` entries after each completed task with cumulative input/output token counts

### Next Steps Ready
- [ ] Foundation laid for Phase 4 (CLI, full documentation, production packaging)
- [ ] All Phase 3 results stored in `ResultsDatabase` format for analysis
- [ ] Performance baseline established for potential Phase 4 hardware backend work

---

## Environment Setup

Phase 3 inherits all Phase 1 and Phase 2 dependencies. Additional requirements:

```txt
# Memory monitoring (for MemoryManager)
psutil>=5.9

# Jinja2 for any template-based code generation inherited from Phase 2 adaptation
jinja2>=3.0

# Optional: for sparse matrix operations in SparseEncoder
scipy>=1.7   # already required by Phase 1
```

---

## Document Update Record
- v1.0 (2026-03-15): Created Phase 3 task breakdown based on RLQAS_Ralph_20260205_EN.md PRD Section 3.6, 3.2 (performance), and Phase 3 spec. Structured as 6 tasks following Phase 1/2 task format conventions.
- v1.2 (2026-03-22): Changed H₆ threshold from relaxed 5.0e-3 to standard chemical accuracy 1.6e-3; removed active_space restriction for H₆ (STO-3G full space is naturally (6,6), so active_space parameter omitted to use full orbital space).
