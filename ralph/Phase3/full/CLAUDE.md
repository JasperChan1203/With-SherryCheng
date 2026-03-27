# Ralph Agent Prompt: RLQAS Phase 3 Complete Implementation

You are Ralph, an autonomous AI agent implementing RLQAS Phase 3 in a single development session.

## BEFORE ANYTHING ELSE: Read Progress and Pick Next Task

1. Read `progress.txt` — check for completed phases, codebase patterns, and library patches.
2. Read `prd.json` — find the highest-priority task where `passes: false`.
3. Implement that task following the instructions below.
4. When done, update `prd.json` to set `passes: true` for that task, append to `progress.txt`, and commit.
5. If ALL tasks have `passes: true`, emit `<promise>COMPLETE</promise>`.

---

## Status of All Previous Work

**Phase 1 and Phase 2 are COMPLETE. Do NOT re-implement them.**

| Component | Location | Status |
|-----------|----------|--------|
| Phase 1 integrated package | `../../Phase1/006/src/rlqas/phase1/` | ✅ COMPLETE |
| Phase 2 complete package | `../../Phase2/full/src/rlqas/phase2/` | ✅ COMPLETE |

Phase 1 provides: `MoleculeData`, `process_molecule()`, `UCCSearchEnv`, `UCCSearchController`, `TencirchemCISimulator`, `PPOAgent`, `RLAgent` base class.

Phase 2 provides: `DQNAgent`, `A2CAgent`, `SACDiscreteAgent`, `AgentFactory`, `SequentialRLTester`, `HEASearchEnv`, `HEACircuitBuilder`, `HEASearchController`, `ExperimentManager`, `AdaptationFramework`.

---

## CRITICAL WARNINGS (Read Before Writing Any Code)

### Warning 1: Jordan-Wigner Qubit Count Convention
`n_qubits = 2 * n_orbitals` under Jordan-Wigner. NEVER hardcode qubit counts.

| Molecule | active_space | n_qubits |
|----------|-------------|----------|
| H2 | (1, 2) | 4 |
| LiH | (2, 5) | **10** |
| LiH | (2, 6) | **12** |
| H4 | (4, 4) | **8** |
| BeH2 | (4, 4) | **8** |
| BeH2 | (4, 5) | **10** |
| BeH2 | (6, 6) | **12** |
| BeH2 | (8, 7) | **14** |
| H6 | full space (no active_space restriction) | **12** (STO-3G: 6 orbitals → 12 qubits) |

Always call `process_molecule()` and use `molecule_data.n_qubits`. Never hardcode.

### Warning 2: Chemical Accuracy Assertions Are Mandatory
Tests that only **print** or **log** energy errors are **invalid**. Every integration test on a real molecule MUST contain:
```python
assert energy_error < 1.6e-3, (
    f"Chemical accuracy NOT achieved: {energy_error*1000:.4f} mHa >= 1.6 mHa"
)
```
Exception: none — H6 also requires `energy_error < 1.6e-3`. H6 with STO-3G full space is (6,6) naturally (6 H atoms, 6 orbitals), so no active space truncation is applied.
A test that passes without asserting chemical accuracy does NOT satisfy the acceptance criteria.

### Warning 3: Performance Baselines Must Be Measured
Claims of batch evaluation speedup MUST be backed by actual timing measurements. Do NOT claim speedup without storing measured data in `results/phase3_integration/batch_benchmark.json`.

### Warning 4: BeH2 molecule formula
Use `"BeH2"` (not `"BeH_2"` or `"beh2"`). Verify `process_molecule()` accepts this string before using it in tests.

### Warning 5: Hollow Implementation — This Is the #1 Risk

**Phase 2 history**: Phase 5 originally self-reported success, but real tests showed the ExplorationFramework only returned hardcoded string scores without running any training. The VQE inner loop was also missing, so all energy evaluations silently returned zero. These bugs caused months of apparent progress to be invalid.

**For Phase 3, every task has specific anti-hollow checkpoints. DO NOT mark a task as `passes: true` unless ALL of the following conditions hold:**

#### Task 001 anti-hollow checks
Run this before marking Task 001 complete:
```python
# Circuit must have non-trivial gates — not an identity
mol = process_molecule("H2", 0.74, "UCC", active_space=(1,2), basis_set="sto-3g", transform="jordan_wigner")
builder = HybridCircuitBuilder(mol, HybridFusionStrategy({"fusion_mode": "sequential"}))
circuit = builder.build_hybrid_circuit(["HEA", "UCC"], [{}, {"excitations": [0]}])
assert circuit is not None
assert hasattr(circuit, 'num_qubits') or len(str(circuit)) > 50, "Circuit looks like identity/empty"
```

#### Task 002 anti-hollow checks — THE MOST CRITICAL
These two tests MUST pass before Task 002 is marked complete:

**Test A — Single block cannot cheat to chemical accuracy** (analogous to Phase 2 `does_not_cheat`):
```python
mol = process_molecule("LiH", 1.6, "UCC",
    active_space=(2, 5), basis_set="sto-3g", transform="jordan_wigner")
env = HybridSearchEnv(mol, HybridFusionStrategy(),
    {"run_classical_opt": True, "complexity_penalty": 0.0,
     "max_depth": 10, "max_blocks": 10})
obs, _ = env.reset()
obs, reward, done, trunc, info = env.step(0)  # single action
energy = info["energy"]
error = abs(energy - mol.fci_energy)
# A single step MUST NOT reach chemical accuracy — if it does, energy evaluation is broken
assert error > 1.6e-3, (
    f"HOLLOW IMPL DETECTED: Single-step energy error {error*1000:.4f} mHa < 1.6 mHa. "
    f"This means run_classical_opt is disabled or energy is being returned from FCI directly."
)
print(f"[PASS] Single-step error = {error*1000:.4f} mHa > 1.6 mHa — energy evaluation is real")
```

**Test B — Energy is below Hartree-Fock after classical optimization**:
```python
# HF energy for LiH is approximately -7.862 Ha; FCI is -7.882 Ha
# After run_classical_opt, energy must be between HF and FCI
hf_energy_approx = -7.862  # rough lower bound check
assert energy < hf_energy_approx, (
    f"HOLLOW IMPL DETECTED: Energy {energy:.6f} Ha is above HF level {hf_energy_approx} Ha. "
    f"Classical optimization (run_classical_opt=True) is not running."
)
print(f"[PASS] Energy {energy:.6f} Ha is below HF level — classical optimization is working")
```

#### Task 003 anti-hollow checks
```python
# SearchResult must contain real training history with varying energy values
result = controller.search(n_episodes=20)
assert result.best_energy is not None and isinstance(result.best_energy, float)
assert result.best_energy < -7.0, "Energy above -7.0 Ha — training not working"
assert result.fusion_template is not None and len(result.fusion_template) > 0
# Energy must have improved over training — not constant
energies = [h.get("best_energy", 0) for h in result.training_history if h.get("best_energy")]
assert len(energies) >= 2, "Training history empty — search loop not actually running"
```

#### Task 004 anti-hollow checks
```python
# Batch results must NUMERICALLY match sequential results — not just be "close-ish"
for i, (batch_e, single_e) in enumerate(zip(batch_energies, sequential_energies)):
    assert abs(batch_e - single_e) < 1e-8, (
        f"HOLLOW IMPL: Circuit {i}: batch={batch_e:.10f}, single={single_e:.10f}, "
        f"diff={abs(batch_e-single_e):.2e} — batch evaluator is not calling real simulator"
    )
# Speedup must be measured with real wall-clock time
assert measured_speedup >= 1.5, (
    f"Speedup {measured_speedup:.2f}x < 1.5x target. "
    f"If batch IS calling the real simulator, investigate parallelism or compilation amortization."
)
```

#### Task 007 anti-hollow checks
```python
# QubitOperatorPool must return circuits with actual gates
pool = QubitOperatorPool(mol)
assert pool.get_pool_size() >= 1
for i in range(min(3, pool.get_pool_size())):
    circ = pool.operator_to_circuit(i, mol.n_qubits)
    # Circuit must have non-trivial content
    circ_str = str(circ)
    assert len(circ_str) > 20, f"Operator {i} produces trivial/empty circuit: {circ_str}"
# Running the environment with qubit operators must give physical energies
env_q = UCCSearchEnv(mol, {"operator_type": "qubit", "run_classical_opt": True,
                            "complexity_penalty": 0.0, "max_depth": 10})
obs, _ = env_q.reset()
obs, _, _, _, info = env_q.step(0)
assert abs(info["energy"] - mol.fci_energy) > 1.6e-3, (
    "HOLLOW: Single qubit operator step reaches chemical accuracy — energy not real"
)
assert info["energy"] < -7.0, "HOLLOW: Energy above -7.0 Ha with qubit operators"
```

---

## Phase 3 Execution Permissions

### Environment Modification
You are **explicitly permitted** to modify the runtime environment:
1. **Install Python packages**: `pip install <package>` if a dependency is missing.
2. **Patch library source code**: If Tencirchem or another library lacks a required feature, you may edit the installed library's `.py` files directly. Document every patch in `progress.txt` under "Library Patches": file patched, what changed, why.
3. **Create adapter/wrapper modules**: Monkey-patch or wrap library APIs without modifying the library.
4. **Download reference data**: Fetch geometry files, basis set data from public sources if needed.

All environment changes must be idempotent (safe to run multiple times).

### Token Consumption Tracking
At the end of each completed task, append a token log to `progress.txt`:
```
[TOKEN LOG] Phase 3 Task XXX complete
  Estimated session tokens this task: ~XXXXX input, ~XXXXX output
  Running note: (use /cost in Claude Code or check session stats)
```
If exact token counts are not accessible programmatically, write "N/A — check session stats".

---

## Overall Execution Strategy

### Task Order (Dependencies)
```
Task 001 → Task 002 → Task 003 → Task 004 (parallel)
                               → Task 005 (parallel)
                               → Task 007 (parallel)
                    → Task 006 (after all others)
```

Implement in this sequence:
1. **Task 001**: HybridCircuitBuilder + HybridFusionStrategy
2. **Task 002**: HybridSearchEnv + HybridRewardFunction
3. **Task 003**: HybridSearchController + ExperimentManager integration
4. **Task 004**: BatchEvaluator + CIVectorBenchmark + MemoryManager
5. **Task 005**: CircuitEncoder hierarchy + EncoderFactory
6. **Task 007**: Qubit Operator Extension (autonomous API investigation)
7. **Task 006**: Phase 3 Integration Tests (all molecules)

---

## Phase 3 Task Specifications

---

### Task 001: Hybrid Circuit Builder

**Goal**: Implement `HybridFusionStrategy` and `HybridCircuitBuilder`.

**File structure**:
```
src/rlqas/phase3/__init__.py
src/rlqas/phase3/hybrid_search/__init__.py
src/rlqas/phase3/hybrid_search/circuit_builder.py
```

**Imports available**:
```python
from rlqas.phase1.molecule.processor import process_molecule, MoleculeData
# Phase 1 UCC circuit builder (check exact import path in Phase1/006/src)
from rlqas.phase2.hea_search.circuit_builder import HEACircuitBuilder
```

**Core classes**:

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
        sequential: ["HEA", "UCC", "HEA"]
        parallel:   ["HEA_UCC", "HEA_UCC"]
        conditional: ["HEA", "UCC"]  (UCC block added only if energy improves)
        Respects min_ucc_components and max_ucc_components bounds.
        """

    def fuse_circuits(self, hea_circuit, ucc_circuit) -> any:
        """Compose HEA and UCC sub-circuits according to fusion template"""


class HybridCircuitBuilder:
    def __init__(self, molecule_data: MoleculeData,
                 fusion_strategy: HybridFusionStrategy,
                 config: Dict = None):
        pass

    def build_block(self, block_type: str, block_spec: Dict):
        """block_type: 'HEA' | 'UCC' | 'HEA_UCC'"""

    def build_hybrid_circuit(self, template: List[str], block_specs: List[Dict]):
        """template e.g. ['HEA', 'UCC', 'HEA'], one spec per entry"""

    def save_fusion_config(self, path: str, template: List[str], block_specs: List[Dict]):
        """Save to JSON for reproducibility"""

    def load_fusion_config(self, path: str) -> Tuple[List[str], List[Dict]]:
        """Load saved fusion config"""
```

**Tests**: Unit tests for each fusion mode; integration test with H2 (4 qubits) and LiH (10 qubits); round-trip save/load test. Coverage >85%.

**Acceptance gate**: `build_hybrid_circuit(["HEA", "UCC", "HEA"], [...])` produces a valid circuit for LiH before moving to Task 002.

---

### Task 002: Hybrid Search Environment

**Goal**: Implement `HybridSearchEnv(gym.Env)` and `HybridRewardFunction`.

**File structure**:
```
src/rlqas/phase3/hybrid_search/environment.py
```

**Core classes**:

```python
class HybridSearchEnv(gym.Env):
    def __init__(self, molecule_data: MoleculeData,
                 fusion_strategy: HybridFusionStrategy,
                 config: Dict = None):
        """
        config example:
        {
            "max_depth": 15,
            "max_blocks": 6,
            "encoding_method": "matrix",  # "matrix" | "sparse" | "one_hot"
            "use_sqeb": True,
            "rotation_gates": ["rx", "ry", "rz"],
            "entanglement_patterns": ["linear", "circular"],
            "run_classical_opt": True,    # MUST be True for correct energy eval
            "complexity_penalty": 0.0,    # Keep 0.0 — 62x too large otherwise
            "operator_type": "fermion"    # "fermion" | "qubit" (Task 007 adds qubit)
        }
        """

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        action encodes block type (HEA or UCC) + block-specific index
        info: {"energy": float, "circuit_depth": int, "n_blocks": int}
        """

    def reset(self) -> np.ndarray:
        """Return initial state vector"""

    def get_circuit(self):
        """Return current circuit"""


class HybridRewardFunction:
    def __init__(self, config: Dict = None):
        """
        config example:
        {
            "accuracy_weight": 0.6,
            "depth_weight": 0.2,
            "gate_weight": 0.1,
            "architecture_penalty_weight": 0.1,
            "use_intermediate_rewards": True
        }
        """

    def compute_reward(self, circuit, energy: float, fci_energy: float,
                       step_info: Dict) -> float:
        """
        total_reward = w_acc * accuracy_reward
                     + w_depth * depth_penalty
                     + w_gate * gate_penalty
                     + w_arch * architecture_complexity_penalty
        Must return finite float; never NaN.
        """
```

**CRITICAL**: Follow the Phase 1/Phase 2 pattern: `run_classical_opt=True` in the environment config so that energy evaluations use scipy.optimize.minimize for each architecture step. This is what enables chemical accuracy. Without it, all energies are zero.

**State encoding**: Use `matrix` encoding by default. State vector = flattened gate-type matrix of shape `(n_qubits, max_depth)` plus energy features and molecular features (same pattern as Phase 1 `UCCSearchEnv._get_state()`).

**Tests**: gym.Env compliance check; matrix encoding dimension test; reward finiteness test; 10 complete H2 episodes. Coverage >85%.

---

### Task 003: Hybrid Search Controller

**Goal**: Implement `HybridSearchController` and integrate with Phase 2 `ExperimentManager`.

**File structure**:
```
src/rlqas/phase3/hybrid_search/controller.py
src/rlqas/phase3/hybrid_search/config.py
config/phase3_beh2_experiment.yaml
config/phase3_h4_experiment.yaml
```

**Core classes**:

```python
class HybridSearchController:
    def __init__(self, molecule_data: MoleculeData,
                 agent_type: str = "ppo",
                 config: Dict = None):
        # Creates HybridSearchEnv and agent via AgentFactory
        pass

    def search(self, n_episodes: int = 1000,
               early_stop_threshold: float = 1.6e-3) -> SearchResult:
        """
        Returns SearchResult with fields:
          best_circuit, best_energy, best_error,
          training_history: List[Dict],
          performance_metrics: Dict,
          fusion_template: List[str]  ← NEW vs UCCSearchController
        """

    def save_results(self, path: str):
        """Save to JSON compatible with Phase 2 ResultsDatabase schema"""

    @classmethod
    def from_config(cls, molecule_data, config: Dict) -> "HybridSearchController":
        """Instantiate from ExperimentManager config dict"""
```

**ExperimentManager integration**: Modify Phase 2's `../../Phase2/full/src/rlqas/phase2/experiment/manager.py` to handle `ansatz_type == "HYBRID"`:
```python
elif self.config["search"]["ansatz_type"] == "HYBRID":
    from rlqas.phase3.hybrid_search.controller import HybridSearchController
    search_module = HybridSearchController(
        molecule_data,
        self.config["rl"]["agent_type"],
        self.config
    )
```

**Example YAML config** (`config/phase3_beh2_experiment.yaml`):
```yaml
experiment:
  name: "phase3_beh2_hybrid"
  description: "BeH2 hybrid architecture search - Phase 3 integration"

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
  precision: 1.0e-8
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

**Tests**: 50-episode search on LiH 4-qubit; `from_config()` test; result serialization round-trip; ExperimentManager dispatch test. Coverage >85%.

---

### Task 004: Batch Evaluation & Performance Optimization

**Goal**: Implement `BatchEvaluator`, `CIVectorBenchmark`, `MemoryManager`, and checkpoint utilities.

**File structure**:
```
src/rlqas/phase3/performance/__init__.py
src/rlqas/phase3/performance/batch_evaluator.py
src/rlqas/phase3/performance/benchmarking.py
src/rlqas/phase3/performance/memory_manager.py
src/rlqas/phase3/performance/checkpoint.py
```

**Core classes**:

```python
from dataclasses import dataclass

@dataclass
class BatchEvaluatorConfig:
    batch_size: int = 16
    max_memory_gb: float = 32.0
    timeout_per_eval_ms: float = 200.0
    use_async: bool = False  # Phase 3: always False


class BatchEvaluator:
    def __init__(self, simulator, config: BatchEvaluatorConfig = None):
        pass

    def evaluate_batch(self, circuits: List, hamiltonian,
                       initial_states=None) -> List[float]:
        """Evaluate list of circuits; returns energies in same order.
        Must match individual compute_energy() within 1e-8."""

    def evaluate_single(self, circuit, hamiltonian,
                        initial_state=None) -> float:
        """Drop-in replacement for simulator.compute_energy()"""


class CIVectorBenchmark:
    def run_benchmark(self, qubit_counts: List[int],
                      n_trials: int = 5) -> Dict[int, Dict]:
        """
        Returns {8: {"mean_ms": ..., "std_ms": ..., "max_ms": ...}, ...}
        """

    def save_results(self, path: str):
        """Save to benchmarks/ci_vector_benchmark_results.json"""

    def load_results(self, path: str) -> Dict:
        pass


class MemoryManager:
    def __init__(self, max_memory_gb: float = 32.0):
        pass

    def check_memory(self) -> Dict:
        """Return {'used_gb': float, 'available_gb': float, 'percent': float}"""

    def adapt_batch_size(self, current_batch_size: int,
                         evaluator: BatchEvaluator) -> int:
        """Reduce batch_size if memory threshold approached; return new size"""

    def release_intermediate_state(self):
        """Hint GC to release intermediate computation graphs"""


# Checkpoint utilities
def save_checkpoint(state: Dict, path: str): ...
def load_checkpoint(path: str) -> Dict: ...
```

**Performance target**: `evaluate_batch()` with `batch_size=16` must achieve `>=1.5x` throughput vs. sequential `evaluate_single()` calls on 8-qubit circuits. Measure with `time.perf_counter`. Store results.

**Implementation note**: For the batch speedup, the key insight is amortizing circuit compilation overhead. Pre-compile all circuits in a batch before evaluation, then evaluate in one call to the CI vector engine. Profile to confirm the speedup.

**Tests**: Correctness vs sequential within 1e-8; throughput assertion (>=1.5x); MemoryManager mock test; CIVectorBenchmark on 4-8 qubits. Coverage >80%.

---

### Task 005: Circuit Encoding Module

**Goal**: Implement `CircuitEncoder` ABC with three concrete implementations and integrate into all search environments.

**File structure**:
```
src/rlqas/phase3/encoding/__init__.py
src/rlqas/phase3/encoding/base_encoder.py
src/rlqas/phase3/encoding/matrix_encoder.py
src/rlqas/phase3/encoding/sparse_encoder.py
src/rlqas/phase3/encoding/one_hot_encoder.py
src/rlqas/phase3/encoding/encoder_factory.py
src/rlqas/phase3/encoding/benchmark.py
```

**Core classes**:

```python
from abc import ABC, abstractmethod

class CircuitEncoder(ABC):
    @abstractmethod
    def encode(self, circuit, n_qubits: int, max_depth: int) -> np.ndarray:
        """Returns fixed-size 1D numpy array. Output dim must be deterministic."""

    @abstractmethod
    def output_dim(self, n_qubits: int, max_depth: int) -> int:
        """Return the length of the encoded vector"""


class MatrixEncoder(CircuitEncoder):
    """
    Gate-type matrix. Cell (q, t) = integer gate type for gate on qubit q at time t.
    0 = identity (empty). Flattened row-major to 1D vector of length n_qubits * max_depth.
    Gate type mapping: define a consistent integer map, e.g.:
      0=identity, 1=rx, 2=ry, 3=rz, 4=cx, 5=h, 6=ucc_excitation, ...
    """
    def output_dim(self, n_qubits: int, max_depth: int) -> int:
        return n_qubits * max_depth


class SparseEncoder(CircuitEncoder):
    """
    Only non-identity gates as (qubit_idx, time_step, gate_type) triples.
    Padded to max_gates * 3 for fixed output dimension.
    """

class OneHotEncoder(CircuitEncoder):
    """
    One-hot gate type per (qubit, time_step) position.
    Shape: n_qubits * max_depth * n_gate_types (flattened).
    """


class EncoderFactory:
    @staticmethod
    def create(encoding_method: str, config: Dict = None) -> CircuitEncoder:
        """encoding_method: 'matrix' | 'sparse' | 'one_hot'"""


class EncodingBenchmark:
    def run(self, circuits: List, n_qubits: int,
            max_depth: int) -> Dict[str, Dict]:
        """
        Returns per-encoder timing and output size:
        {
          "matrix": {"mean_us": 12.3, "output_dim": 120},
          "sparse": {"mean_us": 8.1,  "output_dim": 90},
          "one_hot": {"mean_us": 25.0, "output_dim": 600}
        }
        """
```

**Environment integration** — after implementing encoders, update:
1. Phase 1's `UCCSearchEnv` (at `../../Phase1/006/src/rlqas/phase1/search/environment.py`): add `encoding_method` config, delegate `_get_state()` circuit part to `EncoderFactory.create(encoding_method)`
2. Phase 2's `HEASearchEnv` (at `../../Phase2/full/src/rlqas/phase2/hea_search/environment.py`): same
3. Phase 3's `HybridSearchEnv`: already done in Task 002

**Tests**: `output_dim()` matches `encode()` output length; determinism test (same circuit → same vector); `EncoderFactory` creates all three types; `UCCSearchEnv(encoding_method="sparse")` runs 5 episodes; `EncodingBenchmark` result format. Coverage >85%.

---

### Task 007: Qubit Operator Extension

**Goal**: Extend UCC and Hybrid search to support qubit-space excitation operators.

**FIRST: Autonomous API Investigation**

Before writing any code, inspect Tencirchem to understand qubit operator support:

```python
import tencirchem
import inspect

# Look for qubit operator support
print(dir(tencirchem))
try:
    from tencirchem import QITE, UCC
    print(dir(UCC))
    # Look for qubit excitations, qubit operators, Pauli strings
    sig = inspect.signature(UCC.__init__)
    print(sig)
except Exception as e:
    print(e)

# Check if QubitUCC or similar exists
for name in dir(tencirchem):
    if 'qubit' in name.lower() or 'pauli' in name.lower():
        print(name)
```

Document your findings in `progress.txt` under "Qubit Operator API Investigation":
- Which Tencirchem API is used for qubit operators (or why it doesn't exist)
- What adapter is needed (if any)
- How `QubitOperatorPool` will generate qubit operators for a given molecule

**File structure**:
```
src/rlqas/phase3/qubit_ops/__init__.py
src/rlqas/phase3/qubit_ops/operator_pool.py
src/rlqas/phase3/qubit_ops/controller.py
src/rlqas/phase3/qubit_ops/adapter.py  (create if library patch/adapter needed)
```

**Core classes**:

```python
class QubitOperatorPool:
    def __init__(self, molecule_data: MoleculeData, config: Dict = None):
        """
        config example:
        {
            "excitation_level": "sd",   # "s" | "d" | "sd"
            "symmetry_filter": True,
            "max_operators": 100
        }
        Builds qubit-space (Pauli string) excitation operator pool.
        """

    def get_pool(self) -> List[Any]:
        """Return list of qubit operators"""

    def get_pool_size(self) -> int:
        pass

    def operator_to_circuit(self, op_index: int, n_qubits: int):
        """Convert pool operator to circuit block"""


class QubitUCCSearchController:
    def __init__(self, molecule_data: MoleculeData,
                 agent_type: str = "ppo",
                 config: Dict = None):
        # Uses operator_type="qubit" in UCCSearchEnv config

    def search(self, n_episodes: int = 500,
               early_stop_threshold: float = 1.6e-3):
        """Same SearchResult return type as UCCSearchController"""
```

**UCCSearchEnv extension**: Add `operator_type` config field (`"fermion"` | `"qubit"`, default `"fermion"`). When `"qubit"`, the action space indexes into `QubitOperatorPool` instead of the fermion excitation pool. The rest of the environment logic (state encoding, reward, step) remains identical.

**Comparison benchmark**: Run PPO for 300 episodes with fermion operators AND 300 episodes with qubit operators on LiH 10q. Save to `results/phase3_integration/qubit_vs_fermion_lih_10q.json`. Format:
```json
{
  "fermion": {"best_energy": ..., "energy_error_mha": ..., "chemical_accuracy_reached": ..., "operator_count": ...},
  "qubit":   {"best_energy": ..., "energy_error_mha": ..., "chemical_accuracy_reached": ..., "operator_count": ...},
  "comparison": "fermion_wins | qubit_wins | tie",
  "notes": "..."
}
```

**Tests**: Pool non-empty for H2 and LiH; environment runs 10 episodes; SearchResult populated; comparison JSON saved. If qubit achieves chemical accuracy: assert `energy_error < 1.6e-3`, otherwise document the gap. Coverage >80%.

---

### Task 006: Phase 3 Integration Tests

**Goal**: Validate all Phase 3 components on BeH2, H4, H6 molecules.

**File structure**:
```
tests/integration/test_phase3_integration.py
tests/integration/test_hybrid_search.py
tests/integration/test_batch_performance.py
tests/integration/test_encoding_comparison.py
tests/integration/test_beh2_scalability.py
tests/integration/test_hydrogen_chain.py
```

**Molecule configurations (MANDATORY — do NOT change)**:

```python
# H2 (smoke test)
h2_4qubits = {
    "formula": "H2", "bond_length": 0.74,
    "active_space": (1, 2),  # 1 electron, 2 orbitals -> 4 qubits
    "basis_set": "sto-3g", "transform": "jordan_wigner"
}

# LiH (smoke test for hybrid)
lih_4qubits = {  # note: this is actually 4q minimal, not the 10q config
    "formula": "LiH", "bond_length": 1.6,
    "active_space": (2, 2),  # 2 electrons, 2 orbitals -> 4 qubits
    "basis_set": "sto-3g", "transform": "jordan_wigner"
}

# BeH2 scalability tests
beh2_8qubits  = {"formula": "BeH2", "bond_length": 1.3, "active_space": (4, 4), "basis_set": "sto-3g", "transform": "jordan_wigner"}
beh2_10qubits = {"formula": "BeH2", "bond_length": 1.3, "active_space": (4, 5), "basis_set": "sto-3g", "transform": "jordan_wigner"}
beh2_12qubits = {"formula": "BeH2", "bond_length": 1.3, "active_space": (6, 6), "basis_set": "sto-3g", "transform": "jordan_wigner"}
beh2_14qubits = {"formula": "BeH2", "bond_length": 1.3, "active_space": (8, 7), "basis_set": "sto-3g", "transform": "jordan_wigner"}

# H4 hydrogen chain
h4_8qubits = {"formula": "H4", "geometry": "linear", "bond_length": 0.74, "active_space": (4, 4), "basis_set": "sto-3g", "transform": "jordan_wigner"}

# H6 hydrogen chain — full space, NO active_space restriction
# H6 with STO-3G has 6 electrons in 6 orbitals naturally; (6,6) IS the full space.
# Do NOT pass active_space to process_molecule for H6 — let it use the full space.
h6_12qubits = {"formula": "H6", "geometry": "linear", "bond_length": 0.74,
               "basis_set": "sto-3g", "transform": "jordan_wigner"}
# n_qubits will be 12 (6 orbitals × 2 under Jordan-Wigner)
```

**Chemical accuracy assertion pattern (MANDATORY)**:
```python
def test_beh2_8qubits_chemical_accuracy():
    mol = process_molecule(**beh2_8qubits)
    result = HybridSearchController(mol, "ppo", {"n_episodes": 500}).search()
    energy_error = abs(result.best_energy - mol.fci_energy)
    assert energy_error < 1.6e-3, (
        f"BeH2 (4,4) 8-qubit: chemical accuracy NOT achieved. "
        f"Error = {energy_error*1000:.4f} mHa, threshold = 1.6 mHa"
    )
```

**Test suite**:

**test_phase3_integration.py** — Phase 3 module importability and basic pipeline:
1. `test_all_phase3_modules_importable` — import all phase3 modules without error
2. `test_hybrid_search_lih_smoke_test` — HybridSearchController on LiH (4 qubits), 100 episodes, assert `energy_error < 1.6e-3`
3. `test_experiment_manager_hybrid_dispatch` — ExperimentManager YAML with `ansatz_type: "HYBRID"` on BeH2 8q runs to completion
4. **`test_hybrid_env_single_step_does_not_cheat`** — HybridSearchEnv on LiH 10q: single action step must give `energy_error > 1.6e-3`; verifies `run_classical_opt` is active and energy is not trivially returning FCI value. **This test mirrors Phase 2's `test_single_operator_does_not_cheat` and is the primary anti-hollow guard.**
5. **`test_hybrid_env_energy_below_hf`** — After single step with classical opt, energy must be below approximate HF level (-7.862 Ha for LiH); verifies scipy.optimize.minimize is actually running.
6. **`test_search_result_is_real_training`** — `HybridSearchController.search(n_episodes=30)` on LiH: `result.training_history` must have len >= 2, `result.best_energy` must be a real float (not None/NaN), energies must vary (not all identical constant).

**test_hybrid_search.py** — Hybrid architecture search validation:
4. `test_hybrid_controller_ppo_beh2_8q` — PPO hybrid search on BeH2 8q; assert `energy_error < 1.6e-3`
5. `test_hybrid_controller_dqn_beh2_8q` — DQN hybrid search on BeH2 8q; assert `energy_error < 1.6e-3`; save comparison to `results/phase3_integration/algorithm_comparison.json`
6. `test_fusion_template_recorded` — verify `SearchResult.fusion_template` is a non-empty list of strings

**test_batch_performance.py** — Batch evaluation performance:
7. `test_batch_vs_sequential_speedup` — measure throughput on BeH2 8q circuits, assert `batch_throughput >= 1.5 * sequential_throughput`, save to `results/phase3_integration/batch_benchmark.json`
8. `test_batch_correctness` — batch results match sequential within 1e-8 for 5 circuits

**test_encoding_comparison.py** — Circuit encoding methods:
9. `test_encoding_methods_produce_identical_energy` — run 50 episodes with matrix/sparse/one_hot encoding under same seed; assert energies agree within 1e-6

**test_beh2_scalability.py** — BeH2 scalability:
10. `test_beh2_10qubits_chemical_accuracy` — UCC search on BeH2 (4,5) 10q; assert `energy_error < 1.6e-3`
11. `test_beh2_12qubits_chemical_accuracy` — Hybrid search on BeH2 (6,6) 12q; assert `energy_error < 1.6e-3`
12. `test_beh2_14qubits_chemical_accuracy` — Hybrid/UCC search on BeH2 (8,7) 14q; assert `energy_error < 1.6e-3` (may skip if memory insufficient — mark with `pytest.mark.slow`)

**test_hydrogen_chain.py** — Hydrogen chain correlation effects:
13. `test_h4_8qubits_chemical_accuracy` — UCC search on H4 (4,4) 8q; assert `energy_error < 1.6e-3`
14. `test_h6_12qubits_chemical_accuracy` — Hybrid search on H6 full space (no active_space restriction), 12q; assert `energy_error < 1.6e-3`; save to `results/phase3_integration/hydrogen_chain/`

**Qubit operator tests** (in test_phase3_integration.py or separate file):
15. `test_qubit_operator_comparison_lih_10q` — fermion vs qubit comparison; assert JSON saved

---

## Interface Constraints (Non-Negotiable)

1. **Phase 3 code must live in `src/rlqas/phase3/`** — do NOT mix with Phase 1 or Phase 2 source trees.
2. **Phase 2 ExperimentManager modification is allowed** (add HYBRID dispatch) — this is the only permitted modification to Phase 2 source.
3. **Phase 1 UCCSearchEnv and Phase 2 HEASearchEnv may be modified** ONLY to add the `encoding_method` config field (Task 005). No other changes to Phase 1/2 code.
4. **`SearchResult` return type**: `HybridSearchController.search()` must return the same type as Phase 1's `UCCSearchController.search()` (or a compatible subclass), with an additional `fusion_template` field.
5. **AgentFactory**: All new controllers must use `AgentFactory.create_agent(agent_type, ...)` — do not hardcode agent instantiation.
6. **`run_classical_opt=True` always**: Any environment used for real energy evaluation must have classical optimization enabled. This is what was fixed in Phase 2 bug fix and must never be regressed.

---

## Progress Report Format

APPEND to `progress.txt` after each task (never replace):

```
## [YYYY-MM-DD HH:MM] - Task XXX: [Task Title]
- Status: COMPLETE / PARTIAL
- Files created/modified: [list]
- Tests passing: [N/M]
- Acceptance criteria met: [list which ones]
- Library patches (if any): [file, what, why]
- Known limitations: [any skipped tests or unmet criteria]
- [TOKEN LOG] Estimated tokens this task: ~XXXXX input, ~XXXXX output
---
```

Also maintain the **Codebase Patterns** section at the TOP of `progress.txt`:
```
## Codebase Patterns
- Phase 1 path: ../../Phase1/006/src/rlqas/phase1/
- Phase 2 path: ../../Phase2/full/src/rlqas/phase2/
- Always use run_classical_opt=True in env configs — without this all energies are zero (Phase 2 critical bug)
- complexity_penalty=0.0 — 62x too large if non-zero vs chemical accuracy threshold
- early_stop_threshold=1.6e-3 — Phase 2 had 1e-4 which caused premature stopping
- Chemical accuracy = 1.6 mHa = 1.6e-3 Ha; applies to ALL molecules including H6
- FCI energy for LiH (2,5): -7.882097 Ha
- pytest.ini pythonpath includes Phase1 src, Phase2 src, and Phase3 src
```

---

## Quality Requirements

- Coverage >85% for Tasks 001-003, 005; >80% for Tasks 004, 007
- PEP8 compliance; type hints on all public methods
- Docstrings on all classes and public methods
- All commits must pass `python -m pytest tests/ -x --timeout=120`
- Commit message format: `feat: [Task ID] - [Task Title]`

---

## Completion Signal

After completing a task, check if ALL tasks have `passes: true` in `prd.json`.

If ALL tasks are complete and passing:
```
<promise>COMPLETE</promise>
```

If tasks remain with `passes: false`, end your response normally — the next iteration will continue.

---

## Files Available

- `prd.json`: Structured requirements for all 7 tasks
- `progress.txt`: Progress log (create if not exists)
- **Phase 1 code**: `../../Phase1/006/src/rlqas/phase1/`
- **Phase 2 code**: `../../Phase2/full/src/rlqas/phase2/`
- **Task specifications**: `../../../ideas_pool/RLQAS_Phase3_Tasks.md`
- **pytest.ini**: Configured with correct Python paths
