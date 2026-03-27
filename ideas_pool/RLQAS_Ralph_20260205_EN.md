# RLQAS Modular Requirements Specification

## Version Information
- **Version**: 1.1
- **Creation Date**: 2026-02-05
- **Update Date**: 2026-02-05
- **Status**: Optimized Draft
- **Objective**: Guide Ralph in implementing RLQAS system in modular blocks
- **Key Updates**:
  - Clarified Tencirchem-ng 2024.10 version
  - Configurable simulator interface to adapt to different scale systems
  - RL algorithms support sequential testing, simplified parallel requirements
  - Circuit encoding primarily matrix-based with optional other encoding methods
  - Removed specific time estimates, maintaining flexibility

## 1. Project Overview

### 1.1 Project Objective
Develop a modular Reinforcement Learning Quantum Architecture Search (RLQAS) system that can automatically search and optimize quantum circuit structures based on input molecular information and circuit types.

### 1.2 Core Features
1. **Functional Tool Design**:
   - Input: Molecular formula, bond length, desired circuit type (HEA/UCC/Mixed)
   - Output: Optimized quantum circuit + VQE calculation results

2. **Plug-in RL Support**:
   - Support sequential testing of multiple RL algorithms (PPO, DQN, A2C, SAC)
   - Unified RL agent interface for easy algorithm switching and comparison
   - Algorithm implementation correctness prioritized, parallel testing as optional feature

3. **Modular QAS**:
   - Independent HEA search module
   - Independent UCC search module (Phase 1 priority)
   - Intelligent hybrid architecture search module

4. **High-Performance Quantum Simulation**:
   - Integrated Tencirchem-ng 2024.10 CI vector engine
   - Configurable simulation interface adapting to different scale systems (4-20+ qubits)
   - Automatic optimal simulation strategy selection with performance test verification
   - Circuit encoding primarily matrix-based, supporting multiple encoding method experiments

### 1.3 Performance Goals
1. **Accuracy Goals**:
   - LiH (4 qubits): Chemical accuracy (error < 1.6 mHa)
   - BeH₂ (6-14 qubits): Maintain acceptable accuracy loss
   - Hydrogen chain molecules (H₄, H₆, H₈): Validate scalability

2. **Efficiency Goals**:
   - 30% faster training than baseline RLQAS methods
   - Support parallel experimental configurations

3. **Scalability Goals**:
   - Architecture supports scaling to 20+ qubits
   - Modular design facilitates functional expansion

## 2. System Architecture

```
RLQAS System Architecture
├── User Interface Layer
│   ├── Command Line Tool (CLI)
│   ├── Python API
│   └── Configuration Files (YAML/JSON)
│
├── Core Function Layer (Modular)
│   ├── Molecule Processing Module
│   ├── Quantum Simulator Module (Tencirchem CI vector)
│   ├── RL Agent Module (Plug-in)
│   ├── UCC Search Module (Phase 1 Core)
│   ├── HEA Search Module
│   ├── Hybrid Search Module
│   ├── Evaluation & Validation Module
│   └── Experiment Management Module
│
└── Data Storage Layer
    ├── Experimental Results Database
    ├── Training Logs
    └── Pre-trained Model Library
```

## 3. Module Detailed Requirements

### 3.1 Molecule Processing Module

#### Functional Description
Process molecular information to generate input data required for quantum computations.

#### Input Interface
```python
def process_molecule(
    molecule: str,           # Molecular formula, e.g., 'LiH', 'BeH2', 'H4'
    bond_length: float,      # Bond length (Å)
    ansatz_type: str,        # 'UCC', 'HEA', 'MIXED'
    active_space: Optional[Tuple[int, int]] = None,  # (Number of active electrons, number of active orbitals)
    basis_set: str = "sto-3g",  # Basis set
    transform: str = "parity"   # Fermion-to-qubit transformation
) -> MoleculeData
```

#### Output Data Class
```python
@dataclass
class MoleculeData:
    hamiltonian: QubitOperator      # Qubit Hamiltonian
    n_qubits: int                   # Number of qubits
    reference_state: np.ndarray     # Reference state (Hartree-Fock)
    fci_energy: float               # Exact FCI energy
    molecular_info: Dict            # Original molecular information
```

#### Dependencies
- Tencirchem (molecular integrals, Hamiltonian generation)
- OpenFermion (fermionic operator processing)

#### Testing Requirements
- H₂: 2 qubits, validate basic functionality
- LiH: 4 qubits, core test case
- BeH₂: 6-14 qubits, active space testing

### 3.2 Quantum Simulator Module (Tencirchem CI vector)

#### Functional Description
Provides high-performance quantum circuit simulation based on the Tencirchem-ng 2024.10 CI vector engine. Supports configurable simulation interfaces adapting to different scale systems (4-20+ qubits), with specific performance validated through actual testing.

#### Simulator Interface
```python
class QuantumSimulator(ABC):
    @abstractmethod
    def compute_energy(
        self,
        circuit: QuantumCircuit,    # Quantum circuit
        hamiltonian: QubitOperator, # Hamiltonian
        initial_state: Optional[np.ndarray] = None
    ) -> float:
        """Compute energy expectation value of the circuit"""
        pass

    @abstractmethod
    def get_max_qubits(self) -> int:
        """Return maximum supported qubits"""
        pass

    @abstractmethod
    def estimate_memory(self, n_qubits: int) -> float:
        """Estimate memory usage (GB)"""
        pass
```

#### Tencirchem Implementation
```python
class TencirchemCISimulator(QuantumSimulator):
    def __init__(self, config: Dict = None):
        """
        CI vector engine based on Tencirchem-ng 2024.10

        Configuration example:
        {
            "engine": "ci_vector",           # Primary engine
            "precision": 1e-8,               # Calculation precision
            "use_symmetry": True,            # Use symmetry acceleration
            "max_memory_gb": 32,             # Maximum memory limit
            "fallback_method": "statevector" # Fallback method (if CI vector performance is insufficient)
        }

        Note: CI vector performance at 20+ qubits needs to be validated through actual testing,
              and the system should support automatic selection of optimal simulation strategies during testing.
        """
        self.config = config or {}

    def compute_energy(self, circuit, hamiltonian, initial_state=None):
        # Call Tencirchem CI vector engine
        # Convert quantum circuit to Tencirchem representation
        # Return energy expectation value
        pass
```

#### Simulator Factory
```python
class SimulatorFactory:
    @staticmethod
    def create_simulator(
        n_qubits: int,
        config: Dict = None
    ) -> QuantumSimulator:
        """
        Create simulator based on system scale and configuration

        Strategy:
        1. Priority use of simulator type specified in configuration
        2. Automatic selection based on qubit count and available memory
        3. Support performance testing to validate actual performance of different simulators

        Configuration example:
        {
            "preferred_engine": "ci_vector",  # Preferred engine
            "available_memory_gb": 32,        # Available memory
            "require_exact": False,           # Whether exact simulation is required
            "test_performance": True          # Whether to perform performance testing
        }
        """
        config = config or {}

        # If configuration specifies engine, create directly
        engine = config.get("preferred_engine", "ci_vector")

        if engine == "ci_vector":
            return TencirchemCISimulator(config)
        elif engine == "statevector":
            return StatevectorSimulator(config)
        else:
            # Default to CI vector
            return TencirchemCISimulator(config)
```

#### Performance Requirements and Testing
- **Target Performance**:
  - Single energy evaluation time: < 100ms (8 qubits)
  - Memory usage: Configurable上限 (default 32GB)
  - Support batch evaluation (for RL training)

- **Actual Testing Verification**:
  - CI vector performance at 14-20 qubits needs to be validated through actual testing
  - System should record actual performance metrics of different simulators
  - Support dynamic selection of optimal simulation strategies based on test results

- **Encoding Method Support**:
  - Circuit encoding primarily matrix-based (optimal performance)
  - Support experimentation with other encoding methods (e.g., tensor network, sparse representation)
  - Encoding methods should be configurable for performance comparison

### 3.3 RL Agent Module (Plug-in)

#### Functional Description
Implement standard RL algorithm interfaces supporting sequential testing and rapid switching of multiple algorithms. Algorithm correctness implementation prioritized, parallel testing as optional advanced feature.

#### Agent Interface
```python
class RLAgent(ABC):
    @abstractmethod
    def act(self, state: np.ndarray) -> Tuple[int, Dict]:
        """Select action based on state"""
        pass

    @abstractmethod
    def learn(self, experience: Dict) -> Dict:
        """Learn from experience"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model"""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model"""
        pass
```

#### Supported Algorithm Implementations
```python
# PPO Implementation
class PPOAgent(RLAgent):
    def __init__(self, config: Dict):
        """
        Configuration example:
        {
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
        """
        pass

# DQN Implementation
class DQNAgent(RLAgent):
    def __init__(self, config: Dict):
        """
        Configuration example:
        {
            "learning_rate": 1e-3,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "buffer_size": 10000,
            "batch_size": 64,
            "target_update_freq": 100
        }
        """
        pass

# A2C Implementation
class A2CAgent(RLAgent):
    pass

# SAC Implementation
class SACAgent(RLAgent):
    pass
```

#### Agent Factory
```python
class AgentFactory:
    @staticmethod
    def create_agent(
        agent_type: str,
        state_dim: int,
        action_dim: int,
        config: Dict = None
    ) -> RLAgent:
        """
        Create RL agent of specified type

        Supported agent_type:
        - "ppo": Proximal Policy Optimization
        - "dqn": Deep Q-Network
        - "a2c": Advantage Actor-Critic
        - "sac": Soft Actor-Critic
        """
        agent_types = {
            "ppo": PPOAgent,
            "dqn": DQNAgent,
            "a2c": A2CAgent,
            "sac": SACAgent
        }

        if agent_type not in agent_types:
            raise ValueError(f"Unsupported agent type: {agent_type}")

        return agent_types[agent_type](state_dim, action_dim, config or {})
```

#### Sequential Testing Support
```python
class SequentialRLTester:
    def __init__(self, env_config: Dict, agent_configs: List[Dict]):
        """
        Sequentially test multiple RL algorithms

        agent_configs example:
        [
            {"type": "ppo", "config": {"learning_rate": 3e-4, "n_episodes": 2000}},
            {"type": "dqn", "config": {"learning_rate": 1e-3, "n_episodes": 2000}},
            {"type": "a2c", "config": {"learning_rate": 7e-4, "n_episodes": 2000}}
        ]
        """
        self.env = create_environment(env_config)
        self.agent_configs = agent_configs
        self.results = []

    def run_sequential_tests(self) -> List[Dict]:
        """Sequentially run tests for all agents"""
        results = []

        for i, agent_config in enumerate(self.agent_configs):
            print(f"Testing algorithm {i+1}/{len(self.agent_configs)}: {agent_config['type']}")

            # Create agent
            agent = AgentFactory.create_agent(
                agent_config["type"],
                self.env.observation_space.shape[0],
                self.env.action_space.n,
                agent_config.get("config", {})
            )

            # Train agent
            n_episodes = agent_config.get("config", {}).get("n_episodes", 1000)
            training_result = self._train_agent(agent, n_episodes)

            # Record results
            result = {
                "agent_type": agent_config["type"],
                "config": agent_config.get("config", {}),
                "training_result": training_result
            }
            results.append(result)

            # Reset environment for next agent
            self.env.reset()

        self.results = results
        return results

    def _train_agent(self, agent: RLAgent, n_episodes: int) -> Dict:
        """Train single agent"""
        training_history = []

        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, action_info = agent.act(state)
                next_state, reward, done, info = self.env.step(action)

                # Learn
                experience = {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                }
                agent.learn(experience)

                state = next_state
                total_reward += reward

            training_history.append({
                "episode": episode,
                "total_reward": total_reward,
                "info": info
            })

        return {
            "training_history": training_history,
            "final_performance": training_history[-1] if training_history else None
        }

    def compare_results(self) -> pd.DataFrame:
        """Compare results of different algorithms"""
        # Convert results to DataFrame for analysis
        pass
```

### 3.4 UCC Search Module (Phase 1 Core)

#### Functional Description
Reinforcement learning search specifically for UCC architecture, supporting standard UCC and efficient variants (e.g., sQEB).

#### Search Environment Interface
```python
class UCCSearchEnv(gym.Env):
    """
    UCC-specific search environment

    Action space:
    - Select excitation type (single excitation, double excitation, ...)
    - Select specific excitation operator
    - Select parameter initial value

    State space:
    - Current circuit structure encoding (matrix-based primarily, supporting other encoding experiments)
    - Current energy estimate
    - Used resources (depth, gate count, parameter count)
    - Molecular feature vector
    - Encoding method identifier (facilitating comparison of different encoding strategies)
    """

    def __init__(
        self,
        molecule_data: MoleculeData,
        config: Dict = None
    ):
        """
        Configuration example:
        {
            "max_depth": 10,           # Maximum circuit depth
            "max_excitations": 20,     # Maximum number of excitation operators
            "use_sqeb": True,          # Whether to use sQEB variant
            "param_init_strategy": "random"  # Parameter initialization strategy
        }
        """
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Execute action: add UCC excitation operator
        # Return: new state, reward, done flag, info
        pass

    def reset(self) -> np.ndarray:
        # Reset environment: clear circuit, return initial state
        pass
```

#### UCC Circuit Builder
```python
class UCCCircuitBuilder:
    def __init__(self, molecule_data: MoleculeData):
        self.molecule_data = molecule_data
        self.excitation_pool = self._generate_excitation_pool()

    def _generate_excitation_pool(self) -> List[ExcitationOperator]:
        """
        Generate UCC excitation operator pool

        Strategy:
        1. Generate all possible single and double excitations
        2. Optional pre-screening (based on MP2 importance)
        3. Optional sQEB transformation
        """
        pass

    def add_excitation(
        self,
        circuit: QuantumCircuit,
        excitation_idx: int,
        param_value: float
    ) -> QuantumCircuit:
        """Add specified excitation operator to circuit"""
        pass

    def build_circuit(
        self,
        excitation_sequence: List[Tuple[int, float]]
    ) -> QuantumCircuit:
        """Build complete UCC circuit from excitation sequence"""
        pass
```

#### Reward Function Design
```python
class UCCRewardFunction:
    def __init__(self, config: Dict = None):
        """
        Configuration example:
        {
            "accuracy_weight": 0.6,
            "depth_weight": 0.25,
            "gate_weight": 0.15,
            "use_intermediate_rewards": True
        }
        """
        self.config = config or {}

    def compute_reward(
        self,
        circuit: QuantumCircuit,
        energy: float,
        fci_energy: float,
        step_info: Dict
    ) -> float:
        """
        Compute reward for UCC search

        Reward = w1 * accuracy_reward + w2 * depth_penalty + w3 * gate_penalty
        """
        # Accuracy reward (logarithmic form)
        error = abs(energy - fci_energy)
        if abs(fci_energy) > 1e-10:
            relative_error = error / abs(fci_energy)
        else:
            relative_error = error

        accuracy_reward = -np.log(relative_error + 1e-8)

        # Depth penalty
        depth = circuit.depth()
        depth_penalty = -0.05 * depth

        # Gate count penalty (double-qubit gates weighted higher)
        n_single = count_single_qubit_gates(circuit)
        n_double = count_double_qubit_gates(circuit)
        gate_penalty = -0.01 * (n_single + 3.0 * n_double)

        weights = self.config
        total_reward = (
            weights.get("accuracy_weight", 0.6) * accuracy_reward +
            weights.get("depth_weight", 0.25) * depth_penalty +
            weights.get("gate_weight", 0.15) * gate_penalty
        )

        return total_reward
```

#### UCC Search Controller
```python
class UCCSearchController:
    def __init__(
        self,
        molecule_data: MoleculeData,
        agent_type: str = "ppo",
        config: Dict = None
    ):
        self.env = UCCSearchEnv(molecule_data, config)
        self.agent = AgentFactory.create_agent(
            agent_type,
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            config.get("agent_config", {}) if config else {}
        )

    def search(
        self,
        n_episodes: int = 1000,
        early_stop_threshold: float = 1.6e-3  # Chemical accuracy threshold
    ) -> SearchResult:
        """
        Execute UCC architecture search

        Returns:
        - best_circuit: Optimal circuit
        - best_energy: Optimal energy
        - training_history: Training history
        - performance_metrics: Performance metrics
        """
        pass

    def save_results(self, path: str):
        """Save search results"""
        pass
```

### 3.5 HEA Search Module

#### Functional Description
Reinforcement learning search specifically for HEA architecture, supporting different entanglement patterns and parameterization strategies.

#### Search Environment Interface
```python
class HEASearchEnv(gym.Env):
    """
    HEA-specific search environment

    Action space:
    - Select single-qubit gate type (Rx, Ry, Rz)
    - Select target qubit
    - Select parameter value
    - Select entanglement pattern (linear, circular, fully connected)

    State space:
    - Current circuit structure encoding (matrix-based priority, supporting encoding experiments)
    - Layer information and current depth
    - Entanglement pattern history
    - Molecular feature vector
    - Encoding configuration information
    """

    def __init__(
        self,
        molecule_data: MoleculeData,
        config: Dict = None
    ):
        """
        Configuration example:
        {
            "max_layers": 8,              # Maximum number of layers
            "rotation_gates": ["rx", "ry", "rz"],
            "entanglement_patterns": ["linear", "circular"],
            "parameter_sharing": "layerwise"  # Parameter sharing strategy
        }
        """
        pass
```

#### HEA Circuit Builder
```python
class HEACircuitBuilder:
    def __init__(self, n_qubits: int, config: Dict = None):
        self.n_qubits = n_qubits
        self.config = config or {}

    def add_layer(
        self,
        circuit: QuantumCircuit,
        rotation_gates: List[str],      # Rotation gate for each qubit
        entanglement_pattern: str,      # Entanglement pattern
        entanglement_gate: str = "cx"   # Entanglement gate type
    ) -> QuantumCircuit:
        """Add an HEA layer to the circuit"""
        pass

    def build_circuit(
        self,
        layer_specs: List[Dict]  # Specifications for each layer
    ) -> QuantumCircuit:
        """Build complete HEA circuit based on layer specifications"""
        pass
```

### 3.6 Hybrid Search Module

#### Functional Description
Intelligent fusion of HEA and UCC architecture search module.

#### Fusion Strategy
```python
class HybridFusionStrategy:
    def __init__(self, config: Dict = None):
        """
        Fusion strategy configuration:
        {
            "fusion_mode": "sequential",  # sequential, parallel, conditional
            "min_ucc_components": 1,
            "max_ucc_components": 5,
            "hea_layers_per_block": 2
        }
        """
        pass

    def fuse_circuits(
        self,
        hea_circuit: QuantumCircuit,
        ucc_circuit: QuantumCircuit
    ) -> QuantumCircuit:
        """Fuse HEA and UCC circuits"""
        pass

    def generate_fusion_template(self) -> List[str]:
        """
        Generate fusion template

        Examples:
        - ["HEA", "UCC", "HEA"]: HEA layer → UCC component → HEA layer
        - ["HEA_UCC", "HEA_UCC"]: Parallel fusion
        """
        pass
```

### 3.7 Evaluation and Validation Module

#### Functional Description
Standardized evaluation of quantum circuit performance, generating detailed performance reports.

#### Evaluation Metrics
```python
@dataclass
class CircuitMetrics:
    """Quantum circuit performance metrics"""
    energy: float                      # VQE calculated energy
    fci_energy: float                  # Exact FCI energy
    absolute_error: float              # Absolute error (Hartree)
    relative_error: float              # Relative error (%)
    achieves_chemical_accuracy: bool   # Whether chemical accuracy is achieved
    circuit_depth: int                 # Circuit depth
    n_single_qubit_gates: int          # Number of single-qubit gates
    n_double_qubit_gates: int          # Number of double-qubit gates
    total_gates: int                   # Total number of gates
    n_parameters: int                  # Number of parameters
    hardware_friendliness: float       # Hardware friendliness score (0-1)
    training_steps: int                # Training steps
    training_time: float               # Training time (seconds)
```

#### Evaluator Interface
```python
class CircuitEvaluator:
    def __init__(self, simulator: QuantumSimulator):
        self.simulator = simulator

    def evaluate_circuit(
        self,
        circuit: QuantumCircuit,
        hamiltonian: QubitOperator,
        fci_energy: float
    ) -> CircuitMetrics:
        """Comprehensively evaluate quantum circuit performance"""
        pass

    def compute_hardware_friendliness(
        self,
        depth: int,
        n_double_gates: int,
        n_qubits: int
    ) -> float:
        """
        Calculate hardware friendliness score

        Formula:
        score = 0.6 * (1 / (1 + depth/n_qubits)) +
                0.4 * (1 / (1 + n_double_gates/(n_qubits^2)))
        """
        depth_score = 1.0 / (1.0 + depth / n_qubits)
        gate_score = 1.0 / (1.0 + n_double_gates / (n_qubits ** 2))
        return 0.6 * depth_score + 0.4 * gate_score
```

#### Statistical Analysis
```python
class StatisticalAnalyzer:
    def __init__(self):
        pass

    def analyze_experiment_results(
        self,
        results: List[CircuitMetrics],
        baseline_results: Optional[List[CircuitMetrics]] = None
    ) -> Dict:
        """
        Statistical analysis of experiment results

        Returns:
        - Mean ± standard deviation
        - Statistical significance test (t-test)
        - Effect size calculation (Cohen's d)
        - Confidence intervals
        """
        pass

    def generate_report(
        self,
        analysis_results: Dict,
        format: str = "markdown"
    ) -> str:
        """Generate statistical analysis report"""
        pass
```

### 3.8 Experiment Management Module

#### Functional Description
Manage configuration, execution, and result collection for RLQAS experiments.

#### Experiment Configuration
```yaml
# Example experiment configuration file (experiment_config.yaml)
experiment:
  name: "ucc_search_lih"
  description: "UCC architecture search test - LiH molecule"

molecule:
  formula: "LiH"
  bond_length: 1.6
  active_space: [2, 2]  # Number of active electrons, number of active orbitals
  basis_set: "sto-3g"
  transform: "parity"

search:
  ansatz_type: "UCC"
  max_depth: 12
  use_sqeb: true
  max_excitations: 15

rl:
  # Single algorithm test configuration
  agent_type: "ppo"
  n_episodes: 2000

  # Multi-algorithm sequential test configuration (optional)
  multi_agent_test: false  # Whether to perform multi-algorithm sequential testing
  # If multi_agent_test is true, use the agents list and ignore agent_type above
  agents:
    - type: "ppo"
      config:
        learning_rate: 3e-4
        gamma: 0.99
    - type: "dqn"
      config:
        learning_rate: 1e-3
        gamma: 0.99
    - type: "a2c"
      config:
        learning_rate: 7e-4
        gamma: 0.99

simulation:
  engine: "ci_vector"
  precision: 1e-8
  max_memory_gb: 32

evaluation:
  n_repeats: 10  # Number of experiment repeats
  metrics_to_collect:
    - "energy_error"
    - "circuit_depth"
    - "n_double_gates"
    - "training_time"

output:
  directory: "./results/lih_ucc_search"
  save_circuits: true
  save_training_logs: true
  generate_plots: true
```

#### Experiment Manager
```python
class ExperimentManager:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.results_db = ResultsDatabase()

    def _load_config(self, config_path: str) -> Dict:
        """Load experiment configuration file"""
        pass

    def run_experiment(self) -> ExperimentResult:
        """Execute experiment"""
        # 1. Process molecule
        molecule_data = process_molecule(**self.config["molecule"])

        # 2. Create simulator
        simulator = SimulatorFactory.create_simulator(
            molecule_data.n_qubits,
            self.config["simulation"]["max_memory_gb"]
        )

        # 3. Execute search
        if self.config["search"]["ansatz_type"] == "UCC":
            search_module = UCCSearchController(
                molecule_data,
                self.config["rl"]["agent_type"],
                self.config
            )
        elif self.config["search"]["ansatz_type"] == "HEA":
            search_module = HEASearchController(
                molecule_data,
                self.config["rl"]["agent_type"],
                self.config
            )

        # 4. Repeat experiments
        all_results = []
        for i in range(self.config["evaluation"]["n_repeats"]):
            result = search_module.search(
                n_episodes=self.config["rl"]["n_episodes"]
            )
            all_results.append(result)

        # 5. Statistical analysis
        analyzer = StatisticalAnalyzer()
        analysis = analyzer.analyze_experiment_results(all_results)

        # 6. Save results
        self._save_results(all_results, analysis)

        return ExperimentResult(results=all_results, analysis=analysis)

    def run_batch_experiments(self, config_paths: List[str]):
        """Batch run multiple experiment configurations"""
        pass
```

### 3.9 Command Line Interface Module

#### Functional Description
Provide user-friendly command-line tools for easy use of the RLQAS system.

#### Command Line Interface
```bash
# Basic search command
rlqas search --molecule LiH --bond-length 1.6 --ansatz UCC

# Specify RL algorithm
rlqas search --molecule BeH2 --ansatz HEA --agent ppo

# Multi-algorithm sequential testing
rlqas search --molecule H4 --ansatz UCC --multi-agent --agents ppo,dqn,a2c

# Use configuration file
rlqas experiment --config experiment_config.yaml

# Batch experiments
rlqas batch --molecules LiH,BeH2,H4 --ansatzes UCC,HEA

# Result analysis
rlqas analyze --results-dir ./results --format html
```

#### Python API
```python
import rlqas

# Simple API
result = rlqas.search(
    molecule="LiH",
    bond_length=1.6,
    ansatz_type="UCC",
    agent_type="ppo"
)

# Advanced API
experiment = rlqas.Experiment(
    molecule_config={"formula": "BeH2", "bond_length": 1.3},
    search_config={"ansatz_type": "UCC", "max_depth": 15},
    rl_config={"agent_type": "ppo", "n_episodes": 1500}
)

results = experiment.run()
analysis = experiment.analyze()
```

## 4. Implementation Priority (Phase Planning)

### Phase 1: Core Functionality - UCC Search Priority
**Goal**: Implement basic UCC search functionality, validate core workflow on LiH

1. **Molecule Processing Module**
   - Tencirchem-ng 2024.10 integration
   - Molecule data class definition and interfaces

2. **Quantum Simulator Module**
   - CI vector engine integration (Tencirchem-ng)
   - Simulator factory and configuration interfaces

3. **RL Agent Module**
   - PPO basic implementation (algorithm correctness priority)
   - Standard RL agent interfaces

4. **UCC Search Module** (Core)
   - UCC environment implementation (state/action space definition)
   - Basic search controller
   - LiH validation test

**Deliverables**:
- Runnable UCC search prototype system
- LiH test results (verifying chemical accuracy achievability)
- Basic evaluation and debugging tools

### Phase 2: Extended Functionality - Multi-algorithm and HEA Support
**Goal**: Support multiple RL algorithms and HEA architecture search

1. **Multi-RL Algorithm Support**
   - DQN, A2C, SAC algorithm implementation
   - Sequential testing framework (SequentialRLTester)
   - Basic algorithm performance comparison

2. **HEA Search Module**
   - HEA environment implementation (different entanglement patterns)
   - HEA circuit builder

3. **Experiment Management System**
   - Configuration file support (YAML/JSON)
   - Result data collection and storage

**Testing System**:
- H₂ (2 qubits): Rapid functional validation
- LiH (4 qubits): Core performance testing
- BeH₂ (6 qubits): Preliminary scalability testing

### Phase 3: Advanced Functionality - Hybrid Architecture and Performance Optimization
**Goal**: Hybrid architecture search and system performance optimization

1. **Hybrid Search Module**
   - HEA-UCC fusion strategy
   - Intelligent architecture selection and combination

2. **Performance Optimization**
   - CI vector performance testing and optimization
   - Batch evaluation support
   - Memory usage optimization

3. **Circuit Encoding Optimization**
   - Matrix encoding implementation (performance priority)
   - Optional other encoding method experiments

**Testing System**:
- BeH₂ (10-14 qubits): Medium system performance testing
- H₄ (8 qubits): Hydrogen chain correlation effect testing
- H₆ (12 qubits): Large system scalability testing

### Phase 4: Productionization - User Interface and Complete Ecosystem
**Goal**: User-friendly interfaces and complete documentation ecosystem

1. **Command Line Tool**
   - CLI implementation (rlqas command)
   - Python API improvement

2. **Documentation and Examples**
   - User guides and tutorials
   - API documentation automatic generation
   - Example scripts and case studies

3. **Test Suite Improvement**
   - Complete unit tests
   - Integration test suite
   - Performance benchmark tests

**Flexible Schedule**:
- Each Phase time adjustable based on actual progress
- Focus on ensuring core functionality (Phase 1) quality
- Extended functionality adjustable as needed

## 5. Testing Plan

### 5.1 Test Molecular Systems

| Test Stage | Molecule | Qubit Count | System Characteristics | Test Purpose |
|------------|----------|-------------|-----------------------|--------------|
| **Unit Test** | H₂ | 2 | Simplest system | Basic functionality validation |
| **Integration Test** | LiH | 4 | Ionic bond, medium complexity | Core functionality testing |
| **Performance Test** | BeH₂ | 6-14 | Adjustable active space | Scalability verification |
| **Correlation Test** | H₄ | 8 | Hydrogen chain, strong correlation | Correlation effect handling |
| **Stress Test** | H₆ | 12 | Larger hydrogen chain | Large system performance |

### 5.2 BeH₂ Test Configuration
```python
beh2_test_configs = [
    {
        "formula": "BeH2",
        "bond_length": 1.3,
        "active_space": (4, 4),  # 6 qubits
        "basis_set": "sto-3g"
    },
    {
        "formula": "BeH2",
        "bond_length": 1.3,
        "active_space": (6, 6),  # 10 qubits
        "basis_set": "sto-3g"
    },
    {
        "formula": "BeH2",
        "bond_length": 1.3,
        "active_space": (8, 8),  # 14 qubits
        "basis_set": "sto-3g"
    }
]
```

### 5.3 Hydrogen Chain Test Configuration
```python
hchain_test_configs = [
    {"formula": "H4", "geometry": "linear", "bond_length": 0.74},
    {"formula": "H6", "geometry": "linear", "bond_length": 0.74},
    {"formula": "H8", "geometry": "linear", "bond_length": 0.74}
]
```

### 5.4 Test Metrics
1. **Functional Correctness**:
   - Molecule processing correctness
   - Circuit construction correctness
   - Energy calculation correctness

2. **Performance Metrics**:
   - Chemical accuracy achievement rate (>90%)
   - Training convergence time
   - Memory usage efficiency

3. **Scalability Metrics**:
   - Qubit count scalability
   - Different RL algorithm comparison
   - Different architecture type comparison

## 6. Success Criteria

### 6.1 Technical Success Criteria
1. **Phase 1 Completion**:
   - UCC search achieves chemical accuracy on LiH
   - Single experiment runtime < 2 hours
   - Code coverage > 70%

2. **Phase 2 Completion**:
   - Support at least 3 RL algorithms
   - HEA search functionality normal
   - Sequential testing framework working

3. **Phase 3 Completion**:
   - Hybrid architecture search implemented
   - BeH₂ (14 qubits) test passed
   - Performance optimization effects significant

4. **Phase 4 Completion**:
   - Complete command-line tool
   - User documentation complete
   - Example scripts usable

### 6.2 Acceptance Conditions
1. **Functional Acceptance**:
   - All module interfaces comply with specifications
   - Configuration files correctly parsed
   - Result data format unified

2. **Performance Acceptance**:
   - 30% faster than baseline RLQAS methods
   - Memory usage within configured range
   - Supports statistical analysis of 10 repeat experiments

3. **Quality Acceptance**:
   - Unit test coverage > 80%
   - Code complies with PEP8 standards
   - Documentation complete and accurate

## 7. Risk Management

### 7.1 Technical Risks
1. **Tencirchem Integration Issues**:
   - Risk: CI vector engine interface incompatibility
   - Mitigation: Early integration validation, prepare backup solutions

2. **RL Training Instability**:
   - Risk: Quantum architecture search difficulty high, training divergence
   - Mitigation: Implement multiple exploration strategies, provide expert demonstration options

3. **Large System Performance Issues**:
   - Risk: 14+ qubit computation time too long
   - Mitigation: Optimize CI vector usage, implement checkpoint saving

### 7.2 Schedule Risks
1. **Module Dependency Issues**:
   - Risk: Inter-module interface changes affect schedule
   - Mitigation: Clearly define interface contracts, early integration testing

2. **Algorithm Debugging Time**:
   - Risk: RL algorithm tuning time-consuming
   - Mitigation: Provide reasonable default parameters, implement automatic tuning tools

## 8. Future Extensions

### 8.1 Short-term Extensions (Within Project)
1. **More Quantum Chemistry Features**:
   - Support more basis sets
   - Support more molecular geometries

2. **Advanced RL Features**:
   - Hierarchical reinforcement learning
   - Multi-agent collaboration

3. **Visualization Tools**:
   - Training curve real-time display
   - Circuit structure visualization

### 8.2 Long-term Extensions (Future Projects)
1. **Real Hardware Integration**:
   - Quantum processor backend support
   - Hardware noise model integration

2. **Larger System Support**:
   - Distributed computing support
   - More efficient approximation methods

3. **Application Extensions**:
   - Materials science applications
   - Quantum chemical dynamics

## Appendix

### A. Complete Configuration File Example
```yaml
# Complete experiment configuration file example
version: "1.0"
experiment:
  name: "full_test_suite"
  description: "Complete test suite - UCC and HEA comparison"

molecules:
  - formula: "LiH"
    bond_length: 1.6
    active_space: [2, 2]

  - formula: "BeH2"
    bond_length: 1.3
    active_space: [6, 6]

  - formula: "H4"
    geometry: "linear"
    bond_length: 0.74

ansatzes: ["UCC", "HEA"]

rl:
  multi_agent_test: true  # Multi-algorithm sequential testing
  agents:
    - type: "ppo"
      config:
        learning_rate: 3e-4
        n_episodes: 2000

    - type: "dqn"
      config:
        learning_rate: 1e-3
        n_episodes: 2000

simulation:
  engine: "ci_vector"
  precision: 1e-8
  max_memory_gb: 64

evaluation:
  n_repeats: 10
  metrics:
    - "energy_error"
    - "circuit_depth"
    - "training_time"
    - "convergence_steps"

output:
  directory: "./full_test_results"
  formats: ["csv", "json", "pdf"]
```

### B. Dependency Package List
```txt
Core dependencies:
- python>=3.8
- numpy>=1.21
- scipy>=1.7
- pandas>=1.3

Quantum computing:
- tencirchem-ng>=2024.10  # Core quantum chemistry engine
- openfermion>=1.5        # Fermionic operator processing
- qiskit>=0.34            # Optional, for circuit visualization and basic functions

Reinforcement learning:
- torch>=1.9
- gym>=0.21
- stable-baselines3>=1.6  # Optional, reference implementation

Utility libraries:
- pyyaml>=6.0
- matplotlib>=3.5
- tqdm>=4.62
```

### C. Development Guidelines
1. **Code Standards**:
   - Follow PEP8
   - Use type hints
   - Complete docstrings

2. **Testing Requirements**:
   - Each module has unit tests
   - Integration tests cover main workflows
   - Performance test benchmarks

3. **Documentation Requirements**:
   - API documentation auto-generated
   - User guides complete
   - Example scripts abundant

---

**Document Update Record**:
- v1.1 (2026-02-05): Optimized based on user feedback
  - Clarified Tencirchem-ng 2024.10 version
  - Configurable simulator interface, adapting to different scale systems
  - RL algorithms support sequential testing, simplified parallel requirements
  - Circuit encoding primarily matrix-based, optional other encoding methods
  - Removed specific time estimates, maintaining flexibility
- v1.0 (2026-02-05): Created modular requirements specification, supporting block-based Ralph implementation

**Next Actions**:
1. Review structure and completeness of this requirements specification
2. Create specific Ralph tasks based on this specification
3. Begin Phase 1 implementation (UCC search core functionality)
```