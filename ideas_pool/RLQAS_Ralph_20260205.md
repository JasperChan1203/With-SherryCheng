# RLQAS 模块化需求规范

## 版本信息
- **版本**: 1.1
- **创建日期**: 2026-02-05
- **更新日期**: 2026-02-05
- **状态**: 优化草案
- **目标**: 指导Ralph分块实现RLQAS系统
- **关键更新**:
  - 明确Tencirchem-ng 2024.10版本
  - 模拟器接口可配置化，适应不同规模体系
  - RL算法支持顺序测试，简化并行要求
  - 线路编码支持矩阵形式为主，可选其他编码
  - 移除具体时间预设，保持灵活性

## 1. 项目概述

### 1.1 项目目标
开发一个模块化的强化学习量子架构搜索（RLQAS）系统，能够根据输入的分子信息和线路类型，自动搜索并优化量子线路结构。

### 1.2 核心功能
1. **功能性道具设计**：
   - 输入：分子式、键长、所需线路类型（HEA/UCC/混合）
   - 输出：优化后的量子线路 + VQE计算结果

2. **插件式RL支持**：
   - 支持多种RL算法的顺序测试（PPO, DQN, A2C, SAC）
   - 统一的RL智能体接口，便于算法切换和比较
   - 算法实现正确性优先，并行测试为可选功能

3. **模块化QAS**：
   - 独立的HEA搜索模块
   - 独立的UCC搜索模块（Phase 1优先）
   - 智能混合架构搜索模块

4. **高性能量子模拟**：
   - 集成Tencirchem-ng 2024.10 CI vector引擎
   - 支持可配置的模拟接口，适应不同规模体系（4-20+量子比特）
   - 自动选择最优模拟策略，支持性能测试验证
   - 线路编码以矩阵形式为主，支持多种编码方式实验

### 1.3 性能目标
1. **精度目标**：
   - LiH（4量子比特）：化学精度（误差 < 1.6 mHa）
   - BeH₂（6-14量子比特）：保持可接受的精度损失
   - 氢链分子（H₄, H₆, H₈）：验证扩展性

2. **效率目标**：
   - 比基础RLQAS方法训练速度快30%以上
   - 支持并行实验配置

3. **扩展性目标**：
   - 架构支持扩展到20+量子比特
   - 模块化设计便于功能扩展

## 2. 系统架构

```
RLQAS 系统架构
├── 用户接口层
│   ├── 命令行工具 (CLI)
│   ├── Python API
│   └── 配置文件 (YAML/JSON)
│
├── 核心功能层 (模块化)
│   ├── 分子处理模块
│   ├── 量子模拟器模块 (Tencirchem CI vector)
│   ├── RL智能体模块 (插件式)
│   ├── UCC搜索模块 (Phase 1核心)
│   ├── HEA搜索模块
│   ├── 混合搜索模块
│   ├── 评估验证模块
│   └── 实验管理模块
│
└── 数据存储层
    ├── 实验结果数据库
    ├── 训练日志
    └── 预训练模型库
```

## 3. 模块详细需求

### 3.1 分子处理模块

#### 功能描述
处理分子信息，生成量子计算所需的输入数据。

#### 输入接口
```python
def process_molecule(
    molecule: str,           # 分子式，如 "LiH", "BeH2", "H4"
    bond_length: float,      # 键长（Å）
    ansatz_type: str,        # "UCC", "HEA", "MIXED"
    active_space: Optional[Tuple[int, int]] = None,  # (活性电子数, 活性轨道数)
    basis_set: str = "sto-3g",  # 基组
    transform: str = "parity"   # 费米子-量子比特变换
) -> MoleculeData
```

#### 输出数据类
```python
@dataclass
class MoleculeData:
    hamiltonian: QubitOperator      # 量子比特哈密顿量
    n_qubits: int                   # 量子比特数
    reference_state: np.ndarray     # 参考态（Hartree-Fock）
    fci_energy: float               # 精确FCI能量
    molecular_info: Dict            # 原始分子信息
```

#### 依赖关系
- Tencirchem (分子积分、哈密顿量生成)
- OpenFermion (费米子算符处理)

#### 测试要求
- H₂: 2量子比特，验证基本功能
- LiH: 4量子比特，核心测试案例
- BeH₂: 6-14量子比特，活性空间测试

### 3.2 量子模拟器模块 (Tencirchem CI vector)

#### 功能描述
提供高效的量子线路模拟，基于Tencirchem-ng 2024.10的CI vector引擎。支持可配置的模拟接口，适应不同规模体系（4-20+量子比特），具体性能通过实际测试验证。

#### 模拟器接口
```python
class QuantumSimulator(ABC):
    @abstractmethod
    def compute_energy(
        self,
        circuit: QuantumCircuit,    # 量子线路
        hamiltonian: QubitOperator, # 哈密顿量
        initial_state: Optional[np.ndarray] = None
    ) -> float:
        """计算线路的能量期望值"""
        pass

    @abstractmethod
    def get_max_qubits(self) -> int:
        """返回支持的量子比特上限"""
        pass

    @abstractmethod
    def estimate_memory(self, n_qubits: int) -> float:
        """估计内存使用（GB）"""
        pass
```

#### Tencirchem实现
```python
class TencirchemCISimulator(QuantumSimulator):
    def __init__(self, config: Dict = None):
        """
        基于Tencirchem-ng 2024.10的CI vector引擎

        配置示例:
        {
            "engine": "ci_vector",           # 主要引擎
            "precision": 1e-8,               # 计算精度
            "use_symmetry": True,            # 使用对称性加速
            "max_memory_gb": 32,             # 最大内存限制
            "fallback_method": "statevector" # 备用方法（如CI vector性能不足时）
        }

        注意：CI vector在20+量子比特的性能需要通过实际测试验证，
              系统应支持在测试中自动选择最优模拟策略。
        """
        self.config = config or {}

    def compute_energy(self, circuit, hamiltonian, initial_state=None):
        # 调用Tencirchem CI vector引擎
        # 将量子线路转换为Tencirchem表示
        # 返回能量期望值
        pass
```

#### 模拟器工厂
```python
class SimulatorFactory:
    @staticmethod
    def create_simulator(
        n_qubits: int,
        config: Dict = None
    ) -> QuantumSimulator:
        """
        根据体系规模和配置创建模拟器

        策略:
        1. 优先使用配置指定的模拟器类型
        2. 根据量子比特数和可用内存自动选择
        3. 支持性能测试验证不同模拟器的实际表现

        配置示例:
        {
            "preferred_engine": "ci_vector",  # 首选引擎
            "available_memory_gb": 32,        # 可用内存
            "require_exact": False,           # 是否要求精确模拟
            "test_performance": True          # 是否进行性能测试
        }
        """
        config = config or {}

        # 如果配置指定了引擎，直接创建
        engine = config.get("preferred_engine", "ci_vector")

        if engine == "ci_vector":
            return TencirchemCISimulator(config)
        elif engine == "statevector":
            return StatevectorSimulator(config)
        else:
            # 默认使用CI vector
            return TencirchemCISimulator(config)
```

#### 性能要求与测试
- **目标性能**:
  - 单次能量评估时间：< 100ms (8量子比特)
  - 内存使用：可配置上限（默认32GB）
  - 支持批量评估（用于RL训练）

- **实际测试验证**:
  - CI vector在14-20量子比特的性能需要通过实际测试验证
  - 系统应记录不同模拟器的实际性能指标
  - 支持根据测试结果动态选择最优模拟策略

- **编码方式支持**:
  - 线路编码以矩阵形式为主（性能最优）
  - 支持实验其他编码方式（如张量网络、稀疏表示）
  - 编码方式应可配置，便于性能比较

### 3.3 RL智能体模块 (插件式)

#### 功能描述
实现标准RL算法接口，支持多种算法的顺序测试和快速切换。算法正确实现优先，并行测试为可选高级功能。

#### 智能体接口
```python
class RLAgent(ABC):
    @abstractmethod
    def act(self, state: np.ndarray) -> Tuple[int, Dict]:
        """根据状态选择动作"""
        pass

    @abstractmethod
    def learn(self, experience: Dict) -> Dict:
        """从经验中学习"""
        pass

    @abstractmethod
    def save(self, path: str):
        """保存模型"""
        pass

    @abstractmethod
    def load(self, path: str):
        """加载模型"""
        pass
```

#### 支持的算法实现
```python
# PPO实现
class PPOAgent(RLAgent):
    def __init__(self, config: Dict):
        """
        配置示例:
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

# DQN实现
class DQNAgent(RLAgent):
    def __init__(self, config: Dict):
        """
        配置示例:
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

# A2C实现
class A2CAgent(RLAgent):
    pass

# SAC实现
class SACAgent(RLAgent):
    pass
```

#### 智能体工厂
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
        创建指定类型的RL智能体

        支持的agent_type:
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
            raise ValueError(f"不支持的智能体类型: {agent_type}")

        return agent_types[agent_type](state_dim, action_dim, config or {})
```

#### 顺序测试支持
```python
class SequentialRLTester:
    def __init__(self, env_config: Dict, agent_configs: List[Dict]):
        """
        顺序测试多个RL算法

        agent_configs示例:
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
        """顺序运行所有智能体的测试"""
        results = []

        for i, agent_config in enumerate(self.agent_configs):
            print(f"测试算法 {i+1}/{len(self.agent_configs)}: {agent_config['type']}")

            # 创建智能体
            agent = AgentFactory.create_agent(
                agent_config["type"],
                self.env.observation_space.shape[0],
                self.env.action_space.n,
                agent_config.get("config", {})
            )

            # 训练智能体
            n_episodes = agent_config.get("config", {}).get("n_episodes", 1000)
            training_result = self._train_agent(agent, n_episodes)

            # 记录结果
            result = {
                "agent_type": agent_config["type"],
                "config": agent_config.get("config", {}),
                "training_result": training_result
            }
            results.append(result)

            # 重置环境用于下一个智能体
            self.env.reset()

        self.results = results
        return results

    def _train_agent(self, agent: RLAgent, n_episodes: int) -> Dict:
        """训练单个智能体"""
        training_history = []

        for episode in range(n_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action, action_info = agent.act(state)
                next_state, reward, done, info = self.env.step(action)

                # 学习
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
        """比较不同算法的结果"""
        # 将结果转换为DataFrame便于分析
        pass
```

### 3.4 UCC搜索模块 (Phase 1核心)

#### 功能描述
专门针对UCC架构的强化学习搜索，支持标准UCC和高效变体（如sQEB）。

#### 搜索环境接口
```python
class UCCSearchEnv(gym.Env):
    """
    UCC专用搜索环境

    动作空间:
    - 选择激发类型 (单激发、双激发、...)
    - 选择具体的激发算符
    - 选择参数初始值

    状态空间:
    - 当前线路结构编码（矩阵形式为主，支持其他编码实验）
    - 当前能量估计
    - 已用资源（深度、门数、参数数）
    - 分子特征向量
    - 编码方式标识（便于不同编码策略比较）
    """

    def __init__(
        self,
        molecule_data: MoleculeData,
        config: Dict = None
    ):
        """
        配置示例:
        {
            "max_depth": 10,           # 最大线路深度
            "max_excitations": 20,     # 最大激发算符数
            "use_sqeb": True,          # 是否使用sQEB变体
            "param_init_strategy": "random"  # 参数初始化策略
        }
        """
        pass

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # 执行动作：添加UCC激发算符
        # 返回：新状态，奖励，完成标志，信息
        pass

    def reset(self) -> np.ndarray:
        # 重置环境：清空线路，返回初始状态
        pass
```

#### UCC线路构建器
```python
class UCCCircuitBuilder:
    def __init__(self, molecule_data: MoleculeData):
        self.molecule_data = molecule_data
        self.excitation_pool = self._generate_excitation_pool()

    def _generate_excitation_pool(self) -> List[ExcitationOperator]:
        """
        生成UCC激发算符池

        策略:
        1. 生成所有可能的单、双激发
        2. 可选的预筛选（基于MP2重要性）
        3. 可选的sQEB变换
        """
        pass

    def add_excitation(
        self,
        circuit: QuantumCircuit,
        excitation_idx: int,
        param_value: float
    ) -> QuantumCircuit:
        """向线路添加指定的激发算符"""
        pass

    def build_circuit(
        self,
        excitation_sequence: List[Tuple[int, float]]
    ) -> QuantumCircuit:
        """根据激发序列构建完整UCC线路"""
        pass
```

#### 奖励函数设计
```python
class UCCRewardFunction:
    def __init__(self, config: Dict = None):
        """
        配置示例:
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
        计算UCC搜索的奖励

        奖励 = w1 * 精度奖励 + w2 * 深度惩罚 + w3 * 门数惩罚
        """
        # 精度奖励（对数形式）
        error = abs(energy - fci_energy)
        if abs(fci_energy) > 1e-10:
            relative_error = error / abs(fci_energy)
        else:
            relative_error = error

        accuracy_reward = -np.log(relative_error + 1e-8)

        # 深度惩罚
        depth = circuit.depth()
        depth_penalty = -0.05 * depth

        # 门数惩罚（双量子比特门权重更高）
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

#### UCC搜索控制器
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
        early_stop_threshold: float = 1.6e-3  # 化学精度阈值
    ) -> SearchResult:
        """
        执行UCC架构搜索

        返回:
        - best_circuit: 最优线路
        - best_energy: 最优能量
        - training_history: 训练历史
        - performance_metrics: 性能指标
        """
        pass

    def save_results(self, path: str):
        """保存搜索结果"""
        pass
```

### 3.5 HEA搜索模块

#### 功能描述
专门针对HEA架构的强化学习搜索，支持不同的纠缠模式和参数化策略。

#### 搜索环境接口
```python
class HEASearchEnv(gym.Env):
    """
    HEA专用搜索环境

    动作空间:
    - 选择单量子比特门类型 (Rx, Ry, Rz)
    - 选择目标量子比特
    - 选择参数值
    - 选择纠缠模式 (线性、环状、全连接)

    状态空间:
    - 当前线路结构编码（矩阵形式优先，支持编码实验）
    - 层数信息和当前深度
    - 纠缠模式历史
    - 分子特征向量
    - 编码配置信息
    """

    def __init__(
        self,
        molecule_data: MoleculeData,
        config: Dict = None
    ):
        """
        配置示例:
        {
            "max_layers": 8,              # 最大层数
            "rotation_gates": ["rx", "ry", "rz"],
            "entanglement_patterns": ["linear", "circular"],
            "parameter_sharing": "layerwise"  # 参数共享策略
        }
        """
        pass
```

#### HEA线路构建器
```python
class HEACircuitBuilder:
    def __init__(self, n_qubits: int, config: Dict = None):
        self.n_qubits = n_qubits
        self.config = config or {}

    def add_layer(
        self,
        circuit: QuantumCircuit,
        rotation_gates: List[str],      # 每量子比特的旋转门
        entanglement_pattern: str,      # 纠缠模式
        entanglement_gate: str = "cx"   # 纠缠门类型
    ) -> QuantumCircuit:
        """向线路添加一个HEA层"""
        pass

    def build_circuit(
        self,
        layer_specs: List[Dict]  # 每层的规格
    ) -> QuantumCircuit:
        """根据层规格构建完整HEA线路"""
        pass
```

### 3.6 混合搜索模块

#### 功能描述
智能融合HEA和UCC架构的搜索模块。

#### 融合策略
```python
class HybridFusionStrategy:
    def __init__(self, config: Dict = None):
        """
        融合策略配置:
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
        """融合HEA和UCC线路"""
        pass

    def generate_fusion_template(self) -> List[str]:
        """
        生成融合模板

        示例:
        - ["HEA", "UCC", "HEA"]: HEA层 → UCC组件 → HEA层
        - ["HEA_UCC", "HEA_UCC"]: 并行融合
        """
        pass
```

### 3.7 评估验证模块

#### 功能描述
标准化评估量子线路性能，生成详细的性能报告。

#### 评估指标
```python
@dataclass
class CircuitMetrics:
    """量子线路性能指标"""
    energy: float                      # VQE计算能量
    fci_energy: float                  # 精确FCI能量
    absolute_error: float              # 绝对误差 (Hartree)
    relative_error: float              # 相对误差 (%)
    achieves_chemical_accuracy: bool   # 是否达到化学精度
    circuit_depth: int                 # 线路深度
    n_single_qubit_gates: int          # 单量子比特门数
    n_double_qubit_gates: int          # 双量子比特门数
    total_gates: int                   # 总门数
    n_parameters: int                  # 参数数量
    hardware_friendliness: float       # 硬件友好评分 (0-1)
    training_steps: int                # 训练步数
    training_time: float               # 训练时间 (秒)
```

#### 评估器接口
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
        """全面评估量子线路性能"""
        pass

    def compute_hardware_friendliness(
        self,
        depth: int,
        n_double_gates: int,
        n_qubits: int
    ) -> float:
        """
        计算硬件友好评分

        公式:
        score = 0.6 * (1 / (1 + depth/n_qubits)) +
                0.4 * (1 / (1 + n_double_gates/(n_qubits^2)))
        """
        depth_score = 1.0 / (1.0 + depth / n_qubits)
        gate_score = 1.0 / (1.0 + n_double_gates / (n_qubits ** 2))
        return 0.6 * depth_score + 0.4 * gate_score
```

#### 统计分析
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
        统计分析实验结果

        返回:
        - 均值 ± 标准差
        - 统计显著性检验 (t-test)
        - 效应量计算 (Cohen's d)
        - 置信区间
        """
        pass

    def generate_report(
        self,
        analysis_results: Dict,
        format: str = "markdown"
    ) -> str:
        """生成统计分析报告"""
        pass
```

### 3.8 实验管理模块

#### 功能描述
管理RLQAS实验的配置、执行和结果收集。

#### 实验配置
```yaml
# 实验配置文件示例 (experiment_config.yaml)
experiment:
  name: "ucc_search_lih"
  description: "UCC架构搜索测试 - LiH分子"

molecule:
  formula: "LiH"
  bond_length: 1.6
  active_space: [2, 2]  # 活性电子数, 活性轨道数
  basis_set: "sto-3g"
  transform: "parity"

search:
  ansatz_type: "UCC"
  max_depth: 12
  use_sqeb: true
  max_excitations: 15

rl:
  # 单算法测试配置
  agent_type: "ppo"
  n_episodes: 2000

  # 多算法顺序测试配置（可选）
  multi_agent_test: false  # 是否进行多算法顺序测试
  # 如果multi_agent_test为true，则使用agents列表，忽略上面的agent_type
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
  n_repeats: 10  # 重复实验次数
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

#### 实验管理器
```python
class ExperimentManager:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.results_db = ResultsDatabase()

    def _load_config(self, config_path: str) -> Dict:
        """加载实验配置文件"""
        pass

    def run_experiment(self) -> ExperimentResult:
        """执行实验"""
        # 1. 处理分子
        molecule_data = process_molecule(**self.config["molecule"])

        # 2. 创建模拟器
        simulator = SimulatorFactory.create_simulator(
            molecule_data.n_qubits,
            self.config["simulation"]["max_memory_gb"]
        )

        # 3. 执行搜索
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

        # 4. 重复实验
        all_results = []
        for i in range(self.config["evaluation"]["n_repeats"]):
            result = search_module.search(
                n_episodes=self.config["rl"]["n_episodes"]
            )
            all_results.append(result)

        # 5. 统计分析
        analyzer = StatisticalAnalyzer()
        analysis = analyzer.analyze_experiment_results(all_results)

        # 6. 保存结果
        self._save_results(all_results, analysis)

        return ExperimentResult(results=all_results, analysis=analysis)

    def run_batch_experiments(self, config_paths: List[str]):
        """批量运行多个实验配置"""
        pass
```

### 3.9 命令行接口模块

#### 功能描述
提供用户友好的命令行工具，方便快速使用RLQAS系统。

#### 命令行接口
```bash
# 基本搜索命令
rlqas search --molecule LiH --bond-length 1.6 --ansatz UCC

# 指定RL算法
rlqas search --molecule BeH2 --ansatz HEA --agent ppo

# 多算法顺序测试
rlqas search --molecule H4 --ansatz UCC --multi-agent --agents ppo,dqn,a2c

# 使用配置文件
rlqas experiment --config experiment_config.yaml

# 批量实验
rlqas batch --molecules LiH,BeH2,H4 --ansatzes UCC,HEA

# 结果分析
rlqas analyze --results-dir ./results --format html
```

#### Python API
```python
import rlqas

# 简单API
result = rlqas.search(
    molecule="LiH",
    bond_length=1.6,
    ansatz_type="UCC",
    agent_type="ppo"
)

# 高级API
experiment = rlqas.Experiment(
    molecule_config={"formula": "BeH2", "bond_length": 1.3},
    search_config={"ansatz_type": "UCC", "max_depth": 15},
    rl_config={"agent_type": "ppo", "n_episodes": 1500}
)

results = experiment.run()
analysis = experiment.analyze()
```

## 4. 实现优先级 (Phase规划)

### Phase 1: 核心功能 - UCC搜索优先
**目标**: 实现UCC搜索基本功能，在LiH上验证核心流程

1. **分子处理模块**
   - Tencirchem-ng 2024.10集成
   - 分子数据类定义和接口

2. **量子模拟器模块**
   - CI vector引擎集成（Tencirchem-ng）
   - 模拟器工厂和配置接口

3. **RL智能体模块**
   - PPO基础实现（算法正确性优先）
   - 标准RL智能体接口

4. **UCC搜索模块**（核心）
   - UCC环境实现（状态/动作空间定义）
   - 基础搜索控制器
   - LiH验证测试

**交付物**:
- 可运行的UCC搜索原型系统
- LiH测试结果（验证化学精度可达性）
- 基础评估和调试工具

### Phase 2: 扩展功能 - 多算法与HEA支持
**目标**: 支持多种RL算法和HEA架构搜索

1. **多RL算法支持**
   - DQN, A2C, SAC算法实现
   - 顺序测试框架（SequentialRLTester）
   - 算法性能比较基础

2. **HEA搜索模块**
   - HEA环境实现（不同纠缠模式）
   - HEA线路构建器

3. **实验管理系统**
   - 配置文件支持（YAML/JSON）
   - 结果数据收集和存储

**测试体系**:
- H₂ (2量子比特): 快速功能验证
- LiH (4量子比特): 核心性能测试
- BeH₂ (6量子比特): 扩展性初步测试

### Phase 3: 高级功能 - 混合架构与性能优化
**目标**: 混合架构搜索和系统性能优化

1. **混合搜索模块**
   - HEA-UCC融合策略
   - 智能架构选择和组合

2. **性能优化**
   - CI vector性能测试和优化
   - 批量评估支持
   - 内存使用优化

3. **线路编码优化**
   - 矩阵编码实现（性能优先）
   - 可选其他编码方式实验

**测试体系**:
- BeH₂ (10-14量子比特): 中等体系性能测试
- H₄ (8量子比特): 氢链关联效应测试
- H₆ (12量子比特): 大体系扩展测试

### Phase 4: 生产化 - 用户接口与完整生态
**目标**: 用户友好接口和完整文档生态

1. **命令行工具**
   - CLI实现（rlqas命令）
   - Python API完善

2. **文档和示例**
   - 用户指南和教程
   - API文档自动生成
   - 示例脚本和案例

3. **测试套件完善**
   - 完整单元测试
   - 集成测试套件
   - 性能基准测试

**灵活时间安排**:
- 各Phase时间根据实际进度调整
- 重点保证核心功能（Phase 1）质量
- 扩展功能可按需调整优先级

## 5. 测试计划

### 5.1 测试分子体系

| 测试阶段 | 分子 | 量子比特数 | 体系特点 | 测试目的 |
|---------|------|-----------|---------|---------|
| **单元测试** | H₂ | 2 | 最简单体系 | 基础功能验证 |
| **集成测试** | LiH | 4 | 离子键，中等复杂度 | 核心功能测试 |
| **性能测试** | BeH₂ | 6-14 | 活性空间可调 | 扩展性验证 |
| **关联测试** | H₄ | 8 | 氢链，强关联 | 关联效应处理 |
| **压力测试** | H₆ | 12 | 更大氢链 | 大体系性能 |

### 5.2 BeH₂测试配置
```python
beh2_test_configs = [
    {
        "formula": "BeH2",
        "bond_length": 1.3,
        "active_space": (4, 4),  # 6量子比特
        "basis_set": "sto-3g"
    },
    {
        "formula": "BeH2",
        "bond_length": 1.3,
        "active_space": (6, 6),  # 10量子比特
        "basis_set": "sto-3g"
    },
    {
        "formula": "BeH2",
        "bond_length": 1.3,
        "active_space": (8, 8),  # 14量子比特
        "basis_set": "sto-3g"
    }
]
```

### 5.3 氢链测试配置
```python
hchain_test_configs = [
    {"formula": "H4", "geometry": "linear", "bond_length": 0.74},
    {"formula": "H6", "geometry": "linear", "bond_length": 0.74},
    {"formula": "H8", "geometry": "linear", "bond_length": 0.74}
]
```

### 5.4 测试指标
1. **功能正确性**:
   - 分子处理正确性
   - 线路构建正确性
   - 能量计算正确性

2. **性能指标**:
   - 化学精度达标率 (>90%)
   - 训练收敛时间
   - 内存使用效率

3. **扩展性指标**:
   - 量子比特数扩展性
   - 不同RL算法对比
   - 不同架构类型对比

## 6. 成功标准

### 6.1 技术成功标准
1. **Phase 1完成**:
   - UCC搜索在LiH上达到化学精度
   - 单次实验运行时间 < 2小时
   - 代码覆盖率 > 70%

2. **Phase 2完成**:
   - 支持至少3种RL算法
   - HEA搜索功能正常
   - 顺序测试框架工作

3. **Phase 3完成**:
   - 混合架构搜索实现
   - BeH₂ (14量子比特)测试通过
   - 性能优化效果显著

4. **Phase 4完成**:
   - 完整命令行工具
   - 用户文档齐全
   - 示例脚本可用

### 6.2 验收条件
1. **功能验收**:
   - 所有模块接口符合规范
   - 配置文件正确解析
   - 结果数据格式统一

2. **性能验收**:
   - 比基础RLQAS方法快30%以上
   - 内存使用在配置范围内
   - 支持10次重复实验的统计分析

3. **质量验收**:
   - 单元测试覆盖率 > 80%
   - 代码符合PEP8规范
   - 文档完整且准确

## 7. 风险管理

### 7.1 技术风险
1. **Tencirchem集成问题**:
   - 风险: CI vector引擎接口不兼容
   - 缓解: 早期验证集成，准备备用方案

2. **RL训练不稳定**:
   - 风险: 量子架构搜索难度大，训练发散
   - 缓解: 实现多种探索策略，提供专家演示选项

3. **大体系性能问题**:
   - 风险: 14+量子比特计算时间过长
   - 缓解: 优化CI vector使用，实现检查点保存

### 7.2 进度风险
1. **模块依赖问题**:
   - 风险: 模块间接口变化影响进度
   - 缓解: 明确定义接口契约，早期集成测试

2. **算法调试时间**:
   - 风险: RL算法调优耗时
   - 缓解: 提供合理的默认参数，实现自动调优工具

## 8. 后续扩展

### 8.1 短期扩展 (项目内)
1. **更多量子化学特性**:
   - 支持更多基组
   - 支持更多分子几何

2. **高级RL功能**:
   - 分层强化学习
   - 多智能体协作

3. **可视化工具**:
   - 训练曲线实时显示
   - 线路结构可视化

### 8.2 长期扩展 (未来项目)
1. **真实硬件集成**:
   - 量子处理器后端支持
   - 硬件噪声模型集成

2. **更大体系支持**:
   - 分布式计算支持
   - 更高效的近似方法

3. **应用扩展**:
   - 材料科学应用
   - 量子化学动力学

## 附录

### A. 配置文件完整示例
```yaml
# 完整实验配置文件示例
version: "1.0"
experiment:
  name: "full_test_suite"
  description: "完整测试套件 - UCC和HEA对比"

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
  multi_agent_test: true  # 多算法顺序测试
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

### B. 依赖包清单
```txt
核心依赖:
- python>=3.8
- numpy>=1.21
- scipy>=1.7
- pandas>=1.3

量子计算:
- tencirchem-ng>=2024.10  # 核心量子化学引擎
- openfermion>=1.5        # 费米子算符处理
- qiskit>=0.34            # 可选，用于线路可视化和基础功能

强化学习:
- torch>=1.9
- gym>=0.21
- stable-baselines3>=1.6  # 可选，参考实现

工具库:
- pyyaml>=6.0
- matplotlib>=3.5
- tqdm>=4.62
```

### C. 开发指南
1. **代码规范**:
   - 遵循PEP8
   - 使用类型提示
   - 文档字符串齐全

2. **测试要求**:
   - 每个模块都有单元测试
   - 集成测试覆盖主要流程
   - 性能测试基准

3. **文档要求**:
   - API文档自动生成
   - 用户指南完整
   - 示例脚本丰富

---
**文档更新记录**:
- v1.1 (2026-02-05): 根据用户反馈优化
  - 明确Tencirchem-ng 2024.10版本
  - 模拟器接口可配置化，适应不同规模体系
  - RL算法支持顺序测试，简化并行要求
  - 线路编码支持矩阵形式为主，可选其他编码
  - 移除具体时间预设，保持灵活性
- v1.0 (2026-02-05): 创建模块化需求规范，支持分块Ralph实现

**下一步行动**:
1. 评审本需求规范的结构和完整性
2. 基于本规范创建具体的Ralph任务
3. 开始Phase 1实现（UCC搜索核心功能）
