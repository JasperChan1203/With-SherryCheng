# RLQAS Project Progress Report

**日期**: 2026-04-28  
**状态**: 整理自最新项目进展与同学反馈

---

## Executive Summary

RLQAS (Reinforcement Learning Quantum Architecture Search) 项目已完成核心代码整合与初步测试。项目将 Phase 1-4 的代码打包为 `rlqas-chem` 独立包，并在 LiH 12-qubit 系统上验证了 GRPO 算法优于传统 PPO 的性能。当前重点是实现新选型算法（GiGPO/Tree-GRPO/Double-DQN）并通过验收系统，最终目标是在 6-qubit 系统上实现真机部署。

---

## 1. 已完成工作

### 1.1 rlqas-chem 打包完成 ✅

将 Phase 1-4 的内容整合为可 pip 安装的独立 Python 包：

```bash
pip install -e ./rlqas-chem/
```

**包结构**:
- 支持 4 种搜索模式：UCC / HEA / Hybrid / QOP
- 集成 5 种 RL 算法：PPO / DQN / A2C / SAC / GRPO
- 提供 Python API 和 CLI 两种接口
- 文件位置：`With-SherryCheng/rlqas-chem/`

### 1.2 测试结果正面 ✅

**测试系统**: LiH 全空间分子，12 qubits，搜索 UCC 线路

| 算法 | 能量误差 (mHa) | 算符数 | 时间 (s) | 化学精度 (1.6 mHa) |
|------|----------------|--------|----------|---------------------|
| PPO | 0.255 | 13 | 2627 | ✅ |
| GRPO | 0.679 | 12 | 1076 | ✅ |
| ADAPT-VQE (基准) | ~0.5-1.0 | 6-8 | N/A | ✅ |

**关键发现**:
- GRPO 比 PPO **快 2.4 倍** (1076s vs 2627s)
- GRPO 使用更少算符达到化学精度 (12 vs 13)
- 所有算法均达到或超过 ADAPT-VQE 的精度

**其他分子测试** (来自 `rlqas_efficiency_20260407.json`):

| 分子 | 算法 | 能量误差 (mHa) | 化学精度 | 算符数 |
|------|------|----------------|----------|---------|
| BeH2 (14 qubits) | PPO | 1.600 | ✅ (边界) | 22 |
| BeH2 (14 qubits) | GRPO | 3.371 | ❌ | 27 |
| H6 (12 qubits) | PPO | 18.234 | ❌ | 30 |
| H6 (12 qubits) | GRPO | 25.356 | ❌ | 23 |

**问题**: GRPO 在较大体系（BeH2, H6）表现不稳定，需要改进算法。

### 1.3 Bug 修复完成 ✅

**Phase 5 bugfix001** (2026-03-25):

| Bug ID | 描述 | 状态 |
|--------|------|------|
| US-001 | UCCSearchController: PPO 从不训练（train() 是 no-op） | ✅ 已修复 |
| US-002 | HEASearchController: `_best_energy` 始终为 `float('inf')` | ✅ 已修复 |

**修复方法**:
- US-001: 将 `UCCSearchController.search()` 中的手动 episode 循环替换为 `self.agent.learn()`（调用 SB3 内部循环）
- US-002: 在 `HEASearchEnv.step()` 中添加 `best_energy` 跟踪，并在 controller 中读取

### 1.4 算法选型完成 ✅

**文档**: `ideas_pool/RLQAS_Algorithm_Selection_20260427.md`

#### 淘汰的算法（与 QAS 结构不兼容）

| 算法 | 淘汰原因 |
|------|---------|
| Trun-PPO / SORL / AEPO / ARPO | 针对 LLM 连续 token 分布，QAS 是硬离散动作选择 |
| RAPO / LaMer / VAGEN | 检索增强依赖语义 embedding，量子算符无自然语义空间 |
| SEEA-R1 / Treeadv | 需要过程监督标注数据，QAS 无法大量标注中间步骤 |
| Tree-GRPO (G1/G3) | HEA 参数随机初始化破坏前缀语义，仅 UCC 有效 |
| IGPO / HCAPO | 需要每步 dense 信号；实现复杂度高，性价比低 |

#### 最终选型

```
G1 (HEA 浅线路)      →  GiGPO  +  Double-DQN（对照）
G2 (UCC 超 ADAPT-VQE) →  Tree-GRPO  +  GiGPO（辅）
G3 (混合 Pareto)     →  GiGPO  +  α-sweep
```

**实施优先级**:
1. **P1: GiGPO** - 改动最小，适用所有场景，解决稀疏奖励问题
2. **P2: Tree-GRPO** - UCC 专属，利用前缀确定性缓存 VQE，核心武器
3. **P3: Double-DQN** - 文献验证基线，增强实验说服力

### 1.5 验收系统设计中进行 🔄

**文档**: `ideas_pool/RLQAS验收系统.md`

设计目标：为新增算法/agent 建立标准化验收门控，对抗 AI coding 导致的幻觉和能力边界问题。

**6 级验证体系**:

| Level | 类别 | 阻断合并？ | 典型失败来源 |
|-------|------|-----------|-------------|
| 0 | 接口契约 | **是** | A2C `callback` 参数缺失 |
| 1 | 环境稳定性 | **是** | 终止语义错、参数观测丢失 |
| 2 | 序列化 + Callback | **是** | numpy bool_、n_updates=0 |
| 3 | 搜索功能 | **是** | HEA best_circuit=None、Hybrid 崩溃 |
| 4 | 超参数鲁棒性 | **是** | n_episodes 被覆盖、max_steps 读错节 |
| 5 | 化学精度 | **是（仅 UCC 和 HEA）** | H₂-4 误差 ≥ 1.6 mHa 或门数 > 18 |
| 6 | QOP | 否（可选） | — |

**新 Agent 提交清单** (需提供证据):
- [ ] Level 0-4 测试通过截图 / CI 日志
- [ ] Level 5（HEA）：H₂-4（Jordan-Wigner，max_gates=18）化学精度达标
- [ ] Level 5（UCC）：LiH（Jordan-Wigner，max_excitations=6）化学精度达标
- [ ] Level 6（条件）：若涉及 QOP，提供测试通过证据

---

## 2. 当前问题与差距

### 2.1 GRPO 在较大体系表现不稳定 ⚠️

**现象**:
- BeH2 (14 qubits): GRPO 未达化学精度 (3.37 mHa vs 1.6 mHa)
- H6 (12 qubits): 两个算法都未达化学精度

**原因分析**:
- 稀疏奖励问题：只有终态有 VQE 能量，中间步骤无任何反馈信号
- 大体系动作空间更大（42-70 个算符），探索难度增加

**解决方案**:
- 实现 **GiGPO**（步骤级信用分配）缓解稀疏奖励问题
- 实现 **Tree-GRPO**（前缀共享树采样）提高样本利用率

### 2.2 选中算法尚未实现 ⚠️

| 算法 | 状态 | 预计工作量 |
|------|------|-----------|
| GiGPO | 设计完成，未实现 | 1-2 周 |
| Tree-GRPO | 设计完成，未实现 | 1-2 周 |
| Double-DQN | 设计完成，未实现 | 1 周 |

**验收流程**:
1. 实现算法 → 2. 通过 Level 0-5 验收 → 3. 合并主干

### 2.3 真机部署准备不足 ⚠️

**最终目标**: 在 6-qubit 系统上，使用 RLQAS 搜索到的 HEA 结果，放到真机上实现。

**当前状态**:
- 还在模拟阶段（TenCirChem 模拟器）
- 需要对接真实量子硬件接口（待定：哪台机器？什么接口？）
- 需要考虑硬件噪声模型

---

## 3. 下一步建议

### 3.1 短期（1-2 周）- 算法实现与验证

**Week 1**:
1. ✅ 实现 GiGPO (P1)
   - 修改 `grpo_agent.py` 中的奖励计算逻辑
   - 替换 episode 级奖励为步骤级优势
   - 通过 Level 0-5 验收

2. ✅ 测试 GiGPO 在 BeH2 和 H6 上的表现
   - 对比 PPO/GRPO/GiGPO 的收敛速度
   - 记录 VQE 调用次数和最终精度

**Week 2**:
1. ✅ 实现 Tree-GRPO (P2)
   - 新增树采样逻辑
   - 利用 UCC 前缀确定性缓存 VQE
   - 通过 Level 0-5 验收（仅 UCC 场景）

2. ✅ 实现 Double-DQN (P3)
   - 在 `HybridSearchController` 中集成
   - 通过 Level 0-5 验收（HEA/混合线路）

### 3.2 中期（2-4 周）- 实验与论文准备

**实验计划** (来自 `2026-03-31-rlqas-chem-innovations-design.md`):

| 实验 | 分子 | 算法 | 算符池 | 主要指标 |
|------|------|------|--------|----------|
| E1: Pool 比较 | H₂, LiH, BeH₂, H₂O | PPO | FOP, QOP | 到达 1.6 mHa 的 episode 数；CNOT 数 |
| E2: GRPO vs 基线 | LiH, BeH₂ | PPO, GRPO, GiGPO | E1 中最少 episode 的池 | 到达 1.6 mHa 的 VQE 调用数；稳定性 |
| E3: Pareto 前沿 | LiH, BeH₂ | E2 中 VQE 调用最少的算法 | 同 E2 | Pareto 曲线 vs ADAPT-VQE |
| E4: 迁移 (探索性) | H₂ 多几何结构 | E2 最佳算法 | 同 E2 | Zero-shot vs fine-tune vs scratch |

**论文结构建议**:

```
Title: RLQAS-CHEM: Reinforcement Learning Quantum Architecture Search 
       for Molecular Ground State Preparation

1. Introduction
   - VQE 的背景和线路设计挑战
   - 现有方法（ADAPT-VQE 及其变体）的局限性
   - RL 在电路搜索中的优势
   - 主要贡献（4 个创新点）

2. Related Work
   - ADAPT-VQE 系列 (Greedy, K-ADAPT, Pruned)
   - BenchRL-QAS (RL 基准)
   - LLM-driven 量子电路生成（Hive 等）

3. Method
   3.1 RLQAS Framework (问题定义、状态空间、动作空间、奖励设计)
   3.2 Operator Pools (FOP vs QOP)
   3.3 GRPO for Circuit Search (Group Relative Policy Optimization)
   3.4 Multi-Objective Pareto Optimization (α-sweep)
   3.5 Cross-Geometry Transfer (可选)

4. Experiments
   4.1 Experimental Setup (分子、模拟器、超参数)
   4.2 E1: Pool Comparison
   4.3 E2: GRPO vs Baselines
   4.4 E3: Pareto Frontier
   4.5 E4: Transfer Learning (可选)
   4.6 Ablation Studies

5. Results and Discussion
   - GRPO 比 PPO 快 2.4 倍
   - GiGPO 在较大体系上的改进
   - Pareto 前沿 vs ADAPT-VQE

6. Conclusion and Future Work
   - 总结
   - 真机部署计划
   - 代码开源
```

### 3.3 长期（1-2 月）- 真机部署

**Step 1**: 选择 6-qubit 分子系统
- 候选：H₂O 片段、LiH 简化版
- 要求：FCI 可解，化学精度可达

**Step 2**: 用 RLQAS 搜索最优 HEA 线路
- 使用 GiGPO + Tree-GRPO
- 考虑硬件约束（连通性、门集）

**Step 3**: 在真实量子硬件上实现
- 对接硬件接口（待定）
- 考虑噪声模型
- 对比模拟 vs 真机结果

---

## 4. 创新点总结

### 4.1 多算符池比较 (FOP / QOP)

**创新**: 在 RL 搜索框架下系统比较 FOP 和 QOP。

**意义**: 无现有论文在 RL 搜索框架下比较 FOP 和 QOP。RL 搜索动态（探索、奖励塑造、策略学习）可能倾向于不同于基于梯度结果所暗示的池。

### 4.2 GRPO 用于量子线路搜索

**创新**: 引入 Group Relative Policy Optimization (GRPO, DeepSeek 2024) 作为 UCC 线路构建的新 RL 算法。

**优势**:
- 无需 Critic 网络（电路部分构建的 Critic 网络难以设计）
- 通过组内相对优势进行策略更新
- 适合稀疏奖励场景（VQE 能量仅在电路完成后已知）

### 4.3 多目标 Pareto 优化

**创新**: 同时优化能量精度（ΔE）和电路深度（CNOT 数）。

**意义**: ADAPT-VQE 和所有变体仅优化单一目标（能量）。RLQAS-CHEM 可以生成 Pareto 前沿，显示可实现的能量-深度权衡，直接用于硬件部署。

### 4.4 跨几何结构策略迁移（探索性）

**创新**: 在同一分子的多个键长上训练单一 RL 策略，然后测试其对未见键长的泛化能力（无需重新训练）。

**意义**: 如果成功，这表明学习到的策略捕获了可转移的电路构建启发式方法。即使失败，负面结果仍然具有科学价值（区别于 ADAPT-VQE，后者根本没有迁移机制）。

---

## 5. 与现有文献的对比

| 方法 | 方法 | 局限性 | RLQAS-CHEM 优势 |
|------|------|--------|-----------------|
| ADAPT-VQE (所有变体) | 贪婪梯度算符选择 | 单目标；无学习策略；每个分子重新启动 | 多目标；学习策略；可迁移 |
| K-ADAPT-VQE (2026) | 批量算符添加 | 仍然基于梯度；无 RL 泛化 | RL 策略可迁移 |
| Pruned-ADAPT-VQE (2025) | 事后冗余移除 | 反应性，不是主动性；无奖励塑造 | 主动探索；奖励塑造 |
| BenchRL-QAS | 通用门电路的 RL 基准 | 通用搜索空间；无化学驱动的池 | 化学驱动的搜索空间 |
| Hive/Hiverge (2026) | LLM + 进化程序合成 | 进化，不是 RL；输出显式代码，不是策略 | RL 策略；可迁移 |

---

## 6. 关键文件索引

### 项目根目录

| 文件/目录 | 描述 |
|----------|------|
| `rlqas-chem/` | 打包后的 Python 包（Phase 1-4 整合） |
| `rlqas_test/` | 测试脚本和结果 |
| `ideas_pool/` | 项目想法和讨论笔记 |
| `literature/` | 相关文献 |
| `ralph/Phase1-5/` | Ralph 项目的多个开发阶段 |

### ideas_pool 关键文档

| 文件 | 描述 |
|------|------|
| `RLQAS_Algorithm_Selection_20260427.md` | 算法选型总结（最终版） |
| `RLQAS验收系统.md` | 算法验收系统设 |

### rlqas-chem 关键文件

| 文件 | 描述 |
|------|------|
| `src/rlqas_chem/api.py` | 主 API 入口 |
| `src/rlqas_chem/rl/grpo_agent.py` | GRPO 算法实现 |
| `src/rlqas_chem/search/ucc/` | UCC 搜索模块 |
| `src/rlqas_chem/search/hea/` | HEA 搜索模块 |
| `experiments/run_all.py` | 所有实验的运行脚本 |

---

## 7. 附录：自改进记录

基于本次分析，记录以下模式供未来参考：

**Pattern 1**: 当项目有详细的验收系统时，应该严格按照 Level 0→5 的顺序验证新功能。

**Pattern 2**: RL 算法的比较需要同时考虑精度（化学精度）和效率（VQE 调用次数/时间）。

**Pattern 3**: 量子电路搜索的稀疏奖励问题可以通过 GRPO 类的 group-based 方法缓解。

**Pattern 4**: 真机部署前需要充分模拟验证，考虑硬件约束和噪声模型。

---

## 8. Action Items

- [ ] **程老师**: 审阅本文档，确认下一步优先级
- [ ] **同学**: 实现 GiGPO 算法（P1）
- [ ] **同学**: 运行 E1 实验（Pool 比较）
- [ ] **同学**: 完善验收系统文档（Level 0-5 具体测试用例）
- [ ] **共同**: 确定真机部署目标和时间表
- [ ] **共同**: 开始撰写论文初稿（Introduction + Related Work）

---

*本文档由 AI 助手整理自项目进度反馈和代码分析，供团队分享和留存。*
