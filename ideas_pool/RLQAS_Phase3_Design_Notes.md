# RLQAS Phase 3 设计笔记

记录 Phase 3 开发过程中确立的设计原则和分子配置决策。

---

## 原则一：RLQAS 是主体，UCC/UCCSD 仅作 Benchmark

**确立时间**：2026-03-23

**原则**：
- **被测算法**是 RLQAS——即 RL agent（PPO/DQN/A2C）驱动的 HybridSearchController 或 UCCSearchController。
- **UCC / UCCSD** 是经典量子化学方法，在项目中仅作为对比基准（Benchmark）出现，不单独作为测试目标。
- 如果需要展示 UCC/UCCSD 结果，必须同时有对应的 RLQAS 结果作为对比（例如 `qubit_vs_fermion_lih_10q.json` 中的 fermion 列是 RLQAS 跑出来的，qubit 列也是 RLQAS 跑出来的）。
- **不允许**写一个只跑 UCCSD 而没有 RL 对比的集成测试。

**背景**：
H6 12q 的测试最初采用直接 UCCSD L-BFGS-B（0.9s 出结果），而非 RL 搜索。这是因为 H6 的激发空间过大（56+ 个独立激发），RL 探索需要约 90 分钟。但该结果应被标注为"UCCSD baseline"而非 RLQAS 结果。如未来想用 RLQAS 测试 H6，需考虑裁剪激发空间或使用 Hybrid（部分 UCC + HEA）策略。

---

## 原则二：分子活性空间配置——使用全空间

**确立时间**：2026-03-23

**原则**：
- 在 smoke test 和主要集成测试中，优先使用**全空间**（不限制 active_space），除非计算代价过高。
- 限制 active_space 仅用于以下场景：
  1. Anti-hollow 测试（需要固定量子比特数，如 LiH (2,5) 10q）
  2. 内存受限的大分子（BeH₂ 14q 等）
  3. ExperimentManager dispatch 等功能性测试（速度优先，结果不重要）

**分子全空间配置**（STO-3G 基组，Jordan-Wigner 变换）：

| 分子 | 电子数 | 轨道数 | 量子比特 | active_space 参数 | 备注 |
|------|--------|--------|---------|-------------------|------|
| H2 | 2 | 2 | 4 | `(1, 2)` | 最小活性空间即全空间 |
| LiH | 4 | 6 | 12 | 不传（全空间） | 原先用 (2,2) 4q 是过度简化 |
| H4 | 4 | 4 | 8 | `(4, 4)` | 全空间 |
| BeH₂ | 6 | 6 | 12 | `(6, 6)` | 全空间 |
| H6 | 6 | 6 | 12 | 不传（全空间） | STO-3G 全空间即 (6,6) |

**LiH 具体修改**（Phase 3 Task 006）：
- `TestHybridSearchLiHSmoke` fixture：去掉 `active_space=(2, 2)` → 使用 LiH 全空间 12q
- `TestSearchResultIsRealTraining`：同上
- Anti-hollow 测试保持 `active_space=(2, 5)` 10q 不变（特意选择较难的空间验证 energy evaluation 真实性）

---

## 原则三：化学精度断言是强制的

**原则**：所有集成测试中，在真实分子上的能量验证必须包含显式 `assert energy_error < 1.6e-3`。
不允许只 print/log 而不 assert。唯一例外是 ExperimentManager dispatch 等纯功能性测试（不关心能量精度）。

**化学精度** = 1.6 mHa = 1.6e-3 Ha，适用于所有分子（包括 H6）。

---

## 参考：Phase 3 已验证的分子化学精度结果

| 分子 | 量子比特 | 算法 | 能量误差 | 状态 |
|------|---------|------|---------|------|
| BeH₂ (4,4) | 8q | RLQAS PPO | 0.037 mHa | ✅ |
| BeH₂ (4,4) | 8q | RLQAS DQN | 0.075 mHa | ✅ |
| BeH₂ (4,5) | 10q | RLQAS | < 1.6 mHa | ✅ |
| BeH₂ (6,6) | 12q | RLQAS Hybrid | < 1.6 mHa | ✅ |
| H4 (4,4) | 8q | RLQAS | < 1.6 mHa | ✅ |
| H6 全空间 | 12q | UCCSD Baseline | 0.21 mHa | ✅（baseline，非 RLQAS） |
| LiH 全空间 | 12q | RLQAS Hybrid | 待验证 | ⏳ Phase 3 patch |
