# RLQAS 算法验收系统

**版本：** v1.0  
**日期：** 2026-04-27  
**范围：** `rlqas-chem` 所有 RL agent 及搜索模块  
**目的：** 为新增算法/agent 建立标准化验收门控，确保每个新 agent 在合并主干前通过完整的正确性、稳定性和化学精度验证。

---

## 背景

本验收体系基于以下两个来源设计：

1. **已发现的 Bug**（见 `claudecodefix/ClaudeCodeFix.md`）：A2C callback 接口不兼容、NumPy bool_ 序列化失败、跨 episode 状态错误、HEA best_circuit 丢失等，均属于可通过系统化测试提前拦截的问题。
2. **rlqas_test 运行失败记录**（见 `slurm_logs/`）：SAC/DQN/GRPO/A2C/PPO 五类 agent 全部存在不同层级的失败，涵盖接口、序列化、诊断和化学精度四个维度。

---

## 验收层级总览

```
Level 0  接口契约       第1类   不通过则直接拒绝合并
Level 1  环境稳定性     第6类   RL 环境核心逻辑正确性
Level 2  集成正确性     第7类 + 第8类   序列化、Callback、结果完整性
Level 3  搜索功能       第2、3、4类    UCC / HEA / Hybrid 搜索
Level 4  超参数鲁棒性   第5类   极端 config 不崩溃
Level 5  化学精度       第9类   真实物理验收
Level 6  QOP 搜索       第10类  条件满足时运行（可选）
```

**通过规则：** 新 agent 必须从 Level 0 顺序通过至 Level 5，方可合并主干。Level 6 视新 agent 是否涉及 QOP 模块而定。

---

## 第 1 类 — 算法单元测试（Agent 接口契约）

> **Level 0 — 不通过直接拒绝**  
> 来源：Bug 1（A2C `callback` 参数缺失导致所有 A2C 训练崩溃）

### 1.1 接口签名一致性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `learn(callback=None)` 签名 | `learn()` 必须接受 `callback` 关键字参数 | 不抛 `TypeError` |
| `learn(total_timesteps=N)` | 必须接受 `total_timesteps` 关键字参数 | 不抛 `TypeError` |
| `act(state)` 返回格式 | 返回值为 `(int, dict)` | 类型断言通过 |
| `save(path)` / `load(path)` | 保存后加载，`act()` 输出前后一致 | 两次输出相同 |
| `AgentFactory.create_agent()` 注册 | 通过工厂可正常实例化 | 返回 agent 实例 |

### 1.2 Config 兼容性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| 已知 key 正常读取 | 标准超参数（`lr`、`gamma` 等）被正确读取 | 值与传入一致 |
| 未知 key 处理 | 传入不存在的 key | 抛 `KeyError` 或有明确告警，不静默丢弃 |
| 默认值覆盖规则 | 显式传入值不被 config 默认值覆盖 | 实际值 == 传入值 |

### 1.3 基础训练冒烟测试

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| 最小化训练（100 steps） | agent 能完成 100 timesteps 训练不崩溃 | 无异常 |
| 训练后 `act()` 可用 | 训练后可对任意合法观测给出动作 | 返回合法 action index |
| `learn()` 返回 metrics dict | 返回值为 dict，含 `loss` 或等价 key | 类型断言通过 |

---

## 第 2 类 — UCC 搜索集成测试

> **Level 3**  
> 分子：H₂（最简），验证 UCC 搜索完整流程

### 2.1 基本搜索流程

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `UCCSearchController` 初始化 | agent + env 成功构建 | 无异常 |
| `search()` 正常返回 | 完成 50 episodes 后返回结果 dict | 无异常 |
| 结果字段完整性 | `best_energy`、`best_excitations`、`best_params`、`convergence_reached` 均存在 | 字段存在且非 None |
| `best_energy` 物理合理性 | 不高于 HF 能量 | `best_energy <= hf_energy` |
| `convergence_reached` 类型 | 为 Python `bool`，不是 `numpy.bool_` | `type(...) is bool` |

### 2.2 Action Masking

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `get_valid_action_mask()` 初始状态 | 空电路时全为 True | 全 True |
| 添加算符后 mask 更新 | 已添加算符对应位置变为 False | 已选位置为 False |
| 电路满时 mask | `len(excitations) == max_depth` 时全为 False | 全 False |

### 2.3 终止条件

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| 无效动作不提前终止 | 连续 5 次无效 action，step_count 增加但 episode 不终止 | episode 继续 |
| max_depth 达到正常终止 | 添满 max_depth 个算符后终止 | `terminated=True` |
| early_stop 生效 | 达到精度阈值后训练停止 | 总 timesteps < budget |

---

## 第 3 类 — HEA 搜索集成测试

> **Level 3**  
> 分子：H₂，验证 HEA 架构搜索完整流程

### 3.1 基本搜索流程

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `HEASearchController` 初始化 | 正常构建 | 无异常 |
| `search()` 正常返回 | 完成搜索后返回结果 | 无异常 |
| `best_circuit` 非 None | 搜索完成后电路配置存在 | 非 None，含 `n_layers`、`entanglement_pattern` 等字段 |
| `best_energy` 物理合理性 | 不高于 HF 能量 | `best_energy <= hf_energy` |

### 3.2 Best Circuit 跨 episode 追踪

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `best_circuit_config` 随 best_energy 更新 | 每次发现更低能量时，`best_circuit_config` 同步更新 | `best_circuit_config` 对应 `best_energy` 时刻的电路 |
| 单元测试环境（无真实分子）也追踪 | 无 `molecule_data` 时 `best_energy` 仍被更新 | `best_energy` < 初始值 |

---

## 第 4 类 — 混合搜索集成测试（Hybrid Search）

> **Level 3**

### 4.1 基本流程

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `HybridSearchController` 初始化 | 正常构建 | 无异常 |
| sequential 模式搜索 | `fusion_mode="sequential"` 下完成搜索 | 返回含 `n_hea_blocks`、`n_ucc_components` 的结果 |
| parallel 模式搜索 | `fusion_mode="parallel"` 下完成搜索 | 无异常 |
| `best_circuit` 包含 HEA 和 UCC 部分 | 融合结果中两类成分均存在 | 字段均非空 |

### 4.2 Fusion 正确性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `best_energy` ≤ 纯 UCC 或纯 HEA 最优值 | 混合搜索不差于单一模式 | （软检查，记录对比值） |
| `convergence_reached` 语义一致 | 达到精度阈值时为 True | 与精度检验结果一致 |

---

## 第 5 类 — 超参数搜索验证

> **Level 4**

### 5.1 Config 边界鲁棒性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `n_episodes` 传入生效 | 传入 `n_episodes=2000`，实际运行 2000 轮 | 日志显示 2000，不被默认值覆盖 |
| `max_excitations` 从正确 section 读取 | 从 `environment` 节读取，而非 `controller` 节 | `total_timesteps` 计算正确 |
| `max_depth=1` 极端值 | 最小电路深度下不崩溃 | 完成搜索，best_energy 存在 |
| `learning_rate=0` 极端值 | 零学习率不导致崩溃（参数冻结） | 无 NaN/Inf，无异常 |
| `complexity_penalty=0` | 无惩罚项下收敛 | best_energy 改善 |

### 5.2 超参数影响方向性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| 高 `complexity_penalty` 限制电路深度 | 惩罚增大时平均电路深度下降 | depth(high_penalty) < depth(low_penalty) |
| `baseline_type="current_best"` 跨 episode 保留 | reset 后 baseline 不回退 | 第 2 episode best ≤ 第 1 episode best |

---

## 第 6 类 — 环境稳定性与跨 Episode 一致性测试

> **Level 1**  
> 来源：Bug 2（终止语义）、Bug 3（奖励基线重置）、Bug 4（参数观测编码）

### 6.1 跨 Episode 状态一致性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `global_best_energy` 单调不增 | 运行 10 episodes，每次 reset 后 global_best 不增加 | 序列单调不增 |
| `current_best` baseline 不重置 | reset 后 baseline 仍为历史最优，不回退到 HF | baseline ≤ HF energy |
| per-episode shaping 状态正确重置 | `_first_evaluation` 在每次 reset 后为 True | 每轮第一步 reward 与预期一致 |
| `consecutive_improvements` 归零 | reset 后计数器清零 | reset 后值为 0 |

### 6.2 观测编码正确性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| 高 pool index 算符参数可见 | 选择 pool index ≥ max_depth 的算符后，观测中对应参数非零 | 参数值 ≠ 0 |
| 参数按选择顺序打包 | 观测中第 k 个参数对应第 k 个被选算符 | 顺序一致 |

### 6.3 奖励函数归一化

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| 复杂度惩罚上界 | 任意电路深度下惩罚 ≤ `complexity_penalty` | `penalty ≤ config['complexity_penalty']` |
| 两条计算路径一致 | alpha 加权路径与默认路径在相同输入下惩罚一致 | 差值 < 1e-9 |

---

## 第 7 类 — 序列化与结果完整性测试

> **Level 2**  
> 来源：Bug 2（numpy bool_ 序列化失败）、Bug 7（best_circuit 丢失）

### 7.1 结果 JSON 序列化

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `search()` 结果直接可序列化 | `json.dumps(results)` 无异常 | 无 `TypeError` |
| `convergence_reached` 类型 | `type(results['convergence_reached']) is bool` | True |
| `best_params` 类型 | ndarray 转换为 list 存储 | `isinstance(..., list)` |
| Diagnostics JSON 非空 | 训练完成后 diagnostics 文件 > 2 bytes | 文件大小 > 10 bytes |

### 7.2 模型 save / load 往返

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| save 后 load，act 结果一致 | 相同输入下两次预测动作相同 | action 相同 |
| load 不依赖原始 env 对象 | 新 env 实例上 load 后 act 正常 | 无异常 |

---

## 第 8 类 — Diagnostics / Callback 集成测试

> **Level 2**  
> 来源：Bug 3（None 格式化崩溃）、Bug 4（n_updates=0）、Bug 5（early stop 误判）、Bug 6（GRPO finish 调用）

### 8.1 数据记录完整性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `n_updates > 0` | 100 steps 训练后 callback 记录 ≥1 次 update | `len(callback.updates) >= 1` |
| Callback 实际被调用 | 注入计数器 callback，验证调用次数 | 调用次数 > 0 |
| `global_best_energy` 被记录 | diagnostics 中含有效能量值 | 值 ≤ HF energy |

### 8.2 None 值安全处理

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `explained_variance=None` 时报告显示 N/A | f-string 格式化前有 None 检查 | 输出 "N/A"，不抛 `TypeError` |
| `exploration_rate=None` 时安全 | DQN 训练初期 exploration_rate 为 None 时 | 输出 "N/A"，不抛异常 |

### 8.3 Early Stopping 判断正确性

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| 达到精度后 pass/fail 为通过 | `convergence_reached=True` 时 OVERALL 判定为 PASS | 不误报 FAIL |
| GRPO `finish()` 幂等 | 调用 `finish()` 两次不崩溃 | 无异常 |
| Callback 顺序正确 | EarlyStopCallback 优先于用户 callbacks 执行 | 提前停止生效 |

---

## 第 9 类 — 化学精度回归测试

> **Level 5 — 最终物理验收**  
> 混合搜索（第4类）和 QOP（第10类）**不参与**本类验收，仅验证 UCC 和 HEA 两个系统。

### 9.1 HEA 系统基准

**分子：H₂-4（4 量子比特，Jordan-Wigner 变换）**

| 测试项 | 要求 |
|--------|------|
| 编码方式 | Jordan-Wigner 变换 |
| 最大门数限制 | `max_gates = 18` |
| 精度要求 | 误差 < 1.6 mHa（化学精度） |
| 对比基准 | FCI 能量 |
| 门数效率 | 达到精度时实际使用门数 ≤ 18 |

### 9.2 UCC 系统基准

**分子：LiH（Jordan-Wigner 变换）**

| 测试项 | 要求 |
|--------|------|
| 编码方式 | Jordan-Wigner 变换 |
| 最大独立算符数 | `max_excitations = 6` |
| 精度要求 | 误差 < 1.6 mHa（化学精度） |
| 对比基准 | FCI 能量 |
| 算符效率 | 达到精度时实际使用算符数 ≤ 6 |

### 9.3 混合搜索与 QOP（暂不验收化学精度）

混合搜索和 QOP 模块目前不要求通过化学精度测试，仅须通过各自的搜索功能测试（第 4 类、第 10 类）。

---

## 第 10 类 — QOP 搜索测试（条件可选）

> **Level 6 — 仅当新 agent 涉及 QOP 模块时运行**

| 测试项 | 验证内容 | 预期结果 |
|--------|---------|---------|
| `QubitOperatorPool` 构建 | pool size 正确，算符可生成 | pool size > 0 |
| `SearchResult` 字段完整 | `best_circuit`、`fusion_template`、`convergence_reached` 均存在 | 无 None 字段 |
| QOP 与 UCC 能量可比 | 相同分子 QOP 能量不差于 UCC 超过 10 mHa | 差值 ≤ 10 mHa |
| `SearchResult` JSON 可序列化 | `dataclass` 转 dict 后可 `json.dump` | 无 `TypeError` |

---

## 验收门控总结

| Level | 类别 | 阻断合并？ | 典型失败来源 |
|-------|------|-----------|-------------|
| 0 | 第1类 接口契约 | **是** | A2C `callback` 参数缺失 |
| 1 | 第6类 环境稳定性 | **是** | 终止语义错、参数观测丢失 |
| 2 | 第7、8类 序列化 + Callback | **是** | numpy bool_、n_updates=0 |
| 3 | 第2、3、4类 搜索功能 | **是** | HEA best_circuit=None、Hybrid 崩溃 |
| 4 | 第5类 超参数鲁棒性 | **是** | n_episodes 被覆盖、max_steps 读错节 |
| 5 | 第9类 化学精度 | **是（仅 UCC 和 HEA）** | HEA: H₂-4 误差 ≥ 1.6 mHa 或门数 > 18；UCC: LiH 误差 ≥ 1.6 mHa 或算符数 > 6 |
| 6 | 第10类 QOP | 否（可选） | — |

---

## 新 Agent 提交清单

提交新 agent PR 时，需提供以下证据：

- [ ] Level 0：接口签名测试通过截图 / CI 日志
- [ ] Level 1：环境稳定性测试通过（含跨 episode 状态检验）
- [ ] Level 2：JSON 序列化测试通过 + diagnostics 文件非空
- [ ] Level 3：UCC + HEA 搜索冒烟测试通过（H₂）
- [ ] Level 4：超参数边界测试通过（含 n_episodes override 验证）
- [ ] Level 5（HEA）：H₂-4（Jordan-Wigner，max_gates=18）化学精度达标（误差 < 1.6 mHa）
- [ ] Level 5（UCC）：LiH（Jordan-Wigner，max_excitations=6）化学精度达标（误差 < 1.6 mHa）
- [ ] Level 6（条件）：若涉及 QOP，提供 QOP 测试通过证据

---

*本文档由 Claude Code 根据 rlqas_test slurm_logs 失败记录及 ClaudeCodeFix.md bug 报告归纳整理。*  
*如需扩展新分子基准或新搜索模式，在对应类别下追加测试行即可。*
