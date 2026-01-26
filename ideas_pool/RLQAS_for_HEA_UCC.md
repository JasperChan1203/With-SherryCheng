# 通过Agent实现强化学习量子架构搜索（RLQAS）用于VQE线路优化

## 思路概述
本想法旨在使用强化学习量子架构搜索（RLQAS）方法优化变分量子本征求解器（VQE）的线路结构，重点针对硬件高效ansatz（HEA）和酉耦合簇（UCC）线路。目标是实现比现有方法（如HEA_Rylinear、UCCSD或adaptvqe）更浅的线路深度，以支持更大量子体系（约20量子比特）的精确计算[Reinforcement learning for optimization of variational quantum circuit architectures](https://arxiv.org/abs/2103.16089)。

## 方法论
- **RLQAS框架**：采用强化学习算法自动搜索最优量子线路架构，通过奖励函数引导搜索过程，平衡线路深度与表达能⼒[Reinforcement learning for optimization of variational quantum circuit architectures](https://arxiv.org/abs/2103.16089)。
- **HEA线路搜索空间**：对于HEA线路，搜索空间定义为硬件高效的门集，如Rx、Ry、Rz和CNOT或SWAP门。这部分工作已有文献基础，可直接引用[Reinforcement learning for optimization of variational quantum circuit architectures](https://arxiv.org/abs/2103.16089)。
- **UCC线路创新**：对于UCC线路，算符池可定义为UCC的单双（三）SD(T)激发算符，或采用李震宇提出的sQEB算符以适应硬件约束[sQEB operators](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00119)。这是本想法的创新点。

## 目标
- 通过RLQAS搜索获得浅层线路，减少线路深度，提升计算效率。
- 应用于大体系模拟（如20量子比特），比较RLQAS优化线路与HEA_Rylinear、UCCSD和adaptvqe的性能。
- 在literature文件夹中建立RLQAS文献库，整合相关研究。

## 参考文献
1. [Reinforcement learning for optimization of variational quantum circuit architectures](https://arxiv.org/abs/2103.16089)
2. [sQEB operators](https://pubs.acs.org/doi/10.1021/acs.jctc.5c00119)
