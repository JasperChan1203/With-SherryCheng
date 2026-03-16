# Ralph Task: LiH VQE Circuit Generation with PySCF Control

## 🎯 Project Overview

You are Ralph, an autonomous AI agent tasked with generating a Variational Quantum Eigensolver (VQE) circuit for the LiH molecule at 2.0 Å bond length. This test emphasizes using PySCF for active orbital control and FCI reference energy calculation, ensuring chemical interpretation.

## 📌 当前状态与下一步

**当前进度**：迭代5/10 - 需要增强电路表达能力以达到化学精度

### ✅ 已解决的问题
1. **FCI能量差异问题已解决**：
   - 发现根源：PySCF `symmetry=True` + `sort_mo([2,3,6])` vs `symmetry=False` + 手动core/frozen设置
   - 基准FCI能量：**-7.860153 Hartree**（来自 `tencirchem_benchmark.py`）
   - 解决方案：使用 `symmetry=True` 和 `sort_mo([2,3,6])` 确保复现基准计算
   - 当前实现已修正，生成正确FCI能量：-7.86015321 Hartree

2. **轨道选择已统一**：
   - 使用 1-based 索引 `[2,3,6]`（0-based `[1,2,5]`）
   - 通过 `tencirchem_benchmark.py` 统一了活性轨道选择
   - 积分矩阵与基准一致

### ❌ 待解决的问题
1. **VQE能量未达化学精度**：
   - 当前VQE能量：-7.830906 Hartree（来自 `lih_results.json`）
   - 目标FCI能量：-7.860153 Hartree
   - 差值：**29.2 mHa** > 1.6 mHa 目标精度
   - 问题：当前电路架构或优化策略需要改进（18参数应该足够，但需要更好的设计和优化）

### 🎯 迭代5重点任务：优化电路设计与改进策略
**用户反馈**：对于LiH-4qubits系统，18个门应该足以达到化学精度，需要优化线路设计而非单纯增加参数数量。

1. **优化电路架构**：
   - 保持约18个参数的规模（足够表达4量子位系统）
   - 重新设计电路结构：改进纠缠模式、旋转门排列和层间连接
   - 尝试不同的基本架构：如UCCSD启发式结构（手动实现）、硬件高效ansatz（HEA）的变体、或定制化纠缠模式
   - 优化门顺序和连接性：确保充分纠缠所有量子位

2. **改进优化策略**：
   - **优化器**：使用BFGS优化器（足够有效），调整其参数（增加最大迭代次数至300-500，调整收敛容差）
   - 调整初始参数策略：使用更智能的初始化（基于分子轨道信息或先前的收敛结果）
   - 实现多次随机重启以避免局部最优
   - 检查梯度计算正确性

3. **电路设计建议**：
   - **架构1**：UCCSD启发式结构（手动实现双激发算子）
   - **架构2**：硬件高效ansatz变体（交替单比特旋转和纠缠层）
   - **架构3**：自定义纠缠模式（全连接、循环或特定分子结构）
   - 保持足够但不过度的表达能力

4. **目标**：达到 <1.6 mHa 化学精度，生成 `<promise>COMPLETE</promise>`

### 📚 关键发现（记录于AGENTS.md）
- FCI能量差异：27.93 mHa 源于不同PySCF设置（已解决）
- 必须使用 `symmetry=True` + `sort_mo([2,3,6])` 匹配基准（已实现）
- **用户反馈**：18个门对于4量子位LiH系统应该足以达到化学精度
- **核心问题**：当前电路架构可能不够有效，或优化陷入局部最优
- **需要**：优化电路设计（架构、纠缠模式、门顺序）和改进优化策略，而非单纯增加参数数量

## 📋 Task Description

Your objective is to create a Python implementation that:

1. **Define LiH molecule with PySCF**:
   - Bond length: 2.0 Å
   - Basis set: STO-3G
   - Perform Hartree-Fock calculation using PySCF

2. **Select active orbitals according to benchmark specification**:
   - Active space: (2 electrons, 3 orbitals)
   - Use PySCF's `sort_mo([2,3,6])` method (1-based indices) to select orbitals matching the benchmark
   - This corresponds to 0-based indices [1,2,5] as specified in `tencirchem_benchmark.py`
   - Justify orbital selection by referencing the benchmark unification
   - Note: Active orbitals have been unified via `tencirchem_benchmark.py` to ensure consistency with FCI reference energy

3. **Compute FCI reference energy using PySCF**:
   - Use PySCF's CASCI or FCI implementation to compute exact energy
   - This computed FCI energy serves as reference for VQE accuracy

4. **Perform parity transformation** to obtain a 4-qubit Hamiltonian (using Tencirchem).

5. **MANUALLY DESIGN and implement VQE circuit** using Tencirchem:
   - **DO NOT use pre-defined ansatz functions** like HEA.ry(), UCC(), etc.
   - **Manually construct** quantum circuit using basic gates (Rx, Ry, Rz, CNOT, etc.)
   - Design parameterized circuit suitable for 4-qubit LiH system
   - Use Scipy BFGS optimizer (sufficient for this task, adjust max iterations and tolerance as needed)
   - Track energy convergence during optimization

6. **Output requirements**:
   - Circuit gates and parameters
   - Energy convergence curve (list of energies at each iteration)
   - Final VQE energy
   - Comparison with PySCF-computed FCI energy (should be within 1.6 mHa)
   - Structured JSON output file: `lih_results.json`

7. **Validation**: Your implementation must pass the validation script `../validate_lih_custom_circuit.py`.

## 🔧 Technology Stack

- **Quantum Framework**: Tencirchem
- **Chemistry Library**: PySCF (for HF, orbital selection, FCI)
- **Optimization**: Scipy (BFGS optimizer)
- **Core Libraries**: NumPy, SciPy
- **Output Format**: JSON

## 📖 Reference Example: H2 VQE Circuit

For guidance, here is a working H2 VQE circuit at 0.74 Å bond length using Tencirchem:

```python
# Example H2 VQE circuit structure (conceptual)
circuit_gates = ["rz1", "ry0", "cnot0_1", "rx1", "rx1", "rx1"]
parameters = [-1.1754374504089355, -0.22369137406349182, 0.0,
              1.4277695416969798, 0.2513202621250142, 1.462464690208435]
```

**Key characteristics of this example**:
- Uses Tencirchem framework
- Simple parameterized circuit with rotation and CNOT gates
- Optimized with Scipy BFGS
- Produces chemical accuracy energy

## 🎯 LiH FCI Energy Reference

**IMPORTANT**: You must compute the FCI energy for LiH at 2.0 Å bond length with active_space=(2,3) **using PySCF with symmetry=True and sort_mo([2,3,6])**. This computed FCI energy should match the benchmark value (-7.860153 Hartree) within tolerance. The validation script will use either your PySCF-computed FCI energy or the benchmark value as reference.

**Your target**: Design a VQE circuit that produces energy within 1.6 mHa (0.0016 Hartree) of the FCI reference energy (PySCF-computed or benchmark).

**Note**: You should compute the HF energy as part of your VQE implementation. The validation will compare your VQE final energy against the FCI reference energy (PySCF-computed if available, otherwise benchmark).

## 📁 File Structure

You will work in this directory (`Ralph_Test_LiH_VQE/`). Key files:

```
Ralph_Test_LiH_VQE/
├── CLAUDE.md              # This file (your instructions)
├── prd.json               # Project requirements (referenced)
├── AGENTS.md              # Your knowledge base (update as you learn)
├── ralph.sh               # Execution script
├── progress.txt           # Progress log (update each iteration)
├── ralph_learning_log.txt # Detailed learning log (record thoughts)
└── (you will create):
    ├── generate_lih_vqe.py   # Main implementation script
    └── lih_results.json      # Output results
```

## 🔄 Iterative Learning Workflow

Follow this process for maximum 10 iterations:

### Iteration 1: Understand & Initial Implementation
1. Read and understand `prd.json` requirements
2. Study the H2 VQE reference example
3. Research PySCF documentation for HF, orbital selection, and FCI calculations
4. Research Tencirchem documentation for VQE implementation
5. Write initial implementation in `generate_lih_vqe.py`
6. Generate initial `lih_results.json`
7. Run validation: `python ../validate_lih_custom_circuit.py lih_results.json`
8. Update `AGENTS.md` with learned patterns

### Iteration 2-10: Improve Based on Feedback
1. If validation fails (exit code 1):
   - Read validation error messages carefully
   - Analyze what went wrong
   - Create debug scripts if needed
   - Improve implementation
   - Run validation again
2. If validation passes (exit code 0):
   - Output `<promise>COMPLETE</promise>`
   - Finalize documentation
   - Update knowledge base

### Success Criteria
- **Technical**: VQE energy within 1.6 mHa of PySCF-computed FCI energy
- **Process**: Clear progress logging, knowledge capture, iterative improvement
- **Output**: Complete `lih_results.json` with all required fields

## 📊 Expected Output Format (`lih_results.json`)

Your output JSON must include these fields:

```json
{
  "molecule": {
    "formula": "LiH",
    "bond_length_angstrom": 2.0,
    "active_space": [2, 3],
    "selected_orbitals": [1, 2, 5],
    "n_qubits": 4
  },
  "vqe_settings": {
    "ansatz_type": "your ansatz type",
    "optimizer": "BFGS",
    "framework": "Tencirchem"
  },
  "circuit": {
    "gates": ["gate1", "gate2", ...],
    "parameters": [param1, param2, ...],
    "circuit_depth": 5,
    "n_parameters": 6,
    "design_rationale": "Explanation of circuit architecture"
  },
  "results": {
    "final_energy_hartree": -7.12345678,
    "fci_energy_hartree": -7.12345678,
    "energy_difference_mha": 1.2,
    "converged": true,
    "fci_computation_method": "PySCF CASCI"
  },
  "convergence_data": {
    "energy_curve": [-7.1, -7.12, -7.123, -7.1234, -7.12345],
    "n_iterations": 50,
    "optimization_time_seconds": 12.34
  },
  "orbital_information": {
    "selected_orbitals": [1, 2, 5],
    "orbital_energies": [-0.5, -0.3, 0.1],
    "selection_justification": "Selected HOMO, LUMO, LUMO+1 based on energy ordering"
  },
  "implementation_details": {
    "method": "VQE with Tencirchem, PySCF for orbital selection and FCI",
    "pyscf_version": "2.x.x",
    "tencirchem_version": "x.x.x",
    "script_path": "generate_lih_vqe.py",
    "manual_design_verified": true
  }
}
```

## 🧪 Validation Process

The validation script (`../validate_lih_custom_circuit.py`) will check:

1. **Molecule definition**: Bond length, active space
2. **Energy accuracy**: VQE energy within 1.6 mHa of FCI (from benchmark file)
3. **Circuit properties**: Gates, parameters, qubit count
4. **Convergence data**: Energy curve showing decreasing trend

**CRITICAL**: You must use **PySCF with symmetry=True and sort_mo([2,3,6])** to compute FCI energy that matches the benchmark value (-7.860153 Hartree). The validation script uses this fixed benchmark value. Failure to match this FCI energy within tolerance indicates incorrect PySCF setup.

**Key requirements**:
- Molecule definition: `symmetry=True` (not `symmetry=False`)
- Atom ordering: `[["H", 0, 0, 0], ["Li", 2.0, 0, 0]]`
- Active orbitals: 1-based `[2,3,6]` via `sort_mo([2,3,6])`
- Basis: `sto-3g`

Your computed FCI energy must be within 0.0016 Hartree (1.6 mHa) of -7.860153 Hartree.

**Validation command**:
```bash
python ../validate_lih_custom_circuit.py lih_results.json
```

Exit code 0 = success, exit code 1 = failure with detailed error messages.

## 🔍 Key Technical Challenges

1. **PySCF orbital selection**: Correctly use PySCF to select active orbitals with chemical interpretation for (2,3) active space
2. **PySCF FCI computation**: Compute FCI energy using PySCF's CASCI or FCI implementation
3. **Active space consistency**: Ensure orbital selection matches between PySCF and Tencirchem
4. **Parity transformation**: Transform to 4-qubit Hamiltonian (Tencirchem may handle this automatically)
5. **VQE circuit design**: **MANUALLY DESIGN** parameterized circuit suitable for 4-qubit system (do not use pre-defined ansatz functions)
6. **Circuit design optimization**: Design effective circuit architecture to capture correlation energy (~30 mHa difference from HF to FCI). 18-parameter circuit should be sufficient for 4-qubit LiH system, but requires careful architecture design (entanglement patterns, gate ordering, parameter optimization).
7. **Optimization convergence**: Ensure BFGS optimizer converges to chemical accuracy
8. **Energy tracking**: Record energy at each iteration for convergence curve

## 💡 Learning Resources

1. **PySCF documentation**: Hartree-Fock, CASCI, active space selection
2. **Tencirchem documentation**: VQE implementation examples
3. **H2 reference example**: Study the provided H2 VQE circuit structure
4. **Quantum chemistry basics**: VQE theory and ansatz design

## 📝 Documentation Requirements

As you work, maintain:

1. **`progress.txt`**: Brief progress summary after each iteration
2. **`ralph_learning_log.txt`**: Detailed thought process, decisions, debugging steps
3. **`AGENTS.md`**: Accumulated knowledge about PySCF, Tencirchem, VQE, quantum chemistry
4. **Code comments**: Clear explanations in implementation

## 🚀 Getting Started

1. First, explore the `prd.json` file to understand detailed requirements
2. Research PySCF for HF calculation, orbital selection, and FCI computation
3. Research Tencirchem VQE implementation patterns
4. Start with a simple ansatz design based on the H2 example
5. Implement step-by-step: PySCF molecule → orbital selection → FCI computation → Hamiltonian → circuit → optimization
6. Test each component before full integration
7. Run validation early to identify issues

Remember: You have up to 10 iterations to succeed. Use validation feedback to improve systematically. Document your learning process thoroughly.

Good luck! This is a challenging but achievable quantum computational chemistry task.