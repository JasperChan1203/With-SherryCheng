"""
验证脚本：测试单个激发算符是否真的能达到化学精度。

对比两种模式：
  A) run_classical_opt=True  (Phase2/full 实际使用的，会触发完整 UCCSD-VQE)
  B) run_classical_opt=False (真实的"只用1个算符"的 VQE)

使用方法：
  python verify_single_operator.py

需要在集群计算节点运行，不能在登录节点。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, '/curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase1/006/src')

import numpy as np
from scipy.optimize import minimize

CHEMICAL_ACCURACY_HA = 1.6e-3

def setup_molecule():
    from rlqas.phase1.molecule.processor import process_molecule
    mol_config = {
        "formula": "LiH",
        "bond_length": 1.6,
        "active_space": [2, 5],
        "basis_set": "sto-3g",
        "transform": "jordan_wigner",
    }
    mol_data = process_molecule(mol_config)
    fci_energy = float(mol_data.fci_energy)
    print(f"FCI energy: {fci_energy:.10f} Ha")
    print(f"HF  energy: {mol_data.molecular_info.get('hf_energy', 'N/A'):.10f} Ha")
    return mol_data, fci_energy

def test_mode_A(mol_data, fci_energy, operator):
    """Mode A: run_classical_opt=True — reproduces Phase2/full result."""
    print("\n=== Mode A: run_classical_opt=True (Phase2/full 的实际方式) ===")
    from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
    builder = UCCCircuitBuilder(mol_data, {"param_init_strategy": "zeros"})
    ucc = builder.ucc
    n_params = builder.n_params
    print(f"UCCSD 参数总数: {n_params}")
    print(f"可用激发算符总数: {len(builder.available_excitations)}")
    print(f"选择的算符: {operator}")

    # 模拟 environment.step() 中的 run_classical_opt 行为
    params0 = np.zeros(n_params)
    def energy_func(p):
        return builder.evaluate_energy(None, p)  # = ucc.energy(p), FULL UCCSD!
    result = minimize(energy_func, params0, method='L-BFGS-B', options={'maxiter': 200})
    opt_params = result.x
    energy_A = ucc.energy(opt_params)
    error_A = abs(energy_A - fci_energy)
    print(f"优化后能量: {energy_A:.10f} Ha")
    print(f"误差: {error_A*1000:.6f} mHa")
    print(f"化学精度: {'✅ 达到' if error_A < CHEMICAL_ACCURACY_HA else '❌ 未达到'}")
    print(f"→ 注意：evaluate_energy 调用 ucc.energy(p)，即完整 UCCSD，与 '{operator}' 无关！")

def test_mode_B(mol_data, fci_energy, operator):
    """Mode B: 只用RL选中的单个算符做 VQE，不使用完整 UCCSD。"""
    print(f"\n=== Mode B: 只用算符 {operator} 做 VQE（真实的单算符 VQE）===")
    from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
    builder = UCCCircuitBuilder(mol_data, {"param_init_strategy": "zeros"})
    ucc = builder.ucc
    n_params = builder.n_params
    print(f"UCCSD 参数总数: {n_params}")

    # 找到该算符对应的参数 index
    if operator not in builder.available_excitations:
        print(f"❌ 算符 {operator} 不在可用列表中")
        return
    exc_idx = builder.available_excitations.index(operator)
    param_idx = builder.ex_op_to_param[exc_idx]
    print(f"算符 {operator} 对应参数 index: {param_idx}")

    # 只优化该单个参数，其余固定为 0
    params0 = np.zeros(n_params)
    def energy_func_single(theta):
        p = np.zeros(n_params)
        p[param_idx] = theta[0]
        return ucc.energy(p)   # UCCSD energy with only this operator's param != 0
    result = minimize(energy_func_single, [0.0], method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-10})
    opt_theta = result.x[0]
    opt_params = np.zeros(n_params)
    opt_params[param_idx] = opt_theta
    energy_B = ucc.energy(opt_params)
    error_B = abs(energy_B - fci_energy)
    print(f"最优参数值: {opt_theta:.8f} rad")
    print(f"优化后能量: {energy_B:.10f} Ha")
    print(f"误差: {error_B*1000:.6f} mHa")
    print(f"化学精度 (< 1.6 mHa): {'✅ 达到' if error_B < CHEMICAL_ACCURACY_HA else '❌ 未达到'}")

def test_mode_C_full_uccsd(mol_data, fci_energy):
    """Mode C: 完整 UCCSD 优化，作为参考上限。"""
    print("\n=== Mode C: 完整 UCCSD VQE（真实参考值）===")
    from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
    builder = UCCCircuitBuilder(mol_data, {})
    ucc = builder.ucc
    n_params = builder.n_params
    params0 = np.zeros(n_params)
    def energy_func(p):
        return ucc.energy(p)
    result = minimize(energy_func, params0, method='L-BFGS-B',
                      options={'maxiter': 1000, 'ftol': 1e-15})
    energy_C = ucc.energy(result.x)
    error_C = abs(energy_C - fci_energy)
    print(f"完整 UCCSD 最优能量: {energy_C:.10f} Ha")
    print(f"误差: {error_C*1000:.6f} mHa")
    print(f"化学精度: {'✅ 达到' if error_C < CHEMICAL_ACCURACY_HA else '❌ 未达到'}")

def main():
    print("=" * 60)
    print("验证：1个激发算符能否真的达到化学精度？")
    print("=" * 60)
    mol_data, fci_energy = setup_molecule()

    # PPO 报告的算符
    ppo_operator = (6, 5)
    # DQN 报告的算符
    dqn_operator = (4, 0)

    test_mode_A(mol_data, fci_energy, ppo_operator)
    test_mode_B(mol_data, fci_energy, ppo_operator)
    test_mode_B(mol_data, fci_energy, dqn_operator)
    test_mode_C_full_uccsd(mol_data, fci_energy)

    print("\n" + "=" * 60)
    print("结论预测：")
    print("  Mode A ≈ FCI 精度  （实际是完整 UCCSD，与算符选择无关）")
    print("  Mode B << FCI 精度 （真正只用1个算符时）")
    print("  Mode C ≈ FCI 精度  （完整 UCCSD 的真实上限）")
    print("=" * 60)

if __name__ == "__main__":
    main()
