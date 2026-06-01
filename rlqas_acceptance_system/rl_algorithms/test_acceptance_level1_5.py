"""
RLQAS 验收系统 - Level 1-5 测试

测试目标：
- Level 1: 环境稳定性（与 agent 无关，验证 rlqas-chem 环境本身）
- Level 2: 搜索结果正确性（序列化、字段类型）
- Level 3: 搜索功能（UCC/HEA 能跑通、结果物理合理）
- Level 4: 超参数约束（参数真正生效）
- Level 5: 化学精度（硬门控，误差 < 1.6 mHa）

运行方式：
    # 仅回归测试（用 PPO）
    python3 rlqas_acceptance_system/test_acceptance_level1_5.py

    # 验收新算法（同时跑新算法 + PPO 回归）
    python3 rlqas_acceptance_system/test_acceptance_level1_5.py --agent gigppo
"""

import sys
import os
import json
import argparse
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rlqas-chem', 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rlqas-chem'))

# ==================== 测试框架 ====================

class TestResult:
    def __init__(self, name: str, level: int):
        self.name = name
        self.level = level
        self.passed = False
        self.error_msg = None
        self.details = {}

    def pass_test(self, details: dict = None):
        self.passed = True
        if details:
            self.details = details
        return self

    def fail_test(self, error_msg: str):
        self.passed = False
        self.error_msg = error_msg
        return self

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f"{status} [Level {self.level}] - {self.name}"
        if self.error_msg:
            msg += f"\n    Error: {self.error_msg}"
        if self.details:
            for k, v in self.details.items():
                msg += f"\n    {k}: {v}"
        return msg


def run_test(test_name: str, level: int, test_func, *args, **kwargs):
    result = TestResult(test_name, level)
    try:
        result = test_func(result, *args, **kwargs)
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 1: 环境稳定性（与 agent 无关）====================

def test_global_best_monotonic(result: TestResult):
    """global_best_energy 跨 episode 单调不增"""
    try:
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 6})

        global_bests = []
        for _ in range(5):
            obs, _ = env.reset()
            done = False
            while not done:
                obs, reward, terminated, truncated, info = env.step(np.random.randint(0, env.action_space.n))
                done = terminated or truncated
            global_bests.append(env.global_best_energy)

        monotonic = all(global_bests[i] >= global_bests[i+1] for i in range(len(global_bests)-1))
        if monotonic:
            result.pass_test({"global_bests": [f"{e:.6f}" for e in global_bests]})
        else:
            result.fail_test(f"global_best not monotonic: {global_bests}")

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_global_best_preserved_after_reset(result: TestResult):
    """reset() 后 global_best_energy 不退回 HF 能量"""
    try:
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 6})

        obs, _ = env.reset()
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(np.random.randint(0, env.action_space.n))
            done = terminated or truncated
        best_before = env.global_best_energy

        env.reset()
        best_after = env.global_best_energy

        if best_after <= best_before + 1e-9:
            result.pass_test({"best_before_reset": f"{best_before:.6f}", "best_after_reset": f"{best_after:.6f}"})
        else:
            result.fail_test(f"global_best regressed after reset: {best_before:.6f} -> {best_after:.6f}")

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_step_return_types(result: TestResult):
    """step() 返回值类型符合 gym 规范"""
    try:
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 6})
        obs, _ = env.reset()
        next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = terminated or truncated

        errors = []
        if not isinstance(reward, (float, int, np.floating)):
            errors.append(f"reward type={type(reward)}, expected float")
        if not isinstance(terminated, (bool, np.bool_)):
            errors.append(f"terminated type={type(terminated)}, expected bool")
        if next_obs.shape != env.observation_space.shape:
            errors.append(f"obs shape={next_obs.shape}, expected {env.observation_space.shape}")
        if not isinstance(info, dict):
            errors.append(f"info type={type(info)}, expected dict")

        if not errors:
            result.pass_test({"reward": float(reward), "done": bool(done), "obs_shape": str(next_obs.shape)})
        else:
            result.fail_test("; ".join(errors))

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_obs_no_nan_inf(result: TestResult):
    """观测值连续 5 步不含 NaN/Inf"""
    try:
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 6})
        obs, _ = env.reset()

        for step in range(5):
            if not np.all(np.isfinite(obs)):
                result.fail_test(f"obs contains NaN/Inf at step {step}")
                return result
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

        result.pass_test({"message": "All observations are finite"})

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 2: 搜索结果正确性（用指定 agent）====================

def test_json_serialization(result: TestResult, agent_type: str = 'ppo'):
    """search() 结果可直接 JSON 序列化"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type=agent_type, n_episodes=2)
        json_str = json.dumps(result_dict)
        parsed = json.loads(json_str)
        result.pass_test({"agent": agent_type, "keys": list(parsed.keys())})

    except TypeError as e:
        result.fail_test(f"JSON serialization failed: {e}")
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_convergence_type(result: TestResult, agent_type: str = 'ppo'):
    """chemical_accuracy 是 Python bool，不是 numpy.bool_"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type=agent_type, n_episodes=2)
        conv = result_dict.get('chemical_accuracy')
        if type(conv) is bool:
            result.pass_test({"agent": agent_type, "chemical_accuracy": conv})
        else:
            result.fail_test(f"chemical_accuracy is {type(conv)}, expected Python bool (not numpy.bool_)")

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 3: 搜索功能（用指定 agent）====================

def test_ucc_search_basic(result: TestResult, agent_type: str = 'ppo'):
    """UCC 搜索：结果字段完整"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type=agent_type, n_episodes=5)
        required = ['best_energy', 'fci_energy', 'energy_error_mha', 'n_episodes_run']
        missing = [f for f in required if f not in result_dict]

        if not missing:
            result.pass_test({
                "agent": agent_type,
                "best_energy": f"{result_dict['best_energy']:.6f}" if result_dict['best_energy'] else None,
                "n_operators": result_dict.get('n_operators'),
            })
        else:
            result.fail_test(f"Missing fields: {missing}")

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_ucc_energy_physical(result: TestResult, agent_type: str = 'ppo'):
    """UCC 搜索：best_energy 不高于 FCI 能量（物理合理性）"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type=agent_type, n_episodes=10)
        best_energy = result_dict['best_energy']
        fci_energy = result_dict['fci_energy']

        if best_energy is not None and best_energy <= fci_energy + 0.1:
            result.pass_test({
                "agent": agent_type,
                "best_energy": f"{best_energy:.6f}",
                "fci_energy": f"{fci_energy:.6f}",
            })
        else:
            result.fail_test(
                f"best_energy {best_energy:.6f} > fci_energy {fci_energy:.6f} "
                f"— agent may not be running real VQE"
            )

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_hea_search_basic(result: TestResult, agent_type: str = 'ppo'):
    """HEA 搜索：best_energy 存在且物理合理"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search('H2', 0.74, ansatz_type='HEA', agent_type=agent_type, n_episodes=5)
        best_energy = result_dict.get('best_energy')
        fci_energy = result_dict.get('fci_energy')

        if best_energy is not None and fci_energy is not None and best_energy <= fci_energy + 0.5:
            result.pass_test({
                "agent": agent_type,
                "best_energy": f"{best_energy:.6f}",
                "fci_energy": f"{fci_energy:.6f}",
                "n_operators": result_dict.get('n_operators'),
            })
        else:
            result.fail_test(
                f"HEA search returned implausible result: best_energy={best_energy}, fci_energy={fci_energy}"
            )

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 4: 超参数约束（用指定 agent）====================

def test_n_episodes_respected(result: TestResult, agent_type: str = 'ppo'):
    """n_episodes=15 传入后实际运行 15 轮（检查 n_episodes_run 字段）"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type=agent_type, n_episodes=15)
        n_run = result_dict.get('n_episodes_run')

        if n_run is None:
            result.fail_test("n_episodes_run not in results — cannot verify parameter was respected")
        elif n_run == 15:
            result.pass_test({"agent": agent_type, "n_episodes_run": n_run})
        else:
            result.fail_test(f"n_episodes_run={n_run}, expected 15 — may be overridden by internal default")

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_max_excitations_respected(result: TestResult, agent_type: str = 'ppo'):
    """max_operators=3 时，搜索结果中算符数 <= 3"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search(
            'H2', 0.74, ansatz_type='UCC', agent_type=agent_type, n_episodes=10, max_operators=3
        )
        n_ops = result_dict.get('n_operators')

        if n_ops is None:
            result.fail_test("n_operators not in results")
        elif n_ops <= 3:
            result.pass_test({"agent": agent_type, "n_operators": n_ops, "max_operators": 3})
        else:
            result.fail_test(
                f"n_operators={n_ops} exceeds max_operators=3"
            )

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 5: 化学精度（硬门控，用指定 agent）====================

def test_chemical_accuracy_ucc(result: TestResult, agent_type: str = 'ppo'):
    """UCC LiH 化学精度 < 1.6 mHa（300 episodes 快速门控）"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search('LiH', 1.6, ansatz_type='UCC', agent_type=agent_type, n_episodes=300)
        energy_error_mha = result_dict.get('energy_error_mha')

        if energy_error_mha is None:
            result.fail_test("energy_error_mha not in results — VQE may not be running correctly")
            return result

        if energy_error_mha < 1.6:
            result.pass_test({"agent": agent_type, "energy_error_mha": f"{energy_error_mha:.3f} mHa"})
        else:
            result.fail_test(
                f"Chemical accuracy NOT reached: {energy_error_mha:.3f} mHa >= 1.6 mHa "
                f"(agent={agent_type}). Full run: 1000 episodes on SLURM."
            )

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_chemical_accuracy_hea(result: TestResult, agent_type: str = 'ppo'):
    """HEA H2 化学精度 < 1.6 mHa（300 episodes 快速门控）"""
    try:
        import rlqas_chem

        result_dict = rlqas_chem.search('H2', 0.74, ansatz_type='HEA', agent_type=agent_type, n_episodes=300)
        energy_error_mha = result_dict.get('energy_error_mha')

        if energy_error_mha is None:
            result.fail_test("energy_error_mha not in results — VQE may not be running correctly")
            return result

        if energy_error_mha < 1.6:
            result.pass_test({"agent": agent_type, "energy_error_mha": f"{energy_error_mha:.3f} mHa"})
        else:
            result.fail_test(
                f"Chemical accuracy NOT reached: {energy_error_mha:.3f} mHa >= 1.6 mHa (agent={agent_type})"
            )

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 1 扩展: QOP 物理正确性（与 agent 无关）====================

def test_qop_tencirchem_mode_cnot_reduction(result: TestResult):
    """TenCirChem mode=qubit 比 mode=fermion CNOT 数更少（QOP 的核心物理意义）

    这是 QOP vs FOP 最重要的物理验证：相同精度下 QOP 产生更短的量子电路。
    此测试直接调用 TenCirChem，独立于 rlqas-chem 实现，约 0.5s 完成。
    用 LiH (2,4) 8-qubit 系统——足够大以产生可见的 CNOT 差异，又足够小以快速运行。
    H2 太小（Z 串不产生差异）；LiH (2,5) 也可但稍慢。
    若此测试失败，说明 TenCirChem 版本不支持 mode='qubit'，需升级依赖。
    """
    try:
        import pyscf
        from tencirchem import UCCSD

        # pyscf.M 直接构造，关闭对称性以避免 PointGroupSymmetryError
        mol = pyscf.M(
            atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g",
            symmetry=False, verbose=0,
        )
        active_space = (2, 4)  # 8 qubits

        u_fop = UCCSD(mol, mode="fermion", init_method="zeros", active_space=active_space)
        u_fop.kernel()
        fop_cnots = u_fop.get_circuit(decompose_multicontrol=True).gate_summary().get("cnot", 0)
        fop_err = abs(u_fop.energy() - u_fop.e_fci) * 1000

        u_qop = UCCSD(mol, mode="qubit", init_method="zeros", active_space=active_space)
        u_qop.kernel()
        qop_cnots = u_qop.get_circuit(decompose_multicontrol=True).gate_summary().get("cnot", 0)
        qop_err = abs(u_qop.energy() - u_qop.e_fci) * 1000

        if qop_cnots >= fop_cnots:
            result.fail_test(
                f"QOP CNOTs ({qop_cnots}) >= FOP CNOTs ({fop_cnots}): "
                f"mode='qubit' 未减少 CNOT 数，TenCirChem 行为异常"
            )
        elif qop_err > 1.6:
            result.fail_test(
                f"QOP 精度不足：{qop_err:.4f} mHa >= 1.6 mHa，mode='qubit' 损失了过多精度"
            )
        else:
            result.pass_test({
                "molecule": "LiH (2,4) 8q",
                "fop_cnots": fop_cnots, "qop_cnots": qop_cnots,
                "cnot_reduction": f"{(fop_cnots - qop_cnots) / fop_cnots * 100:.1f}%",
                "fop_error_mha": f"{fop_err:.4f}", "qop_error_mha": f"{qop_err:.4f}",
            })

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 3 扩展: QOP 搜索功能（用指定 agent）====================

def test_qop_search_basic(result: TestResult, agent_type: str = "ppo"):
    """QOP UCC 搜索：必须超越 HF 能量（检测静默失败）

    operator_pool='qop' 必须经由 TenCirChem mode='qubit' 实现。
    H2 的 FCI-HF 相关能约 20 mHa；正确的 QOP 实现在 20 episodes 内可达化学精度 (<1.6 mHa)。
    若 QOP Controller 内部报错但静默返回 HF 能量，误差 ≈ 20 mHa，被 < 10 mHa 门槛捕获。
    """
    try:
        import rlqas_chem

        r = rlqas_chem.search(
            "H2", 0.74, ansatz_type="UCC", operator_pool="qop",
            agent_type=agent_type, n_episodes=20,
        )
        required = ["best_energy", "fci_energy", "energy_error_mha", "n_episodes_run"]
        missing = [f for f in required if f not in r]
        if missing:
            result.fail_test(f"QOP result 缺少字段: {missing}")
            return result

        energy_error_mha = r.get("energy_error_mha")
        if energy_error_mha is None:
            result.fail_test("energy_error_mha 为 None，VQE 未运行")
            return result

        # H2 FCI-HF ≈ 20 mHa；正确 QOP 在 20 episodes 内应 < 10 mHa
        # 若返回 HF 能量（静默失败），误差 ≈ 20 mHa > 10 mHa，被此门控拦截
        if energy_error_mha > 10.0:
            result.fail_test(
                f"energy_error_mha={energy_error_mha:.2f} mHa > 10 mHa："
                f"QOP 未超越 HF 能量（H2 FCI-HF≈20 mHa），搜索静默失败。"
                f"需将 operator_pool='qop' 映射到 TenCirChem mode='qubit'"
            )
        else:
            result.pass_test({
                "agent": agent_type,
                "best_energy": f"{r['best_energy']:.6f}",
                "fci_energy": f"{r['fci_energy']:.6f}",
                "energy_error_mha": f"{energy_error_mha:.3f}",
            })

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_qop_cnot_less_than_fop(result: TestResult, agent_type: str = "ppo"):
    """QOP 搜索结果的 cnot_count 必须小于 FOP（相同算符数下电路更短）

    这是 QOP vs FOP 对比实验（E1）的核心门控：
    若 rlqas-chem 未把 operator_pool='qop' 映射到 TenCirChem mode='qubit'，
    两者 CNOT 数相同，此测试会失败，阻止错误实现进入主干。

    【注意】此测试依赖 search() 返回 'cnot_count' 字段。
    若该字段不存在，说明 api.py 尚未暴露此指标，需 Ralph 补充。
    """
    try:
        import rlqas_chem

        r_fop = rlqas_chem.search(
            "H2", 0.74, ansatz_type="UCC", operator_pool="fop",
            agent_type=agent_type, n_episodes=5,
        )
        r_qop = rlqas_chem.search(
            "H2", 0.74, ansatz_type="UCC", operator_pool="qop",
            agent_type=agent_type, n_episodes=5,
        )

        fop_cnots = r_fop.get("cnot_count")
        qop_cnots = r_qop.get("cnot_count")

        if fop_cnots is None or qop_cnots is None:
            result.fail_test(
                f"search() 结果缺少 'cnot_count' 字段 "
                f"(fop={fop_cnots}, qop={qop_cnots})。"
                f"需在 api.py 中暴露 cnot_count，并在 UCCCircuitBuilder 中"
                f"将 operator_pool='qop' 映射到 TenCirChem mode='qubit'。"
            )
        elif qop_cnots >= fop_cnots:
            result.fail_test(
                f"QOP CNOTs ({qop_cnots}) >= FOP CNOTs ({fop_cnots})。"
                f"operator_pool='qop' 未正确使用 TenCirChem mode='qubit'。"
            )
        else:
            result.pass_test({
                "agent": agent_type,
                "fop_cnot_count": fop_cnots,
                "qop_cnot_count": qop_cnots,
                "reduction": f"{(fop_cnots - qop_cnots) / fop_cnots * 100:.1f}%",
            })

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 5 扩展: QOP 化学精度 ====================

def test_chemical_accuracy_qop(result: TestResult, agent_type: str = "ppo"):
    """QOP UCC LiH 化学精度 < 1.6 mHa（300 episodes 快速门控）

    mode='qubit' 去掉 Z 串后不改变算符池，精度应与 FOP 相当。
    若此测试失败而 FOP 通过，说明 QOP 实现破坏了 VQE 优化逻辑。
    """
    try:
        import rlqas_chem

        r = rlqas_chem.search(
            "LiH", 1.6, ansatz_type="UCC", operator_pool="qop",
            agent_type=agent_type, n_episodes=300,
        )
        energy_error_mha = r.get("energy_error_mha")

        if energy_error_mha is None:
            result.fail_test(
                "energy_error_mha 字段缺失，VQE 可能未正常运行"
            )
        elif energy_error_mha < 1.6:
            result.pass_test({
                "agent": agent_type,
                "energy_error_mha": f"{energy_error_mha:.3f} mHa",
                "operator_pool": "qop",
            })
        else:
            result.fail_test(
                f"QOP 化学精度未达标：{energy_error_mha:.3f} mHa >= 1.6 mHa "
                f"(agent={agent_type}, operator_pool=qop)"
            )

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== 主测试运行器 ====================

EXISTING_AGENTS = ["ppo", "dqn", "a2c", "sac_discrete", "grpo"]


def build_tests(agent_type: str):
    """
    构建测试列表。分层回归策略：

    Level 1  — 与 agent 无关，始终只跑一次（含 QOP TenCirChem 物理验证）
    Level 2  — 用新 agent 跑（序列化/类型问题与 agent 实现有关）
    Level 3  — 新 agent + 所有已有算法冒烟 + QOP 搜索功能
    Level 4  — 用新 agent 跑（超参数约束）
    Level 5  — 新 agent + PPO 回归 + QOP 化学精度
    """
    tests = [
        # Level 1: 环境稳定性 + QOP 底层物理（与 agent 无关，始终只跑一次）
        ("Level 1: global_best 单调性",               1, test_global_best_monotonic,              {}),
        ("Level 1: reset 后 global_best 保留",        1, test_global_best_preserved_after_reset,  {}),
        ("Level 1: step() 返回值类型",                 1, test_step_return_types,                  {}),
        ("Level 1: 观测值无 NaN/Inf",                 1, test_obs_no_nan_inf,                     {}),
        ("Level 1: QOP mode=qubit CNOT 减少 [TC物理]", 1, test_qop_tencirchem_mode_cnot_reduction, {}),

        # Level 2: 新 agent
        (f"Level 2: JSON 序列化 [{agent_type}]",              2, test_json_serialization, {"agent_type": agent_type}),
        (f"Level 2: chemical_accuracy 类型 [{agent_type}]",   2, test_convergence_type,   {"agent_type": agent_type}),

        # Level 3: 新 agent（FOP + HEA + QOP 搜索功能）
        (f"Level 3: UCC FOP 基本搜索 [{agent_type}]",         3, test_ucc_search_basic,        {"agent_type": agent_type}),
        (f"Level 3: UCC FOP 能量物理合理 [{agent_type}]",     3, test_ucc_energy_physical,     {"agent_type": agent_type}),
        (f"Level 3: HEA 基本搜索 [{agent_type}]",             3, test_hea_search_basic,        {"agent_type": agent_type}),
        (f"Level 3: UCC QOP 基本搜索 [{agent_type}]",         3, test_qop_search_basic,        {"agent_type": agent_type}),
        (f"Level 3: UCC QOP CNOT < FOP [{agent_type}]",       3, test_qop_cnot_less_than_fop,  {"agent_type": agent_type}),

        # Level 4: 新 agent
        (f"Level 4: n_episodes 生效 [{agent_type}]",          4, test_n_episodes_respected,       {"agent_type": agent_type}),
        (f"Level 4: max_operators 约束 [{agent_type}]",       4, test_max_excitations_respected,  {"agent_type": agent_type}),

        # Level 5: 新 agent（FOP + HEA + QOP 化学精度）
        (f"Level 5: UCC FOP 化学精度 LiH [{agent_type}]",    5, test_chemical_accuracy_ucc, {"agent_type": agent_type}),
        (f"Level 5: HEA 化学精度 H2 [{agent_type}]",         5, test_chemical_accuracy_hea, {"agent_type": agent_type}),
        (f"Level 5: UCC QOP 化学精度 LiH [{agent_type}]",    5, test_chemical_accuracy_qop, {"agent_type": agent_type}),
    ]

    # Level 3 回归：所有已有算法冒烟（仅 FOP UCC 基本搜索，快速）
    for existing in EXISTING_AGENTS:
        if existing == agent_type:
            continue
        tests.append((
            f"Level 3: UCC FOP 基本搜索 [{existing} 回归]",
            3, test_ucc_search_basic, {"agent_type": existing}
        ))

    # Level 5 回归：仅 PPO（代表共享搜索基础设施）
    if agent_type != "ppo":
        tests.append((
            "Level 5: UCC FOP 化学精度 LiH [ppo 回归]",
            5, test_chemical_accuracy_ucc, {"agent_type": "ppo"}
        ))

    return tests


def run_all_tests(agent_type: str = 'ppo'):
    label = f"新算法验收: {agent_type}" if agent_type != 'ppo' else "回归测试 (PPO)"
    print("=" * 80)
    print(f"RLQAS 验收系统 - Level 1-5  |  {label}")
    print("=" * 80)
    print()

    tests = build_tests(agent_type)
    results = []

    for test_name, level, test_func, kwargs in tests:
        print(f"Running: {test_name}...")
        r = run_test(test_name, level, test_func, **kwargs)
        results.append(r)
        print(f"  {r}")
        print()

    # 按 Level 汇总
    print("=" * 80)
    print("测试汇总（按 Level）")
    print("=" * 80)
    for level in range(1, 6):
        level_results = [r for r in results if r.level == level]
        if not level_results:
            continue
        passed = sum(1 for r in level_results if r.passed)
        print(f"\nLevel {level}: {passed}/{len(level_results)} 通过")
        for r in level_results:
            print(f"  {'✅' if r.passed else '❌'} {r.name}")
            if r.error_msg:
                print(f"       Error: {r.error_msg}")

    # 总体判定
    passed_all = sum(1 for r in results if r.passed)
    failed_all = len(results) - passed_all
    print()
    print("=" * 80)
    print("总体判定")
    print("=" * 80)
    print(f"总计: {len(results)} 个测试  ✅ 通过: {passed_all}  ❌ 失败: {failed_all}")
    print()

    if failed_all > 0:
        print("失败项：")
        for r in results:
            if not r.passed:
                print(f"  ❌ [Level {r.level}] {r.name}")
                print(f"       {r.error_msg}")
        print()
        print("❌ 验收未通过")
        return False
    else:
        print("✅ 验收通过")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLQAS Level 1-5 验收测试")
    parser.add_argument("--agent", default="ppo", help="被验收的 agent 类型（默认 ppo 回归测试）")
    args = parser.parse_args()

    success = run_all_tests(agent_type=args.agent)
    sys.exit(0 if success else 1)
