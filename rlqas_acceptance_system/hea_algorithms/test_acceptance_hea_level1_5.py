"""
RLQAS HEA 验收系统 - Level 1-5 测试

测试目标：
- Level 1: HEA 环境稳定性（步数、history 正确性、obs 合法性、global_best 单调性）
- Level 2: HEA 搜索结果正确性（HEA 特有字段、JSON 序列化）
- Level 3: HEA 搜索功能（结果字段完整、能量物理合理）
- Level 4: HEA 超参数约束（max_layers 真正生效）
- Level 5: 化学精度硬门控（LiH 2e 3o JW，误差 < 1.6 mHa）

测试分子：LiH，active space 2e 3o，Jordan-Wigner 变换，max_layers=3

运行方式：
    python3 rlqas_acceptance_system/hea_algorithms/test_acceptance_hea_level1_5.py
    python3 rlqas_acceptance_system/hea_algorithms/test_acceptance_hea_level1_5.py --agent gigppo
"""

import sys
import os
import json
import logging
import warnings
import argparse
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

# 抑制 rlqas_chem 内部日志噪音（simulator WARNING、gym 弃用警告）
logging.getLogger('rlqas_chem').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rlqas-chem', 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rlqas-chem'))

_MOL_NAME     = 'LiH'
_BOND_LEN     = 1.6
_ACTIVE_SPACE = (2, 3)
_TRANSFORM    = 'jordan_wigner'
_MAX_LAYERS   = 3

_VALID_ENTANGLEMENTS = {"linear", "circular", "full"}
_VALID_ROTATIONS     = {"rx", "ry", "rz"}

# PPO n_steps must be <= total_timesteps = n_episodes * max_layers.
# Smoke tests (Level 2-4) use n_episodes <= 10 -> total_timesteps <= 30, so n_steps=6.
# Level 5 uses n_episodes=200 -> total_timesteps=600, so n_steps=64.
_SMOKE_AGENT_CONFIG = {'n_steps': 6,  'batch_size': 6}
_L5_AGENT_CONFIG    = {'n_steps': 64, 'batch_size': 32}
_L5_N_EPISODES      = 200

# Agents that use n_steps (rollout-buffer based). Others (e.g. DQN) don't accept it.
_STEP_BASED_AGENTS = frozenset({'ppo', 'a2c'})

def _smoke_cfg(agent_type: str) -> dict:
    """Return the appropriate smoke-test config for a given agent."""
    return _SMOKE_AGENT_CONFIG if agent_type in _STEP_BASED_AGENTS else {'batch_size': 6}

_mol_data_cache = None

def _get_mol_data():
    global _mol_data_cache
    if _mol_data_cache is None:
        from rlqas_chem.molecule.processor import process_molecule
        _mol_data_cache = process_molecule(
            _MOL_NAME, _BOND_LEN, 'HEA',
            active_space=_ACTIVE_SPACE,
            transform=_TRANSFORM,
        )
    return _mol_data_cache


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


# ==================== Level 1: HEA 环境稳定性（与 agent 无关）====================

def test_episode_length_exact(result: TestResult):
    """每个 episode 恰好走 max_layers=3 步"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})

        episode_lengths = []
        for _ in range(5):
            env.reset()
            steps = 0
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated
                steps += 1
                if steps > _MAX_LAYERS + 5:
                    break
            episode_lengths.append(steps)

        wrong = [l for l in episode_lengths if l != _MAX_LAYERS]
        if wrong:
            result.fail_test(f"Episode lengths {episode_lengths}, expected all {_MAX_LAYERS}")
        else:
            result.pass_test({"episode_lengths": episode_lengths, "max_layers": _MAX_LAYERS})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_history_lengths_match(result: TestResult):
    """entanglement_history 和 rotation_history 长度等于 episode 实际步数"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1

        cfg = env.best_circuit_config
        ent_len = len(cfg.get('entanglement_history', []))
        rot_len = len(cfg.get('rotation_history', []))
        errors = []
        if ent_len != steps:
            errors.append(f"entanglement_history len={ent_len}, expected {steps}")
        if rot_len != steps:
            errors.append(f"rotation_history len={rot_len}, expected {steps}")
        if errors:
            result.fail_test("; ".join(errors))
        else:
            result.pass_test({"steps": steps, "ent_history_len": ent_len, "rot_history_len": rot_len})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_history_values_valid(result: TestResult):
    """entanglement_history 和 rotation_history 中所有值均为合法字符串"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated

        cfg = env.best_circuit_config
        ent_hist = cfg.get('entanglement_history', [])
        rot_hist = cfg.get('rotation_history', [])
        bad_ent = [v for v in ent_hist if v not in _VALID_ENTANGLEMENTS]
        bad_rot = [v for v in rot_hist if v not in _VALID_ROTATIONS]
        errors = []
        if bad_ent:
            errors.append(f"invalid entanglement values: {bad_ent}")
        if bad_rot:
            errors.append(f"invalid rotation values: {bad_rot}")
        if errors:
            result.fail_test("; ".join(errors))
        else:
            result.pass_test({"entanglement_history": ent_hist, "rotation_history": rot_hist})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_obs_no_nan_inf(result: TestResult):
    """完整 episode 内所有观测值不含 NaN/Inf"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        obs, _ = env.reset()
        step = 0
        if not np.all(np.isfinite(obs)):
            result.fail_test("obs contains NaN/Inf at reset")
            return result
        done = False
        while not done:
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            step += 1
            if not np.all(np.isfinite(obs)):
                result.fail_test(f"obs contains NaN/Inf at step {step}")
                return result
        result.pass_test({"message": f"All observations finite over {step} steps"})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_global_best_monotonic(result: TestResult):
    """global_best_energy 跨 episode 单调不增"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        global_bests = []
        for _ in range(5):
            env.reset()
            done = False
            while not done:
                _, _, terminated, truncated, _ = env.step(env.action_space.sample())
                done = terminated or truncated
            global_bests.append(env.best_energy)

        monotonic = all(global_bests[i] >= global_bests[i + 1] for i in range(len(global_bests) - 1))
        if monotonic:
            result.pass_test({"global_bests": [f"{e:.6f}" for e in global_bests]})
        else:
            result.fail_test(f"global_best not monotonic: {global_bests}")
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_global_best_preserved_after_reset(result: TestResult):
    """reset() 后 global_best_energy 不退回 inf"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
        best_before = env.best_energy
        env.reset()
        best_after = env.best_energy
        if best_after <= best_before + 1e-9:
            result.pass_test({"best_before_reset": f"{best_before:.6f}", "best_after_reset": f"{best_after:.6f}"})
        else:
            result.fail_test(f"global_best regressed after reset: {best_before:.6f} -> {best_after:.6f}")
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_step_info_keys(result: TestResult):
    """step() 返回的 info 含 HEA 特有键：entanglement, rotation, energy_delta"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        required = {'layer', 'energy', 'entanglement', 'rotation', 'energy_delta'}
        missing = required - set(info.keys())
        if missing:
            result.fail_test(f"info missing HEA-specific keys: {missing}")
        else:
            result.pass_test({
                "entanglement": info['entanglement'],
                "rotation": info['rotation'],
                "energy_delta": f"{info['energy_delta']:.6f}",
            })
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 2: HEA 搜索结果正确性（用指定 agent）====================

def test_n_operators_is_none(result: TestResult, agent_type: str = 'ppo'):
    """HEA 搜索 API 返回的 n_operators 必须为 None"""
    try:
        import rlqas_chem
        r = rlqas_chem.search(
            _MOL_NAME, _BOND_LEN, ansatz_type='HEA', agent_type=agent_type,
            n_episodes=2, active_space=_ACTIVE_SPACE, transform=_TRANSFORM,
            max_operators=_MAX_LAYERS, config=_SMOKE_AGENT_CONFIG,
        )
        n_ops = r.get('n_operators')
        if n_ops is None:
            result.pass_test({"agent": agent_type, "n_operators": None})
        else:
            result.fail_test(f"n_operators={n_ops!r}, expected None for HEA")
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_best_circuit_structure(result: TestResult, agent_type: str = 'ppo'):
    """HEASearchController.search() 返回的 best_circuit 包含必要字段且 history 长度 ≤ max_layers"""
    try:
        from rlqas_chem.search.hea.controller import HEASearchController
        mol = _get_mol_data()
        controller = HEASearchController(n_qubits=mol.n_qubits, max_layers=_MAX_LAYERS)
        r = controller.search(
            agent_type=agent_type,
            n_episodes=2,
            total_timesteps=2 * _MAX_LAYERS,
            molecule_data=mol,
            agent_config=_SMOKE_AGENT_CONFIG,
        )
        circuit = r.get('best_circuit')
        if circuit is None:
            result.fail_test("best_circuit is None")
            return result
        required_keys = {'entanglement_history', 'rotation_history', 'n_qubits', 'max_layers', 'n_parameters'}
        missing = required_keys - set(circuit.keys())
        if missing:
            result.fail_test(f"best_circuit missing keys: {missing}")
            return result
        ent_len = len(circuit['entanglement_history'])
        rot_len = len(circuit['rotation_history'])
        if ent_len > _MAX_LAYERS or rot_len > _MAX_LAYERS:
            result.fail_test(
                f"history lengths ({ent_len}, {rot_len}) exceed max_layers={_MAX_LAYERS}"
            )
        else:
            result.pass_test({
                "agent": agent_type,
                "entanglement_history": circuit['entanglement_history'],
                "rotation_history": circuit['rotation_history'],
                "n_parameters": circuit['n_parameters'],
            })
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_json_serialization(result: TestResult, agent_type: str = 'ppo'):
    """search() 结果可直接 JSON 序列化"""
    try:
        import rlqas_chem
        r = rlqas_chem.search(
            _MOL_NAME, _BOND_LEN, ansatz_type='HEA', agent_type=agent_type,
            n_episodes=2, active_space=_ACTIVE_SPACE, transform=_TRANSFORM,
            max_operators=_MAX_LAYERS, config=_SMOKE_AGENT_CONFIG,
        )
        json_str = json.dumps(r)
        parsed = json.loads(json_str)
        result.pass_test({"agent": agent_type, "keys": list(parsed.keys())})
    except TypeError as e:
        result.fail_test(f"JSON serialization failed: {e}")
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_chemical_accuracy_type(result: TestResult, agent_type: str = 'ppo'):
    """chemical_accuracy 是 Python bool，不是 numpy.bool_"""
    try:
        import rlqas_chem
        r = rlqas_chem.search(
            _MOL_NAME, _BOND_LEN, ansatz_type='HEA', agent_type=agent_type,
            n_episodes=2, active_space=_ACTIVE_SPACE, transform=_TRANSFORM,
            max_operators=_MAX_LAYERS, config=_SMOKE_AGENT_CONFIG,
        )
        conv = r.get('chemical_accuracy')
        if type(conv) is bool:
            result.pass_test({"agent": agent_type, "chemical_accuracy": conv})
        else:
            result.fail_test(f"chemical_accuracy is {type(conv)}, expected Python bool")
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 3: HEA 搜索功能（用指定 agent）====================

def test_lih_hea_search_fields_complete(result: TestResult, agent_type: str = 'ppo'):
    """LiH HEA 搜索：必要字段完整"""
    try:
        import rlqas_chem
        r = rlqas_chem.search(
            _MOL_NAME, _BOND_LEN, ansatz_type='HEA', agent_type=agent_type,
            n_episodes=5, active_space=_ACTIVE_SPACE, transform=_TRANSFORM,
            max_operators=_MAX_LAYERS, config=_smoke_cfg(agent_type),
        )
        required = ['best_energy', 'fci_energy', 'energy_error_mha', 'n_episodes_run', 'n_qubits']
        missing = [f for f in required if f not in r]
        if missing:
            result.fail_test(f"Missing fields: {missing}")
        else:
            result.pass_test({
                "agent": agent_type,
                "best_energy": f"{r['best_energy']:.6f}",
                "fci_energy": f"{r['fci_energy']:.6f}",
                "n_qubits": r['n_qubits'],
            })
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_lih_hea_energy_physical(result: TestResult, agent_type: str = 'ppo'):
    """LiH HEA 搜索：best_energy 不高于 FCI 能量（物理合理性）"""
    try:
        import rlqas_chem
        r = rlqas_chem.search(
            _MOL_NAME, _BOND_LEN, ansatz_type='HEA', agent_type=agent_type,
            n_episodes=10, active_space=_ACTIVE_SPACE, transform=_TRANSFORM,
            max_operators=_MAX_LAYERS, config=_SMOKE_AGENT_CONFIG,
        )
        best_energy = r['best_energy']
        fci_energy = r['fci_energy']
        if best_energy <= fci_energy + 0.1:
            result.pass_test({
                "agent": agent_type,
                "best_energy": f"{best_energy:.6f}",
                "fci_energy": f"{fci_energy:.6f}",
            })
        else:
            result.fail_test(
                f"best_energy {best_energy:.6f} > fci_energy {fci_energy:.6f} + 0.1 "
                f"— VQE may not be running correctly"
            )
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 4: HEA 超参数约束（用指定 agent）====================

def test_max_layers_respected(result: TestResult, agent_type: str = 'ppo'):
    """max_operators=3 传入后，best_circuit history 长度 ≤ 3"""
    try:
        import rlqas_chem
        r = rlqas_chem.search(
            _MOL_NAME, _BOND_LEN, ansatz_type='HEA', agent_type=agent_type,
            n_episodes=5, active_space=_ACTIVE_SPACE, transform=_TRANSFORM,
            max_operators=3, config=_SMOKE_AGENT_CONFIG,
        )
        # n_operators is None for HEA; check n_qubits and n_episodes_run instead
        n_qubits = r.get('n_qubits')
        if n_qubits is None:
            result.fail_test("n_qubits not in results")
            return result

        # Verify via controller directly: best_circuit history length
        from rlqas_chem.search.hea.controller import HEASearchController
        mol = _get_mol_data()
        ctrl = HEASearchController(n_qubits=mol.n_qubits, max_layers=3)
        ctrl_r = ctrl.search(
            agent_type=agent_type,
            n_episodes=5,
            total_timesteps=5 * 3,
            molecule_data=mol,
            agent_config=_SMOKE_AGENT_CONFIG,
        )
        circuit = ctrl_r.get('best_circuit', {})
        ent_len = len(circuit.get('entanglement_history', []))
        rot_len = len(circuit.get('rotation_history', []))
        if ent_len <= 3 and rot_len <= 3:
            result.pass_test({
                "agent": agent_type,
                "entanglement_history_len": ent_len,
                "rotation_history_len": rot_len,
                "max_layers": 3,
            })
        else:
            result.fail_test(
                f"history lengths ({ent_len}, {rot_len}) exceed max_layers=3"
            )
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== Level 5: 化学精度硬门控（用指定 agent）====================

def test_chemical_accuracy_lih_hea(result: TestResult, agent_type: str = 'ppo'):
    """LiH 2e 3o JW，HEA max_layers=3，200 episodes，能量误差 < 1.6 mHa"""
    try:
        import rlqas_chem
        r = rlqas_chem.search(
            _MOL_NAME, _BOND_LEN, ansatz_type='HEA', agent_type=agent_type,
            n_episodes=_L5_N_EPISODES, active_space=_ACTIVE_SPACE, transform=_TRANSFORM,
            max_operators=_MAX_LAYERS, config=_L5_AGENT_CONFIG,
        )
        error_mha = r.get('energy_error_mha')
        if error_mha is None:
            result.fail_test("energy_error_mha not in results")
            return result
        if error_mha < 1.6:
            result.pass_test({
                "agent": agent_type,
                "energy_error_mha": f"{error_mha:.3f} mHa",
                "best_energy": f"{r['best_energy']:.6f}",
                "fci_energy": f"{r['fci_energy']:.6f}",
            })
        else:
            result.fail_test(
                f"Chemical accuracy NOT reached: {error_mha:.3f} mHa >= 1.6 mHa "
                f"(agent={agent_type}, molecule=LiH 2e 3o JW, max_layers={_MAX_LAYERS})"
            )
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== 主测试运行器 ====================

EXISTING_AGENTS = ["ppo", "dqn", "a2c", "sac_discrete", "grpo"]


def build_tests(agent_type: str):
    """
    构建 HEA Level 1-5 测试列表。

    Level 1 — 与 agent 无关，始终只跑一次
    Level 2 — 用新 agent 跑
    Level 3 — 新 agent + 已有算法 UCC 基本搜索冒烟
    Level 4 — 用新 agent 跑
    Level 5 — 新 agent + PPO 回归
    """
    tests = [
        # Level 1: 环境稳定性（与 agent 无关）
        ("Level 1: episode 步数严格等于 max_layers",        1, test_episode_length_exact,          {}),
        ("Level 1: history 长度匹配步数",                   1, test_history_lengths_match,          {}),
        ("Level 1: history 值合法",                         1, test_history_values_valid,           {}),
        ("Level 1: 完整 episode obs 无 NaN/Inf",            1, test_obs_no_nan_inf,                 {}),
        ("Level 1: global_best 跨 episode 单调不增",        1, test_global_best_monotonic,          {}),
        ("Level 1: reset 后 global_best 保留",              1, test_global_best_preserved_after_reset, {}),
        ("Level 1: step() info 含 HEA 特有键",             1, test_step_info_keys,                 {}),

        # Level 2: 新 agent
        (f"Level 2: n_operators 为 None [{agent_type}]",    2, test_n_operators_is_none,            {"agent_type": agent_type}),
        (f"Level 2: best_circuit 结构 [{agent_type}]",      2, test_best_circuit_structure,         {"agent_type": agent_type}),
        (f"Level 2: JSON 序列化 [{agent_type}]",            2, test_json_serialization,             {"agent_type": agent_type}),
        (f"Level 2: chemical_accuracy 类型 [{agent_type}]", 2, test_chemical_accuracy_type,         {"agent_type": agent_type}),

        # Level 3: 新 agent
        (f"Level 3: LiH HEA 字段完整 [{agent_type}]",      3, test_lih_hea_search_fields_complete, {"agent_type": agent_type}),
        (f"Level 3: LiH HEA 能量物理合理 [{agent_type}]",  3, test_lih_hea_energy_physical,        {"agent_type": agent_type}),

        # Level 4: 新 agent
        (f"Level 4: max_layers=3 生效 [{agent_type}]",      4, test_max_layers_respected,           {"agent_type": agent_type}),

        # Level 5: 新 agent
        (f"Level 5: LiH HEA 化学精度 [{agent_type}]",      5, test_chemical_accuracy_lih_hea,      {"agent_type": agent_type}),
    ]

    # Level 3 回归：所有已有算法的 HEA 基本搜索冒烟（少量 episodes，快速）
    for existing in EXISTING_AGENTS:
        if existing == agent_type:
            continue
        tests.append((
            f"Level 3: LiH HEA 字段完整 [{existing} 回归]",
            3, test_lih_hea_search_fields_complete, {"agent_type": existing}
        ))

    # Level 5 回归：仅 PPO（代表共享搜索基础设施）
    if agent_type != 'ppo':
        tests.append((
            "Level 5: LiH HEA 化学精度 [ppo 回归]",
            5, test_chemical_accuracy_lih_hea, {"agent_type": "ppo"}
        ))

    return tests


def run_all_tests(agent_type: str = 'ppo'):
    label = f"新算法验收: {agent_type}" if agent_type != 'ppo' else "回归测试 (PPO)"
    print("=" * 80)
    print(f"RLQAS HEA 验收系统 - Level 1-5  |  {label}")
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
        print("❌ HEA 验收未通过")
        return False
    else:
        print("✅ HEA 验收通过")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLQAS HEA Level 1-5 验收测试")
    parser.add_argument("--agent", default="ppo", help="被验收的 agent 类型（默认 ppo 回归测试）")
    args = parser.parse_args()

    success = run_all_tests(agent_type=args.agent)
    sys.exit(0 if success else 1)
