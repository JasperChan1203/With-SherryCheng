"""
RLQAS HEA 验收系统 - Level 0: 接口契约测试

测试目标：
- HEA Env 接口：实例化、obs shape、reset/step 格式、固定步长终止、global_best 持久化
- HEA Controller 接口：实例化、最小化训练不崩溃
- Agent 接口（与 UCC 共用规范）：AgentFactory 注册、act() 格式、save/load、最小化训练

测试分子：LiH，active space 2e 3o，Jordan-Wigner 变换，max_layers=3

通过标准：所有测试通过，否则直接拒绝

运行方式：
    python3 rlqas_acceptance_system/hea_algorithms/test_acceptance_hea_level0.py
    python3 rlqas_acceptance_system/hea_algorithms/test_acceptance_hea_level0.py --agent gigppo
"""

import sys
import os
import logging
import warnings
import argparse
import numpy as np
from typing import Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

# 抑制 rlqas_chem 内部日志噪音（simulator WARNING、gym 弃用警告）
logging.getLogger('rlqas_chem').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rlqas-chem', 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rlqas-chem'))

# LiH 2e 3o JW — 所有 Level 0 测试共用的分子配置
_MOL_NAME   = 'LiH'
_BOND_LEN   = 1.6
_ACTIVE_SPACE = (2, 3)
_TRANSFORM  = 'jordan_wigner'
_MAX_LAYERS = 3
# PPO 默认 n_steps=2048，smoke test 必须 total_timesteps >= n_steps 才能完成一次收集
# 禁用 L-BFGS-B 使每步仅做电路模拟（< 1ms/step），2048 步约 2 秒
_SMOKE_TOTAL_TS = 2048
_SMOKE_OPT = False

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
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error_msg = None
        self.details = {}

    def pass_test(self, details: Dict = None):
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
        msg = f"{status} - {self.name}"
        if self.error_msg:
            msg += f"\n    Error: {self.error_msg}"
        if self.details:
            for k, v in self.details.items():
                msg += f"\n    {k}: {v}"
        return msg


def run_test(test_name: str, test_func, *args, **kwargs):
    result = TestResult(test_name)
    try:
        result = test_func(result, *args, **kwargs)
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== 2.0 分子预处理 ====================

def test_mol_processing(result: TestResult):
    """process_molecule 返回 LiH 2e 3o JW 的有效 MoleculeData"""
    try:
        mol = _get_mol_data()
        errors = []
        if not hasattr(mol, 'n_qubits') or mol.n_qubits <= 0:
            errors.append(f"invalid n_qubits: {getattr(mol, 'n_qubits', None)}")
        if not hasattr(mol, 'fci_energy') or mol.fci_energy is None:
            errors.append("fci_energy missing")
        if not hasattr(mol, 'hamiltonian') or mol.hamiltonian is None:
            errors.append("hamiltonian missing")
        if errors:
            result.fail_test("; ".join(errors))
        else:
            result.pass_test({"n_qubits": mol.n_qubits, "fci_energy": f"{mol.fci_energy:.6f}"})
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== 2.1 HEA Env 接口 ====================

def test_hea_env_instantiation(result: TestResult):
    """HEASearchEnv(mol_data, {'max_layers': 3}) 可实例化，action_space.n == 9"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        n_actions = env.action_space.n
        if n_actions != 9:
            result.fail_test(f"action_space.n={n_actions}, expected 9 (3 patterns × 3 rotations)")
        else:
            result.pass_test({"action_space_n": n_actions, "n_qubits": mol.n_qubits})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_hea_env_obs_shape(result: TestResult):
    """obs.shape == (3 + max_layers * n_qubits,)"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        obs, _ = env.reset()
        expected_shape = (3 + _MAX_LAYERS * mol.n_qubits,)
        if obs.shape == expected_shape:
            result.pass_test({"obs_shape": str(obs.shape)})
        else:
            result.fail_test(f"obs.shape={obs.shape}, expected {expected_shape}")
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_hea_env_reset_format(result: TestResult):
    """reset() 返回 (obs, info)，info 含 'layer' 和 'energy'"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        ret = env.reset()
        if not (isinstance(ret, tuple) and len(ret) == 2):
            result.fail_test(f"reset() returned {type(ret)}, expected 2-tuple")
            return result
        obs, info = ret
        missing = [k for k in ('layer', 'energy') if k not in info]
        if missing:
            result.fail_test(f"info missing keys: {missing}")
        elif info['layer'] != 0:
            result.fail_test(f"info['layer']={info['layer']} after reset, expected 0")
        else:
            result.pass_test({"obs_shape": str(obs.shape), "info_keys": list(info.keys())})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_hea_env_step_format(result: TestResult):
    """step() 返回 (obs, reward, terminated, truncated, info)，类型符合 gym 规范"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        env.reset()
        ret = env.step(env.action_space.sample())
        if not (isinstance(ret, tuple) and len(ret) == 5):
            result.fail_test(f"step() returned {len(ret)}-tuple, expected 5-tuple")
            return result
        obs, reward, terminated, truncated, info = ret
        errors = []
        if not isinstance(reward, (float, int, np.floating)):
            errors.append(f"reward type={type(reward)}")
        if not isinstance(terminated, (bool, np.bool_)):
            errors.append(f"terminated type={type(terminated)}")
        if not isinstance(info, dict):
            errors.append(f"info type={type(info)}")
        if errors:
            result.fail_test("; ".join(errors))
        else:
            result.pass_test({"reward": float(reward), "terminated": bool(terminated)})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_hea_episode_termination(result: TestResult):
    """episode 恰好在第 max_layers 步时 terminated=True，之前不提前终止"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        env.reset()
        steps = 0
        early_term = False
        for i in range(_MAX_LAYERS):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            done = terminated or truncated
            if done and i < _MAX_LAYERS - 1:
                early_term = True
                break
        if early_term:
            result.fail_test(f"Episode terminated early at step {steps}, expected step {_MAX_LAYERS}")
        elif not done:
            result.fail_test(f"Episode did not terminate after {_MAX_LAYERS} steps")
        else:
            result.pass_test({"steps_to_done": steps, "max_layers": _MAX_LAYERS})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_hea_global_best_persistence(result: TestResult):
    """reset() 后 global_best_energy 不退回 inf"""
    try:
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        obs, _ = env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
        best_before = env.best_energy
        env.reset()
        best_after = env.best_energy
        if best_after <= best_before + 1e-9:
            result.pass_test({"best_before": f"{best_before:.6f}", "best_after": f"{best_after:.6f}"})
        else:
            result.fail_test(f"global_best regressed after reset: {best_before:.6f} -> {best_after:.6f}")
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== 2.2 Agent 接口（与 UCC 共용规范）====================

def test_agent_factory_registration(result: TestResult, agent_type: str = 'ppo'):
    """AgentFactory 能用 HEASearchEnv 实例化指定 agent"""
    try:
        from rlqas_chem.rl.agent_factory import AgentFactory
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        agent = AgentFactory.create_agent(agent_type, config={}, env=env)
        result.pass_test({"agent": agent_type, "class": type(agent).__name__})
    except Exception as e:
        result.fail_test(f"AgentFactory could not create '{agent_type}' with HEA env: {e}")
    return result


def test_act_return_format(result: TestResult, agent_type: str = 'ppo'):
    """act(state) 返回 (int, dict)"""
    try:
        from rlqas_chem.rl.agent_factory import AgentFactory
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        agent = AgentFactory.create_agent(agent_type, config={}, env=env)
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        ret = agent.act(state)
        if isinstance(ret, tuple) and len(ret) == 2:
            action, info = ret
            if isinstance(action, (int, np.integer)) and isinstance(info, dict):
                result.pass_test({"agent": agent_type, "action": int(action)})
            else:
                result.fail_test(f"act() returned ({type(action)}, {type(info)}), expected (int, dict)")
        else:
            result.fail_test(f"act() did not return a 2-tuple, got {type(ret)}")
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_save_load_consistency(result: TestResult, agent_type: str = 'ppo'):
    """save/load 往返后 act() 输出一致"""
    try:
        import tempfile
        from rlqas_chem.rl.agent_factory import AgentFactory
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        agent1 = AgentFactory.create_agent(agent_type, config={}, env=env)
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        action1, _ = agent1.act(state)
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'hea_agent')
            agent1.save(save_path)
            agent2 = AgentFactory.create_agent(agent_type, config={}, env=env)
            agent2.load(save_path)
            action2, _ = agent2.act(state)
        if action1 == action2:
            result.pass_test({"agent": agent_type, "action_before": int(action1), "action_after": int(action2)})
        else:
            result.fail_test(f"Actions differ after save/load: {action1} vs {action2}")
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_config_known_keys(result: TestResult, agent_type: str = 'ppo'):
    """已知超参数（gamma）被正确接受，不崩溃"""
    try:
        from rlqas_chem.rl.agent_factory import AgentFactory
        from rlqas_chem.search.hea.environment import HEASearchEnv
        mol = _get_mol_data()
        env = HEASearchEnv(mol, {'max_layers': _MAX_LAYERS})
        config = {'gamma': 0.99}
        AgentFactory.create_agent(agent_type, config=config, env=env)
        result.pass_test({"agent": agent_type, "config": str(config)})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_minimal_training(result: TestResult, agent_type: str = 'ppo'):
    """最小化训练（total_timesteps=2048，禁用 L-BFGS-B）不崩溃"""
    try:
        from rlqas_chem.search.hea.controller import HEASearchController
        mol = _get_mol_data()
        controller = HEASearchController(n_qubits=mol.n_qubits, max_layers=_MAX_LAYERS)
        controller.search(
            agent_type=agent_type,
            n_episodes=2,
            total_timesteps=_SMOKE_TOTAL_TS,
            molecule_data=mol,
            run_classical_opt=_SMOKE_OPT,
        )
        result.pass_test({"agent": agent_type, "message": f"HEA training ({_SMOKE_TOTAL_TS} ts, no L-BFGS-B) completed without crash"})
    except Exception as e:
        result.fail_test(str(e))
    return result


def test_act_after_training(result: TestResult, agent_type: str = 'ppo'):
    """训练后 act() 返回合法 action index（在 HEA action space 范围内）"""
    try:
        from rlqas_chem.search.hea.controller import HEASearchController
        mol = _get_mol_data()
        controller = HEASearchController(n_qubits=mol.n_qubits, max_layers=_MAX_LAYERS)
        controller.search(
            agent_type=agent_type,
            n_episodes=2,
            total_timesteps=_SMOKE_TOTAL_TS,
            molecule_data=mol,
            run_classical_opt=_SMOKE_OPT,
        )
        agent = controller._agent
        env = controller._env
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        action, _ = agent.act(state)
        if isinstance(action, (int, np.integer)) and 0 <= action < env.action_space.n:
            result.pass_test({"agent": agent_type, "action": int(action), "action_space_n": env.action_space.n})
        else:
            result.fail_test(f"Invalid action {action} for action_space.n={env.action_space.n}")
    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== 主测试运行器 ====================

EXISTING_AGENTS = ["ppo", "dqn", "a2c", "sac_discrete", "grpo"]


def build_tests(agent_type: str):
    """
    构建 Level 0 测试列表。

    HEA Env/Controller 接口：每次只跑一次（与 agent 无关）
    Agent 接口（新 agent）：全套 6 项
    已有算法回归：AgentFactory 注册 + 最小化训练（快速）
    """
    tests = [
        # HEA Env/Controller 接口（与 agent 无关）
        ("2.0 分子预处理 [LiH 2e 3o JW]",            test_mol_processing,             {}),
        ("2.1 HEA Env 实例化",                        test_hea_env_instantiation,      {}),
        ("2.1 HEA Env obs shape",                     test_hea_env_obs_shape,          {}),
        ("2.1 HEA Env reset() 格式",                  test_hea_env_reset_format,       {}),
        ("2.1 HEA Env step() 格式",                   test_hea_env_step_format,        {}),
        ("2.1 HEA Env 固定步长终止",                  test_hea_episode_termination,    {}),
        ("2.1 HEA Env global_best 持久化",            test_hea_global_best_persistence,{}),
        # Agent 接口（针对指定 agent）
        (f"2.2 AgentFactory 注册 [{agent_type}]",     test_agent_factory_registration, {"agent_type": agent_type}),
        (f"2.2 act() 返回格式 [{agent_type}]",        test_act_return_format,          {"agent_type": agent_type}),
        (f"2.2 save/load 一致性 [{agent_type}]",      test_save_load_consistency,      {"agent_type": agent_type}),
        (f"2.2 Config 已知 key [{agent_type}]",       test_config_known_keys,          {"agent_type": agent_type}),
        (f"2.2 最小化训练 [{agent_type}]",            test_minimal_training,           {"agent_type": agent_type}),
        (f"2.2 训练后 act() 合法 [{agent_type}]",     test_act_after_training,         {"agent_type": agent_type}),
    ]

    # 所有已有算法的回归：注册 + 最小化训练
    for existing in EXISTING_AGENTS:
        if existing == agent_type:
            continue
        tests += [
            (f"2.2 AgentFactory 注册 [{existing} 回归]", test_agent_factory_registration, {"agent_type": existing}),
            (f"2.2 最小化训练 [{existing} 回归]",         test_minimal_training,           {"agent_type": existing}),
        ]

    return tests


def run_all_level0_tests(agent_type: str = 'ppo'):
    label = f"新算法验收: {agent_type}" if agent_type != 'ppo' else "回归测试 (PPO)"
    print("=" * 80)
    print(f"RLQAS HEA 验收系统 - Level 0: 接口契约测试  |  {label}")
    print("=" * 80)
    print()

    tests = build_tests(agent_type)
    results = []

    for test_name, test_func, kwargs in tests:
        print(f"Running: {test_name}...")
        r = run_test(test_name, test_func, **kwargs)
        results.append(r)
        print(f"  {r}")
        print()

    print("=" * 80)
    print("测试汇总")
    print("=" * 80)
    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"总计: {len(results)} 个测试  ✅ 通过: {passed}  ❌ 失败: {failed}")
    print()

    if failed > 0:
        print("失败项：")
        for r in results:
            if not r.passed:
                print(f"  ❌ {r.name}")
                print(f"       {r.error_msg}")
        print()
        print("❌ Level 0 未通过，拒绝合并")
        return False
    else:
        print("✅ Level 0 通过，可继续 Level 1-5 测试")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLQAS HEA Level 0 接口契约测试")
    parser.add_argument("--agent", default="ppo", help="被验收的 agent 类型（默认 ppo 回归测试）")
    args = parser.parse_args()

    success = run_all_level0_tests(agent_type=args.agent)
    sys.exit(0 if success else 1)
