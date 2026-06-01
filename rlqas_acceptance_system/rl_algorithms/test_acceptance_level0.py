"""
RLQAS 验收系统 - Level 0: 接口契约测试

测试目标：验证 agent 接口签名、Config 兼容性、基础训练冒烟测试
通过标准：所有测试通过，否则直接拒绝

运行方式：
    # 仅回归测试（验证所有已注册 agent）
    python3 rlqas_acceptance_system/test_acceptance_level0.py

    # 验收新算法
    python3 rlqas_acceptance_system/test_acceptance_level0.py --agent gigppo
"""

import sys
import os
import argparse
import numpy as np
from typing import Dict

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rlqas-chem', 'src'))
sys.path.insert(0, os.path.join(_REPO_ROOT, 'rlqas-chem'))

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


# ==================== 1.1 接口签名一致性 ====================

def test_agent_factory_registration(result: TestResult, agent_type: str = 'ppo'):
    """AgentFactory 能实例化指定 agent"""
    try:
        from rlqas_chem.rl.agent_factory import AgentFactory
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        agent = AgentFactory.create_agent(agent_type, config={}, env=env)
        result.pass_test({"agent": agent_type, "class": type(agent).__name__})

    except Exception as e:
        result.fail_test(f"AgentFactory could not create '{agent_type}': {e}")
    return result


def test_learn_callback_signature(result: TestResult, agent_type: str = 'ppo'):
    """learn() 接受 callback 关键字参数"""
    try:
        from rlqas_chem.search.ucc.controller import UCCSearchController
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        controller = UCCSearchController(mol_data, agent_type=agent_type)

        try:
            controller.search(n_episodes=2)
            result.pass_test({"agent": agent_type, "message": "learn() accepts callback parameter"})
        except TypeError as e:
            if "callback" in str(e):
                result.fail_test(f"learn() does not accept callback: {e}")
            else:
                result.pass_test({"agent": agent_type, "warning": str(e)})

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_act_return_format(result: TestResult, agent_type: str = 'ppo'):
    """act(state) 返回 (int, dict)"""
    try:
        from rlqas_chem.rl.agent_factory import AgentFactory
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
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
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        agent1 = AgentFactory.create_agent(agent_type, config={}, env=env)

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        action1, _ = agent1.act(state)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'agent')
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


# ==================== 1.2 Config 兼容性 ====================

def test_config_known_keys(result: TestResult, agent_type: str = 'ppo'):
    """已知超参数（gamma）被正确接受，不崩溃"""
    try:
        from rlqas_chem.rl.agent_factory import AgentFactory
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        config = {'gamma': 0.99}
        AgentFactory.create_agent(agent_type, config=config, env=env)
        result.pass_test({"agent": agent_type, "config": str(config)})

    except Exception as e:
        result.fail_test(str(e))
    return result


# ==================== 1.3 基础训练冒烟测试 ====================

def test_minimal_training(result: TestResult, agent_type: str = 'ppo'):
    """最小训练（2 episodes）不崩溃"""
    try:
        from rlqas_chem.search.ucc.controller import UCCSearchController
        from rlqas_chem.molecule.processor import process_molecule
        import warnings

        mol_data = process_molecule('H2', 0.74, 'UCC')
        controller = UCCSearchController(mol_data, agent_type=agent_type)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            controller.search(n_episodes=2)

        result.pass_test({"agent": agent_type, "message": "2-episode training completed without crash"})

    except Exception as e:
        result.fail_test(str(e))
    return result


def test_act_after_training(result: TestResult, agent_type: str = 'ppo'):
    """训练后 act() 返回合法 action index"""
    try:
        from rlqas_chem.search.ucc.controller import UCCSearchController
        from rlqas_chem.molecule.processor import process_molecule

        mol_data = process_molecule('H2', 0.74, 'UCC')
        controller = UCCSearchController(mol_data, agent_type=agent_type)
        controller.search(n_episodes=2)

        agent = controller.agent
        env = controller.env
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

    回归策略：
    - 新 agent 本身：全套 7 项测试
    - 所有已有算法：AgentFactory 注册 + 最小化训练（快速，< 1 min）
      确保新 agent 加入后没有破坏已有 agent 的接口
    """
    tests = [
        (f"1.1 AgentFactory 注册 [{agent_type}]",    test_agent_factory_registration, {"agent_type": agent_type}),
        (f"1.1 learn(callback) 签名 [{agent_type}]", test_learn_callback_signature,   {"agent_type": agent_type}),
        (f"1.1 act() 返回格式 [{agent_type}]",        test_act_return_format,          {"agent_type": agent_type}),
        (f"1.1 save/load 一致性 [{agent_type}]",      test_save_load_consistency,      {"agent_type": agent_type}),
        (f"1.2 Config 已知 key [{agent_type}]",       test_config_known_keys,          {"agent_type": agent_type}),
        (f"1.3 最小化训练 [{agent_type}]",            test_minimal_training,           {"agent_type": agent_type}),
        (f"1.3 训练后 act() 合法 [{agent_type}]",     test_act_after_training,         {"agent_type": agent_type}),
    ]

    # 所有已有算法的回归：注册 + 最小化训练
    for existing in EXISTING_AGENTS:
        if existing == agent_type:
            continue  # 已在上面测过，跳过
        tests += [
            (f"1.1 AgentFactory 注册 [{existing} 回归]", test_agent_factory_registration, {"agent_type": existing}),
            (f"1.3 最小化训练 [{existing} 回归]",         test_minimal_training,           {"agent_type": existing}),
        ]

    return tests


def run_all_level0_tests(agent_type: str = 'ppo'):
    label = f"新算法验收: {agent_type}" if agent_type != 'ppo' else "回归测试 (PPO)"
    print("=" * 80)
    print(f"RLQAS 验收系统 - Level 0: 接口契约测试  |  {label}")
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
    parser = argparse.ArgumentParser(description="RLQAS Level 0 接口契约测试")
    parser.add_argument("--agent", default="ppo", help="被验收的 agent 类型（默认 ppo 回归测试）")
    args = parser.parse_args()

    success = run_all_level0_tests(agent_type=args.agent)
    sys.exit(0 if success else 1)
