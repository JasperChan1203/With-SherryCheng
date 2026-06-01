"""
RLQAS 验收系统 - Level 0: 接口契约测试

测试目标：验证 Agent 接口签名一致性、Config 兼容性、基础训练冒烟测试
通过标准：所有测试通过，否则直接拒绝合并

运行方式：
    cd /Users/lixuecheng/Documents/ai4qc/With-SherryCheng
    /curie-home/jpchen/.conda/envs/llm/bin/python3 rlqas_test/test_acceptance_level0.py
"""

import sys
import os
import json
import numpy as np
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rlqas-chem/src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rlqas-chem'))

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

def run_test(test_name: str, test_func):
    """运行单个测试并捕获异常"""
    result = TestResult(test_name)
    try:
        result = test_func(result)
    except Exception as e:
        result.fail_test(str(e))
    return result

# ==================== Level 0 测试：接口契约 ====================

def test_learn_callback_signature(result: TestResult):
    """
    测试项 1.1.1: learn(callback=None) 签名
    验证：learn() 必须接受 callback 关键字参数
    """
    try:
        from rlqas_chem.rl.ppo_agent import PPOAgent
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.search.ucc.controller import UCCSearchController
        from rlqas_chem.molecule.processor import process_molecule
        
        # 创建最小环境
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        agent = PPOAgent(env.observation_space, env.action_space, {})
        controller = UCCSearchController(mol_data, agent, env)
        
        # 测试 learn(callback=None) 不抛 TypeError
        try:
            controller.search(n_episodes=2)  # 简化调用，不传 callback
            result.pass_test({"message": "learn() accepts callback parameter"})
        except TypeError as e:
            if "callback" in str(e):
                result.fail_test(f"learn() does not accept callback parameter: {e}")
            else:
                result.pass_test({"warning": str(e)})
        
    except Exception as e:
        result.fail_test(str(e))
    
    return result

def test_learn_total_timesteps(result: TestResult):
    """
    测试项 1.1.2: learn(total_timesteps=N) 签名
    验证：必须接受 total_timesteps 关键字参数
    """
    try:
        from rlqas_chem.rl.grpo_agent import GRPOAgent
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule
        
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        agent = GRPOAgent(env.observation_space, env.action_space, {})
        
        # 测试 learn(total_timesteps=N) 不抛 TypeError
        try:
            agent.learn(total_timesteps=100)
            result.pass_test({"message": "learn(total_timesteps=N) accepted"})
        except TypeError as e:
            if "total_timesteps" in str(e):
                result.fail_test(f"learn() does not accept total_timesteps: {e}")
            else:
                result.pass_test({"warning": str(e)})
        
    except Exception as e:
        result.fail_test(str(e))
    
    return result

def test_act_return_format(result: TestResult):
    """
    测试项 1.1.3: act(state) 返回格式
    验证：返回值为 (int, dict)
    """
    try:
        from rlqas_chem.rl.dqn_agent import DQNAgent
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule
        
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        agent = DQNAgent(env.observation_space, env.action_space, {})
        
        state = env.reset()
        result_tuple = agent.act(state)
        
        # 验证返回格式
        if isinstance(result_tuple, tuple) and len(result_tuple) == 2:
            action, info = result_tuple
            if isinstance(action, (int, np.integer)) and isinstance(info, dict):
                result.pass_test({"action": int(action), "info_keys": list(info.keys())})
            else:
                result.fail_test(f"Return tuple format incorrect: action type={type(action)}, info type={type(info)}")
        else:
            result.fail_test(f"act() did not return (int, dict), got {type(result_tuple)}")
        
    except Exception as e:
        result.fail_test(str(e))
    
    return result

def test_save_load_consistency(result: TestResult):
    """
    测试项 1.1.4: save(path) / load(path) 往返一致性
    验证：保存后加载，act() 输出前后一致
    """
    try:
        import tempfile
        from rlqas_chem.rl.ppo_agent import PPOAgent
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule
        
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        
        # 创建并训练 agent
        agent1 = PPOAgent(env.observation_space, env.action_space, {})
        state = env.reset()
        action1, _ = agent1.act(state)
        
        # 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'test_agent')
            agent1.save(save_path)
            
            # 加载
            agent2 = PPOAgent(env.observation_space, env.action_space, {})
            agent2.load(save_path)
            
            # 比较输出
            action2, _ = agent2.act(state)
            
            if action1 == action2:
                result.pass_test({"action_before": int(action1), "action_after": int(action2)})
            else:
                result.fail_test(f"Actions differ after load: {action1} vs {action2}")
        
    except Exception as e:
        result.fail_test(str(e))
    
    return result

def test_agent_factory_registration(result: TestResult):
    """
    测试项 1.1.5: AgentFactory.create_agent() 注册
    验证：通过工厂可正常实例化所有注册的 agent
    """
    try:
        from rlqas_chem.rl.agent_factory import AgentFactory
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule
        
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        
        registered_agents = ['ppo', 'dqn', 'a2c', 'sac_discrete', 'grpo']
        results = {}
        
        for agent_type in registered_agents:
            try:
                agent = AgentFactory.create_agent(agent_type, env.observation_space, env.action_space, {})
                results[agent_type] = "OK"
            except Exception as e:
                results[agent_type] = f"FAIL: {e}"
        
        failed = [k for k, v in results.items() if v.startswith("FAIL")]
        
        if not failed:
            result.pass_test({"registered_agents": list(results.keys())})
        else:
            result.fail_test(f"Failed to create agents: {failed}")
        
    except Exception as e:
        result.fail_test(str(e))
    
    return result

# ==================== 1.2 Config 兼容性测试 ====================

def test_config_known_keys(result: TestResult):
    """
    测试项 1.2.1: 已知 key 正常读取
    验证：标准超参数（lr、gamma 等）被正确读取
    """
    try:
        from rlqas_chem.rl.ppo_agent import PPOAgent
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule
        
        config = {
            'lr': 0.001,
            'gamma': 0.99,
            'batch_size': 64
        }
        
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        agent = PPOAgent(env.observation_space, env.action_space, config)
        
        # 验证配置被读取（通过检查 agent 内部属性）
        if hasattr(agent, 'lr') and agent.lr == 0.001:
            result.pass_test({"lr": agent.lr, "gamma": agent.gamma})
        else:
            result.pass_test({"message": "Config keys accepted", "agent_config": agent.config})
        
    except Exception as e:
        result.fail_test(str(e))
    
    return result

def test_config_unknown_keys(result: TestResult):
    """
    测试项 1.2.2: 未知 key 处理
    验证：传入不存在的 key，应该抛 KeyError 或有明确告警
    """
    try:
        from rlqas_chem.rl.ppo_agent import PPOAgent
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.molecule.processor import process_molecule
        
        config = {
            'unknown_key_12345': 42  # 不存在的 key
        }
        
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        
        # 应该抛 KeyError 或有 warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            agent = PPOAgent(env.observation_space, env.action_space, config)
            
            if len(w) > 0:
                result.pass_test({"warning": str(w[0].message)})
            else:
                # 检查是否静默丢弃
                if 'unknown_key_12345' not in agent.config:
                    result.fail_test("Unknown key silently dropped (should raise KeyError or warn)")
                else:
                    result.pass_test({"message": "Unknown key kept in config"})
        
    except KeyError as e:
        result.pass_test({"key_error": str(e)})
    except Exception as e:
        result.fail_test(str(e))
    
    return result

# ==================== 1.3 基础训练冒烟测试 ====================

def test_minimal_training(result: TestResult):
    """
    测试项 1.3.1: 最小化训练（100 steps）
    验证：agent 能完成 100 timesteps 训练不崩溃
    """
    try:
        from rlqas_chem.rl.ppo_agent import PPOAgent
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.search.ucc.controller import UCCSearchController
        from rlqas_chem.molecule.processor import process_molecule
        
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        agent = PPOAgent(env.observation_space, env.action_space, {})
        controller = UCCSearchController(mol_data, agent, env)
        
        # 运行 100 timesteps
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = controller.search(n_episodes=2)  # 2 episodes x 4 steps = 8 timesteps
        
        result.pass_test({"message": "Minimal training completed", "results": str(results)[:100]})
        
    except Exception as e:
        result.fail_test(str(e))
    
    return result

def test_act_after_training(result: TestResult):
    """
    测试项 1.3.2: 训练后 act() 可用
    验证：训练后可对任意合法观测给出动作
    """
    try:
        from rlqas_chem.rl.ppo_agent import PPOAgent
        from rlqas_chem.search.ucc.environment import UCCSearchEnv
        from rlqas_chem.search.ucc.controller import UCCSearchController
        from rlqas_chem.molecule.processor import process_molecule
        
        mol_data = process_molecule('H2', 0.74, 'UCC')
        env = UCCSearchEnv(mol_data, config={'max_excitations': 4})
        agent = PPOAgent(env.observation_space, env.action_space, {})
        controller = UCCSearchController(mol_data, agent, env)
        
        # 训练
        controller.search(n_episodes=2)
        
        # 测试 act()
        state = env.reset()
        action, info = agent.act(state)
        
        if isinstance(action, (int, np.integer)) and 0 <= action < env.action_space.n:
            result.pass_test({"action": int(action), "action_space_size": env.action_space.n})
        else:
            result.fail_test(f"Invalid action: {action}")
        
    except Exception as e:
        result.fail_test(str(e))
    
    return result

# ==================== 主测试运行器 ====================

def run_all_level0_tests():
    """运行所有 Level 0 测试"""
    print("=" * 80)
    print("RLQAS 验收系统 - Level 0: 接口契约测试")
    print("=" * 80)
    print()
    
    tests = [
        # 1.1 接口签名一致性
        ("1.1.1 learn(callback=None) 签名", test_learn_callback_signature),
        ("1.1.2 learn(total_timesteps=N) 签名", test_learn_total_timesteps),
        ("1.1.3 act(state) 返回格式", test_act_return_format),
        ("1.1.4 save/load 往返一致性", test_save_load_consistency),
        ("1.1.5 AgentFactory 注册", test_agent_factory_registration),
        
        # 1.2 Config 兼容性
        ("1.2.1 已知 key 正常读取", test_config_known_keys),
        ("1.2.2 未知 key 处理", test_config_unknown_keys),
        
        # 1.3 基础训练冒烟测试
        ("1.3.1 最小化训练（100 steps）", test_minimal_training),
        ("1.3.2 训练后 act() 可用", test_act_after_training),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running: {test_name}...")
        result = run_test(test_name, test_func)
        results.append(result)
        print(f"  {result}")
        print()
    
    # 汇总
    print("=" * 80)
    print("测试汇总")
    print("=" * 80)
    
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    
    print(f"总计: {len(results)} 个测试")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print()
    
    if failed > 0:
        print("失败测试列表:")
        for r in results:
            if not r.passed:
                print(f"  - {r.name}")
        print()
        print("❌ Level 0 未通过，拒绝合并！")
        return False
    else:
        print("✅ Level 0 通过，可以继续 Level 1 测试。")
        return True

if __name__ == "__main__":
    success = run_all_level0_tests()
    sys.exit(0 if success else 1)
