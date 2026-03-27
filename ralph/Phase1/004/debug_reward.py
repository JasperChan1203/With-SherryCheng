import sys
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")
from src.modules.ucc_search.reward_function import UCCRewardFunction

config = {
    "reward_function": {
        "baseline_type": "current_best",
        "shaping_rewards": True,
    }
}
rf = UCCRewardFunction(config)
print("baseline_type:", rf.baseline_type)
print("use_shaping:", rf.use_shaping)
print("energy_weight:", rf.energy_weight)
print("complexity_penalty:", rf.complexity_penalty)
print("first evaluation")
reward1 = rf.compute_reward(-1.0, 1)
print("reward1:", reward1)
print("best_energy:", rf.best_energy)
print("last_energy:", rf.last_energy)
print("consecutive_improvements:", rf.consecutive_improvements)
print("second evaluation")
reward2 = rf.compute_reward(-1.01, 1)
print("reward2:", reward2)
print("best_energy:", rf.best_energy)
print("last_energy:", rf.last_energy)
print("consecutive_improvements:", rf.consecutive_improvements)
print("shaping reward computed:", rf._compute_shaping_reward(-1.01))
# Let's manually compute
baseline = rf.best_energy
energy_improvement = baseline - (-1.01)
weighted = energy_improvement * rf.energy_weight
penalty = rf.complexity_penalty * 1
shaping = rf._compute_shaping_reward(-1.01)
print("baseline:", baseline)
print("energy_improvement:", energy_improvement)
print("weighted:", weighted)
print("penalty:", penalty)
print("shaping:", shaping)
print("total:", weighted - penalty + shaping)