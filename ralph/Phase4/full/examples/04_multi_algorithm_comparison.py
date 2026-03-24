"""
RLQAS Example 04: Multi-Algorithm Comparison on LiH
Molecule    : LiH, bond_length=1.6 Ang, active_space=(2,5) -> 10 qubits
Ansatz      : HYBRID (supports PPO, DQN, A2C)
Agents      : PPO, DQN, A2C — 100 episodes each
Expected    : Comparison table of energy errors
Note        : UCC ansatz only supports PPO; use HYBRID or HEA for multi-agent comparison
"""
import rlqas

common = dict(molecule="LiH", bond_length=1.6, ansatz_type="HYBRID",
              active_space=(2, 5), n_episodes=100)
results = {agent: rlqas.search(agent_type=agent, **common)
           for agent in ["ppo", "dqn", "a2c"]}

print(f"\n{'Agent':<14} {'Error (mHa)':>12} {'Accurate':>10}")
print("-" * 40)
for agent, r in results.items():
    acc = "Yes" if r["chemical_accuracy"] else "No"
    print(f"{agent:<14} {r['energy_error_mha']:>12.3f} {acc:>10}")
