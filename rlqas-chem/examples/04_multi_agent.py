"""Example 04: H2 UCC multi-agent comparison (PPO vs DQN)."""
import rlqas_chem

print("H2 UCC multi-agent comparison...")

for agent in ['ppo', 'dqn']:
    result = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type=agent, n_episodes=50)
    print(f"  [{agent.upper():3s}] energy={result['best_energy']:.4f} Ha, "
          f"error={result['energy_error_mha']:.3f} mHa, "
          f"chem_acc={'YES' if result['chemical_accuracy'] else 'NO'}")

print("Done.")
