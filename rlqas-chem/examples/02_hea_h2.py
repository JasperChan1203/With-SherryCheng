"""Example 02: H2 HEA PPO, 100 episodes."""
import rlqas_chem

print("Running H2 HEA PPO (100 episodes)...")
result = rlqas_chem.search('H2', 0.74, ansatz_type='HEA', agent_type='ppo', n_episodes=100)

print(f"\nResults:")
print(f"  Best energy:      {result['best_energy']:.6f} Ha")
print(f"  FCI energy:       {result['fci_energy']:.6f} Ha")
print(f"  Error:            {result['energy_error_mha']:.3f} mHa")
print(f"  Chemical accuracy: {'YES' if result['chemical_accuracy'] else 'NO'}")
