"""Example 01: LiH UCC PPO with active_space=(2,5), 300 episodes."""
import rlqas_chem

print("Running LiH UCC PPO benchmark (300 episodes, active_space=(2,5))...")
result = rlqas_chem.search(
    'LiH', 1.6,
    ansatz_type='UCC',
    agent_type='ppo',
    n_episodes=300,
    active_space=(2, 5),
)

print(f"\nResults:")
print(f"  Best energy:      {result['best_energy']:.6f} Ha")
print(f"  FCI energy:       {result['fci_energy']:.6f} Ha")
print(f"  Error:            {result['energy_error_mha']:.3f} mHa")
print(f"  Chemical accuracy: {'YES' if result['chemical_accuracy'] else 'NO'}")
print(f"  Qubits:           {result['n_qubits']}")

assert result['energy_error_mha'] < 1.6, (
    f"LiH failed chemical accuracy: {result['energy_error_mha']:.3f} mHa >= 1.6 mHa"
)
print("\nBenchmark PASSED: error < 1.6 mHa")
