"""
RLQAS Example 01: UCC Architecture Search on LiH
Molecule    : LiH, bond_length=1.6 Ang, active_space=(2,5) -> 10 qubits
Ansatz      : UCC (fermion excitation operators)
Agent       : PPO, 200 episodes (early stop at < 1.6 mHa)
Expected    : Chemical accuracy (< 1.6 mHa); may need more episodes on first run
"""
import rlqas

result = rlqas.search(
    "LiH", bond_length=1.6, ansatz_type="UCC",
    agent_type="ppo", n_episodes=200, active_space=(2, 5)
)
print(f"Best energy : {result['best_energy']:.6f} Ha")
print(f"FCI energy  : {result['fci_energy']:.6f} Ha")
print(f"Error       : {result['energy_error_mha']:.3f} mHa")
print(f"Operators   : {result['n_operators']}")
if result["chemical_accuracy"]:
    print("Chemical accuracy achieved")
else:
    print(f"Note: {result['energy_error_mha']:.3f} mHa (> 1.6 mHa threshold; increase n_episodes for reliable convergence)")
