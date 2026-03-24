"""
RLQAS Example 03: Hybrid Architecture Search on BeH2
Molecule    : BeH2, bond_length=1.3 Å, active_space=(4,4) -> 8 qubits
Ansatz      : HYBRID (UCC + HEA fusion)
Agent       : PPO, 200 episodes
Expected    : Prints fusion_template and energy result
"""
import rlqas

result = rlqas.search(
    "BeH2", bond_length=1.3, ansatz_type="HYBRID",
    agent_type="ppo", n_episodes=200, active_space=(4, 4)
)
print(f"Best energy     : {result['best_energy']:.6f} Ha")
print(f"FCI energy      : {result['fci_energy']:.6f} Ha")
print(f"Error           : {result['energy_error_mha']:.3f} mHa")
print(f"Qubits          : {result['n_qubits']}")
print(f"Fusion template : {result['fusion_template']}")
acc = "Chemical accuracy achieved" if result["chemical_accuracy"] else "Chemical accuracy NOT achieved"
print(acc)
