"""
RLQAS Example 02: HEA Architecture Search on BeH2
Molecule    : BeH2, bond_length=1.3 Å, active_space=(4,4) -> 8 qubits
Ansatz      : HEA (Hardware Efficient Ansatz)
Agent       : DQN, 200 episodes
Expected    : Prints energy result
"""
import rlqas

result = rlqas.search(
    "BeH2", bond_length=1.3, ansatz_type="HEA",
    agent_type="dqn", n_episodes=200, active_space=(4, 4)
)
print(f"Best energy : {result['best_energy']:.6f} Ha")
print(f"FCI energy  : {result['fci_energy']:.6f} Ha")
print(f"Error       : {result['energy_error_mha']:.3f} mHa")
print(f"Qubits      : {result['n_qubits']}")
acc = "Chemical accuracy achieved" if result["chemical_accuracy"] else "Chemical accuracy NOT achieved"
print(acc)
