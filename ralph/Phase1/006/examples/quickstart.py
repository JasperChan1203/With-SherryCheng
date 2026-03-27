#!/usr/bin/env python3
"""
Quickstart example for RLQAS Phase 1 integrated package.

Demonstrates the complete workflow from molecule processing to UCC search.
"""

import sys
sys.path.insert(0, '../src')

import rlqas.phase1 as rlqas
from rlqas.phase1.molecule import process_molecule
from rlqas.phase1.search import UCCSearchController


def main():
    print("RLQAS Phase 1 Quickstart Example")
    print("=================================")

    # 1. Process H2 molecule
    print("\n1. Processing H2 molecule...")
    molecule_data = process_molecule(
        molecule="H2",
        bond_length=0.74,
        ansatz_type="UCC",
        active_space=(2, 2),  # Minimal active space for demonstration
        basis_set="sto-3g",
        transform="jordan_wigner"
    )
    print(f"   • Number of qubits: {molecule_data.n_qubits}")
    print(f"   • FCI energy: {molecule_data.fci_energy:.6f} Hartree")
    print(f"   • Hamiltonian terms: {len(molecule_data.hamiltonian.terms)}")

    # 2. Create UCC search controller
    print("\n2. Creating UCC search controller...")
    controller = UCCSearchController(
        molecule_data,
        agent_type='ppo',
        config={
            'controller': {
                'n_episodes': 10,  # Small number for demonstration
                'early_stop_threshold': 0.01,  # Loose threshold
                'use_gpu': False,
                'verbose': 0
            }
        }
    )
    print("   • Controller created with PPO agent")

    # 3. Run search
    print("\n3. Running UCC search (10 episodes)...")
    results = controller.search(n_episodes=10, early_stop_threshold=0.01)

    # 4. Display results
    print("\n4. Search Results:")
    print(f"   • Best energy: {results['best_energy']:.6f} Hartree")
    print(f"   • FCI energy: {molecule_data.fci_energy:.6f} Hartree")
    error_hartree = results['best_energy'] - molecule_data.fci_energy
    error_mha = error_hartree * 1000
    print(f"   • Error: {error_mha:.2f} mHa")
    print(f"   • Episodes completed: {len(results['episode_energies'])}")
    print(f"   • Circuit depth: {results.get('episode_depths', [])[-1] if results.get('episode_depths') else 'N/A'}")
    print(f"   • Excitations selected: {len(results.get('best_excitations', []))}")

    # 5. Check chemical accuracy
    chemical_accuracy = abs(error_mha) < 1.6
    if chemical_accuracy:
        print(f"\n✓ Chemical accuracy achieved ({error_mha:.2f} mHa < 1.6 mHa)")
    else:
        print(f"\n⚠ Chemical accuracy not achieved ({error_mha:.2f} mHa >= 1.6 mHa)")

    print("\nQuickstart example completed successfully!")


if __name__ == "__main__":
    main()