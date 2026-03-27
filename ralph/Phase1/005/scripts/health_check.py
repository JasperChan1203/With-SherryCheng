#!/usr/bin/env python3
"""Health check script for Phase 1 modules."""

import sys
import os

def check_task_001():
    """Check Task 001 (Molecule Processing)."""
    try:
        sys.path.append('../001')
        from src.modules.molecule_processor import process_molecule, MoleculeData
        # Test with H2 (simple molecule)
        data = process_molecule('H2', 0.74, 'UCC')
        assert hasattr(data, 'fci_energy'), "MoleculeData missing fci_energy"
        assert data.n_qubits == 2, f"Expected 2 qubits for H2, got {data.n_qubits}"
        print("✓ Task 001: Molecule processing OK")
        return True
    except Exception as e:
        print(f"✗ Task 001 failed: {e}")
        return False

def check_task_002():
    """Check Task 002 (Quantum Simulator)."""
    try:
        sys.path.append('../002')
        from src.modules.quantum_simulator import SimulatorFactory
        simulator = SimulatorFactory.create_simulator(4)  # 4 qubits for LiH
        assert simulator is not None, "Simulator creation failed"
        print("✓ Task 002: Simulator creation OK")
        return True
    except Exception as e:
        print(f"✗ Task 002 failed: {e}")
        return False

def check_task_003():
    """Check Task 003 (PPO RL Agent)."""
    try:
        sys.path.append('../003')
        from src.modules.rl_agents import PPOAgent
        agent = PPOAgent(config={'seed': 42, 'use_gpu': False})
        assert agent is not None, "Agent creation failed"
        print("✓ Task 003: RL agent creation OK")
        return True
    except Exception as e:
        print(f"✗ Task 003 failed: {e}")
        return False

def check_task_004():
    """Check Task 004 (UCC Search Module)."""
    try:
        sys.path.append('../004')
        from src.modules.ucc_search.controller import UCCSearchController
        # Note: This may require mock dependencies for testing
        print("✓ Task 004: Module imports OK (may need mock dependencies for full test)")
        return True
    except Exception as e:
        print(f"✗ Task 004 failed: {e}")
        return False

def check_lih_processing():
    """Check LiH molecule processing with correct parameters."""
    try:
        sys.path.append('../001')
        from src.modules.molecule_processor import process_molecule
        # Process LiH with validation test parameters
        transform = 'parity'
        data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,2), basis_set='sto-3g', transform=transform)
        # Compute expected qubits: active_space (2,2) => 4 spin orbitals
        # Parity transformation reduces qubit count by 2 (particle number and spin symmetry)
        # Jordan-Wigner uses 4 qubits
        n_spin_orbitals = 2 * 2  # 2 orbitals * 2 spins
        if transform == 'parity':
            expected_qubits = n_spin_orbitals - 2  # parity reduction
        elif transform == 'jordan_wigner':
            expected_qubits = n_spin_orbitals
        elif transform == 'bravyi_kitaev':
            expected_qubits = n_spin_orbitals
        else:
            expected_qubits = n_spin_orbitals
        assert data.n_qubits == expected_qubits, f"Expected {expected_qubits} qubits for LiH (2,2) active space with {transform} transform, got {data.n_qubits}"
        assert hasattr(data, 'fci_energy') and data.fci_energy is not None, "FCI energy missing"
        print(f"✓ LiH processing OK: {data.n_qubits} qubits, FCI energy = {data.fci_energy:.6f} Hartree")
        return True
    except Exception as e:
        print(f"✗ LiH processing failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Phase 1 Module Health Check ===")
    results = [
        check_task_001(),
        check_task_002(),
        check_task_003(),
        check_task_004(),
        check_lih_processing()
    ]
    if all(results):
        print("\n✓ All Phase 1 modules pass basic health checks!")
        sys.exit(0)
    else:
        print("\n✗ Some modules failed health checks. Fix dependencies before proceeding.")
        sys.exit(1)