#!/usr/bin/env python3
"""
Test parity transformation processing for LiH (2,3) active space.
Expected: 4 qubits with parity transformation.
"""

import sys
sys.path.insert(0, 'src')

from rlqas.phase1.molecule import process_molecule

def test_parity_lih():
    print("Testing LiH with active_space=(2,3) and parity transformation...")
    try:
        data = process_molecule("LiH", 1.6, "UCC", active_space=(2,3), transform="parity")
        print(f"  Number of qubits: {data.n_qubits}")
        print(f"  Expected: 4 qubits (2*3 - 2 = 4)")
        print(f"  Hamiltonian terms: {len(data.hamiltonian.terms)}")
        print(f"  Reference state shape: {data.reference_state.shape}")

        # Check qubit count
        if data.n_qubits == 4:
            print("  ✓ Parity transformation yields correct qubit count (4)")
            return True
        else:
            print(f"  ✗ Unexpected qubit count: {data.n_qubits}")
            return False
    except Exception as e:
        print(f"  ✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parity_lih()
    sys.exit(0 if success else 1)