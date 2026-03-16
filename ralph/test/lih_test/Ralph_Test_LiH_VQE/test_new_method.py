#!/usr/bin/env python3
"""
Test new method using sort_mo for orbital selection.
"""

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, fci
from tencirchem import UCC, parity
from openfermion import QubitOperator

def test_new_method():
    d = 2.0
    mol = gto.M(
        atom=[["H", 0, 0, 0], ["Li", d, 0, 0]],
        basis='sto-3g',
        symmetry=True
    )
    hf = scf.RHF(mol)
    hf.kernel()
    print(f"HF energy: {hf.e_tot:.8f}")

    # Create CASCI object and use sort_mo
    cas = mcscf.CASCI(hf, 3, 2)  # 3 orbitals, 2 electrons
    mo = cas.sort_mo([2, 3, 6])  # 1-based indices [2,3,6] -> orbitals 1,2,5 (0-based)
    cas.kernel(mo)
    print(f"CASCI total energy: {cas.e_tot:.8f}")
    print(f"CASCI active energy: {cas.e_cas:.8f}")

    # Get integrals for active space
    int1e, e_core = cas.get_h1eff()
    int2e = cas.get_h2eff()
    int2e_full = ao2mo.restore(1, int2e, 3)
    print(f"Core energy: {e_core:.8f}")
    print(f"int1e shape: {int1e.shape}")
    print(f"int2e shape: {int2e.shape}")

    # Compute FCI to verify
    fci_solver = fci.direct_spin0.FCI()
    e_fci, _ = fci_solver.kernel(int1e, int2e_full, 3, 2, ecore=e_core)
    print(f"FCI energy: {e_fci:.8f}")
    print(f"Difference CASCI-FCI: {cas.e_tot - e_fci:.2e}")

    # Build Hamiltonian using Tencirchem
    ucc = UCC.from_integral(int1e, int2e, n_elec=2, e_core=e_core, hcb=False)
    fermion_op = ucc.h_fermion_op
    n_modes = 2 * int1e.shape[0]  # spin orbitals
    h_qubit_op = parity(fermion_op, n_modes=n_modes, n_elec=2)

    # Count qubits
    max_idx = 0
    for term in h_qubit_op.terms:
        for idx, _ in term:
            if idx > max_idx:
                max_idx = idx
    n_qubits = max_idx + 1
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of Pauli terms: {len(h_qubit_op.terms)}")

    # Check energy expectation with initial guess
    import tensorcircuit as tc
    def manual_circuit(n_qubits, params):
        c = tc.Circuit(n_qubits)
        param_idx = 0
        for i in range(n_qubits):
            c.ry(i, theta=params[param_idx])
            param_idx += 1
        for i in range(n_qubits - 1):
            c.cnot(i, i+1)
            param_idx += 1
        for i in range(n_qubits):
            c.rz(i, theta=params[param_idx])
            param_idx += 1
        for i in range(n_qubits):
            c.ry(i, theta=params[param_idx])
            param_idx += 1
        for i in range(n_qubits - 2, -1, -1):
            c.cnot(i, i+1)
            param_idx += 1
        return c

    def energy_function(params, h_qubit_op, n_qubits):
        c = manual_circuit(n_qubits, params)
        energy = 0.0
        for term, coeff in h_qubit_op.terms.items():
            x_list = []
            y_list = []
            z_list = []
            for idx, pauli in term:
                if pauli == 'X':
                    x_list.append(idx)
                elif pauli == 'Y':
                    y_list.append(idx)
                elif pauli == 'Z':
                    z_list.append(idx)
                else:
                    raise ValueError(f"Unknown Pauli {pauli}")
            exp_val = c.expectation_ps(x=x_list, y=y_list, z=z_list)
            energy += coeff * exp_val
        return energy.real

    n_params = n_qubits * 3 + (n_qubits - 1) * 2
    init_params = np.random.uniform(-np.pi, np.pi, size=n_params)
    init_energy = energy_function(init_params, h_qubit_op, n_qubits)
    print(f"Initial VQE energy: {init_energy:.8f}")
    print(f"Target FCI energy: {e_fci:.8f}")
    print(f"Difference: {init_energy - e_fci:.8f}")

    return cas.e_tot, e_fci, n_qubits

if __name__ == "__main__":
    casci, fci_energy, nq = test_new_method()
    print(f"\nSummary:")
    print(f"CASCI: {casci:.8f}, FCI: {fci_energy:.8f}, Qubits: {nq}")
    print(f"Benchmark target: -7.860153")
    print(f"Diff from benchmark: {casci + 7.860153:.8f}")