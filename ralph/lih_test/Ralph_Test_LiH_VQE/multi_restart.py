#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
import patch_openfermion
import numpy as np
from pyscf import gto, scf, mcscf, ao2mo
import tencirchem
from tencirchem import UCC, parity
import tensorcircuit as tc
from scipy.optimize import minimize
import time

bond_length = 2.0
mol = gto.M(
    atom=[["H", 0, 0, 0], ["Li", bond_length, 0, 0]],
    basis='sto-3g',
    symmetry=True,
    verbose=0
)
hf = scf.RHF(mol)
hf.kernel()
cas = mcscf.CASCI(hf, 3, 2)
mo_coeff = cas.sort_mo([2, 3, 6])
cas.kernel(mo_coeff)
int1e, e_core = cas.get_h1eff()
int2e = cas.get_h2eff()
n_orb = int1e.shape[0]
int2e = ao2mo.restore(1, int2e, n_orb)
ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core, hcb=False)
fermion_op = ucc.h_fermion_op
n_modes = 2 * int1e.shape[0]
h_qubit_op = parity(fermion_op, n_modes=n_modes, n_elec=2)
n_qubits = 4

# Define circuit (same as generate_lih_vqe_new.py manual_circuit)
def manual_circuit(params):
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
    for i in range(n_qubits - 2, -1, -1):
        c.cnot(i, i+1)
        param_idx += 1
    c.rz(n_qubits - 1, theta=params[param_idx])
    param_idx += 1
    for i in range(n_qubits - 1):
        c.cnot(i, i+1)
        param_idx += 1
    assert param_idx == len(params)
    return c

def energy_function(params):
    c = manual_circuit(params)
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
        exp_val = c.expectation_ps(x=x_list, y=y_list, z=z_list)
        energy += coeff * exp_val
    return energy.real

n_params = 18
n_restarts = 5
best_energy = float('inf')
best_params = None
for r in range(n_restarts):
    print(f"\n--- Restart {r+1}/{n_restarts} ---")
    init_params = np.random.uniform(-np.pi, np.pi, size=n_params)
    # Optionally set central Rz parameter to something non-zero to break symmetry
    init_params[4+3+4+3] = np.random.uniform(-1, 1)  # index of central Rz
    energy_curve = []
    def callback(xk):
        energy = energy_function(xk)
        energy_curve.append(energy)
        print(f"Iter {len(energy_curve)}: energy = {energy:.8f}")
    start = time.time()
    result = minimize(
        fun=energy_function,
        x0=init_params,
        method='BFGS',
        callback=callback,
        options={'disp': False, 'maxiter': 100, 'gtol': 1e-6}
    )
    elapsed = time.time() - start
    print(f"Final energy: {result.fun:.8f}, success: {result.success}, iterations: {len(energy_curve)}")
    if result.fun < best_energy:
        best_energy = result.fun
        best_params = result.x
        print(f"New best energy!")
print(f"\nBest energy across restarts: {best_energy:.8f}")
print(f"HF energy: {hf.e_tot:.8f}")
print(f"FCI energy: {cas.e_tot:.8f}")
print(f"Difference from HF: {(best_energy - hf.e_tot)*1000:.3f} mHa")
print(f"Difference from FCI: {(best_energy - cas.e_tot)*1000:.3f} mHa")