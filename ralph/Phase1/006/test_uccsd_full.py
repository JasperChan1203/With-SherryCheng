#!/usr/bin/env python3
"""Test full UCCSD energy for LiH (2,3) with Jordan-Wigner."""

import sys
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')

from rlqas.phase1.molecule.processor import process_molecule
import tencirchem
from tencirchem import UCCSD
import numpy as np

print("Processing LiH with active_space=(2,3), transform='jordan_wigner'")
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
print(f"n_qubits: {data.n_qubits}")
print(f"FCI energy: {data.fci_energy}")
print(f"HF energy: {data.molecular_info.get('hf_energy')}")

# Get UCCSD object from molecule data
ucc = data.ucc_sd_object
print(f"\nUCCSD object type: {type(ucc)}")
print(f"UCCSD n_params: {ucc.n_params}")
print(f"UCCSD ex_ops: {ucc.ex_ops}")
print(f"UCCSD param_ids: {ucc.param_ids}")

# Get MP2 initial guess
if hasattr(ucc, 'init_guess'):
    init_params = ucc.init_guess
    print(f"MP2 initial parameters: {init_params}")
else:
    init_params = np.zeros(ucc.n_params)
    print("No MP2 init_guess, using zeros")

# Compute energy with initial parameters
energy_init = ucc.energy(init_params)
print(f"Energy with MP2/zero parameters: {energy_init}")
print(f"Difference from HF: {energy_init - data.molecular_info.get('hf_energy')}")
print(f"Difference from FCI: {energy_init - data.fci_energy}")

# Run kernel to optimize parameters (full optimization)
print("\nRunning UCCSD kernel optimization...")
try:
    ucc.kernel()
    optimized_params = ucc.params
    optimized_energy = ucc.energy()
    print(f"Optimized energy: {optimized_energy}")
    print(f"Difference from FCI: {optimized_energy - data.fci_energy}")
    print(f"Error in mHa: {(optimized_energy - data.fci_energy) * 1000}")
except Exception as e:
    print(f"Kernel failed: {e}")
    # Try simple gradient descent using scipy
    from scipy.optimize import minimize
    def loss(params):
        return ucc.energy(params)
    result = minimize(loss, init_params, method='L-BFGS-B', options={'maxiter': 100})
    print(f"Optimization result: {result.success}")
    print(f"Optimized energy: {result.fun}")
    print(f"Difference from FCI: {result.fun - data.fci_energy}")

# Evaluate energy with random parameters (like environment)
print("\nTesting random parameters (uniform -0.1, 0.1)")
for i in range(5):
    rand_params = np.random.uniform(-0.1, 0.1, size=ucc.n_params)
    e = ucc.energy(rand_params)
    print(f"Random set {i}: energy {e:.6f}, diff FCI {(e - data.fci_energy)*1000:.2f} mHa")

# What about subset of excitations? Let's pick first 3 excitations
print("\nTesting subset of excitations (first 3)")
subset = ucc.ex_ops[:3]
print(f"Subset excitations: {subset}")
# Need to map to parameters: each excitation maps to parameter id
param_indices = set()
for ex in subset:
    idx = ucc.ex_ops.index(ex)
    param_idx = ucc.param_ids[idx]
    param_indices.add(param_idx)
print(f"Active parameter indices: {param_indices}")
# Set other parameters to zero
params_subset = np.zeros(ucc.n_params)
for idx in param_indices:
    params_subset[idx] = init_params[idx] if idx < len(init_params) else 0.0
energy_subset = ucc.energy(params_subset)
print(f"Energy with subset (MP2 init): {energy_subset}")
print(f"Diff FCI: {(energy_subset - data.fci_energy)*1000:.2f} mHa")