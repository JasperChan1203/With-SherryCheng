#!/usr/bin/env python3
import numpy as np
from pyscf import gto, scf, mcscf

bond_length = 2.0
mol = gto.M(
    atom=f'Li 0 0 0; H 0 0 {bond_length}',
    basis='sto-3g',
    symmetry=False,
    verbose=0
)
hf = scf.RHF(mol)
hf.kernel()
print(f"HF energy: {hf.e_tot:.8f}")
mo_energy = hf.mo_energy
print("Orbital energies:")
for i, e in enumerate(mo_energy):
    print(f"  {i}: {e:.6f}")

selections = {
    "HOMO-LUMO-LUMO+1": [1,2,3],  # 0-indexed
    "Benchmark [0,1,4]": [0,1,4],
    "All low energy": [0,1,2],
    "High energy virtual": [0,1,5],
}

for name, active in selections.items():
    print(f"\n--- {name} active orbitals {active} ---")
    # Compute CASCI with proper freezing
    core_orbitals = [i for i in range(mo_energy.size) if i not in active and i < 2]  # occupied not active
    virtual_frozen = [i for i in range(mo_energy.size) if i not in active and i >= 2]
    print(f"  Core orbitals: {core_orbitals}")
    print(f"  Virtual frozen: {virtual_frozen}")
    cas = mcscf.CASCI(hf, len(active), 2)
    cas.ncore = len(core_orbitals)
    cas.frozen = virtual_frozen
    cas.mo_coeff = hf.mo_coeff
    cas.kernel()
    print(f"  CASCI total energy: {cas.e_tot:.8f}")
    # Get integrals
    int1e, e_core = cas.get_h1eff()
    int2e = cas.get_h2eff()
    from pyscf import ao2mo
    int2e_full = ao2mo.restore(1, int2e, len(active))
    from pyscf import fci
    fci_solver = fci.direct_spin0.FCI()
    e_fci, _ = fci_solver.kernel(int1e, int2e_full, len(active), 2, ecore=e_core)
    print(f"  FCI total energy: {e_fci:.8f}")
    print(f"  Difference from benchmark (-7.860153): {e_fci - (-7.860153):.8f} Ha")

# Also compute full FCI within active space of all orbitals? Not possible.
# Let's compute MP2 natural orbital occupation to see which orbitals are important.
print("\n--- MP2 natural orbital occupation ---")
from pyscf import mp
mp2 = mp.MP2(hf).run()
print("MP2 natural orbital occupation numbers (approx):")
for i, occ in enumerate(mp2.mo_occ):
    if occ > 0.01:
        print(f"  Orbital {i}: {occ:.4f}")