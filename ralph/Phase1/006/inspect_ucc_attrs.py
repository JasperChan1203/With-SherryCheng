#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from rlqas.phase1.molecule.processor import process_molecule

data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='parity')
ucc = data.ucc_object
print("UCC attributes:")
for attr in ['init_state', 'hf', 'n_qubits', 'h_qubit_op', 'h_fermion_op']:
    if hasattr(ucc, attr):
        val = getattr(ucc, attr)
        print(f"  {attr}: {type(val)}")
        if attr == 'init_state':
            print(f"    shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
            print(f"    first few: {val[:4] if len(val) > 4 else val}")
        elif attr == 'hf':
            print(f"    hf energy: {val.e_tot if hasattr(val, 'e_tot') else val}")
print("Checking parity mapping...")
# Try to see if there is a parity mapping function
import tencirchem
print(f"tencirchem.parity: {tencirchem.parity}")