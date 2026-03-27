#!/usr/bin/env python3
"""Inspect UCCSD object."""

import sys
sys.path.append("../001")
sys.path.append("../002")
from src.modules.molecule_processor import process_molecule
from src.modules.ucc_search.circuit_builder import UCCCircuitBuilder

mol = process_molecule('H2',0.74,'UCC')
builder = UCCCircuitBuilder(mol)
ucc = builder.ucc
print("UCCSD attributes:")
for attr in dir(ucc):
    if not attr.startswith('_'):
        val = getattr(ucc, attr)
        if not callable(val):
            print(f"  {attr}: {type(val)}")
        else:
            print(f"  {attr}(): callable")
# Look for hamiltonian related
print("\nChecking for hamiltonian, h_qubit, h_fermion, etc:")
for attr in dir(ucc):
    if 'ham' in attr.lower() or 'qubit' in attr.lower() or 'ferm' in attr.lower():
        val = getattr(ucc, attr)
        print(f"  {attr}: {type(val)}")
        if not callable(val):
            print(f"    {val}")
# Check if there is a property h_qubit_op
print("\nUCCSD ex_ops:", ucc.ex_ops)
print("UCCSD param_ids:", ucc.param_ids)
print("UCCSD n_params:", ucc.n_params)