#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)

ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
print("UCC attributes:")
for attr in dir(ucc):
    if not attr.startswith('_'):
        try:
            val = getattr(ucc, attr)
            if callable(val):
                continue
            print(f"  {attr}: {type(val).__name__}")
        except:
            pass

# Look for fermion operator
if hasattr(ucc, 'h_fermion_op'):
    print(f"\nh_fermion_op: {ucc.h_fermion_op}")
if hasattr(ucc, 'fermion_op'):
    print(f"fermion_op: {ucc.fermion_op}")
if hasattr(ucc, 'h'):
    print(f"h shape: {ucc.h.shape}")
if hasattr(ucc, 'int1e'):
    print(f"int1e shape: {ucc.int1e.shape}")
if hasattr(ucc, 'int2e'):
    print(f"int2e shape: {ucc.int2e.shape}")
if hasattr(ucc, 'n_spatial'):
    print(f"n_spatial: {ucc.n_spatial}")
if hasattr(ucc, 'n_spin'):
    print(f"n_spin: {ucc.n_spin}")

# Check if there is a method to get fermion operator
print("\nMethods containing fermion:")
for attr in dir(ucc):
    if 'fermion' in attr.lower():
        print(f"  {attr}")

# Get fermion operator via maybe get_fermion_operator
if hasattr(ucc, 'get_fermion_operator'):
    ferm_op = ucc.get_fermion_operator()
    print(f"Fermion operator: {ferm_op}")