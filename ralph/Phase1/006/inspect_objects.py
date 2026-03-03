#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from rlqas.phase1.molecule.processor import process_molecule

# LiH with parity
data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='parity')
print(f"n_qubits: {data.n_qubits}")
print(f"transform: {data.molecular_info['transform']}")
print(f"ucc_object type: {type(data.ucc_object)}")
print(f"ucc_sd_object type: {type(data.ucc_sd_object)}")

ucc = data.ucc_object
print("\nUCC object attributes:")
for attr in dir(ucc):
    if not attr.startswith('_'):
        print(f"  {attr}")

# Check if has ex_ops, param_ids, get_circuit
print("\nChecking important attributes:")
print(f"has ex_ops: {hasattr(ucc, 'ex_ops')}")
if hasattr(ucc, 'ex_ops'):
    print(f"ex_ops: {ucc.ex_ops}")
print(f"has param_ids: {hasattr(ucc, 'param_ids')}")
print(f"has get_circuit: {hasattr(ucc, 'get_circuit')}")
print(f"has n_qubits: {hasattr(ucc, 'n_qubits')}")
if hasattr(ucc, 'n_qubits'):
    print(f"n_qubits: {ucc.n_qubits}")
print(f"has h_fermion_op: {hasattr(ucc, 'h_fermion_op')}")
print(f"has h_qubit_op: {hasattr(ucc, 'h_qubit_op')}")

# Check UCCSD object
print("\nUCCSD object attributes (sample):")
uccsd = data.ucc_sd_object
print(f"has ex_ops: {hasattr(uccsd, 'ex_ops')}")
print(f"has param_ids: {hasattr(uccsd, 'param_ids')}")
print(f"has get_circuit: {hasattr(uccsd, 'get_circuit')}")
print(f"n_qubits: {getattr(uccsd, 'n_qubits', 'N/A')}")