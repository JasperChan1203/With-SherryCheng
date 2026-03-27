#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from rlqas.phase1.molecule.processor import process_molecule

data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='parity')
ucc = data.ucc_object
print(f"UCC type: {type(ucc)}")
print(f"has ex_ops: {hasattr(ucc, 'ex_ops')}")
print(f"has get_ex_ops: {hasattr(ucc, 'get_ex_ops')}")
if hasattr(ucc, 'get_ex_ops'):
    ex_ops = ucc.get_ex_ops()
    print(f"ex_ops length: {len(ex_ops)}")
    print(f"ex_ops sample: {ex_ops[:3]}")
print(f"has param_ids: {hasattr(ucc, 'param_ids')}")
try:
    param_ids = ucc.param_ids
    print(f"param_ids: {param_ids}")
except Exception as e:
    print(f"param_ids error: {e}")
print(f"has get_circuit: {hasattr(ucc, 'get_circuit')}")
if hasattr(ucc, 'get_circuit'):
    circuit = ucc.get_circuit()
    print(f"circuit type: {type(circuit)}")
print(f"n_qubits: {getattr(ucc, 'n_qubits', 'N/A')}")
print(f"h_qubit_op type: {type(ucc.h_qubit_op) if hasattr(ucc, 'h_qubit_op') else 'N/A'}")