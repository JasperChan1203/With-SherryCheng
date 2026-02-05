#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from generate_lih_vqe import define_molecule, select_active_orbitals, get_active_integrals
from tencirchem import UCC

mol, hf = define_molecule()
active_orbitals, _, _ = select_active_orbitals(hf)
int1e, int2e, e_core = get_active_integrals(hf, active_orbitals)

print("Testing engine parameter:")
for engine in [None, 'tensornetwork', 'cirq', 'qulacs']:
    try:
        ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core, engine=engine)
        print(f"  engine={engine}: n_qubits={ucc.n_qubits}")
    except Exception as e:
        print(f"  engine={engine}: error {e}")

# Also try to pass hcb=False and engine='parity' maybe
try:
    ucc = UCC.from_integral(int1e, int2e, 2, e_core=e_core, hcb=False, engine='parity')
    print(f"  engine='parity': n_qubits={ucc.n_qubits}")
except Exception as e:
    print(f"  engine='parity': error {e}")

# Check if there is a mapping attribute
ucc_default = UCC.from_integral(int1e, int2e, 2, e_core=e_core)
print(f"\nDefault mapping attribute: {getattr(ucc_default, 'mapping', 'not found')}")
print(f"hcb attribute: {ucc_default.hcb}")