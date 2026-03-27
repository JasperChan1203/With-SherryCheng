import sys
sys.path.insert(0, 'src')
import numpy as np
import tensorcircuit as tc
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder

data = process_molecule('LiH', 1.6, 'UCC', active_space=(2,3), transform='jordan_wigner')
builder = UCCCircuitBuilder(data)
circuit = builder.ucc.get_circuit()
print("circuit._nqubits =", circuit._nqubits)
print("has circuit.gates?", hasattr(circuit, 'gates'))
if hasattr(circuit, 'gates'):
    print("number of gates:", len(circuit.gates))
    for i, gate in enumerate(circuit.gates):
        print(f"  {i}: {gate}")
        if i > 5: break
print("has circuit._ops?", hasattr(circuit, '_ops'))
if hasattr(circuit, '_ops'):
    print("_ops length:", len(circuit._ops))
    for i, op in enumerate(circuit._ops[:5]):
        print(f"  {i}: {op}")
# Check inputs
print("circuit.inputs =", circuit.inputs)
# Try to set inputs
if circuit.inputs is None:
    # Try to set inputs to reference state
    psi = data.reference_state
    # reshape to tensor?
    print("psi shape", psi.shape)
    # circuit.inputs = psi maybe?
    # Try circuit.replace_inputs
    if hasattr(circuit, 'replace_inputs'):
        print("has replace_inputs")
        # need to know signature
        # circuit.replace_inputs(psi)
        pass
# Try to create new circuit with same gates but different initial state
# Use copy
if hasattr(circuit, 'copy'):
    c2 = circuit.copy()
    print("copied circuit")
    # maybe we can modify c2.inputs
    if hasattr(c2, 'inputs'):
        print("c2.inputs =", c2.inputs)
# Compute expectation with initial_state by manually applying gates to initial_state
# Get state vector of circuit starting from |0> (default)
state0 = circuit.state()
print("state0 shape", state0.shape)
print("max prob index", np.argmax(np.abs(state0)**2))
# Now we want state starting from reference state psi
# We can compute U|psi> where U is the circuit unitary.
# Since we don't have unitary, we can compute state = circuit.state() with initial_state?
# Let's see if there is a method to compute state with custom initial_state
if hasattr(circuit, 'wavefunction'):
    print("has wavefunction")
    # wavefunction(inputs) maybe
    try:
        wf = circuit.wavefunction(inputs=psi)
        print("wavefunction shape", wf.shape)
    except Exception as e:
        print("wavefunction error", e)
if hasattr(circuit, 'get_state_as_quvector'):
    print("has get_state_as_quvector")
    try:
        sv = circuit.get_state_as_quvector(initial_state=psi)
        print("sv shape", sv.shape)
    except Exception as e:
        print("error", e)