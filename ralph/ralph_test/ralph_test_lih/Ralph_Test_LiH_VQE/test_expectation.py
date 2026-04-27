#!/usr/bin/env python3
import numpy as np
import tensorcircuit as tc
from openfermion import QubitOperator

# Create a simple circuit
c = tc.Circuit(2)
c.ry(0, theta=0.5)
c.cnot(0,1)
c.rz(1, theta=0.3)

# Test expectation of single Pauli Z
print("Expectation Z0:", c.expectation_ps(z=[0]))
print("Expectation Z1:", c.expectation_ps(z=[1]))
print("Expectation X0:", c.expectation_ps(x=[0]))
print("Expectation Y0:", c.expectation_ps(y=[0]))

# Expectation of product Z0 * Z1
print("Expectation Z0*Z1:", c.expectation_ps(z=[0,1]))
# Expectation of X0 * Y1
print("Expectation X0*Y1:", c.expectation_ps(x=[0], y=[1]))

# Now test with QubitOperator
op = QubitOperator('Z0', 1.0) + QubitOperator('X0 Y1', 2.0)
print("\nQubitOperator terms:", op.terms)

# Compute expectation manually
energy = 0.0
for term, coeff in op.terms.items():
    # term is tuple of (index, pauli)
    # Build lists for each Pauli type
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
    # Compute expectation
    exp = c.expectation_ps(x=x_list, y=y_list, z=z_list)
    print(f"Term {term}, coeff {coeff}, exp {exp}")
    energy += coeff * exp
print("Total energy:", energy)

# Check if tensorcircuit has a built-in expectation for QubitOperator
if hasattr(tc, 'expectation'):
    print("tc.expectation exists")
if hasattr(c, 'expectation'):
    print("c.expectation exists")