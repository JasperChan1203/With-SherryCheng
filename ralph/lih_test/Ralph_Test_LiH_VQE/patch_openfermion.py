#!/usr/bin/env python3
"""
Monkey-patch QubitOperator to accept numpy numeric types.
"""
import numpy as np
from openfermion.ops.operators.qubit_operator import QubitOperator as OriginalQubitOperator

# Store original __init__
original_init = OriginalQubitOperator.__init__

def patched_init(self, term=None, coefficient=1.0):
    # Convert numpy numeric types to Python built-in types
    if isinstance(coefficient, np.integer):
        coefficient = int(coefficient)
    elif isinstance(coefficient, np.floating):
        coefficient = float(coefficient)
    elif isinstance(coefficient, np.complexfloating):
        coefficient = complex(coefficient)
    # Call original
    original_init(self, term, coefficient)

# Apply patch
OriginalQubitOperator.__init__ = patched_init

print("Patched QubitOperator to accept numpy types")