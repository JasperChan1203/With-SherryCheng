import qiskit
print(qiskit.__file__)
import qiskit.quantum_info as qi
print(qi.__file__)
print(dir(qi))
# Check if SparsePauliOp exists
if hasattr(qi, 'SparsePauliOp'):
    print("SparsePauliOp exists")
else:
    print("SparsePauliOp missing")
    # List all attributes
    for attr in dir(qi):
        if 'Pauli' in attr:
            print(attr)