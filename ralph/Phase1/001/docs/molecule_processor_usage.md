# Molecule Processing Module Usage Guide

## Overview

The molecule processing module converts molecular information into quantum computation inputs using Tencirchem-ng 2024.10 and OpenFermion. It provides a single function `process_molecule()` that returns a `MoleculeData` dataclass containing the qubit Hamiltonian, number of qubits, reference state (Hartree-Fock), exact FCI energy, and molecular metadata.

## Installation

Ensure dependencies are installed:

```bash
pip install tencirchem-ng>=2024.10 openfermion>=1.5 pyscf numpy scipy
```

## Basic Usage

```python
from src.modules.molecule_processor import process_molecule

# Process H2 molecule at 0.74 Å bond length
result = process_molecule(
    molecule="H2",
    bond_length=0.74,
    ansatz_type="UCC",
    active_space=None,
    basis_set="sto-3g",
    transform="parity"
)

print(f"Number of qubits: {result.n_qubits}")
print(f"FCI energy: {result.fci_energy}")
print(f"Reference state shape: {result.reference_state.shape}")
print(f"Number of Hamiltonian terms: {len(result.hamiltonian.terms)}")
```

## Supported Molecules

Currently supported molecules:
- **H₂**: Hydrogen molecule (linear)
- **LiH**: Lithium hydride (linear)
- **BeH₂**: Beryllium hydride (linear, H-Be-H geometry)

Bond length is the distance between the two atoms for diatomic molecules, or the metal‑hydrogen distance for BeH₂ (H‑Be‑H linear with Be at origin).

## Active Space Support

The `active_space` parameter allows selecting a subset of electrons and orbitals for the quantum computation. Provide a tuple `(n_electrons_active, n_orbitals_active)`. If `None`, the full space is used.

Example: LiH with active space (2,2) (2 electrons in 2 orbitals):

```python
result = process_molecule(
    molecule="LiH",
    bond_length=1.6,
    ansatz_type="UCC",
    active_space=(2, 2),
    basis_set="sto-3g",
    transform="parity"
)
```

The module automatically selects the highest‑energy (valence) orbitals for the active space. For production calculations, manual orbital selection may be needed.

## Fermion-to-Qubit Transformations

Three transformations are supported:

1. **`parity`** (default): Uses Tencirchem's parity mapping with particle‑number conservation, reducing qubit count.
2. **`jordan_wigner`**: Standard Jordan‑Wigner mapping (one qubit per spin orbital).
3. **`bravyi_kitaev`**: Bravyi‑Kitaev mapping (same qubit count as Jordan‑Wigner but with fewer Pauli terms).

Example with Jordan‑Wigner:

```python
result = process_molecule(
    molecule="H2",
    bond_length=0.74,
    ansatz_type="UCC",
    transform="jordan_wigner"
)
```

## Output Data Structure

`process_molecule()` returns a `MoleculeData` object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `hamiltonian` | `openfermion.QubitOperator` | Qubit Hamiltonian as a sum of Pauli terms |
| `n_qubits` | `int` | Number of qubits required |
| `reference_state` | `np.ndarray` | Hartree‑Fock state as a complex unit vector (one‑hot) |
| `fci_energy` | `float` | Exact Full CI energy (Hartree) |
| `molecular_info` | `dict` | Metadata including formula, bond length, basis set, active space, transform, HF energy, electron/orbital counts |

Example accessing metadata:

```python
print(f"Molecule: {result.molecular_info['formula']}")
print(f"Bond length: {result.molecular_info['bond_length_angstrom']} Å")
print(f"HF energy: {result.molecular_info['hf_energy']} Hartree")
```

## Error Handling

The function raises appropriate exceptions for invalid inputs:

- `ValueError` for negative bond length, unsupported ansatz type, unsupported transform, or unsupported molecule.
- `RuntimeError` if the Hartree‑Fock calculation fails to converge.
- `ImportError` if required dependencies are missing.

## Integration with Quantum Simulator

The `hamiltonian` and `reference_state` are directly usable with quantum simulation libraries (e.g., Qiskit, Cirq, PennyLane). Example expectation value computation:

```python
import numpy as np
from openfermion import expectation

# Compute expectation value of Hamiltonian with respect to reference state
energy = expectation(result.hamiltonian, result.reference_state)
print(f"HF energy from expectation: {energy}")
```

## Performance Notes

- **HF convergence**: For difficult molecules, the module increases the maximum SCF iterations to 1000 and uses atomic guess. If convergence still fails, consider adjusting geometry or basis set.
- **Active space integrals**: When an active space smaller than the full space is requested, the module uses PySCF's CASCI to compute integrals and core energy. This adds computational overhead but is necessary for accurate reduced‑space Hamiltonians.
- **Reference state search**: For systems with up to ~12 qubits, the reference state is found by brute‑force search over all computational basis states. For larger systems, this becomes prohibitive; future versions may implement direct mapping of the HF Slater determinant.

## Example: Full Workflow

```python
import numpy as np
from src.modules.molecule_processor import process_molecule

# 1. Generate quantum inputs for H2
h2_data = process_molecule("H2", 0.74, "UCC")

# 2. Extract Hamiltonian and reference state
hamiltonian = h2_data.hamiltonian
psi0 = h2_data.reference_state

# 3. Use with a quantum algorithm (example: exact diagonalization)
from scipy.sparse.linalg import eigsh
# Build sparse matrix representation of Hamiltonian (not shown)
# ...

print(f"H2 FCI energy: {h2_data.fci_energy} Hartree")
print(f"Number of qubits: {h2_data.n_qubits}")
```

## Troubleshooting

- **HF non‑convergence**: Try a different bond length or basis set. The module already uses robust settings.
- **Unexpected qubit count**: Parity mapping reduces qubit count due to particle‑number conservation. Use Jordan‑Wigner if you need one qubit per spin orbital.
- **Large Hamiltonian terms**: The number of Pauli terms grows as O(N⁴) with orbital count. Consider using active space to reduce system size.

## References

- Tencirchem Documentation: https://tencirchem.readthedocs.io/
- OpenFermion Documentation: https://quantumai.google/openfermion
- PySCF Documentation: https://pyscf.org/

## License

Part of the RLQAS project. See project LICENSE for details.