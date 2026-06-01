"""
VQE on HEA circuit: test if energy can approach FCI for H2.

Tests multiple architectures and depths using scipy L-BFGS-B optimizer,
with 5 random restarts per configuration.
"""

import sys
import importlib.util
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem/src")

def _load(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

_base = "/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem/src/rlqas_chem"
_load("rlqas_chem.utils.logger",    f"{_base}/utils/logger.py")
_load("rlqas_chem.utils.transforms", f"{_base}/utils/transforms.py")
_load("rlqas_chem.molecule.processor", f"{_base}/molecule/processor.py")
_load("rlqas_chem.search.hea.circuit_builder", f"{_base}/search/hea/circuit_builder.py")

from rlqas_chem.search.hea.circuit_builder import HEACircuitBuilder


def build_energy_fn(n_qubits, n_layers, entanglement, rotation_gates, hamiltonian, reference_state):
    """Return a closure that computes electronic energy for given params.

    The HEACircuitBuilder is constructed once; only params change each call.
    """
    import tensorcircuit as tc

    builder = HEACircuitBuilder(
        n_qubits=n_qubits,
        n_layers=n_layers,
        entanglement_pattern=entanglement,
        rotation_gates=rotation_gates,
        parameter_sharing="none",
    )
    n_params = builder._count_total_parameters()

    # Pre-parse Hamiltonian terms once
    parsed_terms = []
    for term, coeff in hamiltonian.terms.items():
        x_list, y_list, z_list = [], [], []
        for idx, pauli in term:
            if pauli == "X": x_list.append(idx)
            elif pauli == "Y": y_list.append(idx)
            elif pauli == "Z": z_list.append(idx)
        parsed_terms.append((float(np.real(coeff)), x_list, y_list, z_list))

    def energy_fn(params):
        builder.build(np.array(params))
        circuit = builder.to_tensorcircuit(inputs=reference_state)
        e = 0.0
        for coeff, x_list, y_list, z_list in parsed_terms:
            exp_val = circuit.expectation_ps(x=x_list, y=y_list, z=z_list)
            e += coeff * float(np.real(exp_val))
        return float(e)

    return energy_fn, n_params


def run_vqe(mol, n_layers, entanglement, rotation_gates, n_restarts=5):
    """Run VQE with multiple random restarts. Returns best total energy."""
    energy_fn, n_params = build_energy_fn(
        mol.n_qubits, n_layers, entanglement, rotation_gates,
        mol.hamiltonian, mol.reference_state,
    )

    best_electronic = float("inf")
    for _ in range(n_restarts):
        x0 = np.random.uniform(-np.pi, np.pi, n_params)
        result = minimize(energy_fn, x0, method="L-BFGS-B",
                          options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9})
        if result.fun < best_electronic:
            best_electronic = result.fun

    return best_electronic + mol.nuclear_repulsion


def main():
    import warnings
    warnings.filterwarnings("ignore")

    # Import here so logging is already suppressed
    from rlqas_chem.molecule.processor import process_molecule

    print("Processing H2 @ 0.74 Å, STO-3G ...", flush=True)
    mol = process_molecule("H2", 0.74, "HEA")
    fci_energy = mol.fci_energy
    hf_energy = mol.molecular_info["hf_energy"] + mol.nuclear_repulsion

    print(f"  n_qubits      : {mol.n_qubits}", flush=True)
    print(f"  HF  energy    : {hf_energy:.6f} Ha", flush=True)
    print(f"  FCI energy    : {fci_energy:.6f} Ha", flush=True)
    print(f"  Correlation   : {abs(hf_energy - fci_energy)*1000:.2f} mHa", flush=True)
    print(f"  Nuclear repul : {mol.nuclear_repulsion:.6f} Ha", flush=True)
    print(flush=True)

    configs = [
        (1, "linear",          ["ry"]),
        (2, "linear",          ["ry"]),
        (2, "circular",        ["ry"]),
        (2, "fully_connected", ["ry"]),
        (4, "linear",          ["ry"]),
        (4, "circular",        ["ry"]),
        (4, "fully_connected", ["ry"]),
        (4, "linear",          ["rx", "ry", "rz"]),
        (4, "circular",        ["rx", "ry", "rz"]),
        (4, "fully_connected", ["rx", "ry", "rz"]),
        (8, "fully_connected", ["ry"]),
        (8, "fully_connected", ["rx", "ry", "rz"]),
    ]

    header = f"{'layers':>6}  {'entanglement':>15}  {'gates':>18}  {'energy (Ha)':>12}  {'error (mHa)':>11}  {'chem.acc':>8}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for n_layers, entanglement, rotation_gates in configs:
        energy = run_vqe(mol, n_layers, entanglement, rotation_gates, n_restarts=5)
        error_mha = abs(energy - fci_energy) * 1000
        above_fci = energy >= fci_energy - 1e-6
        chem_acc = "YES" if error_mha < 1.6 else "no"
        vp_ok = "" if above_fci else " [VIOLATION!]"
        print(f"{n_layers:>6}  {entanglement:>15}  {str(rotation_gates):>18}  "
              f"{energy:>12.6f}  {error_mha:>11.2f}  {chem_acc:>8}{vp_ok}", flush=True)


if __name__ == "__main__":
    np.random.seed(42)
    main()
