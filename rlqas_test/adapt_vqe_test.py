"""
ADAPT-VQE baseline: record how many operators are needed to reach chemical accuracy
for LiH, BeH2, H6 at bond_length=1.0 Å.

Molecule setup mirrors rlqas-chem processor.py for consistency.
"""
import json
import time
import numpy as np
from pyscf import gto, scf
from tencirchem import UCCSD

CHEM_ACC = 1.6e-3  # Hartree
MAX_ITER = 50
EPSILON = 1e-4     # gradient norm convergence

MOLECULES = {
    "LiH": {
        "atoms": [("Li", 0, 0, 0), ("H", 0, 0, 1.0)],
        "active_space": (4, 6),
    },
    "BeH2": {
        "atoms": [("H", -1.0, 0, 0), ("Be", 0, 0, 0), ("H", 1.0, 0, 0)],
        "active_space": (6, 7),
    },
    "H6": {
        "atoms": [
            ("H", 0.0, 0, 0), ("H", 1.0, 0, 0), ("H", 2.0, 0, 0),
            ("H", 3.0, 0, 0), ("H", 4.0, 0, 0), ("H", 5.0, 0, 0),
        ],
        "active_space": (6, 6),
    },
}


def run_adapt_vqe(name, atoms, active_space):
    print(f"\n{'='*60}")
    print(f"ADAPT-VQE: {name}  active_space={active_space}")
    print(f"{'='*60}")

    mol = gto.M(atom=atoms, basis="sto-3g", unit="angstrom", symmetry=True, verbose=0)
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-8
    hf.max_cycle = 1000
    hf.kernel()

    ucc = UCCSD(mol, active_space=active_space, init_method="mp2")
    ucc.kernel()
    fci_energy = ucc.fci_energy if hasattr(ucc, "fci_energy") else ucc.e_fci
    print(f"FCI energy: {fci_energy:.6f} Ha")
    print(f"n_qubits:   {ucc.n_qubits}")

    # Build operator pool (individual operators, not grouped)
    ex1_ops, ex1_param_ids, _ = ucc.get_ex1_ops(ucc.t1)
    ex2_ops, ex2_param_ids, ex2_init_guess = ucc.get_ex2_ops(ucc.t2)
    ex2_ops, ex2_param_ids, _ = ucc.pick_and_sort(
        ex2_ops, ex2_param_ids, ex2_init_guess, ucc.pick_ex2, ucc.sort_ex2
    )
    op_pool = [[op] for op in ex1_ops] + [[op] for op in ex2_ops]
    print(f"Operator pool size: {len(op_pool)}")

    # Reset ansatz
    ucc.ex_ops = []
    ucc.params = []
    ucc.param_ids = []

    t0 = time.time()
    n_ops_to_chem_acc = None
    history = []

    for i in range(MAX_ITER):
        psi = ucc.civector()
        bra = ucc.hamiltonian(psi)

        grad_list = []
        for op_list in op_pool:
            grad = bra.conj() @ ucc.apply_excitation(psi, op_list[0])
            if len(op_list) == 2:
                grad += bra.conj() @ ucc.apply_excitation(psi, op_list[1])
            grad_list.append(2 * grad)

        grad_norm = np.linalg.norm(grad_list)
        chosen_op_list = op_pool[np.argmax(np.abs(grad_list))]

        ucc.ex_ops.extend(chosen_op_list)
        ucc.params = list(ucc.params) + [0]
        ucc.param_ids.extend([len(ucc.params) - 1] * len(chosen_op_list))
        ucc.init_guess = ucc.params
        ucc.kernel()

        energy = ucc.e_ucc
        error_mha = abs(energy - fci_energy) * 1000
        n_ops = len(ucc.ex_ops)
        history.append({"iter": i, "n_ops": n_ops, "energy": energy, "error_mha": error_mha})

        print(f"  Iter {i:2d} | ops={n_ops:2d} | E={energy:.6f} Ha | err={error_mha:.3f} mHa | |grad|={grad_norm:.4f}")

        if n_ops_to_chem_acc is None and error_mha < 1.6:
            n_ops_to_chem_acc = n_ops
            print(f"  *** Chemical accuracy reached at {n_ops} operators ***")

        if grad_norm < EPSILON:
            print(f"  Gradient converged (norm={grad_norm:.2e})")
            break

    elapsed = time.time() - t0
    final_error = abs(ucc.e_ucc - fci_energy) * 1000

    return {
        "molecule": name,
        "fci_energy": fci_energy,
        "final_energy": ucc.e_ucc,
        "final_error_mha": final_error,
        "n_ops_total": len(ucc.ex_ops),
        "n_ops_to_chem_acc": n_ops_to_chem_acc,
        "chem_acc_reached": n_ops_to_chem_acc is not None,
        "elapsed_s": elapsed,
        "history": history,
    }


if __name__ == "__main__":
    results = {}
    for name, cfg in MOLECULES.items():
        result = run_adapt_vqe(name, cfg["atoms"], cfg["active_space"])
        results[name] = result

    print(f"\n{'='*60}")
    print(f"{'Molecule':<8} {'ChemAcc':>8} {'Ops(chem)':>10} {'Ops(final)':>11} {'FinalErr(mHa)':>14} {'Time(s)':>8}")
    print(f"{'-'*60}")
    for name, r in results.items():
        ops_chem = r["n_ops_to_chem_acc"] if r["chem_acc_reached"] else "N/A"
        print(f"{name:<8} {'YES' if r['chem_acc_reached'] else 'NO':>8} {str(ops_chem):>10} {r['n_ops_total']:>11} {r['final_error_mha']:>14.3f} {r['elapsed_s']:>8.1f}")
    print(f"{'='*60}")

    with open("results/adapt_vqe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/adapt_vqe_results.json")
