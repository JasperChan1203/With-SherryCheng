"""
Tests verifying the partial-circuit architecture search bug fix.

Bug A (environment.py): The classical optimizer was given ALL n_params, so starting
from zeros it always converged to the full-UCCSD minimum regardless of which operators
the agent selected.  Fix: only optimise the active parameter slots.

Bug B (tencirchem.py): The shortcut `ucc.energy(circuit.params)` is safe only when
circuit.params has zeros for non-selected operators — which Bug A now guarantees.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def lih_mol():
    from rlqas.phase1.molecule.processor import process_molecule
    return process_molecule(
        'LiH', 1.6, 'UCC',
        active_space=(2, 5),
        basis_set='sto-3g',
        transform='jordan_wigner',
    )


def _make_env(lih_mol):
    from rlqas.phase1.search.environment import UCCSearchEnv
    return UCCSearchEnv(lih_mol, {
        'run_classical_opt': True,
        'complexity_penalty': 0.0,
        'param_init_strategy': 'zeros',
        'max_depth': 10,
    })


def test_single_operator_does_not_cheat(lih_mol):
    """One operator must NOT give FCI accuracy — it requires genuine search."""
    env = _make_env(lih_mol)
    obs, _ = env.reset()
    obs, r, terminated, truncated, info = env.step(0)

    fci_energy = lih_mol.fci_energy
    err = abs(env.current_energy - fci_energy)
    assert err > 1.6e-3, (
        f"BUG STILL PRESENT: 1 operator achieved chemical accuracy "
        f"(error={err*1000:.4f} mHa < 1.6 mHa). "
        "Optimizer must be constrained to active parameters only."
    )


def test_optimizer_only_touches_active_params(lih_mol):
    """After one step, exactly 1 parameter slot must be non-zero."""
    env = _make_env(lih_mol)
    obs, _ = env.reset()
    obs, r, terminated, truncated, info = env.step(0)

    params = env.current_params
    non_zero = [i for i, x in enumerate(params) if abs(x) > 1e-10]
    assert len(non_zero) == 1, (
        f"BUG STILL PRESENT: {len(non_zero)} parameter slots non-zero after "
        f"selecting 1 operator (indices {non_zero}). "
        "Optimizer must not modify inactive parameter slots."
    )


def test_full_uccsd_matches_fci(lih_mol):
    """Full UCCSD optimisation over all params should reach chemical accuracy."""
    from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
    from scipy.optimize import minimize

    builder = UCCCircuitBuilder(lih_mol)
    n_params = builder.n_params

    def energy_func(p):
        return builder.evaluate_energy(None, p)

    x0 = np.zeros(n_params)
    result = minimize(energy_func, x0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10})

    fci_energy = lih_mol.fci_energy
    err = abs(result.fun - fci_energy)
    assert err < 1.6e-3, (
        f"Full UCCSD optimisation did not reach chemical accuracy: "
        f"error={err*1000:.4f} mHa. "
        "Check molecule setup."
    )


def test_two_operators_improve_over_one(lih_mol):
    """Adding a second operator should further reduce the energy (monotone improvement)."""
    env = _make_env(lih_mol)
    obs, _ = env.reset()
    obs, r1, t1, tr1, info1 = env.step(0)
    energy_1op = env.current_energy

    if not (t1 or tr1):
        obs, r2, t2, tr2, info2 = env.step(1)
        energy_2op = env.current_energy
        # Two operators should give lower or equal energy (variational principle)
        assert energy_2op <= energy_1op + 1e-8, (
            f"Adding a second operator raised energy: {energy_1op:.8f} -> {energy_2op:.8f}"
        )
