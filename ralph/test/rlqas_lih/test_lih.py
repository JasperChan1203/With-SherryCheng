#!/usr/bin/env python3
"""
RLQAS LiH Benchmark Test
Tests LiH molecule with multiple configurations via the unified rlqas.search() API.
"""

import sys
import time
import json
from pathlib import Path

PYTHON = sys.executable
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Chemical accuracy threshold: 1.6 mHa
CHEM_ACC_MHA = 1.6

# ── Test configurations ────────────────────────────────────────────────────────
# Baseline configs (no hyperparameter tuning)
_BASELINE = [
    {
        "name": "LiH_UCC_PPO_baseline_12q",
        "molecule": "LiH",
        "bond_length": 1.6,
        "ansatz_type": "UCC",
        "agent_type": "ppo",
        "n_episodes": 500,
        "active_space": None,
        "config": None,
        "description": "Baseline: 12-qubit, no active space, default hyperparams",
    },
    {
        "name": "LiH_UCC_PPO_baseline_10q",
        "molecule": "LiH",
        "bond_length": 1.6,
        "ansatz_type": "UCC",
        "agent_type": "ppo",
        "n_episodes": 500,
        "active_space": (2, 5),
        "config": None,
        "description": "Baseline: active_space=(2,5) 10-qubit, default hyperparams",
    },
]

# Tuned configs: hartree_fock baseline + higher ent_coef
_TUNED = [
    {
        "name": "LiH_UCC_PPO_tuned_12q",
        "molecule": "LiH",
        "bond_length": 1.6,
        "ansatz_type": "UCC",
        "agent_type": "ppo",
        "n_episodes": 500,
        "active_space": None,
        "config": {
            "reward_function": {"baseline_type": "hartree_fock", "energy_weight": 100.0},
            "controller": {"ent_coef": 0.05, "n_steps": 4096},
        },
        "description": "Tuned: 12-qubit, hf-baseline, energy_weight=100, ent_coef=0.05",
    },
    {
        "name": "LiH_UCC_PPO_tuned_10q",
        "molecule": "LiH",
        "bond_length": 1.6,
        "ansatz_type": "UCC",
        "agent_type": "ppo",
        "n_episodes": 500,
        "active_space": (2, 5),
        "config": {
            "reward_function": {"baseline_type": "hartree_fock", "energy_weight": 100.0},
            "controller": {"ent_coef": 0.05, "n_steps": 4096},
        },
        "description": "Tuned: active_space=(2,5) 10-qubit, hf-baseline, energy_weight=100, ent_coef=0.05",
    },
    {
        "name": "LiH_UCC_PPO_tuned_10q_highent",
        "molecule": "LiH",
        "bond_length": 1.6,
        "ansatz_type": "UCC",
        "agent_type": "ppo",
        "n_episodes": 500,
        "active_space": (2, 5),
        "config": {
            "reward_function": {"baseline_type": "hartree_fock", "energy_weight": 100.0},
            "controller": {"ent_coef": 0.1, "n_steps": 4096},
        },
        "description": "Tuned: active_space=(2,5) 10-qubit, hf-baseline, energy_weight=100, ent_coef=0.1",
    },
]

TESTS = _BASELINE + _TUNED


def run_test(cfg: dict) -> dict:
    import rlqas

    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"  Test: {name}")
    print(f"  {cfg['description']}")
    print(f"{'='*60}")

    kwargs = dict(
        ansatz_type=cfg["ansatz_type"],
        agent_type=cfg["agent_type"],
        n_episodes=cfg["n_episodes"],
    )
    if cfg.get("active_space"):
        kwargs["active_space"] = cfg["active_space"]
    if cfg.get("config"):
        kwargs["config"] = cfg["config"]

    t0 = time.time()
    try:
        result = rlqas.search(cfg["molecule"], cfg["bond_length"], **kwargs)
        elapsed = time.time() - t0

        energy_error_mha = result.get("energy_error_mha", float("nan"))
        n_operators = result.get("n_operators", -1)
        best_energy = result.get("best_energy", float("nan"))
        passed = (
            energy_error_mha > 0
            and energy_error_mha < CHEM_ACC_MHA
            and best_energy != float("inf")
        )

        print(f"  best_energy    = {best_energy:.6f} Ha")
        print(f"  energy_error   = {energy_error_mha:.3f} mHa  (threshold: {CHEM_ACC_MHA} mHa)")
        print(f"  n_operators    = {n_operators}")
        print(f"  elapsed        = {elapsed:.1f} s")
        print(f"  chemical_acc   = {'PASS ✓' if passed else 'FAIL ✗'}")

        record = {
            "name": name,
            "description": cfg["description"],
            "config": cfg,
            "best_energy_ha": best_energy,
            "energy_error_mha": energy_error_mha,
            "n_operators": n_operators,
            "elapsed_s": elapsed,
            "chemical_accuracy_pass": passed,
            "raw_result": {k: v for k, v in result.items() if k != "best_params"},
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ERROR: {e}")
        record = {
            "name": name,
            "description": cfg["description"],
            "config": cfg,
            "error": str(e),
            "elapsed_s": elapsed,
            "chemical_accuracy_pass": False,
        }

    return record


def main():
    print("=" * 60)
    print("  RLQAS LiH Benchmark Test")
    print(f"  Python: {sys.executable}")
    print("=" * 60)

    # Import check
    try:
        import rlqas
        print(f"  rlqas version: {getattr(rlqas, '__version__', 'unknown')}")
    except ImportError as e:
        print(f"ERROR: cannot import rlqas — {e}")
        sys.exit(1)

    records = []
    for cfg in TESTS:
        rec = run_test(cfg)
        records.append(rec)

        # Save incremental results after each test
        out_path = RESULTS_DIR / f"{cfg['name']}.json"
        with open(out_path, "w") as f:
            json.dump(rec, f, indent=2, default=str)
        print(f"  Saved: {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    passed_count = 0
    for rec in records:
        status = "PASS ✓" if rec.get("chemical_accuracy_pass") else "FAIL ✗"
        err = rec.get("energy_error_mha", rec.get("error", "N/A"))
        ops = rec.get("n_operators", "-")
        if isinstance(err, float):
            err_str = f"{err:.3f} mHa"
        else:
            err_str = str(err)
        print(f"  [{status}] {rec['name']:45s}  error={err_str}  ops={ops}")
        if rec.get("chemical_accuracy_pass"):
            passed_count += 1

    print(f"\n  Passed: {passed_count}/{len(records)}")

    # Save combined summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(records, f, indent=2, default=str)
    print(f"  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
