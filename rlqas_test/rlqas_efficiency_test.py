"""
RLQAS efficiency comparison: PPO and GRPO with early stopping at chemical accuracy.
Records n_operators when chemical accuracy is first reached, for comparison with ADAPT-VQE.

Key differences from test_molecules.py:
  - use_early_stop=True (stop episode when energy < FCI + 1.6 mHa)
  - max_operators=30 (larger budget to give RLQAS room to find the right circuit)
  - n_episodes=1000
"""
import argparse
import json
import os
import time
from datetime import datetime

import rlqas_chem

MOLECULES = [
    ("LiH",  1.0),
    ("BeH2", 1.0),
    ("H6",   1.0),
]

AGENTS = ["ppo", "grpo"]
DEFAULT_EPISODES = 1000
MAX_OPERATORS = 50


def run_search(molecule, bond_length, agent_type, n_episodes):
    print(f"  [{agent_type.upper()}] {molecule} @ {bond_length} Å  ({n_episodes} ep, max_ops={MAX_OPERATORS}) ...", flush=True)
    t0 = time.time()
    result = rlqas_chem.search(
        molecule,
        bond_length,
        ansatz_type="UCC",
        agent_type=agent_type,
        n_episodes=n_episodes,
        use_early_stop=True,
        max_operators=MAX_OPERATORS,
    )
    elapsed = time.time() - t0
    status = "✓ CHEM_ACC" if result["chemical_accuracy"] else "✗ not converged"
    print(
        f"    → energy={result['best_energy']:.6f} Ha  "
        f"error={result['energy_error_mha']:.2f} mHa  "
        f"ops={result['n_operators']}  "
        f"{status}  ({elapsed:.1f}s)",
        flush=True,
    )
    return {**result, "elapsed_s": round(elapsed, 1)}


def print_summary(results):
    print("\n" + "=" * 70)
    print(f"{'Molecule':<8} {'Agent':<6} {'Energy (Ha)':>14} {'Error (mHa)':>12} {'Ops':>5} {'ChemAcc':>8}")
    print("-" * 70)
    for r in results:
        acc = "YES" if r["chemical_accuracy"] else "NO"
        ops = str(r["n_operators"]) if r["n_operators"] is not None else "-"
        print(
            f"{r['molecule']:<8} {r['agent_type']:<6} "
            f"{r['best_energy']:>14.6f} "
            f"{r['energy_error_mha']:>12.3f} "
            f"{ops:>5} "
            f"{acc:>8}"
        )
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,
                        default=int(os.environ.get("RLQAS_N_EPISODES", DEFAULT_EPISODES)))
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    n_episodes = args.episodes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"results/rlqas_efficiency_{timestamp}.json"

    print(f"RLQAS Efficiency Test  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Molecules  : {[m for m, _ in MOLECULES]}")
    print(f"Agents     : {AGENTS}")
    print(f"Episodes   : {n_episodes}")
    print(f"max_ops    : {MAX_OPERATORS}")
    print(f"early_stop : chemical accuracy (1.6 mHa)")
    print(f"Output     : {output_path}")
    print("=" * 70)

    all_results = []
    for molecule, bond_length in MOLECULES:
        print(f"\n--- {molecule} (bond_length={bond_length} Å) ---")
        for agent in AGENTS:
            try:
                r = run_search(molecule, bond_length, agent, n_episodes)
                all_results.append(r)
            except Exception as e:
                print(f"  [{agent.upper()}] ERROR: {e}", flush=True)
                all_results.append({
                    "molecule": molecule,
                    "bond_length": bond_length,
                    "agent_type": agent,
                    "error": str(e),
                })

    print_summary([r for r in all_results if "error" not in r])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
