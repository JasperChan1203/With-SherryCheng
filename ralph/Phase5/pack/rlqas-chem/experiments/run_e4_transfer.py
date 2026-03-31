#!/usr/bin/env python3
"""E4: Cross-geometry policy transfer experiment."""
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rlqas_chem.experiment.transfer import train_multi_geometry, evaluate_transfer

MOLECULE = 'H2'
TRAIN_BOND_LENGTHS = [0.5, 0.7, 0.9, 1.1, 1.3]
TEST_BOND_LENGTHS = [1.5, 1.7, 2.0]
N_EPISODES_PER_GEOMETRY = int(os.environ.get('RLQAS_N_EPISODES', '100'))
N_EPISODES_FINETUNE = int(os.environ.get('RLQAS_N_FINETUNE', '50'))
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def run_e4():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"E4: Training H2 policy on bond lengths: {TRAIN_BOND_LENGTHS}")
    print(f"    n_episodes_per_geometry={N_EPISODES_PER_GEOMETRY}")

    try:
        agent = train_multi_geometry(
            molecule=MOLECULE,
            bond_lengths_train=TRAIN_BOND_LENGTHS,
            agent_type='ppo',
            n_episodes_per_geometry=N_EPISODES_PER_GEOMETRY,
        )
    except Exception as e:
        print(f"Training failed: {e}")
        raise

    print(f"\nEvaluating transfer on bond lengths: {TEST_BOND_LENGTHS}")
    print(f"    n_episodes_finetune={N_EPISODES_FINETUNE}")

    try:
        transfer_results = evaluate_transfer(
            agent=agent,
            molecule=MOLECULE,
            bond_lengths_test=TEST_BOND_LENGTHS,
            n_episodes_finetune=N_EPISODES_FINETUNE,
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise

    # Prepare JSON-serializable output
    output = {
        "molecule": MOLECULE,
        "train_bond_lengths": TRAIN_BOND_LENGTHS,
        "test_bond_lengths": TEST_BOND_LENGTHS,
        "n_episodes_per_geometry": N_EPISODES_PER_GEOMETRY,
        "n_episodes_finetune": N_EPISODES_FINETUNE,
        "results": {str(bl): v for bl, v in transfer_results.items()},
    }

    output_path = os.path.join(RESULTS_DIR, 'e4_transfer.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n=== E4 Transfer Summary ===")
    print(f"{'Bond Length':<14} {'Zero-shot(mHa)':<17} {'Finetune(mHa)':<16} {'Scratch(mHa)'}")
    print("-" * 62)

    finetune_errors = []
    scratch_errors = []

    for bl in TEST_BOND_LENGTHS:
        entry = transfer_results.get(bl, {})
        zs = entry.get("zero_shot_error")
        ft = entry.get("finetune_error")
        sc = entry.get("scratch_error")

        zs_str = f"{zs*1000:.3f}" if zs is not None else "N/A"
        ft_str = f"{ft*1000:.3f}" if ft is not None else "N/A"
        sc_str = f"{sc*1000:.3f}" if sc is not None else "N/A"
        print(f"{bl:<14.1f} {zs_str:<17} {ft_str:<16} {sc_str}")

        if ft is not None:
            finetune_errors.append(ft)
        if sc is not None:
            scratch_errors.append(sc)

    # Conclusion
    if finetune_errors and scratch_errors:
        mean_ft = np.mean(finetune_errors)
        mean_sc = np.mean(scratch_errors)
        if mean_ft < 0.5 * mean_sc:
            conclusion = "Transfer effective"
        else:
            conclusion = "Transfer inconclusive"
    else:
        conclusion = "Transfer inconclusive (insufficient data)"

    print(f"\nConclusion: {conclusion}")
    output["conclusion"] = conclusion

    # Save updated output with conclusion
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    return output


if __name__ == '__main__':
    run_e4()
    sys.exit(0)
