"""LiH validation for RLQAS Phase 1 integrated package.

This module provides validation functions for the integrated RLQAS Phase 1 package.
It implements the run_lih_validation() function as specified in RLQAS_Phase1_Tasks.md.
"""

import sys
import os
import json
import time
import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import random

# Import integrated RLQAS Phase 1 modules
from ..molecule.processor import process_molecule, MoleculeData
from ..simulator.factory import SimulatorFactory
from ..rl.ppo_agent import PPOAgent
from ..search.controller import UCCSearchController
from ..search.config import UCCSearchConfig
from .metrics import MetricsCollector
from .report import ReportGenerator


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # Set PyTorch CUDA seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def check_module_health():
    """Check that all required Phase 1 modules are accessible and functional.

    Returns:
        tuple: (success, errors) where success is bool, errors is list of strings
    """
    errors = []

    try:
        # Test molecule processing
        from ..molecule.processor import process_molecule
        # Quick test with H2
        data = process_molecule('H2', 0.74, 'UCC')
        if not hasattr(data, 'fci_energy'):
            errors.append("MoleculeData missing fci_energy attribute")
    except Exception as e:
        errors.append(f"Molecule processing failed: {e}")

    try:
        # Test simulator creation
        from ..simulator.factory import SimulatorFactory
        simulator = SimulatorFactory.create_simulator(4)  # 4 qubits test
        if simulator is None:
            errors.append("Simulator creation returned None")
    except Exception as e:
        errors.append(f"Simulator creation failed: {e}")

    try:
        # Test RL agent creation
        from ..rl.ppo_agent import PPOAgent
        agent = PPOAgent(config={'seed': 42, 'use_gpu': False})
        if agent is None:
            errors.append("PPOAgent creation returned None")
    except Exception as e:
        errors.append(f"RL agent creation failed: {e}")

    try:
        # Test UCC search controller import
        from ..search.controller import UCCSearchController
        # Just test import, instantiation requires molecule data
    except Exception as e:
        errors.append(f"UCC search module failed: {e}")

    return len(errors) == 0, errors


def run_lih_validation(
    bond_length: float = 1.6,
    active_space: Tuple[int, int] = (2, 2),
    basis_set: str = 'sto-3g',
    transform: str = 'jordan_wigner',
    n_episodes: int = 500,
    early_stop_threshold: float = 1.6e-3,
    output_dir: str = 'results/lih_test_results',
    generate_report: bool = True
) -> Dict[str, Any]:
    """Run complete LiH validation test.

    Args:
        bond_length: LiH bond length in Å
        active_space: Active space (electrons, orbitals)
        basis_set: Basis set for quantum chemistry calculation
        transform: Fermion-to-qubit transformation
        n_episodes: Maximum number of RL episodes
        early_stop_threshold: Convergence threshold in Hartree
        output_dir: Directory to save results
        generate_report: Whether to generate comprehensive validation report (default: True)

    Returns:
        Dictionary containing validation results and metrics
    """
    results = {
        'validation_start_time': time.time(),
        'configuration': {
            'molecule': 'LiH',
            'bond_length': bond_length,
            'active_space': list(active_space),
            'basis_set': basis_set,
            'transform': transform,
            'n_episodes': n_episodes,
            'early_stop_threshold': early_stop_threshold,
            'output_dir': output_dir
        },
        'metrics': {},
        'success': False,
        'errors': [],
        'warnings': []
    }

    try:
        # Set random seeds for reproducibility
        set_seed(42)

        # Step 0: Check module health
        print("Step 0: Checking module health...")
        health_ok, health_errors = check_module_health()
        if not health_ok:
            results['errors'].extend(health_errors)
            print(f"  ✗ Module health check failed: {len(health_errors)} errors")
            for err in health_errors:
                print(f"    - {err}")
            # Early return with failure
            results['validation_end_time'] = time.time()
            results['total_time_seconds'] = results['validation_end_time'] - results['validation_start_time']
            return results
        print("  ✓ All modules pass health checks")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # 1. Process LiH molecule using integrated molecule processor
        print("Step 1: Processing LiH molecule...")
        molecule_data = process_molecule(
            molecule='LiH',
            bond_length=bond_length,
            ansatz_type='UCC',
            active_space=active_space,
            basis_set=basis_set,
            transform=transform
        )
        results['molecule_info'] = molecule_data.molecular_info
        results['fci_energy'] = molecule_data.fci_energy
        results['n_qubits'] = molecule_data.n_qubits
        print(f"  LiH processed: {molecule_data.n_qubits} qubits, FCI energy = {molecule_data.fci_energy:.6f} Hartree")

        # 2. Create simulator using integrated simulator factory
        print("Step 2: Creating quantum simulator...")
        simulator_config = {"max_memory_gb": 32}
        simulator = SimulatorFactory.create_simulator(molecule_data.n_qubits, simulator_config)
        print(f"  Simulator created: max qubits = {simulator.get_max_qubits()}")

        # 3. Create UCC search controller using integrated controller
        print("Step 3: Creating UCC search controller...")
        controller_config = {
            "environment": {
                "max_depth": 12,
                "max_excitations": 15,
                "use_sqeb": True,
                "param_init_strategy": "random",
                "observation_normalization": True,
                "action_masking": True,
            },
            "reward_function": {
                "energy_weight": 10.0,
                "complexity_penalty": 0.0001,
                "baseline_type": "current_best",
                "shaping_rewards": True,
                "rolling_window_size": 10
            },
            "controller": {
                "agent_type": "ppo",
                "n_episodes": n_episodes,
                "early_stop_threshold": early_stop_threshold,
                "checkpoint_frequency": 0,  # disable checkpoints for validation
                "log_frequency": 10,
                "use_gpu": False,
                "seed": 42,
                "policy_type": "MlpPolicy",
                "learning_rate": 3e-4,
                "n_steps": 256,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.1,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "verbose": 1,
                "n_envs": 1,
                "train_frequency": 1
            }
        }
        controller = UCCSearchController(molecule_data, agent_type='ppo', config=controller_config)
        print("  Controller created with PPO agent")

        # 4. Run search
        print(f"Step 4: Running UCC search for up to {n_episodes} episodes...")
        search_results = controller.search(
            n_episodes=n_episodes,
            early_stop_threshold=early_stop_threshold
        )
        results['search_results'] = search_results

        # 5. Collect metrics
        print("Step 5: Collecting metrics...")
        # Basic metrics from search results
        final_energy = search_results.get('best_energy')
        fci_energy = molecule_data.fci_energy
        if final_energy is not None and fci_energy is not None:
            error_hartree = final_energy - fci_energy
            error_mha = error_hartree * 1000
            chemical_accuracy_achieved = abs(error_mha) < 1.6
        else:
            error_hartree = None
            error_mha = None
            chemical_accuracy_achieved = False

        metrics = {
            'final_energy': final_energy,
            'fci_energy': fci_energy,
            'error_hartree': error_hartree,
            'error_mha': error_mha,
            'chemical_accuracy_achieved': chemical_accuracy_achieved,
            'circuit_depth': search_results.get('episode_depths', [])[-1] if search_results.get('episode_depths') is not None and len(search_results.get('episode_depths', [])) > 0 else None,
            'n_excitations': len(search_results.get('best_excitations', [])) if search_results.get('best_excitations') is not None else None,
            'n_parameters': len(search_results.get('best_params', [])) if search_results.get('best_params') is not None else None,
            'convergence_reached': search_results.get('convergence_reached', False),
            'episodes_completed': len(search_results.get('episode_energies', [])),
            'final_reward': search_results.get('episode_rewards', [])[-1] if search_results.get('episode_rewards') is not None and len(search_results.get('episode_rewards', [])) > 0 else None,
        }
        results['metrics'] = metrics

        # 6. Evaluate against success criteria
        print("Step 6: Evaluating success criteria...")
        success = True
        # Chemical accuracy criterion
        if chemical_accuracy_achieved:
            print(f"  ✓ Chemical accuracy achieved: {error_mha:.2f} mHa error (< 1.6 mHa)")
        else:
            print(f"  ✗ Chemical accuracy NOT achieved: {error_mha:.2f} mHa error (>= 1.6 mHa)")
            success = False

        # Convergence criterion
        if search_results.get('convergence_reached'):
            print("  ✓ Convergence reached (early stopping)")
        else:
            print("  ✗ Convergence not reached (max episodes exhausted)")
            # Not necessarily a failure, but note
            results['warnings'].append("Convergence not reached within episode limit")

        # Functional success criterion (no errors)
        if len(results['errors']) == 0:
            print("  ✓ No errors during validation")
        else:
            print(f"  ✗ Errors occurred: {len(results['errors'])}")
            success = False

        results['success'] = success

        # 7. Save results
        print("Step 7: Saving results...")
        results['validation_end_time'] = time.time()
        results['total_time_seconds'] = results['validation_end_time'] - results['validation_start_time']

        # Save raw results
        results_path = os.path.join(output_dir, 'validation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Results saved to {results_path}")

        # 8. Generate report
        if generate_report:
            print("Step 8: Generating validation report...")
            try:
                # Create metrics collector
                metrics_collector = MetricsCollector(output_dir=output_dir)

                # Record energy metrics per episode
                episode_energies = search_results.get('episode_energies', [])
                for i, energy in enumerate(episode_energies):
                    metrics_collector.record_energy_metric(i, energy, fci_energy)

                # Record circuit metrics (final)
                episode_depths = search_results.get('episode_depths', [])
                final_depth = episode_depths[-1] if episode_depths else None
                best_excitations = search_results.get('best_excitations')
                n_excitations = len(best_excitations) if best_excitations is not None else None
                best_params = search_results.get('best_params')
                n_parameters = len(best_params) if best_params is not None else None
                if final_depth is not None or n_excitations is not None or n_parameters is not None:
                    metrics_collector.record_circuit_metric(
                        final_depth if final_depth is not None else 0,
                        n_excitations if n_excitations is not None else 0,
                        n_parameters if n_parameters is not None else 0,
                        iteration=len(episode_energies)-1 if episode_energies else 0
                    )

                # Record training metrics per episode
                episode_rewards = search_results.get('episode_rewards', [])
                for i, reward in enumerate(episode_rewards):
                    energy = episode_energies[i] if i < len(episode_energies) else None
                    depth = episode_depths[i] if episode_depths and i < len(episode_depths) else None
                    best_energy = min(episode_energies[:i+1]) if episode_energies else None
                    metrics_collector.record_training_metric(i, reward, energy, depth, best_energy)

                # Record timing metrics (simple)
                metrics_collector.record_timing_metric('total_validation', results['total_time_seconds'])

                # Record validation metrics
                metrics_collector.record_validation_metric('final_energy', final_energy)
                metrics_collector.record_validation_metric('fci_energy', fci_energy)
                metrics_collector.record_validation_metric('error_mha', error_mha)
                metrics_collector.record_validation_metric('chemical_accuracy_achieved', chemical_accuracy_achieved)
                metrics_collector.record_validation_metric('convergence_reached', search_results.get('convergence_reached', False))
                metrics_collector.record_validation_metric('episodes_completed', len(episode_energies))

                # Save metrics
                metrics_collector.save_metrics()
                metrics_collector.generate_csv_report()

                # Generate report
                report_generator = ReportGenerator(metrics_collector.metrics, results)
                report_path = os.path.join(output_dir, 'validation_report.md')
                report_generator.generate_markdown_report(report_path)

                # Generate visualizations
                viz_dir = os.path.join(output_dir, 'visualizations')
                report_generator.generate_visualizations(viz_dir)

                print(f"  Report generated: {report_path}")
                print(f"  Visualizations saved to: {viz_dir}")

            except Exception as e:
                print(f"  Warning: Failed to generate report: {e}")
                import traceback
                traceback.print_exc()

        # 9. Print summary
        print("\n=== Validation Summary ===")
        print(f"Status: {'SUCCESS' if success else 'FAILURE'}")
        print(f"Chemical accuracy: {'YES' if chemical_accuracy_achieved else 'NO'}")
        print(f"Final VQE energy: {final_energy:.6f} Hartree")
        print(f"FCI reference energy: {fci_energy:.6f} Hartree")
        if error_mha is not None:
            print(f"Energy error: {error_mha:.2f} mHa")
        print(f"Convergence reached: {search_results.get('convergence_reached', False)}")
        print(f"Episodes completed: {len(search_results.get('episode_energies', []))}")
        print(f"Total time: {results['total_time_seconds']:.2f} seconds")

    except Exception as e:
        error_msg = f"Validation failed with exception: {str(e)}"
        print(f"ERROR: {error_msg}")
        import traceback
        traceback.print_exc()
        results['errors'].append(error_msg)
        results['success'] = False
        results['validation_end_time'] = time.time()
        results['total_time_seconds'] = results['validation_end_time'] - results['validation_start_time']

    return results


def main():
    """Command-line interface for LiH validation."""
    import argparse
    parser = argparse.ArgumentParser(description='Run LiH validation test for RLQAS Phase 1')
    parser.add_argument('--bond-length', type=float, default=1.6,
                        help='LiH bond length in Å (default: 1.6)')
    parser.add_argument('--active-space', type=int, nargs=2, default=[2, 2],
                        help='Active space (electrons, orbitals) (default: 2 2)')
    parser.add_argument('--basis-set', type=str, default='sto-3g',
                        help='Basis set (default: sto-3g)')
    parser.add_argument('--transform', type=str, default='jordan_wigner',
                        choices=['parity', 'jordan_wigner', 'bravyi_kitaev'],
                        help='Fermion-to-qubit transformation (default: jordan_wigner)')
    parser.add_argument('--n-episodes', type=int, default=500,
                        help='Maximum number of RL episodes (default: 500)')
    parser.add_argument('--early-stop-threshold', type=float, default=1.6e-3,
                        help='Convergence threshold in Hartree (default: 1.6e-3)')
    parser.add_argument('--output-dir', type=str, default='results/lih_test_results',
                        help='Output directory (default: results/lih_test_results)')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast configuration for debugging (n_episodes=50, threshold=0.01)')

    args = parser.parse_args()

    # Apply fast configuration if requested
    if args.fast:
        print("Using fast configuration for debugging...")
        args.n_episodes = 50
        args.early_stop_threshold = 0.01
        args.output_dir = 'results/lih_test_fast'

    # Run validation
    results = run_lih_validation(
        bond_length=args.bond_length,
        active_space=tuple(args.active_space),
        basis_set=args.basis_set,
        transform=args.transform,
        n_episodes=args.n_episodes,
        early_stop_threshold=args.early_stop_threshold,
        output_dir=args.output_dir
    )

    # Exit with appropriate code
    if results.get('success', False):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()