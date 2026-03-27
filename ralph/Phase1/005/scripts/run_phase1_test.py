#!/usr/bin/env python3
"""Run Phase 1 validation test.

This script provides a convenient entry point for running the RLQAS Phase 1
validation test with configurable options.
"""

import sys
import os
import argparse

# Ensure we can import validate_lih module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validate_lih import run_lih_validation


def main():
    parser = argparse.ArgumentParser(
        description='Run RLQAS Phase 1 validation test'
    )
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'fast', 'full'],
                        help='Configuration preset (default: default)')
    parser.add_argument('--output-dir', type=str,
                        help='Override output directory')
    parser.add_argument('--no-report', action='store_true',
                        help='Skip report generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Configuration presets
    configs = {
        'default': {
            'bond_length': 1.6,
            'active_space': (2, 2),
            'basis_set': 'sto-3g',
            'transform': 'parity',
            'n_episodes': 500,
            'early_stop_threshold': 1.6e-3,
            'output_dir': 'results/lih_test_results'
        },
        'fast': {
            'bond_length': 1.6,
            'active_space': (2, 2),
            'basis_set': 'sto-3g',
            'transform': 'parity',
            'n_episodes': 50,
            'early_stop_threshold': 0.01,
            'output_dir': 'results/lih_test_fast'
        },
        'full': {
            'bond_length': 1.6,
            'active_space': (2, 2),
            'basis_set': 'sto-3g',
            'transform': 'parity',
            'n_episodes': 1000,
            'early_stop_threshold': 1.0e-4,
            'output_dir': 'results/lih_test_full'
        }
    }

    config = configs[args.config]
    if args.output_dir:
        config['output_dir'] = args.output_dir

    if args.verbose:
        print(f"Running Phase 1 validation with {args.config} configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")

    # Run validation
    results = run_lih_validation(**config)

    # Generate report unless disabled
    if not args.no_report and results.get('success', False):
        # Try to import report generator
        try:
            sys.path.append('src/evaluation')
            from report_generator import ReportGenerator
            from metrics_collector import MetricsCollector

            # Create metrics collector and report generator
            # This is a placeholder - actual implementation would load metrics
            print("\nGenerating validation report...")
            # For now, just save results as JSON
            report_path = os.path.join(config['output_dir'], 'validation_report.md')
            with open(report_path, 'w') as f:
                f.write(f"# RLQAS Phase 1 Validation Report\n\n")
                f.write(f"Generated from {args.config} configuration\n\n")
                f.write(f"Success: {results.get('success', False)}\n")
                f.write(f"Chemical accuracy achieved: {results.get('metrics', {}).get('chemical_accuracy_achieved', False)}\n")
                f.write(f"Total time: {results.get('total_time_seconds', 0):.2f} seconds\n")
            print(f"Report saved to {report_path}")
        except ImportError:
            print("Note: Report generation modules not available. Skipping report.")

    # Exit with appropriate code
    if results.get('success', False):
        print("\n✓ Phase 1 validation test completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Phase 1 validation test failed")
        sys.exit(1)


if __name__ == "__main__":
    main()