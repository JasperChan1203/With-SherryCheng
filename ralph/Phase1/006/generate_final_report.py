#!/usr/bin/env python3
"""
Generate final validation report for RLQAS Phase 1 integration.

This script runs:
1. Chemical accuracy validation (LiH with active_space=(2,3))
2. Performance benchmark (8-qubit simulation)
3. Integration test verification

Generates a comprehensive validation report in Markdown format.
"""

import sys
import os
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'src')

def run_chemical_accuracy_validation():
    """Run LiH validation with chemical accuracy target."""
    from rlqas.phase1.validation.validator import run_lih_validation

    print("Running chemical accuracy validation...")
    results = run_lih_validation(
        bond_length=1.6,
        active_space=(2, 3),
        basis_set='sto-3g',
        transform='jordan_wigner',
        n_episodes=500,
        early_stop_threshold=1.6e-3,
        output_dir='results/final_validation',
        generate_report=True
    )
    return results

def run_performance_benchmark():
    """Run 8-qubit performance benchmark."""
    # Import benchmark function from our script
    import sys
    sys.path.insert(0, 'examples')
    from benchmark_8qubit import benchmark_8qubit
    print("Running 8-qubit performance benchmark...")
    results = benchmark_8qubit(n_trials=5, warmup=2)
    return results

def run_integration_tests():
    """Run integration tests and capture results."""
    import subprocess
    print("Running integration tests...")
    # Run test_integration.py and capture output
    result = subprocess.run(
        [sys.executable, 'test_integration.py'],
        capture_output=True,
        text=True
    )
    success = result.returncode == 0
    return {
        'success': success,
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

def generate_markdown_report(chem_results, perf_results, integration_results):
    """Generate comprehensive validation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# RLQAS Phase 1 Validation Report

**Generated**: {timestamp}

## Executive Summary

- **Chemical Accuracy**: {'✓ ACHIEVED' if chem_results.get('success') else '✗ NOT ACHIEVED'}
- **Performance Target (8-qubit <500ms)**: {'✓ ACHIEVED' if perf_results.get('target_met') else '✗ NOT ACHIEVED'}
- **Integration Tests**: {'✓ ALL PASSED' if integration_results['success'] else '✗ FAILED'}
- **Overall Status**: {'✅ ALL OBJECTIVES MET' if chem_results.get('success') and perf_results.get('target_met') and integration_results['success'] else '❌ OBJECTIVES NOT MET'}

## 1. Chemical Accuracy Validation

### Configuration
- **Molecule**: LiH
- **Bond length**: 1.6 Å
- **Active space**: (2, 3)
- **Transformation**: Jordan-Wigner
- **Maximum episodes**: 500
- **Early stop threshold**: 1.6 mHa (1.6e-3 Hartree)

### Results
"""
    if chem_results.get('success'):
        error_mha = chem_results.get('metrics', {}).get('error_mha')
        report += f"- **Status**: SUCCESS\n"
        report += f"- **Final VQE energy**: {chem_results.get('metrics', {}).get('final_energy'):.6f} Hartree\n"
        report += f"- **FCI reference energy**: {chem_results.get('fci_energy'):.6f} Hartree\n"
        report += f"- **Error**: {error_mha:.2f} mHa\n"
        report += f"- **Chemical accuracy achieved**: {abs(error_mha):.2f} mHa < 1.6 mHa\n"
        report += f"- **Episodes completed**: {chem_results.get('metrics', {}).get('episodes_completed')}\n"
        report += f"- **Total validation time**: {chem_results.get('total_time_seconds', 0):.2f} seconds\n"
    else:
        report += f"- **Status**: FAILURE\n"
        report += f"- **Errors**: {chem_results.get('errors', [])}\n"

    report += f"""
## 2. Performance Benchmark

### Configuration
- **Qubit count**: {perf_results.get('n_qubits', 8)}
- **Trials**: {perf_results.get('n_trials', 5)}
- **Target**: <500 ms per energy evaluation

### Results
- **Average time**: {perf_results.get('avg_time_ms', 0):.2f} ms
- **Minimum time**: {perf_results.get('min_time_ms', 0):.2f} ms
- **Maximum time**: {perf_results.get('max_time_ms', 0):.2f} ms
- **Standard deviation**: {perf_results.get('std_time_ms', 0):.2f} ms
- **Target met**: {'Yes' if perf_results.get('target_met') else 'No'}
- **Estimated memory usage**: {perf_results.get('memory_gb', 0):.4f} GB

## 3. Integration Tests

### Status
{'All integration tests passed.' if integration_results['success'] else 'Integration tests failed.'}

### Output Summary
```
{integration_results['stdout'][-500:] if integration_results['stdout'] else 'No output'}
```

## 4. System Information

- **Python version**: {sys.version.split()[0]}
- **Platform**: {sys.platform}
- **Working directory**: {os.getcwd()}
- **Package version**: RLQAS Phase 1 (integrated)

## 5. Notes

- Chemical accuracy validation uses Jordan-Wigner transformation due to parity transformation incompatibility with circuit builder.
- Performance benchmark uses random Hamiltonian and circuit; actual chemical systems may have different performance characteristics.
- Integration tests cover all major modules: molecule processing, simulator, circuit builder, environment, and RL agent.

## 6. Conclusion

RLQAS Phase 1 integrated package {'meets all specified objectives' if chem_results.get('success') and perf_results.get('target_met') and integration_results['success'] else 'does not meet all objectives'}.

"""
    return report

def main():
    """Main function to generate validation report."""
    print("=" * 60)
    print("RLQAS Phase 1 Final Validation Report Generation")
    print("=" * 60)

    # Create output directory
    os.makedirs('results/final_validation', exist_ok=True)

    # Run validations
    chem_results = run_chemical_accuracy_validation()
    perf_results = run_performance_benchmark()
    integration_results = run_integration_tests()

    # Generate report
    report = generate_markdown_report(chem_results, perf_results, integration_results)

    # Save report
    report_path = 'results/final_validation/validation_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nValidation report saved to: {report_path}")
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Chemical accuracy: {'✓' if chem_results.get('success') else '✗'}")
    print(f"  Performance target: {'✓' if perf_results.get('target_met') else '✗'}")
    print(f"  Integration tests: {'✓' if integration_results['success'] else '✗'}")
    print("=" * 60)

    # Overall success
    overall_success = (
        chem_results.get('success') and
        perf_results.get('target_met') and
        integration_results['success']
    )

    if overall_success:
        print("\n✅ ALL RLQAS PHASE 1 OBJECTIVES ACHIEVED")
    else:
        print("\n❌ SOME OBJECTIVES NOT MET")

    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())