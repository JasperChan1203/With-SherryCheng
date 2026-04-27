#!/usr/bin/env python
"""
Validate Ralph's H2 Hamiltonian implementation against benchmark values.
This script compares Ralph's results with reference values and provides feedback.
"""
import json
import sys
import os
import numpy as np
from typing import Dict, Any, Tuple

def load_benchmark(benchmark_path: str = "h2_benchmark.json") -> Dict[str, Any]:
    """Load benchmark values from JSON file."""
    # First try the provided path
    if os.path.exists(benchmark_path):
        try:
            with open(benchmark_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing benchmark file: {e}")
            sys.exit(1)

    # If not found, try in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_benchmark_path = os.path.join(script_dir, benchmark_path)

    if os.path.exists(script_benchmark_path):
        try:
            with open(script_benchmark_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"✗ Error parsing benchmark file: {e}")
            sys.exit(1)

    # If still not found, show error
    print(f"✗ Benchmark file not found: {benchmark_path}")
    print("  Tried current directory and script directory.")
    print("  Run generate_h2_benchmark.py first to create benchmark values.")
    sys.exit(1)

def load_ralph_results(results_path: str) -> Dict[str, Any]:
    """Load Ralph's results from JSON file."""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"✗ Results file not found: {results_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"✗ Error parsing results file: {e}")
        sys.exit(1)

def validate_molecule(ralph_results: Dict[str, Any], benchmark: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate molecule definition."""
    errors = []

    # Check bond length
    ralph_bond_length = ralph_results.get("molecule", {}).get("bond_length_angstrom")
    bench_bond_length = benchmark["molecule"]["bond_length_angstrom"]
    tolerance = benchmark["verification_tolerances"]["bond_length_tolerance_angstrom"]

    if ralph_bond_length is None:
        errors.append("Missing bond length information")
    elif abs(ralph_bond_length - bench_bond_length) > tolerance:
        errors.append(f"Bond length mismatch: {ralph_bond_length} Å vs {bench_bond_length} Å (±{tolerance} Å)")

    # Check basis set
    ralph_basis = ralph_results.get("molecule", {}).get("basis_set")
    bench_basis = benchmark["molecule"]["basis_set"]

    if ralph_basis != bench_basis:
        errors.append(f"Basis set mismatch: '{ralph_basis}' vs '{bench_basis}'")

    return len(errors) == 0, "; ".join(errors) if errors else "OK"

def validate_energies(ralph_results: Dict[str, Any], benchmark: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate energy calculations."""
    errors = []

    # Get energies
    ralph_energies = ralph_results.get("energies", {})
    bench_energies = benchmark["energies"]
    hf_tol = benchmark["verification_tolerances"]["hf_energy_tolerance_hartree"]
    fci_tol = benchmark["verification_tolerances"]["fci_energy_tolerance_hartree"]

    # HF energy validation
    ralph_hf = ralph_energies.get("hf_hartree")
    bench_hf = bench_energies["hf_hartree"]

    if ralph_hf is None:
        errors.append("Missing HF energy")
    elif abs(ralph_hf - bench_hf) > hf_tol:
        errors.append(f"HF energy mismatch: {ralph_hf:.6f} vs {bench_hf:.6f} Hartree (±{hf_tol})")

    # FCI energy validation
    ralph_fci = ralph_energies.get("fci_hartree")
    bench_fci = bench_energies["fci_hartree"]

    if ralph_fci is None:
        errors.append("Missing FCI energy")
    elif abs(ralph_fci - bench_fci) > fci_tol:
        errors.append(f"FCI energy mismatch: {ralph_fci:.6f} vs {bench_fci:.6f} Hartree (±{fci_tol})")

    # Check HF > FCI (HF should be less accurate, i.e., higher energy)
    if ralph_hf is not None and ralph_fci is not None:
        if ralph_hf <= ralph_fci:
            errors.append(f"Energy relationship incorrect: HF({ralph_hf:.6f}) ≤ FCI({ralph_fci:.6f})")

    return len(errors) == 0, "; ".join(errors) if errors else "OK"

def validate_hamiltonian(ralph_results: Dict[str, Any], benchmark: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate Hamiltonian properties."""
    errors = []

    ralph_ham = ralph_results.get("hamiltonian", {})
    bench_ham = benchmark["hamiltonian"]

    # Check qubit count
    ralph_qubits = ralph_ham.get("n_qubits")
    if ralph_qubits is not None:
        min_q = bench_ham["n_qubits_min"]
        max_q = bench_ham["n_qubits_max"]
        if not (min_q <= ralph_qubits <= max_q):
            errors.append(f"Qubit count {ralph_qubits} outside expected range [{min_q}, {max_q}]")

    # Check Hermiticity
    ralph_hermitian = ralph_ham.get("h1_hermitian")
    if ralph_hermitian is not None and not ralph_hermitian:
        errors.append("Hamiltonian is not Hermitian")

    return len(errors) == 0, "; ".join(errors) if errors else "OK"

def validate_implementation(ralph_results: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    """Run all validation checks."""
    validation = {
        "molecule": validate_molecule(ralph_results, benchmark),
        "energies": validate_energies(ralph_results, benchmark),
        "hamiltonian": validate_hamiltonian(ralph_results, benchmark),
        "overall": (False, "Pending")
    }

    # Overall validation
    all_passed = all(check[0] for check in [
        validation["molecule"],
        validation["energies"],
        validation["hamiltonian"]
    ])

    validation["overall"] = (all_passed, "All checks passed" if all_passed else "Some checks failed")

    return validation

def print_validation_report(validation: Dict[str, Any], ralph_results: Dict[str, Any], benchmark: Dict[str, Any]):
    """Print human-readable validation report."""
    print("\n" + "="*60)
    print("H2 Hamiltonian Validation Report")
    print("="*60)

    print("\n📋 VALIDATION SUMMARY:")
    overall_passed, overall_msg = validation["overall"]
    status_icon = "✅" if overall_passed else "❌"
    print(f"  {status_icon} Overall: {overall_msg}")

    print("\n🔬 DETAILED CHECKS:")
    for check_name, (passed, message) in validation.items():
        if check_name == "overall":
            continue
        icon = "✅" if passed else "❌"
        print(f"  {icon} {check_name.capitalize()}: {message}")

    print("\n📊 RESULTS COMPARISON:")
    print("  Energies (Hartree):")
    print(f"    HF:  Ralph={ralph_results.get('energies', {}).get('hf_hartree', 'N/A'):.6f}, "
          f"Benchmark={benchmark['energies']['hf_hartree']:.6f}")
    print(f"    FCI: Ralph={ralph_results.get('energies', {}).get('fci_hartree', 'N/A'):.6f}, "
          f"Benchmark={benchmark['energies']['fci_hartree']:.6f}")

    print("\n📐 Molecular setup:")
    ralph_mol = ralph_results.get('molecule', {})
    print(f"    Bond length: Ralph={ralph_mol.get('bond_length_angstrom', 'N/A')} Å, "
          f"Benchmark={benchmark['molecule']['bond_length_angstrom']} Å")
    print(f"    Basis set: Ralph='{ralph_mol.get('basis_set', 'N/A')}', "
          f"Benchmark='{benchmark['molecule']['basis_set']}'")

    print("\n💡 RECOMMENDATIONS:")
    if overall_passed:
        print("  ✓ Excellent! All validation checks passed.")
        print("  ✓ Your implementation correctly generates the H2 Hamiltonian.")
    else:
        print("  ✗ Some issues detected. Please check:")
        for check_name, (passed, message) in validation.items():
            if check_name != "overall" and not passed:
                print(f"    - {check_name.capitalize()}: {message}")
        print("\n  🔧 Suggestions:")
        print("    1. Verify the H2 molecule definition (0.5 Å bond length, STO-3G basis)")
        print("    2. Check Hartree-Fock and FCI calculation methods")
        print("    3. Ensure Hamiltonian is properly extracted and Hermitian")
        print("    4. Review PySCF documentation for molecular setup")

    print("\n" + "="*60)

def save_validation_summary(validation: Dict[str, Any], summary_path: str = "validation_summary.txt"):
    """Save validation summary to file."""
    with open(summary_path, 'w') as f:
        f.write("H2 Hamiltonian Validation Summary\n")
        f.write("="*40 + "\n\n")

        overall_passed, overall_msg = validation["overall"]
        f.write(f"Overall Status: {'PASS' if overall_passed else 'FAIL'} - {overall_msg}\n\n")

        f.write("Detailed Results:\n")
        for check_name, (passed, message) in validation.items():
            if check_name == "overall":
                continue
            status = "PASS" if passed else "FAIL"
            f.write(f"  {check_name.upper():12} {status:6} - {message}\n")

    print(f"\n📄 Validation summary saved to: {summary_path}")

def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python validate_h2.py <ralph_results.json>")
        print("Example: python validate_h2.py ../Ralph_Test_H2_Hamiltonian/h2_results.json")
        sys.exit(1)

    results_path = sys.argv[1]

    print("🔍 Validating Ralph's H2 Hamiltonian implementation...")

    # Load data
    benchmark = load_benchmark()
    ralph_results = load_ralph_results(results_path)

    print(f"✓ Loaded benchmark from: h2_benchmark.json")
    print(f"✓ Loaded Ralph's results from: {results_path}")

    # Run validation
    validation = validate_implementation(ralph_results, benchmark)

    # Print report
    print_validation_report(validation, ralph_results, benchmark)

    # Save summary
    summary_path = os.path.join(os.path.dirname(results_path), "validation_summary.txt")
    save_validation_summary(validation, summary_path)

    # Exit code
    overall_passed, _ = validation["overall"]
    sys.exit(0 if overall_passed else 1)

if __name__ == "__main__":
    main()