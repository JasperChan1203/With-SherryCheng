#!/usr/bin/env python
"""
Validate Ralph's LiH VQE implementation against FCI energy (PySCF-computed or benchmark).
This script validates manual circuit design requirements and energy accuracy.
"""

import json
import sys
import os
import numpy as np
from typing import Dict, Any, Tuple

def load_benchmark(benchmark_path: str = "lih_benchmark_corrected.json") -> Dict[str, Any]:
    """Load CORRECTED benchmark values from JSON file."""
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
    print("  The corrected benchmark file should be: lih_benchmark_corrected.json")
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

    # Check active space
    ralph_active_space = ralph_results.get("molecule", {}).get("active_space")
    bench_active_space = benchmark["molecule"]["active_space"]

    if ralph_active_space is None:
        errors.append("Missing active space information")
    elif ralph_active_space != bench_active_space:
        errors.append(f"Active space mismatch: {ralph_active_space} vs {bench_active_space}")

    # Check orbital selection (flexible for PySCF-controlled orbitals)
    # First check for new field "selected_orbitals", then old field "orbital_selection"
    ralph_orbitals = ralph_results.get("molecule", {}).get("selected_orbitals")
    if ralph_orbitals is None:
        ralph_orbitals = ralph_results.get("molecule", {}).get("orbital_selection")

    bench_orbitals = benchmark["molecule"]["active_orbitals"]

    if ralph_orbitals is None:
        errors.append("Missing orbital selection information")
    elif sorted(ralph_orbitals) != sorted(bench_orbitals):
        # Only warning, not error, for PySCF-controlled orbitals
        # Check if fci_computation_method indicates PySCF
        fci_method = ralph_results.get("results", {}).get("fci_computation_method", "")
        if "pyscf" in fci_method.lower():
            # PySCF-controlled orbitals allowed to differ
            print(f"  ⚠️  Note: Orbital selection differs from benchmark (PySCF-controlled: {ralph_orbitals} vs benchmark: {bench_orbitals})")
        else:
            errors.append(f"Orbital selection mismatch: {ralph_orbitals} vs {bench_orbitals}")

    return len(errors) == 0, "; ".join(errors) if errors else "OK"

def validate_energies(ralph_results: Dict[str, Any], benchmark: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate energy calculations against FCI energy (PySCF-computed or benchmark)."""
    errors = []

    # Get energies
    ralph_results_data = ralph_results.get("results", {})
    bench_energies = benchmark["energies"]
    vqe_tolerance = benchmark["verification_tolerances"]["vqe_energy_tolerance_hartree"]

    # VQE final energy validation
    ralph_vqe = ralph_results_data.get("final_energy_hartree")

    # Determine FCI reference: use PySCF-computed if available, otherwise benchmark
    ralph_fci = ralph_results_data.get("fci_energy_hartree")
    fci_method = ralph_results_data.get("fci_computation_method", "")
    use_pyscf_fci = (ralph_fci is not None and "pyscf" in fci_method.lower())

    if use_pyscf_fci:
        bench_fci = ralph_fci
        fci_source = "PySCF-computed"
    else:
        bench_fci = bench_energies["fci_hartree"]
        fci_source = "benchmark"

    if ralph_vqe is None:
        errors.append("Missing VQE final energy")
    else:
        energy_diff = abs(ralph_vqe - bench_fci)
        if energy_diff > vqe_tolerance:
            errors.append(f"VQE energy outside tolerance: {energy_diff:.6f} Hartree > {vqe_tolerance} Hartree (1.6 mHa)")
            errors.append(f"VQE: {ralph_vqe:.6f} Hartree, FCI ({fci_source}): {bench_fci:.6f} Hartree")

    # Check if energy curve exists
    convergence_data = ralph_results.get("convergence_data", {})
    energy_curve = convergence_data.get("energy_curve")
    if energy_curve is None:
        errors.append("Missing energy convergence curve")
    elif not isinstance(energy_curve, list) or len(energy_curve) == 0:
        errors.append("Energy curve must be a non-empty list")
    else:
        # Check if curve shows decreasing trend (last energy should be lowest or near lowest)
        # Allow for some noise in optimization
        min_energy = min(energy_curve)
        last_energy = energy_curve[-1]
        if last_energy > min_energy + 0.001:  # Allow 1 mHa difference
            errors.append(f"Energy curve does not converge properly: last={last_energy:.6f}, min={min_energy:.6f}")

    return len(errors) == 0, "; ".join(errors) if errors else "OK"

def validate_circuit(ralph_results: Dict[str, Any], benchmark: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate circuit properties."""
    errors = []

    ralph_circuit = ralph_results.get("circuit", {})
    bench_qubits = benchmark["qubit_information"]["n_qubits_parity"]

    # Check circuit gates
    ralph_gates = ralph_circuit.get("gates")
    if ralph_gates is None:
        errors.append("Missing circuit gates")
    elif not isinstance(ralph_gates, list) or len(ralph_gates) == 0:
        errors.append("Circuit gates must be a non-empty list")

    # Check parameters
    ralph_params = ralph_circuit.get("parameters")
    if ralph_params is None:
        errors.append("Missing circuit parameters")
    elif not isinstance(ralph_params, list):
        errors.append("Circuit parameters must be a list")
    elif len(ralph_gates) > 0 and len(ralph_params) != len(ralph_gates):
        errors.append(f"Parameter count mismatch: {len(ralph_params)} parameters for {len(ralph_gates)} gates")

    # Check qubit count
    ralph_qubits = ralph_results.get("molecule", {}).get("n_qubits")
    if ralph_qubits is not None and ralph_qubits != bench_qubits:
        errors.append(f"Qubit count mismatch: {ralph_qubits} vs expected {bench_qubits}")

    return len(errors) == 0, "; ".join(errors) if errors else "OK"

def validate_implementation(ralph_results: Dict[str, Any], benchmark: Dict[str, Any]) -> Dict[str, Any]:
    """Run all validation checks."""
    validation = {
        "molecule": validate_molecule(ralph_results, benchmark),
        "energies": validate_energies(ralph_results, benchmark),
        "circuit": validate_circuit(ralph_results, benchmark),
        "overall": (False, "Pending")
    }

    # Overall validation
    all_passed = all(check[0] for check in [
        validation["molecule"],
        validation["energies"],
        validation["circuit"]
    ])

    validation["overall"] = (all_passed, "All checks passed" if all_passed else "Some checks failed")

    return validation

def print_validation_report(validation: Dict[str, Any], ralph_results: Dict[str, Any], benchmark: Dict[str, Any]):
    """Print human-readable validation report."""
    print("\n" + "="*60)
    print("LiH VQE Validation Report (Custom Circuit Test)")
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
    ralph_final = ralph_results.get("results", {}).get("final_energy_hartree", "N/A")
    if isinstance(ralph_final, (int, float)):
        print(f"    VQE:  {ralph_final:.8f}")
    else:
        print(f"    VQE:  {ralph_final}")

    # Determine FCI source
    ralph_fci = ralph_results.get("results", {}).get("fci_energy_hartree")
    fci_method = ralph_results.get("results", {}).get("fci_computation_method", "")
    use_pyscf_fci = (ralph_fci is not None and "pyscf" in fci_method.lower())

    if use_pyscf_fci:
        fci_energy = ralph_fci
        fci_source = "PySCF-computed"
    else:
        fci_energy = benchmark['energies']['fci_hartree']
        fci_source = "benchmark"

    print(f"    FCI ({fci_source}):  {fci_energy:.8f}")

    if isinstance(ralph_final, (int, float)):
        energy_diff = abs(ralph_final - fci_energy)
        print(f"    Difference: {energy_diff:.8f} Hartree ({energy_diff*1000:.3f} mHa)")
        tolerance = benchmark["verification_tolerances"]["vqe_energy_tolerance_hartree"]
        print(f"    Tolerance:  {tolerance:.6f} Hartree ({tolerance*1000:.2f} mHa)")
        if energy_diff <= tolerance:
            print(f"    ✅ Within chemical accuracy (1.6 mHa)")
        else:
            print(f"    ❌ Outside chemical accuracy")

    print("\n📐 Molecular setup:")
    ralph_mol = ralph_results.get('molecule', {})
    print(f"    Bond length: Ralph={ralph_mol.get('bond_length_angstrom', 'N/A')} Å, "
          f"Benchmark={benchmark['molecule']['bond_length_angstrom']} Å")
    print(f"    Active space: Ralph={ralph_mol.get('active_space', 'N/A')}, "
          f"Benchmark={benchmark['molecule']['active_space']}")
    # Show orbital selection (prefer selected_orbitals field)
    ralph_orbitals = ralph_mol.get('selected_orbitals', ralph_mol.get('orbital_selection', 'N/A'))
    print(f"    Orbitals: Ralph={ralph_orbitals}, "
          f"Benchmark={benchmark['molecule']['active_orbitals']}")

    # Circuit info
    ralph_circuit = ralph_results.get('circuit', {})
    print("\n🔌 Circuit info:")
    print(f"    Gates: {len(ralph_circuit.get('gates', []))} gates")
    print(f"    Parameters: {len(ralph_circuit.get('parameters', []))} parameters")
    print(f"    Qubits: {ralph_mol.get('n_qubits', 'N/A')}")

    # Convergence info
    conv_data = ralph_results.get('convergence_data', {})
    energy_curve = conv_data.get('energy_curve', [])
    if energy_curve and len(energy_curve) > 0:
        print(f"\n📈 Convergence:")
        print(f"    Iterations: {len(energy_curve)}")
        print(f"    Initial energy: {energy_curve[0]:.6f}")
        print(f"    Final energy: {energy_curve[-1]:.6f}")
        print(f"    Improvement: {energy_curve[0] - energy_curve[-1]:.6f}")

    print("\n💡 RECOMMENDATIONS:")
    if overall_passed:
        print("  ✅ Excellent! All validation checks passed.")
        print("  ✅ Your VQE implementation correctly computes LiH energy within chemical accuracy.")
        print("  ✅ Circuit design and optimization are successful.")
        print("  ⚠️  NOTE: This test requires MANUAL circuit design (not using pre-defined ansatz functions).")
        print("     Manual code review will verify circuit was designed from scratch.")
    else:
        print("  ❌ Some issues detected. Please check:")
        for check_name, (passed, message) in validation.items():
            if check_name != "overall" and not passed:
                print(f"    - {check_name.capitalize()}: {message}")
        print("\n  🔧 Suggestions:")
        print("    1. Verify LiH molecule definition (2.0 Å bond length, active_space=(2,3), appropriate orbital selection)")
        print("    2. Check parity transformation to ensure 4-qubit Hamiltonian")
        print("    3. MANUALLY DESIGN VQE circuit (do not use pre-defined ansatz functions)")
        print("    4. Ensure BFGS optimizer is properly configured")
        print("    5. Track energy curve during optimization")
        print("    6. Compare final VQE energy with FCI reference (PySCF-computed or benchmark)")

    print("\n" + "="*60)

def save_validation_summary(validation: Dict[str, Any], summary_path: str = "validation_summary.txt"):
    """Save validation summary to file."""
    with open(summary_path, 'w') as f:
        f.write("LiH VQE Validation Summary (Custom Circuit Test)\n")
        f.write("="*40 + "\n\n")

        overall_passed, overall_msg = validation["overall"]
        f.write(f"Overall Status: {'PASS' if overall_passed else 'FAIL'} - {overall_msg}\n\n")

        f.write("Detailed Results:\n")
        for check_name, (passed, message) in validation.items():
            if check_name == "overall":
                continue
            status = "PASS" if passed else "FAIL"
            f.write(f"  {check_name.upper():12} {status:6} - {message}\n")

        f.write("\nNOTE: This test requires MANUAL circuit design (not using pre-defined ansatz functions).\n")
        f.write("Manual code review is needed to verify circuit was designed from scratch.\n")

    print(f"\n📄 Validation summary saved to: {summary_path}")

def main():
    """Main validation function."""
    if len(sys.argv) != 2:
        print("Usage: python validate_lih_custom_circuit.py <ralph_results.json>")
        print("Example: python validate_lih_custom_circuit.py ../Ralph_Test_LiH_VQE/lih_results.json")
        print("\nNOTE: This validation uses FCI energy (PySCF-computed or benchmark)")
        print("      and requires MANUAL circuit design (not pre-defined ansatz).")
        sys.exit(1)

    results_path = sys.argv[1]

    print("🔍 Validating Ralph's LiH VQE Custom Circuit Implementation...")
    print("⚠️  This test uses FCI energy (PySCF-computed or benchmark)")
    print("⚠️  and requires MANUAL circuit design (no pre-defined ansatz functions)")

    # Load data
    benchmark = load_benchmark()
    ralph_results = load_ralph_results(results_path)

    print(f"✓ Loaded benchmark from: lih_benchmark_corrected.json")
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