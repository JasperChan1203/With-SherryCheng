"""
Report generator for RLQAS validation.

Generates validation reports for RLQAS system.
"""

import json
import os
import datetime
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt


class ReportGenerator:
    """Generates validation reports for RLQAS system."""

    def __init__(self, metrics: Dict[str, Any], results: Dict[str, Any]):
        """Initialize report generator.

        Args:
            metrics: Collected performance metrics (from MetricsCollector)
            results: Validation results (from run_lih_validation)
        """
        self.metrics = metrics
        self.results = results
        self.report = {
            'header': {},
            'executive_summary': {},
            'test_configuration': {},
            'results_and_metrics': {},
            'analysis_and_conclusions': {},
            'recommendations': []
        }

    def check_chemical_accuracy(self) -> bool:
        """Check if chemical accuracy was achieved.

        Returns:
            True if chemical accuracy achieved (<1.6 mHa error)
        """
        validation_metrics = self.metrics.get('validation_metrics', {})
        if 'chemical_accuracy_achieved' in validation_metrics:
            return validation_metrics['chemical_accuracy_achieved']

        # Fallback: compute from energy metrics
        energy_metrics = self.metrics.get('energy_metrics', [])
        if energy_metrics:
            final_error = energy_metrics[-1].get('error_mha')
            if final_error is not None:
                return abs(final_error) < 1.6

        # Fallback to results
        results_metrics = self.results.get('metrics', {})
        if 'chemical_accuracy_achieved' in results_metrics:
            return results_metrics['chemical_accuracy_achieved']

        return False

    def generate_markdown_report(self, output_path: str = 'validation_report.md'):
        """Generate markdown validation report.

        Args:
            output_path: Path to save markdown report
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Build report sections
        report_lines = [
            "# RLQAS Phase 1 - LiH Validation Test Report",
            "",
            f"**Generated**: {datetime.datetime.now().isoformat()}",
            "",
            "## Executive Summary",
            "",
            self._generate_executive_summary(),
            "",
            "## Test Configuration",
            "",
            self._generate_test_configuration(),
            "",
            "## Results and Metrics",
            "",
            self._generate_results_and_metrics(),
            "",
            "## Analysis and Conclusions",
            "",
            self._generate_analysis_and_conclusions(),
            "",
            "## Recommendations",
            "",
            self._generate_recommendations(),
            "",
            "## Appendix",
            "",
            self._generate_appendix(),
            ""
        ]

        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Markdown report generated: {output_path}")

    def _generate_executive_summary(self) -> str:
        """Generate executive summary section."""
        success = self.results.get('success', False)
        chemical_accuracy = self.check_chemical_accuracy()
        total_time = self.results.get('total_time_seconds', 0)

        lines = [
            f"- **Validation Status**: {'SUCCESS' if success else 'FAILURE'}",
            f"- **Chemical Accuracy Achieved**: {'YES' if chemical_accuracy else 'NO'}",
            f"- **Total Time**: {total_time:.2f} seconds ({total_time/3600:.2f} hours)",
            f"- **System Integration**: {'All modules integrated successfully' if success else 'Integration issues detected'}",
            ""
        ]

        # Add key findings
        if success and chemical_accuracy:
            lines.append("### Key Findings")
            lines.append("- ✓ All Phase 1 modules work together correctly")
            lines.append("- ✓ System achieves chemical accuracy target (<1.6 mHa error)")
            lines.append("- ✓ RLQAS prototype demonstrates functional UCC search")
        elif success and not chemical_accuracy:
            lines.append("### Key Findings")
            lines.append("- ✓ All Phase 1 modules work together correctly")
            lines.append("- ✗ System does NOT achieve chemical accuracy target")
            lines.append("- ⚠ RLQAS prototype functional but accuracy insufficient")
        else:
            lines.append("### Key Findings")
            lines.append("- ✗ Validation test failed")
            lines.append("- ⚠ System integration issues or runtime errors")

        return '\n'.join(lines)

    def _generate_test_configuration(self) -> str:
        """Generate test configuration section."""
        config = self.results.get('configuration', {})
        molecule_info = self.results.get('molecule_info', {})

        lines = [
            "### Molecule Configuration",
            "```json",
            json.dumps({
                'molecule': config.get('molecule', 'LiH'),
                'bond_length': config.get('bond_length'),
                'active_space': config.get('active_space'),
                'basis_set': config.get('basis_set'),
                'transform': config.get('transform'),
                'n_qubits': self.results.get('n_qubits'),
                'fci_energy': self.results.get('fci_energy')
            }, indent=2),
            "```",
            "",
            "### Search Configuration",
            "```json",
            json.dumps({
                'n_episodes': config.get('n_episodes'),
                'early_stop_threshold': config.get('early_stop_threshold'),
                'max_depth': 12,  # from controller config
                'max_excitations': 15  # from controller config
            }, indent=2),
            "```",
            "",
            "### RL Agent Configuration",
            "```json",
            json.dumps({
                'agent_type': 'ppo',
                'policy_type': 'MlpPolicy',
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5
            }, indent=2),
            "```",
            "",
            "### Simulator Configuration",
            "```json",
            json.dumps({
                'max_memory_gb': 32,
                'engine': 'ci_vector',
                'fallback_method': 'statevector'
            }, indent=2),
            "```"
        ]

        return '\n'.join(lines)

    def _generate_results_and_metrics(self) -> str:
        """Generate results and metrics section."""
        metrics = self.results.get('metrics', {})
        search_results = self.results.get('search_results', {})

        # Energy results
        final_energy = metrics.get('final_energy')
        fci_energy = metrics.get('fci_energy')
        error_mha = metrics.get('error_mha')
        chemical_accuracy = metrics.get('chemical_accuracy_achieved', False)

        # Training results
        convergence = search_results.get('convergence_reached', False)
        episodes = metrics.get('episodes_completed', 0)
        circuit_depth = metrics.get('circuit_depth')
        n_excitations = metrics.get('n_excitations')
        n_parameters = metrics.get('n_parameters')

        lines = [
            "### Energy Results",
            f"- **Final VQE Energy**: {final_energy:.6f} Hartree" if final_energy is not None else "- **Final VQE Energy**: N/A",
            f"- **FCI Reference Energy**: {fci_energy:.6f} Hartree" if fci_energy is not None else "- **FCI Reference Energy**: N/A",
            f"- **Energy Error**: {error_mha:.2f} mHa" if error_mha is not None else "- **Energy Error**: N/A",
            f"- **Chemical Accuracy Target**: <1.6 mHa",
            f"- **Chemical Accuracy Achieved**: {'YES' if chemical_accuracy else 'NO'}",
            "",
            "### Search Performance",
            f"- **Convergence Reached**: {convergence}",
            f"- **Episodes Completed**: {episodes}",
            f"- **Final Reward**: {search_results.get('episode_rewards', [])[-1] if search_results.get('episode_rewards') else 'N/A'}",
            "",
            "### Circuit Characteristics",
            f"- **Circuit Depth**: {circuit_depth}" if circuit_depth is not None else "- **Circuit Depth**: N/A",
            f"- **Number of Excitations**: {n_excitations}" if n_excitations is not None else "- **Number of Excitations**: N/A",
            f"- **Number of Parameters**: {n_parameters}" if n_parameters is not None else "- **Number of Parameters**: N/A",
            "",
            "### Timing and Resources",
        ]

        # Add timing metrics if available
        timing_summary = self.metrics.get('summary', {}).get('timing', {})
        if timing_summary:
            lines.append(f"- **Total Time**: {timing_summary.get('total_time', 0):.2f} seconds")
            if 'stage_breakdown' in timing_summary:
                lines.append("- **Stage Breakdown**:")
                for stage, duration in timing_summary['stage_breakdown'].items():
                    lines.append(f"  - {stage}: {duration:.2f} seconds")

        # Add resource metrics if available
        resource_summary = self.metrics.get('summary', {}).get('resources', {})
        if resource_summary:
            lines.append("- **Resource Usage**:")
            if resource_summary.get('peak_memory_mb'):
                lines.append(f"  - Peak Memory: {resource_summary['peak_memory_mb']:.1f} MB")
            if resource_summary.get('average_cpu_percent'):
                lines.append(f"  - Average CPU: {resource_summary['average_cpu_percent']:.1f}%")

        return '\n'.join(lines)

    def _generate_analysis_and_conclusions(self) -> str:
        """Generate analysis and conclusions section."""
        success = self.results.get('success', False)
        chemical_accuracy = self.check_chemical_accuracy()
        convergence = self.results.get('search_results', {}).get('convergence_reached', False)
        total_time = self.results.get('total_time_seconds', 0)

        lines = ["### Analysis"]

        # Success analysis
        if success:
            lines.append("1. **System Integration**: All Phase 1 modules (Tasks 001-004) work together correctly.")
            lines.append("   - Molecule processing, simulator, RL agent, and UCC search environment integrate seamlessly.")
        else:
            lines.append("1. **System Integration**: Validation test failed due to errors.")
            errors = self.results.get('errors', [])
            if errors:
                lines.append(f"   - Errors encountered: {len(errors)}")
                for i, error in enumerate(errors[:3]):  # Show first 3 errors
                    lines.append(f"   - Error {i+1}: {error[:100]}...")
            else:
                lines.append("   - Failure reason unknown (check logs).")

        # Accuracy analysis
        if chemical_accuracy:
            lines.append("2. **Chemical Accuracy**: System achieves target accuracy (<1.6 mHa error).")
            lines.append("   - VQE energy is within chemical accuracy of FCI reference.")
            lines.append("   - UCC search successfully found circuit parameters approximating ground state.")
        else:
            lines.append("2. **Chemical Accuracy**: System does NOT achieve target accuracy.")
            lines.append("   - Possible reasons:")
            lines.append("     - Circuit may lack expressive power (insufficient excitations)")
            lines.append("     - Parameter optimization may be stuck in local minimum")
            lines.append("     - RL agent may need more training or better exploration")
            lines.append("     - FCI reference energy may be inaccurate")

        # Performance analysis
        if convergence:
            lines.append("3. **Convergence**: Search converged before reaching episode limit.")
            lines.append("   - Early stopping threshold was met.")
            lines.append("   - RL agent successfully learned to optimize circuit.")
        else:
            lines.append("3. **Convergence**: Search did NOT converge within episode limit.")
            lines.append("   - May need more episodes or adjusted convergence threshold.")
            lines.append("   - RL training may require hyperparameter tuning.")

        # Timing analysis
        if total_time < 7200:  # 2 hours
            lines.append("4. **Performance**: Validation completed within performance goal (<2 hours).")
            lines.append(f"   - Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours).")
        else:
            lines.append("4. **Performance**: Validation exceeded performance goal (>2 hours).")
            lines.append(f"   - Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours).")
            lines.append("   - Optimization opportunities exist in simulator or RL training.")

        lines.append("")
        lines.append("### Conclusions")

        if success and chemical_accuracy:
            lines.append("The RLQAS Phase 1 prototype **successfully meets all validation criteria**:")
            lines.append("1. ✓ System integration works correctly")
            lines.append("2. ✓ Chemical accuracy achieved")
            lines.append("3. ✓ Performance within reasonable bounds")
            lines.append("")
            lines.append("The system is ready for Phase 2 development (multi-algorithm support).")
        elif success and not chemical_accuracy:
            lines.append("The RLQAS Phase 1 prototype **partially meets validation criteria**:")
            lines.append("1. ✓ System integration works correctly")
            lines.append("2. ✗ Chemical accuracy NOT achieved")
            lines.append("3. ⚠ Performance may need improvement")
            lines.append("")
            lines.append("System is functional but requires accuracy improvements before Phase 2.")
        else:
            lines.append("The RLQAS Phase 1 prototype **fails validation criteria**.")
            lines.append("System requires debugging and fixes before proceeding.")

        return '\n'.join(lines)

    def _generate_recommendations(self) -> str:
        """Generate recommendations section."""
        success = self.results.get('success', False)
        chemical_accuracy = self.check_chemical_accuracy()
        convergence = self.results.get('search_results', {}).get('convergence_reached', False)
        total_time = self.results.get('total_time_seconds', 0)

        lines = []

        if not success:
            lines.append("1. **Debug System Integration**")
            lines.append("   - Examine error logs to identify failing module")
            lines.append("   - Run module health checks individually")
            lines.append("   - Verify dependency versions and compatibility")

        if not chemical_accuracy:
            lines.append("2. **Improve Chemical Accuracy**")
            lines.append("   - Increase maximum circuit depth in UCC search")
            lines.append("   - Allow more excitation operators")
            lines.append("   - Tune RL agent hyperparameters (learning rate, entropy coefficient)")
            lines.append("   - Verify FCI reference energy with independent calculation")

        if not convergence:
            lines.append("3. **Improve Convergence**")
            lines.append("   - Increase maximum episode count")
            lines.append("   - Adjust early stopping threshold")
            lines.append("   - Implement better reward shaping")
            lines.append("   - Add exploration incentives for RL agent")

        if total_time > 7200:  # >2 hours
            lines.append("4. **Optimize Performance**")
            lines.append("   - Profile time spent in each module")
            lines.append("   - Consider GPU acceleration for simulator")
            lines.append("   - Implement more efficient circuit evaluation")
            lines.append("   - Add checkpointing to resume interrupted runs")

        # General recommendations
        lines.append("5. **General Improvements**")
        lines.append("   - Upgrade from Gym to Gymnasium for NumPy 2.0 compatibility")
        lines.append("   - Add more comprehensive logging and monitoring")
        lines.append("   - Implement visualization tools for circuit analysis")
        lines.append("   - Create benchmark suite for systematic evaluation")

        if not lines:
            lines.append("1. **Proceed to Phase 2**")
            lines.append("   - System meets all Phase 1 validation criteria")
            lines.append("   - Begin implementation of multi-algorithm support (HEA, hybrid search)")
            lines.append("   - Expand validation to additional molecules and basis sets")

        return '\n'.join(lines)

    def _generate_appendix(self) -> str:
        """Generate appendix section."""
        lines = [
            "### Software Versions",
            "- Python: 3.8+",
            "- Tencirchem-ng: >=2024.10",
            "- OpenFermion: >=1.5",
            "- PySCF: >=2.0.0",
            "- Stable-Baselines3: >=2.0.0",
            "- PyTorch: >=1.9.0",
            "- Gym: >=0.21.0",
            "",
            "### Random Seeds",
            "- All random seeds set to 42 for reproducibility",
            "",
            "### Output Files",
            "- `validation_results.json`: Complete validation results",
            "- `metrics.json`: Detailed performance metrics",
            "- `energy_metrics.csv`, `training_metrics.csv`, etc.: CSV exports",
            "- `validation_report.md`: This report",
            "",
            "### References",
            "- RLQAS Phase 1 Specification (Sections 5.1, 6.1)",
            "- RLQAS Phase 1 Tasks Document",
            "- Tencirchem Documentation: https://tencirchem.readthedocs.io/",
            "- OpenFermion Documentation: https://quantumai.google/openfermion"
        ]

        return '\n'.join(lines)

    def generate_visualizations(self, output_dir: str):
        """Generate visualization plots.

        Args:
            output_dir: Directory to save visualization plots
        """
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Energy convergence plot
            energy_metrics = self.metrics.get('energy_metrics', [])
            if energy_metrics:
                iterations = [m['iteration'] for m in energy_metrics]
                energies = [m['energy'] for m in energy_metrics]
                fci_energy = energy_metrics[0]['fci_energy'] if energy_metrics else None

                plt.figure(figsize=(10, 6))
                plt.plot(iterations, energies, 'b-', label='VQE Energy')
                if fci_energy is not None:
                    plt.axhline(y=fci_energy, color='r', linestyle='--', label='FCI Energy')
                plt.xlabel('Iteration')
                plt.ylabel('Energy (Hartree)')
                plt.title('Energy Convergence')
                plt.legend()
                plt.grid(True)
                energy_plot_path = os.path.join(output_dir, 'energy_convergence.png')
                plt.savefig(energy_plot_path, dpi=150)
                plt.close()

            # Training rewards plot
            training_metrics = self.metrics.get('training_metrics', [])
            if training_metrics:
                episodes = [m['episode'] for m in training_metrics]
                rewards = [m['reward'] for m in training_metrics]

                plt.figure(figsize=(10, 6))
                plt.plot(episodes, rewards, 'g-', label='Episode Reward')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Training Rewards')
                plt.legend()
                plt.grid(True)
                reward_plot_path = os.path.join(output_dir, 'training_rewards.png')
                plt.savefig(reward_plot_path, dpi=150)
                plt.close()

            print(f"Visualizations saved to {output_dir}")

        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")