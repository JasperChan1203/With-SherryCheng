"""
Metrics collector for RLQAS validation.

Collects and analyzes performance metrics for RLQAS validation.
"""

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import time
import psutil


class MetricsCollector:
    """Collects and analyzes performance metrics for RLQAS validation."""

    def __init__(self, output_dir: str = 'results/lih_test_results'):
        """Initialize metrics collector.

        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = output_dir
        self.metrics = {
            'energy_metrics': [],
            'circuit_metrics': [],
            'training_metrics': [],
            'timing_metrics': [],
            'resource_metrics': [],
            'validation_metrics': {}
        }
        self.process = psutil.Process()
        self.start_time = time.time()

    def record_energy_metric(self, iteration: int, energy: float, fci_energy: float):
        """Record energy metric at specific iteration.

        Args:
            iteration: Iteration number
            energy: Current VQE energy
            fci_energy: Reference FCI energy
        """
        error_hartree = energy - fci_energy
        error_mha = error_hartree * 1000  # Convert to mHa
        self.metrics['energy_metrics'].append({
            'iteration': iteration,
            'energy': energy,
            'fci_energy': fci_energy,
            'error_hartree': error_hartree,
            'error_mha': error_mha,
            'chemical_accuracy_achieved': abs(error_mha) < 1.6,
            'timestamp': time.time() - self.start_time
        })

    def record_circuit_metric(self, circuit_depth: int, n_excitations: int,
                              n_parameters: int, iteration: Optional[int] = None):
        """Record circuit metric.

        Args:
            circuit_depth: Circuit depth
            n_excitations: Number of excitation operators
            n_parameters: Number of parameters
            iteration: Optional iteration number (defaults to next index)
        """
        if iteration is None:
            iteration = len(self.metrics['circuit_metrics'])
        self.metrics['circuit_metrics'].append({
            'iteration': iteration,
            'circuit_depth': circuit_depth,
            'n_excitations': n_excitations,
            'n_parameters': n_parameters,
            'timestamp': time.time() - self.start_time
        })

    def record_training_metric(self, episode: int, reward: float, energy: float,
                               depth: int, best_energy: float):
        """Record training metric for RL episode.

        Args:
            episode: Episode number
            reward: Episode reward
            energy: Episode energy
            depth: Circuit depth at episode end
            best_energy: Best energy found so far
        """
        self.metrics['training_metrics'].append({
            'episode': episode,
            'reward': reward,
            'energy': energy,
            'depth': depth,
            'best_energy': best_energy,
            'timestamp': time.time() - self.start_time
        })

    def record_timing_metric(self, stage: str, duration: float):
        """Record timing metric for a stage.

        Args:
            stage: Stage name (e.g., 'molecule_processing', 'rl_training')
            duration: Duration in seconds
        """
        self.metrics['timing_metrics'].append({
            'stage': stage,
            'duration': duration,
            'timestamp': time.time() - self.start_time
        })

    def record_resource_metric(self):
        """Record current resource usage."""
        memory_mb = self.process.memory_info().rss / 1024**2
        cpu_percent = self.process.cpu_percent()
        self.metrics['resource_metrics'].append({
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'timestamp': time.time() - self.start_time
        })

    def record_validation_metric(self, key: str, value: Any):
        """Record validation metric (e.g., final results).

        Args:
            key: Metric key
            value: Metric value
        """
        self.metrics['validation_metrics'][key] = value

    def save_metrics(self, filename: str = 'metrics.json'):
        """Save metrics to JSON file.

        Args:
            filename: Output filename
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        # Convert numpy types to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_for_json(item) for item in obj)
            else:
                return obj

        serializable_metrics = convert_for_json(self.metrics)
        with open(filepath, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")

    def load_metrics(self, filepath: str) -> Dict[str, Any]:
        """Load metrics from JSON file.

        Args:
            filepath: Path to metrics JSON file

        Returns:
            Loaded metrics dictionary
        """
        with open(filepath, 'r') as f:
            loaded_metrics = json.load(f)
        # Convert lists back to numpy arrays where appropriate
        # (simple approach, could be enhanced)
        self.metrics = loaded_metrics
        return loaded_metrics

    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics from collected metrics.

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        # Energy metrics summary
        energy_metrics = self.metrics['energy_metrics']
        if energy_metrics:
            energies = [m['energy'] for m in energy_metrics]
            errors_mha = [m['error_mha'] for m in energy_metrics]
            summary['energy'] = {
                'final_energy': energies[-1] if energies else None,
                'min_energy': min(energies) if energies else None,
                'max_energy': max(energies) if energies else None,
                'final_error_mha': errors_mha[-1] if errors_mha else None,
                'min_error_mha': min(errors_mha) if errors_mha else None,
                'max_error_mha': max(errors_mha) if errors_mha else None,
                'chemical_accuracy_achieved': any(m['chemical_accuracy_achieved'] for m in energy_metrics)
            }

        # Circuit metrics summary
        circuit_metrics = self.metrics['circuit_metrics']
        if circuit_metrics:
            depths = [m['circuit_depth'] for m in circuit_metrics]
            excitations = [m['n_excitations'] for m in circuit_metrics]
            parameters = [m['n_parameters'] for m in circuit_metrics]
            summary['circuit'] = {
                'final_depth': depths[-1] if depths else None,
                'max_depth': max(depths) if depths else None,
                'final_excitations': excitations[-1] if excitations else None,
                'max_excitations': max(excitations) if excitations else None,
                'final_parameters': parameters[-1] if parameters else None,
                'max_parameters': max(parameters) if parameters else None,
            }

        # Training metrics summary
        training_metrics = self.metrics['training_metrics']
        if training_metrics:
            rewards = [m['reward'] for m in training_metrics]
            energies = [m['energy'] for m in training_metrics]
            best_energies = [m['best_energy'] for m in training_metrics]
            summary['training'] = {
                'episodes_completed': len(training_metrics),
                'final_reward': rewards[-1] if rewards else None,
                'average_reward': np.mean(rewards) if rewards else None,
                'final_energy': energies[-1] if energies else None,
                'final_best_energy': best_energies[-1] if best_energies else None,
                'convergence_episode': self._find_convergence_episode(best_energies) if best_energies else None
            }

        # Timing metrics summary
        timing_metrics = self.metrics['timing_metrics']
        if timing_metrics:
            stage_durations = {}
            for metric in timing_metrics:
                stage = metric['stage']
                duration = metric['duration']
                if stage not in stage_durations:
                    stage_durations[stage] = []
                stage_durations[stage].append(duration)
            summary['timing'] = {
                'total_time': time.time() - self.start_time,
                'stage_breakdown': {stage: sum(durs) for stage, durs in stage_durations.items()},
                'stage_average': {stage: np.mean(durs) for stage, durs in stage_durations.items()}
            }

        # Resource metrics summary
        resource_metrics = self.metrics['resource_metrics']
        if resource_metrics:
            memories = [m['memory_mb'] for m in resource_metrics]
            cpus = [m['cpu_percent'] for m in resource_metrics]
            summary['resources'] = {
                'peak_memory_mb': max(memories) if memories else None,
                'average_memory_mb': np.mean(memories) if memories else None,
                'peak_cpu_percent': max(cpus) if cpus else None,
                'average_cpu_percent': np.mean(cpus) if cpus else None
            }

        # Validation metrics
        summary['validation'] = self.metrics['validation_metrics']

        return summary

    def _find_convergence_episode(self, best_energies: List[float], threshold: float = 1e-4) -> Optional[int]:
        """Find episode where energy converged within threshold.

        Args:
            best_energies: List of best energies per episode
            threshold: Convergence threshold for relative change

        Returns:
            Episode number where convergence first achieved, or None
        """
        if len(best_energies) < 2:
            return None
        for i in range(1, len(best_energies)):
            if abs(best_energies[i] - best_energies[i-1]) < threshold:
                return i
        return None

    def to_dataframe(self, metric_type: str) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame for analysis.

        Args:
            metric_type: Type of metrics ('energy', 'circuit', 'training', 'timing', 'resource')

        Returns:
            pandas DataFrame with metrics
        """
        if metric_type == 'energy':
            return pd.DataFrame(self.metrics['energy_metrics'])
        elif metric_type == 'circuit':
            return pd.DataFrame(self.metrics['circuit_metrics'])
        elif metric_type == 'training':
            return pd.DataFrame(self.metrics['training_metrics'])
        elif metric_type == 'timing':
            return pd.DataFrame(self.metrics['timing_metrics'])
        elif metric_type == 'resource':
            return pd.DataFrame(self.metrics['resource_metrics'])
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

    def generate_csv_report(self, output_dir: Optional[str] = None):
        """Generate CSV reports for all metric types.

        Args:
            output_dir: Optional override output directory
        """
        if output_dir is None:
            output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        for metric_type in ['energy', 'circuit', 'training', 'timing', 'resource']:
            try:
                df = self.to_dataframe(metric_type)
                if not df.empty:
                    csv_path = os.path.join(output_dir, f'{metric_type}_metrics.csv')
                    df.to_csv(csv_path, index=False)
                    print(f"Saved {metric_type} metrics to {csv_path}")
            except Exception as e:
                print(f"Warning: Could not generate CSV for {metric_type}: {e}")