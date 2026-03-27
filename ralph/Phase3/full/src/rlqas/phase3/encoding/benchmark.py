"""Encoding benchmark for comparing encoder performance."""
import time
import json
import os
import numpy as np
from typing import Dict, List
from .encoder_factory import EncoderFactory


class EncodingBenchmark:
    """Benchmarks different encoding methods for speed and output size."""

    def run(
        self,
        circuits: List,
        n_qubits: int,
        max_depth: int,
        n_repeats: int = 10,
    ) -> Dict[str, Dict]:
        """Benchmark all encoding methods.

        Args:
            circuits: List of circuit objects to encode
            n_qubits: Number of qubits
            max_depth: Maximum circuit depth
            n_repeats: Number of timing repetitions

        Returns:
            Per-encoder timing and output size::

                {
                  "matrix": {"mean_us": 12.3, "output_dim": 120},
                  "sparse": {"mean_us": 8.1,  "output_dim": 90},
                  "one_hot": {"mean_us": 25.0, "output_dim": 600}
                }
        """
        results = {}
        n_circuits = max(len(circuits), 1)

        for method in ("matrix", "sparse", "one_hot"):
            encoder = EncoderFactory.create(method)
            expected_dim = encoder.output_dim(n_qubits, max_depth)

            times_us = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                for circ in circuits:
                    encoder.encode(circ, n_qubits, max_depth)
                elapsed_us = (time.perf_counter() - t0) * 1e6 / n_circuits
                times_us.append(elapsed_us)

            results[method] = {
                "mean_us": float(np.mean(times_us)),
                "std_us": float(np.std(times_us)),
                "output_dim": expected_dim,
            }
        return results

    def save_results(self, results: Dict, path: str) -> None:
        """Save benchmark results to JSON.

        Args:
            results: Results dict from run()
            path: File path to save JSON
        """
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
