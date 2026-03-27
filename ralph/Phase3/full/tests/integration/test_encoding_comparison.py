"""Integration tests comparing circuit encoding methods.

Runs H2 episodes with matrix/sparse/one_hot encoding under a fixed random-action
policy (so actions are identical regardless of encoding), then asserts that
energy values agree within 1e-6.  This validates that encoding selection does
not affect the underlying physics computation.
"""

import json
import os
import pytest
import numpy as np

from rlqas.phase1.molecule.processor import process_molecule


@pytest.fixture(scope="module")
def h2_mol():
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2),
        basis_set="sto-3g",
        transform="jordan_wigner",
    )


class TestEncodingMethodsProduceIdenticalEnergy:
    """All three encoding methods must produce identical energies for same actions."""

    def _run_random_episodes(self, mol, encoding_method: str, n_episodes: int, seed: int):
        """Run episodes using a fixed random-action policy (ignores state observation).

        Because actions are determined by np.random (seeded), they are identical
        across encoding methods.  Energies therefore depend only on the UCC
        computation, not on the encoding.
        """
        from rlqas.phase3.hybrid_search.environment import HybridSearchEnv
        from rlqas.phase3.hybrid_search.circuit_builder import HybridFusionStrategy

        env = HybridSearchEnv(
            mol,
            HybridFusionStrategy({"fusion_mode": "sequential"}),
            {
                "run_classical_opt": True,
                "complexity_penalty": 0.0,
                "max_depth": 6,
                "max_blocks": 4,
                "encoding_method": encoding_method,
            },
        )

        rng = np.random.default_rng(seed)
        best_energies = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            episode_done = False
            ep_best = None
            while not episode_done:
                action = int(rng.integers(0, env.action_space.n))
                obs, reward, terminated, truncated, info = env.step(action)
                episode_done = terminated or truncated
                energy = info.get("energy")
                if energy is not None and np.isfinite(energy):
                    if ep_best is None or energy < ep_best:
                        ep_best = energy
            if ep_best is not None:
                best_energies.append(ep_best)
        return best_energies

    def test_encoding_methods_produce_identical_energy(self, h2_mol):
        """matrix/sparse/one_hot encodings give same energies for same random actions."""
        from rlqas.phase3.encoding.encoder_factory import EncoderFactory

        # Verify all three encoders are importable and functional
        for method in ("matrix", "sparse", "one_hot"):
            enc = EncoderFactory.create(method)
            assert enc is not None, f"EncoderFactory failed to create '{method}' encoder"

        mol = h2_mol
        n_episodes = 30
        seed = 42

        energies_matrix = self._run_random_episodes(mol, "matrix", n_episodes, seed)
        energies_sparse = self._run_random_episodes(mol, "sparse", n_episodes, seed)
        energies_one_hot = self._run_random_episodes(mol, "one_hot", n_episodes, seed)

        # All methods must have produced valid energies
        assert len(energies_matrix) > 0
        assert len(energies_sparse) > 0
        assert len(energies_one_hot) > 0

        # Best energy across all episodes must agree within 1e-6
        # (independent of encoding because energy comes from ucc.energy(), not encoding)
        best_matrix = min(energies_matrix)
        best_sparse = min(energies_sparse)
        best_one_hot = min(energies_one_hot)

        assert abs(best_matrix - best_sparse) < 1e-6, (
            f"matrix vs sparse best_energy differs: "
            f"{best_matrix:.10f} vs {best_sparse:.10f}"
        )
        assert abs(best_matrix - best_one_hot) < 1e-6, (
            f"matrix vs one_hot best_energy differs: "
            f"{best_matrix:.10f} vs {best_one_hot:.10f}"
        )

        # Save encoding benchmark
        results_dir = "results/phase3_integration"
        os.makedirs(results_dir, exist_ok=True)
        bench_path = os.path.join(results_dir, "encoding_benchmark.json")
        with open(bench_path, "w") as fh:
            json.dump(
                {
                    "molecule": "H2",
                    "n_episodes": n_episodes,
                    "seed": seed,
                    "best_energy": {
                        "matrix": best_matrix,
                        "sparse": best_sparse,
                        "one_hot": best_one_hot,
                    },
                    "agreement_within_1e6": True,
                },
                fh,
                indent=2,
            )
        print(
            f"[PASS] Encoding agreement: matrix={best_matrix:.8f}, "
            f"sparse={best_sparse:.8f}, one_hot={best_one_hot:.8f}"
        )
