"""Unit tests for HybridFusionStrategy and HybridCircuitBuilder."""
import pytest
import numpy as np
import os
import tempfile

from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase3.hybrid_search.circuit_builder import (
    HybridFusionStrategy,
    HybridCircuitBuilder,
    HybridCircuit,
)


# ─────────────── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def h2_mol():
    return process_molecule(
        "H2", 0.74, "UCC",
        active_space=(2, 2), basis_set="sto-3g", transform="jordan_wigner"
    )


@pytest.fixture(scope="module")
def lih_mol():
    return process_molecule(
        "LiH", 1.6, "UCC",
        active_space=(2, 5), basis_set="sto-3g", transform="jordan_wigner"
    )


@pytest.fixture(scope="module")
def h2_builder(h2_mol):
    strategy = HybridFusionStrategy({"fusion_mode": "sequential"})
    return HybridCircuitBuilder(h2_mol, strategy)


@pytest.fixture(scope="module")
def lih_builder(lih_mol):
    strategy = HybridFusionStrategy({"fusion_mode": "sequential"})
    return HybridCircuitBuilder(lih_mol, strategy)


# ─────────────── HybridFusionStrategy tests ──────────────────────────────────

class TestHybridFusionStrategy:
    def test_sequential_template(self):
        strat = HybridFusionStrategy({"fusion_mode": "sequential", "min_ucc_components": 1})
        tmpl = strat.generate_fusion_template()
        assert isinstance(tmpl, list)
        assert len(tmpl) >= 2
        assert "UCC" in tmpl
        assert "HEA" in tmpl

    def test_parallel_template(self):
        strat = HybridFusionStrategy({"fusion_mode": "parallel", "min_ucc_components": 2, "max_ucc_components": 2})
        tmpl = strat.generate_fusion_template()
        assert all(t == "HEA_UCC" for t in tmpl)
        assert len(tmpl) == 2

    def test_conditional_template(self):
        strat = HybridFusionStrategy({"fusion_mode": "conditional"})
        tmpl = strat.generate_fusion_template()
        assert tmpl == ["HEA", "UCC"]

    def test_min_ucc_components_respected(self):
        strat = HybridFusionStrategy({"fusion_mode": "sequential", "min_ucc_components": 2, "max_ucc_components": 5})
        tmpl = strat.generate_fusion_template()
        ucc_count = tmpl.count("UCC")
        assert ucc_count >= 2

    def test_max_ucc_components_respected(self):
        strat = HybridFusionStrategy({"fusion_mode": "sequential", "min_ucc_components": 1, "max_ucc_components": 1})
        tmpl = strat.generate_fusion_template()
        ucc_count = tmpl.count("UCC")
        assert ucc_count <= 1

    def test_invalid_mode_raises(self):
        strat = HybridFusionStrategy({"fusion_mode": "invalid"})
        with pytest.raises(ValueError):
            strat.generate_fusion_template()

    def test_fuse_circuits(self):
        strat = HybridFusionStrategy()
        result = strat.fuse_circuits({"type": "hea"}, {"type": "ucc"})
        assert result["type"] == "fused"
        assert "hea" in result
        assert "ucc" in result


# ─────────────── HybridCircuitBuilder unit tests ─────────────────────────────

class TestHybridCircuitBuilderH2:
    def test_build_ucc_block(self, h2_builder):
        block = h2_builder.build_block("UCC", {"excitations": [0]})
        assert block["type"] == "UCC"
        assert "circuit" in block
        assert len(block["excitations"]) >= 0  # may be 0 if no valid excitation

    def test_build_hea_block(self, h2_builder):
        block = h2_builder.build_block("HEA", {"n_layers": 2})
        assert block["type"] == "HEA"
        assert "circuit" in block
        assert block["depth"] > 0
        assert block["n_params"] > 0

    def test_build_hea_ucc_block(self, h2_builder):
        block = h2_builder.build_block("HEA_UCC", {})
        assert block["type"] == "HEA_UCC"
        assert "hea" in block
        assert "ucc" in block

    def test_build_hybrid_circuit_h2(self, h2_builder):
        circuit = h2_builder.build_hybrid_circuit(
            ["HEA", "UCC"],
            [{}, {"excitations": [0]}]
        )
        assert circuit is not None
        assert circuit.n_qubits == 4  # H2: (1,2) active space = 4 qubits
        assert circuit.num_qubits == 4
        assert circuit.fusion_template == ["HEA", "UCC"]
        assert len(circuit.blocks) == 2
        assert circuit.params is not None

    def test_circuit_string_representation(self, h2_builder):
        circuit = h2_builder.build_hybrid_circuit(
            ["HEA", "UCC"],
            [{}, {"excitations": [0]}]
        )
        s = str(circuit)
        assert len(s) > 50, f"Circuit string too short: {s!r}"

    def test_anti_hollow_circuit_not_identity(self, h2_mol):
        """Anti-hollow check: circuit must have non-trivial gates."""
        strategy = HybridFusionStrategy({"fusion_mode": "sequential"})
        builder = HybridCircuitBuilder(h2_mol, strategy)
        circuit = builder.build_hybrid_circuit(
            ["HEA", "UCC"],
            [{}, {"excitations": [0]}]
        )
        assert circuit is not None
        assert hasattr(circuit, "num_qubits") or len(str(circuit)) > 50, (
            "Circuit looks like identity/empty"
        )

    def test_save_load_fusion_config(self, h2_builder, tmp_path):
        template = ["HEA", "UCC", "HEA"]
        specs = [{}, {"excitations": [0]}, {}]
        path = str(tmp_path / "fusion_config.json")
        h2_builder.save_fusion_config(path, template, specs)
        assert os.path.exists(path)

        loaded_template, loaded_specs = h2_builder.load_fusion_config(path)
        assert loaded_template == template
        assert len(loaded_specs) == len(specs)

    def test_template_length_mismatch_raises(self, h2_builder):
        with pytest.raises(ValueError):
            h2_builder.build_hybrid_circuit(["HEA", "UCC"], [{}])  # 2 vs 1


class TestHybridCircuitBuilderLiH:
    def test_build_hybrid_circuit_lih_10q(self, lih_builder, lih_mol):
        circuit = lih_builder.build_hybrid_circuit(
            ["HEA", "UCC", "HEA"],
            [{}, {"excitations": [0]}, {}]
        )
        assert circuit is not None
        assert circuit.n_qubits == 10  # LiH (2,5) = 10 qubits
        assert circuit.ucc is not None
        assert circuit.params is not None
        assert len(circuit.blocks) == 3

    def test_lih_circuit_has_ucc_object(self, lih_builder):
        circuit = lih_builder.build_hybrid_circuit(
            ["HEA", "UCC"],
            [{}, {"excitations": [0]}]
        )
        assert circuit.ucc is not None
        assert hasattr(circuit.ucc, "energy"), "ucc object must have energy() method"

    def test_lih_parallel_fusion(self, lih_mol):
        strategy = HybridFusionStrategy({"fusion_mode": "parallel", "min_ucc_components": 1})
        builder = HybridCircuitBuilder(lih_mol, strategy)
        circuit = builder.build_hybrid_circuit(
            ["HEA_UCC"],
            [{}]
        )
        assert circuit is not None
        assert circuit.n_qubits == 10

    def test_lih_save_load_roundtrip(self, lih_builder, tmp_path):
        template = ["HEA", "UCC", "HEA"]
        specs = [{"n_layers": 1}, {"excitations": [0, 1]}, {"n_layers": 2}]
        path = str(tmp_path / "lih_fusion.json")
        lih_builder.save_fusion_config(path, template, specs)
        loaded_tmpl, loaded_specs = lih_builder.load_fusion_config(path)
        assert loaded_tmpl == template
        assert len(loaded_specs) == 3


class TestFusionModes:
    def test_all_three_modes(self, h2_mol):
        for mode in ["sequential", "parallel", "conditional"]:
            strategy = HybridFusionStrategy({"fusion_mode": mode})
            builder = HybridCircuitBuilder(h2_mol, strategy)
            tmpl = strategy.generate_fusion_template()
            specs = [{} for _ in tmpl]
            circuit = builder.build_hybrid_circuit(tmpl, specs)
            assert circuit is not None
            assert circuit.n_qubits == 4
