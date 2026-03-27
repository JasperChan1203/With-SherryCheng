"""Tests for UCC environment (Phase 5 duplicate-action fix)."""
import pytest
from rlqas_chem.molecule import process_molecule
from rlqas_chem.search.ucc import UCCSearchEnv, UCCSearchController


def test_duplicate_terminates_episode():
    mol = process_molecule('H2', 0.74, 'UCC')
    env = UCCSearchEnv(mol, config={})
    obs, _ = env.reset()
    # First action
    obs, r, term, _, _ = env.step(0)
    assert not term, "First action should not terminate"
    # Duplicate action
    obs, r, term, _, _ = env.step(0)
    assert term is True, "Duplicate action must terminate episode"
    assert r == -1.0, "Duplicate action must give reward=-1.0"


def test_ent_coef_default():
    """Verify ent_coef default is 0.01 (Phase 5 Fix B)."""
    import pathlib
    ctrl_path = pathlib.Path(__file__).parent.parent / "src" / "rlqas_chem" / "search" / "ucc" / "controller.py"
    src = ctrl_path.read_text()
    assert 'ent_coef' in src and '0.01' in src, "ent_coef default must be 0.01"
    # Verify it's actually using 0.01 as the default
    import re
    match = re.search(r'"ent_coef",\s*([\d.]+)', src)
    assert match and float(match.group(1)) == 0.01, f"ent_coef default should be 0.01, got {match.group(1) if match else 'not found'}"


def test_env_reset():
    mol = process_molecule('H2', 0.74, 'UCC')
    env = UCCSearchEnv(mol, config={})
    obs, info = env.reset()
    assert obs is not None
    assert len(obs) > 0
