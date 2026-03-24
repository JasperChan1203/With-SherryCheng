"""Smoke tests for example scripts (Task 003)."""
import subprocess
import sys
import os
import pytest

EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "examples")


def run_example(script_name, timeout=300):
    path = os.path.join(EXAMPLES_DIR, script_name)
    return subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True, timeout=timeout
    )


def test_example_01_ucc_search_lih():
    result = run_example("01_ucc_search_lih.py")
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert "Best energy" in result.stdout


def test_example_02_hea_search_beh2():
    result = run_example("02_hea_search_beh2.py")
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert "Best energy" in result.stdout


def test_example_03_hybrid_search_beh2():
    result = run_example("03_hybrid_search_beh2.py")
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert "Best energy" in result.stdout


def test_example_04_multi_algorithm_comparison():
    result = run_example("04_multi_algorithm_comparison.py")
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"
    assert "ppo" in result.stdout or "Agent" in result.stdout
