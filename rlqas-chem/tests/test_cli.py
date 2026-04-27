"""Tests for rlqas-chem CLI."""
import subprocess
import sys


PYTHON = "/curie-home/jpchen/.conda/envs/llm/bin/python3"
RLQAS_CLI = "/curie-home/jpchen/.conda/envs/llm/bin/rlqas-chem"


def test_cli_search_exits_zero():
    result = subprocess.run(
        [RLQAS_CLI, "search",
         "--molecule", "H2",
         "--bond-length", "0.74",
         "--ansatz", "UCC",
         "--agent", "ppo",
         "--episodes", "20"],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    assert "Best energy" in result.stdout


def test_cli_help():
    result = subprocess.run(
        [RLQAS_CLI, "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0
    assert "search" in result.stdout
