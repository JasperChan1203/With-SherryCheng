"""Smoke tests for rlqas CLI (Task 002)."""
import json
import subprocess
import sys
import tempfile
import os
import pytest


def run_cli(*args, check=True):
    return subprocess.run(
        ["rlqas"] + list(args),
        capture_output=True, text=True, check=check
    )


def test_cli_help():
    result = run_cli("--help")
    assert "rlqas" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_cli_search_help():
    result = run_cli("search", "--help")
    assert "--molecule" in result.stdout


def test_cli_experiment_help():
    result = run_cli("experiment", "--help")
    assert "--config" in result.stdout


def test_cli_search_h2():
    result = run_cli(
        "search", "--molecule", "H2", "--bond-length", "0.74",
        "--ansatz", "UCC", "--agent", "ppo", "--episodes", "50"
    )
    assert result.returncode == 0
    assert "RLQAS Result" in result.stdout
    assert "H2" in result.stdout


def test_cli_search_with_output():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        result = run_cli(
            "search", "--molecule", "H2", "--bond-length", "0.74",
            "--ansatz", "UCC", "--agent", "ppo", "--episodes", "20",
            "--output", path
        )
        assert result.returncode == 0
        with open(path) as f:
            data = json.load(f)
        assert data["molecule"] == "H2"
        assert "energy_error_mha" in data
    finally:
        os.unlink(path)


def test_cli_experiment_from_yaml():
    yaml_content = """
molecule:
  formula: H2
  bond_length: 0.74
search:
  ansatz_type: UCC
rl:
  agent_type: ppo
  n_episodes: 20
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        yaml_path = f.name
    try:
        result = run_cli("experiment", "--config", yaml_path)
        assert result.returncode == 0
        assert "RLQAS Result" in result.stdout
    finally:
        os.unlink(yaml_path)


def test_cli_invalid_ansatz_exits_nonzero():
    result = subprocess.run(
        ["rlqas", "search", "--molecule", "H2", "--bond-length", "0.74",
         "--ansatz", "INVALID"],
        capture_output=True, text=True
    )
    assert result.returncode != 0
