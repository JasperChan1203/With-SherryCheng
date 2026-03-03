#!/bin/bash
# SLURM interactive job script for Phase 1 Task 003
# Usage:
# 1. First request interactive resources: salloc --job-name=ralph-003-interactive --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=24:00:00 --partition=CPU
# 2. Then run this script: ./slurm_interactive.sh

echo "=== RLQAS Phase 1 Task 003 Interactive Testing ==="
echo "Start time: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=1

# Activate Python environment
PYTHON_PATH="python3"
if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    if conda activate llm 2>/dev/null; then
        echo "Conda environment activated: $(which python3)"
        PYTHON_PATH="$(which python3)"
    else
        echo "Warning: Failed to activate conda environment, using default python3"
        if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
            PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
            echo "Using direct path: $PYTHON_PATH"
        fi
    fi
else
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
        PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        echo "Using direct path: $PYTHON_PATH"
    fi
fi

# Check Python environment
echo "Python version: $($PYTHON_PATH --version)"
echo "Checking Task 003 dependencies..."

$PYTHON_PATH -c "
import sys
print(f'Python path: {sys.executable}')

# Check core dependencies
deps_ok = True

# Quantum chemistry dependencies (shared with Tasks 001/002)
try:
    import tencirchem
    print(f'✓ tencirchem version: {tencirchem.__version__}')
except ImportError as e:
    print(f'✗ tencirchem import error: {e}')
    deps_ok = False

try:
    import openfermion
    print(f'✓ openfermion version: {openfermion.__version__}')
except ImportError as e:
    print(f'✗ openfermion import error: {e}')
    deps_ok = False

try:
    import numpy
    print(f'✓ numpy version: {numpy.__version__}')
except ImportError as e:
    print(f'✗ numpy import error: {e}')
    deps_ok = False

# RL-specific dependencies
try:
    import stable_baselines3
    print(f'✓ stable-baselines3 version: {stable_baselines3.__version__}')
except ImportError as e:
    print(f'✗ stable-baselines3 import error: {e}')
    deps_ok = False

try:
    import gym
    print(f'✓ gym version: {gym.__version__}')
except ImportError as e:
    print(f'✗ gym import error: {e}')
    deps_ok = False

try:
    import torch
    print(f'✓ PyTorch version: {torch.__version__}')
except ImportError as e:
    print(f'⚠️  PyTorch import warning: {e}')
    # PyTorch is optional but recommended

# Optional: Check Task 001 import for UCC compatibility testing
import os
task001_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../001')
sys.path.append(task001_dir)

try:
    from src.modules.molecule_processor import MoleculeData, process_molecule
    print('✓ Successfully imported Task 001 modules (UCC compatibility available)')
except ImportError as e:
    print(f'⚠️  Task 001 module import warning: {e}')
    print(f'Task 001 path: {task001_dir}')
    print('  (UCC compatibility testing may be limited)')

if deps_ok:
    print('All dependency checks passed, ready to run Ralph')
else:
    print('Dependency check failed, please resolve dependency issues first')
    sys.exit(1)
"

# Check required files
echo "Checking required files..."
if [ ! -f "prd.json" ]; then
    echo "✗ prd.json not found"
    exit 1
else
    echo "✓ prd.json exists"
fi

if [ ! -f "CLAUDE.md" ]; then
    echo "✗ CLAUDE.md not found"
    exit 1
else
    echo "✓ CLAUDE.md exists"
fi

if [ ! -f "ralph.sh" ]; then
    echo "✗ ralph.sh not found"
    exit 1
else
    echo "✓ ralph.sh exists"
fi

echo "All checks passed, starting Ralph..."

# Run Ralph (can start with fewer iterations for testing)
echo "========================================"
echo "Running Ralph (10 iteration test)..."
echo "Task: RLQAS Phase 1 Task 003 - PPO RL Agent"
echo "Goal: Implement PPO agent using Stable-Baselines3"
echo "      with UCC compatibility helpers"
echo "========================================"

./ralph.sh --tool claude 10

echo "========================================"
echo "Ralph run completed"
echo "End time: $(date)"
echo ""
echo "Check these files for progress:"
echo "  - progress.txt - Ralph progress log"
echo "  - ralph_learning_log.txt - Learning process record"
echo "  - AGENTS.md - Knowledge base"
echo ""
echo "To continue running, execute: ./ralph.sh --tool claude 10"