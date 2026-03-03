#!/bin/bash
#SBATCH --job-name=ralph-phase1-003
#SBATCH --output=ralph_phase1_003_%j.out
#SBATCH --error=ralph_phase1_003_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G                # RL训练通常需要较少内存
#SBATCH --time=72:00:00          # 48小时，RL训练可能耗时
#SBATCH --partition=4V100        # 根据集群调整
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Ralph Phase 1 Task 003 Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"

# Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Activate Python environment (reference path from 001/002 scripts)
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
    echo "Warning: conda not found at /software/devtools/anaconda3"
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
        PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        echo "Using direct path: $PYTHON_PATH"
    fi
fi
export PYTHON_PATH

# Check Python environment
echo "Python version: $($PYTHON_PATH --version)"
echo "Checking key dependencies for RLQAS Phase 1 Task 003..."
$PYTHON_PATH -c "
import sys
import traceback
print(f'Python path: {sys.executable}')

# Check quantum chemistry dependencies (shared with Tasks 001/002)
try:
    import tencirchem
    print(f'tencirchem version: {tencirchem.__version__}')
    print('✓ tencirchem import successful')
except ImportError as e:
    print(f'✗ tencirchem import error: {e}')
    traceback.print_exc()

try:
    import openfermion
    print(f'openfermion version: {openfermion.__version__}')
    print('✓ openfermion import successful')
except ImportError as e:
    print(f'✗ openfermion import error: {e}')
    traceback.print_exc()

try:
    import numpy
    print(f'numpy version: {numpy.__version__}')
except ImportError as e:
    print(f'✗ numpy import error: {e}')
    traceback.print_exc()

try:
    import scipy
    print(f'scipy version: {scipy.__version__}')
except ImportError as e:
    print(f'✗ scipy import error: {e}')
    traceback.print_exc()

# Check RL-specific dependencies
try:
    import stable_baselines3
    print(f'stable-baselines3 version: {stable_baselines3.__version__}')
    print('✓ stable-baselines3 import successful')
except ImportError as e:
    print(f'✗ stable-baselines3 import error: {e}')
    traceback.print_exc()

try:
    import gym
    print(f'gym version: {gym.__version__}')
    print('✓ gym import successful')
except ImportError as e:
    print(f'✗ gym import error: {e}')
    traceback.print_exc()

try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print('✓ PyTorch import successful (GPU acceleration optional)')
except ImportError as e:
    print(f'PyTorch import warning: {e} (optional dependency)')
"

# Check Task 001 dependency (optional, for UCC compatibility testing)
echo "Checking Task 001 module import (optional for UCC compatibility)..."
$PYTHON_PATH -c "
import sys
import os

# Add Task 001 path
task001_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../001')
sys.path.append(task001_dir)

print(f'Trying to import Task 001 modules from: {task001_dir}')
print(f'Task 001 directory contents:')
import os
if os.path.exists(task001_dir):
    for item in os.listdir(task001_dir):
        print(f'  - {item}')
else:
    print(f'  Task 001 directory not found: {task001_dir}')

try:
    from src.modules.molecule_processor import MoleculeData, process_molecule
    print('✓ Successfully imported Task 001 modules: MoleculeData, process_molecule')
    print('  (UCC compatibility testing will be available)')
except ImportError as e:
    print(f'⚠️  Cannot import Task 001 modules: {e}')
    print('  UCC compatibility testing may be limited')
    print('  This is acceptable for basic RL agent implementation')
"

# Check required files
REQUIRED_FILES=("prd.json" "CLAUDE.md" "ralph.sh")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ Required file not found: $file"
        exit 1
    else
        echo "✓ Required file exists: $file"
    fi
done

# Run Ralph (20 iterations, can be adjusted)
echo "Starting Ralph for RLQAS Phase 1 Task 003 at: $(date)"
echo "========================================"
echo "Maximum iterations: 20 (48 hour time limit)"
echo "Task: RLQAS Phase 1 Task 003 - PPO RL Agent"
echo "Ralph will implement:"
echo "  - RLAgent abstract base class"
echo "  - PPOAgent using Stable-Baselines3"
echo "  - UCC compatibility helper methods"
echo "  - Configuration management with defaults from RLQAS spec"
echo "  - >80% test coverage with fixed random seeds"
echo "========================================"
./ralph.sh --tool claude 20
RALPH_EXIT_CODE=$?
echo "========================================"
echo "Ralph exit code: $RALPH_EXIT_CODE"

# Check results
echo "Checking progress..."
if [ -f "progress.txt" ]; then
    echo "Last 30 lines of progress.txt:"
    tail -30 progress.txt
else
    echo "progress.txt not found"
fi

# Check learning log
if [ -f "ralph_learning_log.txt" ]; then
    echo ""
    echo "Last 20 lines of learning log:"
    tail -20 ralph_learning_log.txt
fi

# Check AGENTS.md for knowledge capture
if [ -f "AGENTS.md" ]; then
    echo ""
    echo "Knowledge base updated (AGENTS.md exists)"
fi

# Check for generated code files
echo ""
echo "Checking for generated implementation files:"
if [ -d "src" ]; then
    echo "✓ src/ directory exists"
    find src -name "*.py" -type f | head -10
else
    echo "✗ src/ directory not found"
fi

if [ -d "tests" ]; then
    echo "✓ tests/ directory exists"
    find tests -name "*.py" -type f | head -10
else
    echo "✗ tests/ directory not found"
fi

echo "Job completed at: $(date)"
exit $RALPH_EXIT_CODE