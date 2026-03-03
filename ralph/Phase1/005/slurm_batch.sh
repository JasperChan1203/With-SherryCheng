#!/bin/bash
#SBATCH --job-name=ralph-phase1-005
#SBATCH --output=ralph_phase1_005_%j.out
#SBATCH --error=ralph_phase1_005_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G                # RL训练需要更多内存
#SBATCH --time=48:00:00          # 48小时，完整验证可能耗时
#SBATCH --partition=4V100        # 根据集群调整
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Ralph Phase 1 Task 005 Batch Job ==="
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
echo "Checking key dependencies for RLQAS Phase 1 Task 005..."

# Check core dependencies
$PYTHON_PATH -c "
import sys
print('Python path:', sys.executable)

# Check quantum chemistry dependencies
try:
    import tencirchem
    print('✓ tencirchem version:', tencirchem.__version__)
except ImportError as e:
    print('✗ tencirchem import failed:', str(e))
    sys.exit(1)

try:
    import openfermion
    print('✓ openfermion version:', openfermion.__version__)
except ImportError as e:
    print('✗ openfermion import failed:', str(e))
    sys.exit(1)

try:
    import pyscf
    print('✓ PySCF version:', pyscf.__version__)
except ImportError as e:
    print('✗ PySCF import failed:', str(e))
    sys.exit(1)

# Check RL dependencies
try:
    import stable_baselines3
    print('✓ stable-baselines3 version:', stable_baselines3.__version__)
except ImportError as e:
    print('✗ stable-baselines3 import failed:', str(e))
    sys.exit(1)

try:
    import gym
    print('✓ gym version:', gym.__version__)
except ImportError as e:
    print('✗ gym import failed:', str(e))
    sys.exit(1)

try:
    import torch
    print('✓ PyTorch version:', torch.__version__, '(GPU acceleration optional)')
except ImportError as e:
    print('✗ PyTorch import failed:', str(e))
    sys.exit(1)

# Check analysis dependencies (specific to Task 005)
try:
    import pandas
    print('✓ pandas version:', pandas.__version__)
except ImportError as e:
    print('✗ pandas import failed:', str(e))
    sys.exit(1)

try:
    import matplotlib
    print('✓ matplotlib version:', matplotlib.__version__)
except ImportError as e:
    print('✗ matplotlib import failed:', str(e))
    sys.exit(1)

# Check Task dependencies
print('Checking Task 001 module import (for molecule data)...')
sys.path.append('../001')
try:
    from src.modules.molecule_processor import MoleculeData, process_molecule
    print('✓ Task 001 module import successful')
except ImportError as e:
    print('✗ Task 001 import failed:', str(e))
    print('  Note: Task 001 must be completed before running Task 005')
    sys.exit(1)

print('Checking Task 002 module import (for quantum simulator)...')
sys.path.append('../002')
try:
    from src.modules.quantum_simulator import QuantumSimulator, SimulatorFactory
    print('✓ Task 002 module import successful')
except ImportError as e:
    print('✗ Task 002 import failed:', str(e))
    print('  Note: Task 002 must be completed before running Task 005')
    sys.exit(1)

print('Checking Task 003 module import (for RL agent)...')
sys.path.append('../003')
try:
    from src.modules.rl_agents import RLAgent, PPOAgent
    print('✓ Task 003 module import successful')
except ImportError as e:
    print('✗ Task 003 import failed:', str(e))
    print('  Note: Task 003 must be completed before running Task 005')
    sys.exit(1)

print('Checking Task 004 module import (for UCC search)...')
sys.path.append('../004')
try:
    from src.modules.ucc_search.controller import UCCSearchController
    from src.modules.ucc_search.environment import UCCSearchEnv
    print('✓ Task 004 module import successful')
except ImportError as e:
    print('✗ Task 004 import failed:', str(e))
    print('  Note: Task 004 must be completed before running Task 005')
    sys.exit(1)

print('All dependency checks completed.')
"

# Check for required files
echo "Checking for required Ralph input files..."
REQUIRED_FILES=("prd.json" "CLAUDE.md" "ralph.sh" "progress.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Required file exists: $file"
    else
        echo "✗ Missing required file: $file"
        exit 1
    fi
done

# Create output directory for logs
mkdir -p slurm_logs
if [ -f "ralph_phase1_005_${SLURM_JOB_ID}.out" ]; then
    mv ralph_phase1_005_${SLURM_JOB_ID}.out slurm_logs/
fi
if [ -f "ralph_phase1_005_${SLURM_JOB_ID}.err" ]; then
    mv ralph_phase1_005_${SLURM_JOB_ID}.err slurm_logs/
fi

# Start Ralph
echo "Starting Ralph for RLQAS Phase 1 Task 005 at: $(date)"
echo "========================================"
echo "Maximum iterations: 20 (48 hour time limit)"
echo "Task: RLQAS Phase 1 Task 005 - LiH Validation Test"
echo "Ralph will implement:"
echo "  - Complete validation script with run_lih_validation() function"
echo "  - Performance metrics collection and analysis"
echo "  - Validation criteria implementation"
echo "  - Comprehensive test report generation"
echo "  - Basic evaluation tools"
echo "  - Integration testing of all Phase 1 modules"
echo "  - Chemical accuracy verification (<1.6 mHa error)"
echo "  - Performance monitoring (<2 hours goal)"
echo "========================================"

# Run Ralph
./ralph.sh

echo ""
echo "Ralph agent completed for Task 005."
echo "End time: $(date)"

# Check if validation script exists and run it
if [ -f "scripts/validate_lih.py" ]; then
    echo ""
    echo "Running validation script (if implemented)..."
    $PYTHON_PATH scripts/validate_lih.py --output_dir="results/lih_test_results"
fi

echo "Job completed at: $(date)"