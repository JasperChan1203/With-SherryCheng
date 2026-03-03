#!/bin/bash
#SBATCH --job-name=ralph-phase1-006
#SBATCH --output=ralph_phase1_006_%j.out
#SBATCH --error=ralph_phase1_006_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=48G                # 集成任务可能需要更多内存
#SBATCH --time=72:00:00          # 72小时，集成优化可能耗时较长
#SBATCH --partition=4V100        # 根据集群调整
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Ralph Phase 1 Task 006 Batch Job ==="
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
echo "Checking key dependencies for RLQAS Phase 1 Task 006..."

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

# Check RL dependencies (note: Task 006 requires gymnasium, not gym)
try:
    import gymnasium
    print('✓ gymnasium version:', gymnasium.__version__)
except ImportError as e:
    print('✗ gymnasium import failed:', str(e))
    print('  Note: Task 006 requires gymnasium >= 1.0.0')
    sys.exit(1)

try:
    import stable_baselines3
    print('✓ stable-baselines3 version:', stable_baselines3.__version__)
except ImportError as e:
    print('✗ stable-baselines3 import failed:', str(e))
    sys.exit(1)

try:
    import torch
    print('✓ PyTorch version:', torch.__version__, '(GPU acceleration optional)')
except ImportError as e:
    print('✗ PyTorch import failed:', str(e))
    sys.exit(1)

# Check analysis dependencies
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

# Check Task 006 specific dependencies
try:
    import tqdm
    print('✓ tqdm version:', tqdm.__version__)
except ImportError as e:
    print('✗ tqdm import failed:', str(e))
    sys.exit(1)

print('All dependency checks completed.')
"

# Check for required files
echo "Checking for required Ralph input files..."
REQUIRED_FILES=("prd.json" "CLAUDE.md" "ralph.sh" "progress.txt" "requirements.txt" "pyproject.toml")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Required file exists: $file"
    else
        echo "✗ Missing required file: $file"
        exit 1
    fi
done

# Check that Task 001-005 directories exist (for reference)
echo "Checking for Phase 1 Task directories..."
TASK_DIRS=("001" "002" "003" "004" "005")
for task_dir in "${TASK_DIRS[@]}"; do
    if [ -d "../$task_dir" ]; then
        echo "✓ Task $task_dir directory exists"
    else
        echo "⚠ Task $task_dir directory not found (may affect reference comparisons)"
    fi
done

# Create output directory for logs
mkdir -p slurm_logs
if [ -f "ralph_phase1_006_${SLURM_JOB_ID}.out" ]; then
    mv ralph_phase1_006_${SLURM_JOB_ID}.out slurm_logs/
fi
if [ -f "ralph_phase1_006_${SLURM_JOB_ID}.err" ]; then
    mv ralph_phase1_006_${SLURM_JOB_ID}.err slurm_logs/
fi

# Start Ralph
echo "Starting Ralph for RLQAS Phase 1 Task 006 at: $(date)"
echo "========================================"
echo "Maximum iterations: 30 (72 hour time limit)"
echo "Task: RLQAS Phase 1 Task 006 - Integration and Optimization"
echo "Ralph will implement:"
echo "  1. architecture-integration: Create unified package structure"
echo "  2. gym-migration: Migrate from Gym to Gymnasium"
echo "  3. chemical-accuracy-optimization: Achieve <1.6 mHa for LiH (2,3)"
echo "  4. performance-validation: Validate 8-qubit <500ms and other targets"
echo "  5. algorithm-optimizations: Improve reference state calculation"
echo "  6. documentation-examples: Provide comprehensive docs and examples"
echo "========================================"

# Run Ralph
./ralph.sh

echo ""
echo "Ralph agent completed for Task 006."
echo "End time: $(date)"

# Check if package can be installed and basic import works
echo ""
echo "Testing integrated package installation..."
if [ -f "pyproject.toml" ]; then
    echo "Attempting to install package in development mode..."
    pip install -e . > /tmp/install_log.txt 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ Package installed successfully"

        # Test basic import
        echo "Testing basic import..."
        $PYTHON_PATH -c "
import sys
try:
    import rlqas.phase1
    print('✓ rlqas.phase1 import successful')

    # Try to import key components
    from rlqas.phase1.molecule import process_molecule
    print('✓ process_molecule import successful')

    from rlqas.phase1.simulator import SimulatorFactory
    print('✓ SimulatorFactory import successful')

    from rlqas.phase1.rl import PPOAgent
    print('✓ PPOAgent import successful')

    from rlqas.phase1.search import UCCSearchController
    print('✓ UCCSearchController import successful')

    print('All key imports successful!')

except Exception as e:
    print('✗ Import test failed:', str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)
"
    else
        echo "⚠ Package installation had issues (check /tmp/install_log.txt)"
    fi
else
    echo "⚠ pyproject.toml not found, skipping package installation test"
fi

echo "Job completed at: $(date)"