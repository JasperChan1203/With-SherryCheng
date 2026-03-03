#!/bin/bash
#SBATCH --job-name=ralph-phase1-001
#SBATCH --output=ralph_phase1_001_%j.out
#SBATCH --error=ralph_phase1_001_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --partition=CPU
#SBATCH --mail-type=NONE

echo "=== RLQAS Ralph Phase 1 Task 001 Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM job directory: $SLURM_SUBMIT_DIR"
echo "Current directory: $(pwd)"

# Print SLURM environment
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"

# Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
echo "Python unbuffered mode enabled (PYTHONUNBUFFERED=1)"

# Load necessary modules (adjust for your cluster)
# module load python/3.9
# module load cuda/12.4  # if GPU needed

# Setup Python environment
PYTHON_PATH="python3"
if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    if conda activate llm 2>/dev/null; then
        echo "Conda environment activated: $(which python3)"
        PYTHON_PATH="$(which python3)"
    else
        echo "Warning: Failed to activate conda environment, using default python3"
        # Try alternative: direct path to conda environment
        if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
            PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
            echo "Using direct path: $PYTHON_PATH"
        fi
    fi
else
    echo "Warning: conda not found at /software/devtools/anaconda3"
    # Try alternative: direct path to conda environment
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
        PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        echo "Using direct path: $PYTHON_PATH"
    fi
fi
export PYTHON_PATH

# Check Python environment
echo "Python version: $($PYTHON_PATH --version)"
echo "Checking key dependencies for RLQAS Phase 1 Task 001..."
$PYTHON_PATH -c "
import sys
import traceback
print(f'Python path: {sys.executable}')
try:
    import tencirchem
    print(f'tencirchem version: {tencirchem.__version__}')
    # Check for specific features needed for molecule processing
    from tencirchem import UCC
    print('✓ tencirchem import successful')
except ImportError as e:
    print(f'✗ tencirchem import error: {e}')
    traceback.print_exc()
try:
    import openfermion
    print(f'openfermion version: {openfermion.__version__}')
    from openfermion import QubitOperator
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
"

# Change to RLQAS directory if needed
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change to submit directory"; exit 1; }

# Go to Ralph test directory (already in Phase1/001 from submission)
# The job should be submitted from Phase1/001 directory
echo "Current working directory: $(pwd)"
echo "Checking directory contents:"
ls -la

# Check for required files
REQUIRED_FILES=("prd.json" "CLAUDE.md" "ralph.sh")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ Required file not found: $file"
        echo "  Please ensure all required files exist in the submission directory."
        exit 1
    else
        echo "✓ Required file exists: $file"
    fi
done

# Run Ralph with 20 iterations as requested
echo "Starting Ralph for RLQAS Phase 1 Task 001 at: $(date)"
echo "========================================"
echo "Running with maximum 20 iterations (72 hour time limit)"
./ralph.sh --tool claude 20
RALPH_EXIT_CODE=$?
echo "========================================"
echo "Ralph finished with exit code: $RALPH_EXIT_CODE"

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