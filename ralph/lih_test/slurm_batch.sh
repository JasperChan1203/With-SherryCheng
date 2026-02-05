#!/bin/bash
#SBATCH --job-name=ralph-lih-vqe
#SBATCH --output=ralph_lih_vqe_%j.out
#SBATCH --error=ralph_lih_vqe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G  # LiH VQE may need more memory than H2
#SBATCH --time=24:00:00  # LiH VQE is more complex, allow 6 hours
#SBATCH --partition=CPU  # Change to your partition if needed
#SBATCH --mail-type=NONE  # Disable email notifications for test

echo "=== RLQAS Ralph LiH VQE Batch Job ==="
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

# Load necessary modules
# module load python/3.9
# module load cuda/11.4  # if GPU needed

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
echo "Checking key dependencies for LiH VQE..."
$PYTHON_PATH -c "
import sys
import traceback
print(f'Python path: {sys.executable}')
try:
    import qiskit
    print(f'qiskit version: {qiskit.__version__}')
    from qiskit.quantum_info import SparsePauliOp
    print('✓ SparsePauliOp import successful')
except ImportError as e:
    print(f'✗ qiskit import error: {e}')
    traceback.print_exc()
try:
    import pyscf
    print(f'PySCF version: {pyscf.__version__}')
except ImportError as e:
    print(f'✗ PySCF import error: {e}')
    traceback.print_exc()
try:
    import tencirchem
    print(f'tencirchem version: {tencirchem.__version__}')
    from tencirchem import UCC, parity
    print('✓ tencirchem import successful')
except ImportError as e:
    print(f'✗ tencirchem import error: {e}')
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
try:
    import openfermion
    print(f'openfermion version: {openfermion.__version__}')
except ImportError as e:
    print(f'✗ openfermion import error: {e}')
    traceback.print_exc()
"

# Change to RLQAS directory if needed
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change to submit directory"; exit 1; }

# Check if CORRECTED benchmark exists
if [ ! -f "lih_benchmark_corrected.json" ]; then
    echo "✗ CORRECTED benchmark file not found: lih_benchmark_corrected.json"
    echo "  This file contains the correct FCI energy (-7.860153 Hartree)"
    echo "  Please ensure the corrected benchmark file exists."
    exit 1
else
    echo "✓ CORRECTED benchmark file exists (FCI: -7.860153 Hartree)"
fi

# Go to Ralph test directory
RALPH_TEST_DIR="Ralph_Test_LiH_VQE"
if [ ! -d "$RALPH_TEST_DIR" ]; then
    echo "✗ Ralph test directory not found: $RALPH_TEST_DIR"
    echo "  Current directory: $(pwd)"
    ls -la
    exit 1
fi

cd "$RALPH_TEST_DIR" || { echo "Failed to enter test directory"; exit 1; }
echo "✓ Entered Ralph test directory: $(pwd)"

# Run Ralph
echo "Starting Ralph for LiH VQE at: $(date)"
echo "========================================"
./ralph.sh --tool claude 10
RALPH_EXIT_CODE=$?
echo "========================================"
echo "Ralph finished with exit code: $RALPH_EXIT_CODE"

# Check results
echo "Checking progress..."
if [ -f "progress.txt" ]; then
    echo "Last 20 lines of progress.txt:"
    tail -20 progress.txt
else
    echo "progress.txt not found"
fi

# Check validation summary
if [ -f "validation_summary.txt" ]; then
    echo ""
    echo "Validation summary:"
    cat validation_summary.txt
fi

echo "Job completed at: $(date)"
exit $RALPH_EXIT_CODE
