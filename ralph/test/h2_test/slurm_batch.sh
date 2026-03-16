#!/bin/bash
#SBATCH --job-name=ralph-h2-test
#SBATCH --output=ralph_h2_test_%j.out
#SBATCH --error=ralph_h2_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=2:00:00  # H2 test is quick, 2 hours is enough
#SBATCH --partition=CPU  # Change to your partition if needed
#SBATCH --mail-type=NONE  # Disable email notifications for test

echo "=== RLQAS Ralph Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM job directory: $SLURM_SUBMIT_DIR"
echo "Current directory: $(pwd)"

# Print SLURM environment
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"

# Load necessary modules
# module load python/3.9
# module load cuda/11.4  # if GPU needed

# Setup Python environment
# conda activate rlqas_env
# source venv/bin/activate

# Check Python environment
echo "Python version: $(python3 --version)"
echo "Checking key dependencies..."
python3 -c "
import sys
print(f'Python path: {sys.executable}')
try:
    import tencirchem
    print(f'tencirchem version: {tencirchem.__version__}')
except ImportError as e:
    print(f'tencirchem import error: {e}')
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
except ImportError as e:
    print(f'PyTorch import error: {e}')
try:
    import stable_baselines3
    print(f'stable-baselines3 version: {stable_baselines3.__version__}')
except ImportError as e:
    print(f'stable-baselines3 import error: {e}')
"

# Change to RLQAS directory if needed
cd "$SLURM_SUBMIT_DIR" || { echo "Failed to change to submit directory"; exit 1; }

# Go to Ralph test directory
RALPH_TEST_DIR="Ralph_Test_H2_Hamiltonian"
if [ ! -d "$RALPH_TEST_DIR" ]; then
    echo "✗ Ralph test directory not found: $RALPH_TEST_DIR"
    echo "  Current directory: $(pwd)"
    ls -la
    exit 1
fi

cd "$RALPH_TEST_DIR" || { echo "Failed to enter test directory"; exit 1; }
echo "✓ Entered Ralph test directory: $(pwd)"

# Run Ralph
echo "Starting Ralph at: $(date)"
echo "========================================"
./ralph.sh --tool claude 5
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

echo "Job completed at: $(date)"
exit $RALPH_EXIT_CODE