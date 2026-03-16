#!/bin/bash
# SLURM interactive job script for H2 Ralph testing
# Run with: salloc --job-name=ralph-h2-test --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=2:00:00 --partition=CPU

echo "=== H2 Ralph Test Interactive Job ==="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# Check and navigate to Ralph test directory
RALPH_TEST_DIR="Ralph_Test_H2_Hamiltonian"
if [ -f "$RALPH_TEST_DIR/prd.json" ]; then
    # In RLQAS root directory, go to test directory
    cd "$RALPH_TEST_DIR" || { echo "Failed to enter test directory"; exit 1; }
    echo "✓ Entered Ralph test directory: $(pwd)"
elif [ -f "prd.json" ]; then
    # Already in test directory
    echo "✓ Already in Ralph test directory"
else
    echo "✗ prd.json not found. Please run from RLQAS directory or Ralph_Test_H2_Hamiltonian directory."
    echo "  Current directory contents:"
    ls -la
    exit 1
fi

# Check Python and dependencies for H2 test
echo "Python version: $(python3 --version)"
echo "Checking H2 test dependencies..."
python3 -c "
import sys
try:
    import pyscf
    print(f'✓ PySCF version: {pyscf.__version__}')
except ImportError as e:
    print(f'✗ PySCF import error: {e}')
    print('  Install with: pip install pyscf numpy')
    sys.exit(1)
try:
    import numpy
    print(f'✓ NumPy version: {numpy.__version__}')
except ImportError as e:
    print(f'✗ NumPy import error: {e}')
    sys.exit(1)
print('✓ All required dependencies are available')
" || exit 1

# Run Ralph with specified parameters
echo "Starting Ralph (claude tool, max 5 iterations)..."
./ralph.sh --tool claude 5

echo "Job completed at: $(date)"
echo "========================================"
echo "Check validation results in validation_summary.txt"
echo "Check Ralph progress in progress.txt and ralph_learning_log.txt"