#!/bin/bash
# SLURM interactive job script for LiH VQE Ralph testing
# Run with: salloc --job-name=ralph-lih-vqe --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=6:00:00 --partition=CPU

echo "=== LiH VQE Ralph Test Interactive Job ==="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# Check if benchmark exists, generate if needed
if [ ! -f "lih_benchmark.json" ]; then
    echo "Benchmark file not found. Generating benchmark..."
    python generate_lih_benchmark.py
    if [ ! -f "lih_benchmark.json" ]; then
        echo "✗ Failed to generate benchmark file"
        exit 1
    fi
    echo "✓ Benchmark generated"
else
    echo "✓ Benchmark file exists"
fi

# Check and navigate to Ralph test directory
RALPH_TEST_DIR="Ralph_Test_LiH_VQE"
if [ -f "$RALPH_TEST_DIR/prd.json" ]; then
    # In lih_test directory, go to test directory
    cd "$RALPH_TEST_DIR" || { echo "Failed to enter test directory"; exit 1; }
    echo "✓ Entered Ralph test directory: $(pwd)"
elif [ -f "prd.json" ]; then
    # Already in test directory
    echo "✓ Already in Ralph test directory"
else
    echo "✗ prd.json not found. Please run from lih_test directory or Ralph_Test_LiH_VQE directory."
    echo "  Current directory contents:"
    ls -la
    exit 1
fi

# Check Python and dependencies for LiH VQE test
echo "Python version: $(python3 --version)"
echo "Checking LiH VQE test dependencies..."
python3 -c "
import sys
try:
    import pyscf
    print(f'✓ PySCF version: {pyscf.__version__}')
except ImportError as e:
    print(f'✗ PySCF import error: {e}')
    print('  Install with: pip install pyscf')
    sys.exit(1)
try:
    import tencirchem
    print(f'✓ tencirchem version: {tencirchem.__version__}')
except ImportError as e:
    print(f'✗ tencirchem import error: {e}')
    print('  Install with: pip install tencirchem-ng')
    sys.exit(1)
try:
    import numpy
    print(f'✓ NumPy version: {numpy.__version__}')
except ImportError as e:
    print(f'✗ NumPy import error: {e}')
    sys.exit(1)
try:
    import scipy
    print(f'✓ SciPy version: {scipy.__version__}')
except ImportError as e:
    print(f'✗ SciPy import error: {e}')
    sys.exit(1)
print('✓ All required dependencies are available for LiH VQE test')
" || exit 1

# Run Ralph with specified parameters
echo "Starting Ralph for LiH VQE (claude tool, max 10 iterations)..."
./ralph.sh --tool claude 10

echo "Job completed at: $(date)"
echo "========================================"
echo "Check validation results in validation_summary.txt"
echo "Check Ralph progress in progress.txt and ralph_learning_log.txt"
echo "Check circuit and energy results in lih_results.json"