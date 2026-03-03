#!/bin/bash
# SLURM interactive job script for Phase 1 Task 006
# Usage:
# 1. First request interactive resources:
#    # For CPU debugging/testing:
#    salloc --job-name=ralph-006-interactive --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=24:00:00 --partition=CPU
#    # For GPU debugging (if available):
#    # salloc --job-name=ralph-006-gpu --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=8 --mem=48G --time=12:00:00 --partition=4V100
# 2. Then run this script: ./slurm_interactive.sh

echo "=== RLQAS Phase 1 Task 006 Interactive Testing ==="
echo "Start time: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

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
echo "Checking Task 006 dependencies..."

$PYTHON_PATH -c "
import sys
print(f'Python path: {sys.executable}')

# Check core dependencies
deps_ok = True

# Quantum chemistry
try:
    import tencirchem
    print(f'✓ tencirchem version: {tencirchem.__version__}')
except ImportError as e:
    print(f'✗ tencirchem import failed: {e}')
    deps_ok = False

try:
    import openfermion
    print(f'✓ openfermion version: {openfermion.__version__}')
except ImportError as e:
    print(f'✗ openfermion import failed: {e}')
    deps_ok = False

try:
    import pyscf
    print(f'✓ PySCF version: {pyscf.__version__}')
except ImportError as e:
    print(f'✗ PySCF import failed: {e}')
    deps_ok = False

# RL dependencies
try:
    import gymnasium
    print(f'✓ gymnasium version: {gymnasium.__version__}')
except ImportError as e:
    print(f'✗ gymnasium import failed: {e}')
    print('  Note: Task 006 requires gymnasium >= 1.0.0')
    deps_ok = False

try:
    import stable_baselines3
    print(f'✓ stable-baselines3 version: {stable_baselines3.__version__}')
except ImportError as e:
    print(f'✗ stable-baselines3 import failed: {e}')
    deps_ok = False

try:
    import torch
    print(f'✓ PyTorch version: {torch.__version__}')
    if torch.cuda.is_available():
        print(f'  CUDA available: {torch.cuda.get_device_name(0)}')
    else:
        print('  CUDA not available (CPU only)')
except ImportError as e:
    print(f'✗ PyTorch import failed: {e}')
    deps_ok = False

# Additional dependencies
try:
    import tensorcircuit
    print(f'✓ tensorcircuit version: {tensorcircuit.__version__}')
except ImportError as e:
    print(f'✗ tensorcircuit import failed: {e}')
    deps_ok = False

try:
    import pandas
    print(f'✓ pandas version: {pandas.__version__}')
except ImportError as e:
    print(f'✗ pandas import failed: {e}')
    deps_ok = False

# Check if all dependencies are available
if not deps_ok:
    print('\\n⚠ Some dependencies are missing. Task 006 may not work correctly.')
    print('   Consider installing missing packages or using a different environment.')
else:
    print('\\n✓ All core dependencies available.')
"

echo ""
echo "Task 006 Status:"
echo "  - PRD file: $(ls -la prd.json 2>/dev/null | wc -l) (should be 1)"
echo "  - CLAUDE prompt: $(ls -la CLAUDE.md 2>/dev/null | wc -l) (should be 1)"
echo "  - Ralph script: $(ls -la ralph.sh 2>/dev/null | wc -l) (should be 1)"
echo "  - Progress file: $(ls -la progress.txt 2>/dev/null | wc -l) (should be 1)"

echo ""
echo "Available commands:"
echo "  1. Run Ralph: ./ralph.sh"
echo "  2. Run specific number of iterations: ./ralph.sh 10"
echo "  3. Test package installation: pip install -e ."
echo "  4. Run quick health check: python -c \"import sys; sys.path.append('..'); exec(open('scripts/health_check.py').read())\""
echo "  5. Check progress: tail -f progress.txt"

echo ""
echo "Note: Task 006 is an integration task. It will create a unified package structure"
echo "in src/rlqas/phase1/. Check the progress.txt file for current status."

# If we're in an interactive session, show prompt
if [ -n "$SLURM_JOB_ID" ]; then
    echo ""
    echo "You are in SLURM interactive session. Job ID: $SLURM_JOB_ID"
    echo "To exit and release resources, type 'exit'"
fi