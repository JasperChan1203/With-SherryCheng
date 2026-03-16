#!/bin/bash
# SLURM interactive job script for Phase 2 Task 001
# Usage:
# 1. First request interactive resources:
#    # For CPU debugging/testing:
#    salloc --job-name=ralph-phase2-001-interactive --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=12:00:00 --partition=CPU
#    # For GPU debugging (if available):
#    # salloc --job-name=ralph-phase2-001-gpu --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=8 --mem=48G --time=8:00:00 --partition=4V100
# 2. Then run this script: ./slurm_interactive.sh

echo "=== RLQAS Phase 2 Task 001 Interactive Testing ==="
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
echo "Checking Phase 2 Task 001 dependencies..."

$PYTHON_PATH -c "
import sys
print(f'Python path: {sys.executable}')

# First, check Phase 1 dependency
try:
    import rlqas.phase1
    print('✓ Phase 1 package available')
    try:
        from rlqas.phase1.rl.base_agent import RLAgent
        print('✓ RLAgent interface available')
    except ImportError as e:
        print(f'✗ RLAgent import failed: {e}')
except ImportError as e:
    print(f'✗ Phase 1 package not found: {e}')
    print('  Phase 2 Task 001 requires Phase 1 integrated package.')
    print('  Install Phase 1 first: pip install -e ../Phase1/006')

# Check core dependencies (same as Phase 1)
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
    print('  Note: Phase 2 requires gymnasium >= 1.0.0')
    deps_ok = False

try:
    import stable_baselines3
    print(f'✓ stable-baselines3 version: {stable_baselines3.__version__}')
    # Check DQN is available
    try:
        from stable_baselines3 import DQN
        print('✓ DQN implementation available in stable-baselines3')
    except ImportError as e:
        print(f'✗ DQN import failed: {e}')
        deps_ok = False
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
    import yaml
    print(f'✓ PyYAML available for configuration loading')
except ImportError as e:
    print(f'✗ PyYAML import failed: {e}')
    deps_ok = False

# Check if all dependencies are available
if not deps_ok:
    print('\\n⚠ Some dependencies are missing. Phase 2 Task 001 may not work correctly.')
    print('   Consider installing missing packages or using a different environment.')
else:
    print('\\n✓ All core dependencies available for Phase 2 Task 001.')
"

echo ""
echo "Phase 2 Task 001 Status:"
echo "  - PRD file: $(ls -la prd.json 2>/dev/null | wc -l) (should be 1)"
echo "  - CLAUDE prompt: $(ls -la CLAUDE.md 2>/dev/null | wc -l) (should be 1)"
echo "  - Ralph script: $(ls -la ralph.sh 2>/dev/null | wc -l) (should be 1)"
echo "  - Progress file: $(ls -la progress.txt 2>/dev/null | wc -l) (should be 1)"
echo "  - AGENTS.md: $(ls -la AGENTS.md 2>/dev/null | wc -l) (should be 0 initially, Ralph will create)"

echo ""
echo "Phase 2 Task 001 Overview:"
echo "  - Extends Phase 1 with DQN algorithm support"
echo "  - Creates DQNAgent implementing RLAgent interface"
echo "  - Extends AgentFactory to support both PPO and DQN"
echo "  - Maintains backward compatibility with Phase 1"
echo ""
echo "Available commands:"
echo "  1. Run Ralph: ./ralph.sh"
echo "  2. Run specific number of iterations: ./ralph.sh 10"
echo "  3. Test Phase 1 installation: python -c \"from rlqas.phase1.rl.ppo_agent import PPOAgent; print('Phase 1 OK')\""
echo "  4. Check progress: tail -f progress.txt"
echo "  5. View PRD summary: jq '.project, .description' prd.json"

echo ""
echo "Note: Phase 2 Task 001 builds directly upon Phase 1 integrated package."
echo "Make sure Phase 1 is properly installed before running Ralph."

# If we're in an interactive session, show prompt
if [ -n "$SLURM_JOB_ID" ]; then
    echo ""
    echo "You are in SLURM interactive session. Job ID: $SLURM_JOB_ID"
    echo "To exit and release resources, type 'exit'"
fi