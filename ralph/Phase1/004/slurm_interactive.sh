#!/bin/bash
# SLURM interactive job script for Phase 1 Task 004
# Usage:
# 1. First request interactive resources: salloc --job-name=ralph-004-interactive --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=24:00:00 --partition=CPU
# 2. Then run this script: ./slurm_interactive.sh

echo "=== RLQAS Phase 1 Task 004 Interactive Testing ==="
echo "Start time: $(date)"
echo "Running on host: $(hostname)"
echo "Current directory: $(pwd)"

# Set Python to unbuffered mode for real-time output
export PYTHONUNBUFFERED=1

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
echo "Checking Task 004 dependencies..."

$PYTHON_PATH -c "
import sys
print(f'Python path: {sys.executable}')

# Check core dependencies
deps_ok = True

# Quantum chemistry dependencies (shared with Tasks 001/002/003)
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

# RL dependencies (from Task 003)
try:
    import stable_baselines3
    print(f'✓ stable-baselines3 version: {stable_baselines3.__version__}')
except ImportError as e:
    print(f'✗ stable-baselines3 import failed: {e}')
    deps_ok = False

try:
    import gym
    print(f'✓ gym version: {gym.__version__}')
except ImportError as e:
    print(f'✗ gym import failed: {e}')
    deps_ok = False

try:
    import torch
    print(f'✓ PyTorch version: {torch.__version__}')
except ImportError as e:
    print(f'✗ PyTorch import failed: {e}')
    deps_ok = False

# Check Task dependencies
print('\\nChecking Task dependencies...')
try:
    sys.path.append('../001')
    from src.modules.molecule_processor import MoleculeData, process_molecule
    print('✓ Task 001 (Molecule Processing) import successful')
except ImportError as e:
    print(f'✗ Task 001 import failed: {e}')
    deps_ok = False

try:
    sys.path.append('../002')
    from src.modules.quantum_simulator import QuantumSimulator, SimulatorFactory
    print('✓ Task 002 (Quantum Simulator) import successful')
except ImportError as e:
    print(f'✗ Task 002 import failed: {e}')
    deps_ok = False

try:
    sys.path.append('../003')
    from src.modules.rl_agents import RLAgent, PPOAgent
    print('✓ Task 003 (PPO RL Agent) import successful')
except ImportError as e:
    print(f'✗ Task 003 import failed: {e}')
    deps_ok = False

if not deps_ok:
    print('\\n✗ Some dependencies are missing. Please install them before continuing.')
    print('Required packages:')
    print('  - Quantum chemistry: tencirchem-ng, openfermion, pyscf')
    print('  - RL: stable-baselines3, gym, torch')
    print('\\nAlso ensure Tasks 001, 002, 003 are completed and importable.')
    sys.exit(1)
else:
    print('\\n✓ All dependencies check out!')
"

# Run tests
echo ""
echo "Running tests for Task 004..."
$PYTHON_PATH -m pytest tests/ -v --tb=short

# Check for src directory
if [ -d "src" ]; then
    echo ""
    echo "Source code structure:"
    find src -name "*.py" | sort
fi

echo ""
echo "Interactive session ready for Task 004 development."
echo "You can now run:"
echo "  ./ralph.sh          # Start Ralph agent"
echo "  python -m pytest tests/ -v  # Run tests"
echo ""
echo "Session started at: $(date)"