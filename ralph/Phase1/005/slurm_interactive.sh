#!/bin/bash
# SLURM interactive job script for Phase 1 Task 005
# Usage:
# 1. First request interactive resources:
#    # For CPU debugging/testing:
#    salloc --job-name=ralph-005-interactive --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=24:00:00 --partition=CPU
#    # For GPU debugging (if available):
#    # salloc --job-name=ralph-005-gpu --nodes=1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 --mem=32G --time=12:00:00 --partition=4V100
# 2. Then run this script: ./slurm_interactive.sh

echo "=== RLQAS Phase 1 Task 005 Interactive Testing ==="
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
echo "Checking Task 005 dependencies..."

$PYTHON_PATH -c "
import sys
print(f'Python path: {sys.executable}')

# Check core dependencies
deps_ok = True

# Quantum chemistry dependencies (shared with Tasks 001-004)
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
    if torch.cuda.is_available():
        print(f'  CUDA available: {torch.cuda.device_count()} GPU(s)')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('  CUDA not available - will use CPU')
except ImportError as e:
    print(f'✗ PyTorch import failed: {e}')
    deps_ok = False

# Analysis dependencies (specific to Task 005)
try:
    import pandas
    print(f'✓ pandas version: {pandas.__version__}')
except ImportError as e:
    print(f'✗ pandas import failed: {e}')
    deps_ok = False

try:
    import matplotlib
    print(f'✓ matplotlib version: {matplotlib.__version__}')
except ImportError as e:
    print(f'✗ matplotlib import failed: {e}')
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

try:
    sys.path.append('../004')
    from src.modules.ucc_search.environment import UCCSearchEnv
    from src.modules.ucc_search.controller import UCCSearchController
    print('✓ Task 004 (UCC Search Module) import successful')
except ImportError as e:
    print(f'✗ Task 004 import failed: {e}')
    deps_ok = False

if not deps_ok:
    print('\\n✗ Some dependencies are missing. Please install them before continuing.')
    print('Required packages:')
    print('  - Quantum chemistry: tencirchem-ng, openfermion, pyscf')
    print('  - RL: stable-baselines3, gym, torch')
    print('  - Analysis: pandas, matplotlib')
    print('\\nAlso ensure Tasks 001, 002, 003, 004 are completed and importable.')
    sys.exit(1)
else:
    print('\\n✓ All dependencies check out!')
"

# Run tests if they exist
echo ""
echo "Checking for tests in Task 005..."
if [ -d "tests" ]; then
    echo "Running tests for Task 005..."
    $PYTHON_PATH -m pytest tests/ -v --tb=short
else
    echo "No tests directory found. Tests will be created during implementation."
fi

# Check for existing source code
if [ -d "src" ]; then
    echo ""
    echo "Source code structure:"
    find src -name "*.py" | sort
fi

if [ -d "scripts" ]; then
    echo ""
    echo "Scripts directory:"
    find scripts -name "*.py" | sort
fi

echo ""
echo "Interactive session ready for Task 005 development."
echo "You can now run:"
echo "  ./ralph.sh          # Start Ralph agent"
echo "  python scripts/validate_lih.py  # Run validation script (when implemented)"
echo ""
echo "Session started at: $(date)"