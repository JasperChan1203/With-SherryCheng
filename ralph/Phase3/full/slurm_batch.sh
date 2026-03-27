#!/bin/bash
#SBATCH --job-name=ralph-phase3-full
#SBATCH --output=slurm_logs/ralph_phase3_full_%j.out
#SBATCH --error=slurm_logs/ralph_phase3_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=96:00:00          # 96 hours for complete Phase 3 development
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Ralph Phase 3 Complete (7 Tasks) Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Add claude CLI to PATH (installed via VSCode extension)
CLAUDE_BIN="/curie-home/jpchen/.vscode-server/extensions/anthropic.claude-code-2.1.77-linux-x64/resources/native-binary"
if [ -f "$CLAUDE_BIN/claude" ]; then
  export PATH="$CLAUDE_BIN:$PATH"
  echo "claude binary: $(which claude)"
else
  echo "Warning: claude binary not found at $CLAUDE_BIN — ralph.sh may fail"
fi

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
    echo "Warning: conda not found at /software/devtools/anaconda3"
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
        PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        echo "Using direct path: $PYTHON_PATH"
    fi
fi
export PYTHON_PATH

echo "Python version: $($PYTHON_PATH --version)"
echo "Checking dependencies for RLQAS Phase 3 Complete Implementation..."

$PYTHON_PATH -c "
import sys
print('Python path:', sys.executable)

# Phase 1 dependency (CRITICAL)
try:
    import rlqas.phase1
    from rlqas.phase1.rl.base_agent import RLAgent
    from rlqas.phase1.molecule.processor import process_molecule
    from rlqas.phase1.search.environment import UCCSearchEnv
    print('✓ Phase 1 package available (RLAgent, process_molecule, UCCSearchEnv)')
except ImportError as e:
    print('✗ Phase 1 package not found:', str(e))
    print('  Install Phase 1 first: pip install -e ../../Phase1/006')
    sys.exit(1)

# Phase 2 dependency (CRITICAL)
try:
    import rlqas.phase2
    from rlqas.phase2.rl.agent_factory import AgentFactory
    from rlqas.phase2.hea_search.environment import HEASearchEnv
    from rlqas.phase2.experiment.manager import ExperimentManager
    from rlqas.phase2.adaptation.exploration_framework import ExplorationFramework
    print('✓ Phase 2 package available (AgentFactory, HEASearchEnv, ExperimentManager)')
except ImportError as e:
    print('✗ Phase 2 package not found:', str(e))
    print('  Install Phase 2 first: pip install -e ../../Phase2/full')
    sys.exit(1)

# Quantum chemistry dependencies
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

# RL dependencies
try:
    import gymnasium
    print('✓ gymnasium version:', gymnasium.__version__)
except ImportError as e:
    print('✗ gymnasium import failed:', str(e))
    sys.exit(1)

try:
    import stable_baselines3
    from stable_baselines3 import PPO, DQN, A2C
    print('✓ stable-baselines3 version:', stable_baselines3.__version__)
except ImportError as e:
    print('✗ stable-baselines3 import failed:', str(e))
    sys.exit(1)

try:
    import torch
    print('✓ PyTorch version:', torch.__version__)
    if torch.cuda.is_available():
        print('  CUDA available:', torch.cuda.get_device_name(0))
    else:
        print('  CUDA not available (CPU only)')
except ImportError as e:
    print('✗ PyTorch import failed:', str(e))
    sys.exit(1)

# Phase 3 additional dependencies
try:
    import psutil
    print('✓ psutil available (MemoryManager dependency)')
except ImportError as e:
    print('⚠ psutil not installed — MemoryManager will need: pip install psutil')

try:
    import yaml
    print('✓ PyYAML available (ExperimentManager config)')
except ImportError as e:
    print('✗ PyYAML import failed:', str(e))
    sys.exit(1)

try:
    import scipy
    print('✓ SciPy version:', scipy.__version__)
except ImportError as e:
    print('✗ SciPy import failed:', str(e))
    sys.exit(1)

try:
    import numpy as np
    print('✓ NumPy version:', np.__version__)
except ImportError as e:
    print('✗ NumPy import failed:', str(e))
    sys.exit(1)

print('All dependency checks completed.')
"

# Check required Ralph input files
echo "Checking for required Ralph input files..."
REQUIRED_FILES=("prd.json" "CLAUDE.md" "ralph.sh" "progress.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Required file exists: $file"
    else
        echo "✗ Missing required file: $file"
        if [ "$file" == "progress.txt" ]; then
            echo "# RLQAS Phase 3 Complete - Progress Log" > "$file"
            echo "Started: $(date)" >> "$file"
            echo "---" >> "$file"
            echo "✓ Created: $file"
        else
            echo "  Error: $file is required but not found."
            exit 1
        fi
    fi
done

# Check Phase 1 dependency
echo "Checking for Phase 1 dependency..."
if [ -d "../../Phase1/006" ]; then
    echo "✓ Phase 1 Task 006 directory exists"
else
    echo "✗ Phase 1 Task 006 directory not found at ../../Phase1/006"
    echo "  Phase 3 requires Phase 1 integrated package."
    exit 1
fi

# Check Phase 2 dependency
echo "Checking for Phase 2 dependency..."
if [ -d "../../Phase2/full" ]; then
    echo "✓ Phase 2 full directory exists"
    if [ -f "../../Phase2/full/src/rlqas/phase2/rl/agent_factory.py" ]; then
        echo "✓ Phase 2 AgentFactory found"
    else
        echo "⚠ Phase 2 AgentFactory not found — Phase 3 may not integrate correctly"
    fi
else
    echo "✗ Phase 2 full directory not found at ../../Phase2/full"
    echo "  Phase 3 requires Phase 2 complete package."
    exit 1
fi

mkdir -p slurm_logs results/phase3_integration benchmarks

echo ""
echo "Starting Ralph for RLQAS Phase 3 Complete Implementation at: $(date)"
echo "========================================"
echo "Maximum iterations: 70 (96 hour time limit)"
echo "Tasks to implement:"
echo "  001: Hybrid Circuit Builder (HybridFusionStrategy + HybridCircuitBuilder)"
echo "  002: Hybrid Search Environment (HybridSearchEnv + HybridRewardFunction)"
echo "  003: Hybrid Search Controller (integrated with Phase 2 ExperimentManager)"
echo "  004: Batch Evaluation & Performance Optimization (>=1.5x speedup target)"
echo "  005: Circuit Encoding Module (MatrixEncoder / SparseEncoder / OneHotEncoder)"
echo "  006: Phase 3 Integration Tests (BeH2 8-14q, H4 8q, H6 12q)"
echo "  007: Qubit Operator Extension (autonomous Tencirchem API investigation)"
echo ""
echo "Performance targets:"
echo "  - Chemical accuracy (<1.6 mHa) on BeH2 8q, 10q, 12q and H4 8q"
echo "  - H6 12q: relaxed threshold <5.0 mHa (strongly correlated)"
echo "  - Batch evaluation: >=1.5x throughput on 8-qubit circuits"
echo "========================================"

./ralph.sh

echo ""
echo "Ralph agent completed for Phase 3 Complete Implementation."
echo "End time: $(date)"
echo ""
echo "Validation steps:"
echo "  1. Run integration tests: python -m pytest tests/integration/ -v"
echo "  2. Check results: ls results/phase3_integration/"
echo "  3. Check benchmarks: cat benchmarks/ci_vector_benchmark_results.json"
echo "  4. Run full test suite: sbatch slurm_test_integration.sh"
