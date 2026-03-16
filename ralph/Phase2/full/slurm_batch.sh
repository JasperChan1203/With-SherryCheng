#!/bin/bash
#SBATCH --job-name=ralph-phase2-full
#SBATCH --output=ralph_phase2_full_%j.out
#SBATCH --error=ralph_phase2_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G                # Phase 2 full needs more memory (all 6 tasks)
#SBATCH --time=72:00:00          # 72 hours for complete Phase 2 development
#SBATCH --partition=4V100        # 根据集群调整
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Ralph Phase 2 Complete (All 6 Tasks) Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
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
    echo "Warning: conda not found at /software/devtools/anaconda3"
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
        PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        echo "Using direct path: $PYTHON_PATH"
    fi
fi
export PYTHON_PATH

# Check Python environment
echo "Python version: $($PYTHON_PATH --version)"
echo "Checking dependencies for RLQAS Phase 2 Complete Implementation..."

# Check core dependencies
$PYTHON_PATH -c "
import sys
print('Python path:', sys.executable)

# First, check Phase 1 dependency (CRITICAL for Phase 2)
try:
    import rlqas.phase1
    print('✓ Phase 1 package available')
    try:
        from rlqas.phase1.rl.base_agent import RLAgent
        print('✓ RLAgent interface available')
    except ImportError as e:
        print('✗ RLAgent import failed:', str(e))
        sys.exit(1)
except ImportError as e:
    print('✗ Phase 1 package not found:', str(e))
    print('  Phase 2 requires Phase 1 integrated package.')
    print('  Install Phase 1 first: pip install -e ../../Phase1/006')
    sys.exit(1)

# Check Phase 2 Task 001 dependency (DQN implementation)
try:
    # Try to import Phase 2 Task 001 components
    from rlqas.phase2.rl import DQNAgent, AgentFactory
    print('✓ Phase 2 Task 001 components available')
    print('  DQNAgent and AgentFactory found')
except ImportError as e:
    print('⚠ Phase 2 Task 001 components not found:', str(e))
    print('  Phase 2 complete implementation will integrate Task 001 code from ../001/')
    print('  If Task 001 not completed, the unified implementation will include it.')

# Check quantum chemistry dependencies (same as Phase 1)
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

# Check RL dependencies (note: Phase 2 requires gymnasium)
try:
    import gymnasium
    print('✓ gymnasium version:', gymnasium.__version__)
except ImportError as e:
    print('✗ gymnasium import failed:', str(e))
    print('  Note: Phase 2 requires gymnasium >= 1.0.0')
    sys.exit(1)

try:
    import stable_baselines3
    print('✓ stable-baselines3 version:', stable_baselines3.__version__)
    # Check DQN is available (specifically needed for Phase 2)
    try:
        from stable_baselines3 import DQN
        print('✓ DQN implementation available in stable-baselines3')
    except ImportError as e:
        print('✗ DQN import failed:', str(e))
        print('  DQN is required for Phase 2')
        sys.exit(1)
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

# Check additional dependencies for Phase 2 full
try:
    import yaml
    print('✓ PyYAML available for configuration loading (Task 004)')
except ImportError as e:
    print('✗ PyYAML import failed:', str(e))
    print('  Required for experiment configuration system (Task 004)')
    sys.exit(1)

try:
    import tensorcircuit
    print('✓ tensorcircuit version:', tensorcircuit.__version__)
except ImportError as e:
    print('✗ tensorcircuit import failed:', str(e))
    sys.exit(1)

try:
    import numpy as np
    print('✓ NumPy version:', np.__version__)
except ImportError as e:
    print('✗ NumPy import failed:', str(e))
    sys.exit(1)

try:
    import scipy
    print('✓ SciPy version:', scipy.__version__)
except ImportError as e:
    print('✗ SciPy import failed:', str(e))
    sys.exit(1)

print('All dependency checks completed.')
"

# Check for required Ralph input files
echo "Checking for required Ralph input files..."
REQUIRED_FILES=("prd.json" "CLAUDE.md" "ralph.sh" "progress.txt")
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Required file exists: $file"
    else
        echo "✗ Missing required file: $file"
        echo "Creating missing files..."
        if [ "$file" == "progress.txt" ]; then
            echo "# RLQAS Phase 2 Complete - Progress Log" > "$file"
            echo "# Task: Complete Phase 2 Implementation (All 6 tasks)" >> "$file"
            echo "" >> "$file"
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
    # Check Phase 1 has required files
    if [ -f "../../Phase1/006/prd.json" ] && [ -f "../../Phase1/006/CLAUDE.md" ]; then
        echo "✓ Phase 1 Task 006 has PRD and CLAUDE files"
    else
        echo "⚠ Phase 1 Task 006 missing some files (may affect Phase 2 development)"
    fi
else
    echo "✗ Phase 1 Task 006 directory not found"
    echo "  Phase 2 requires Phase 1 integrated package."
    echo "  Make sure ../../Phase1/006 exists and is properly set up."
    exit 1
fi

# Check Phase 2 Task 001 dependency (code should exist)
echo "Checking for Phase 2 Task 001 dependency..."
if [ -d "../001" ]; then
    echo "✓ Phase 2 Task 001 directory exists"
    if [ -f "../001/src/rlqas/phase2/rl/dqn_agent.py" ]; then
        echo "✓ Phase 2 Task 001 DQN implementation exists"
    else
        echo "⚠ Phase 2 Task 001 DQN implementation not found"
        echo "  Unified implementation will create it if needed."
    fi
else
    echo "✗ Phase 2 Task 001 directory not found"
    echo "  Phase 2 complete implementation will create Task 001 components."
fi

# Create output directory for logs
mkdir -p slurm_logs
if [ -f "ralph_phase2_full_${SLURM_JOB_ID}.out" ]; then
    mv ralph_phase2_full_${SLURM_JOB_ID}.out slurm_logs/
fi
if [ -f "ralph_phase2_full_${SLURM_JOB_ID}.err" ]; then
    mv ralph_phase2_full_${SLURM_JOB_ID}.err slurm_logs/
fi

# Start Ralph
echo "Starting Ralph for RLQAS Phase 2 Complete Implementation at: $(date)"
echo "========================================"
echo "Maximum iterations: 50 (72 hour time limit)"
echo "Task: RLQAS Phase 2 Complete (All 6 Tasks)"
echo "Ralph will implement in sequence:"
echo "  Phase 0: Task 001 Verification and Integration"
echo "  Phase 1: Task 002 - Sequential Testing Framework"
echo "  Phase 2: Task 003 - HEA Search Module"
echo "  Phase 3: Task 004 - Experiment Management System"
echo "  Phase 4: Task 005 - Agent Autonomous RL Exploration"
echo "  Phase 5: Task 006 - Phase 2 Integration Test"
echo ""
echo "Key dependencies:"
echo "  - Phase 1 integrated package (../../Phase1/006)"
echo "  - Phase 2 Task 001 DQN implementation (../001) - will be integrated"
echo "  - Stable-Baselines3 with DQN support"
echo "  - PyTorch (GPU recommended)"
echo "  - Gymnasium environments"
echo "  - PyYAML for configuration"
echo ""
echo "Expected deliverables:"
echo "  - Complete multi-algorithm quantum architecture search system"
echo "  - HEA support with configurable entanglement patterns"
echo "  - Experiment management with configuration files"
echo "  - Autonomous RL exploration framework (key innovation)"
echo "  - Comprehensive integration tests"
echo "  - >90% test coverage"
echo "========================================"

# Run Ralph
echo "Starting Ralph execution..."
./ralph.sh

echo ""
echo "Ralph agent completed for Phase 2 Complete Implementation."
echo "End time: $(date)"
echo ""
echo "If Ralph completed successfully, check for:"
echo "  - Complete Phase 2 package structure (src/rlqas/phase2/)"
echo "  - All 6 Phase 2 tasks implemented and integrated"
echo "  - HEA search module with entanglement patterns"
echo "  - Experiment management system"
echo "  - Autonomous RL exploration framework"
echo "  - Integration test results"
echo "  - progress.txt with detailed implementation notes"
echo ""
echo "Validation steps:"
echo "  1. Run integration tests: python -m pytest tests/integration/"
echo "  2. Test LiH 10-qubit chemical accuracy"
echo "  3. Test multi-algorithm comparison"
echo "  4. Verify HEA search functionality"