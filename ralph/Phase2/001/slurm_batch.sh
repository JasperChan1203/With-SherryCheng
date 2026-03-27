#!/bin/bash
#SBATCH --job-name=ralph-phase2-001
#SBATCH --output=ralph_phase2_001_%j.out
#SBATCH --error=ralph_phase2_001_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G                # Phase 2 tasks may need less memory than integration tasks
#SBATCH --time=48:00:00          # 48 hours for Phase 2 development
#SBATCH --partition=4V100        # 根据集群调整
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Ralph Phase 2 Task 001 Batch Job ==="
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
echo "Checking dependencies for RLQAS Phase 2 Task 001..."

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
    print('  Phase 2 Task 001 requires Phase 1 integrated package.')
    print('  Install Phase 1 first: pip install -e ../../Phase1/006')
    sys.exit(1)

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
        print('  DQN is required for Phase 2 Task 001')
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

# Check additional dependencies
try:
    import tensorcircuit
    print('✓ tensorcircuit version:', tensorcircuit.__version__)
except ImportError as e:
    print('✗ tensorcircuit import failed:', str(e))
    sys.exit(1)

try:
    import yaml
    print('✓ PyYAML available for configuration loading')
except ImportError as e:
    print('✗ PyYAML import failed:', str(e))
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
        exit 1
    fi
done

# Check that Phase 1 directory exists
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
    echo "  Phase 2 Task 001 requires Phase 1 integrated package."
    echo "  Make sure ../../Phase1/006 exists and is properly set up."
    exit 1
fi

# Create output directory for logs
mkdir -p slurm_logs
if [ -f "ralph_phase2_001_${SLURM_JOB_ID}.out" ]; then
    mv ralph_phase2_001_${SLURM_JOB_ID}.out slurm_logs/
fi
if [ -f "ralph_phase2_001_${SLURM_JOB_ID}.err" ]; then
    mv ralph_phase2_001_${SLURM_JOB_ID}.err slurm_logs/
fi

# Start Ralph
echo "Starting Ralph for RLQAS Phase 2 Task 001 at: $(date)"
echo "========================================"
echo "Maximum iterations: 25 (48 hour time limit)"
echo "Task: RLQAS Phase 2 Task 001 - Multi-RL Algorithm Support (DQN Implementation)"
echo "Ralph will implement:"
echo "  1. DQN agent implementation conforming to RLAgent interface"
echo "  2. Extended AgentFactory supporting both PPO and DQN agents"
echo "  3. Configuration system with DQN-specific hyperparameters"
echo "  4. Integration testing with Phase 1 components"
echo "  5. Unit tests with >90% coverage"
echo "  6. Documentation and examples"
echo ""
echo "Key dependencies:"
echo "  - Phase 1 integrated package (../../Phase1/006)"
echo "  - Stable-Baselines3 with DQN support"
echo "  - PyTorch (GPU optional)"
echo "  - Gymnasium environments"
echo "========================================"

# Run Ralph
./ralph.sh

echo ""
echo "Ralph agent completed for Phase 2 Task 001."
echo "End time: $(date)"
echo ""
echo "If Ralph completed successfully, check for:"
echo "  - src/rlqas/phase2/ directory structure"
echo "  - DQNAgent class implementation"
echo "  - Extended AgentFactory"
echo "  - Test suite with >90% coverage"
echo "  - Integration with Phase 1 components"
echo "  - progress.txt with detailed implementation notes"