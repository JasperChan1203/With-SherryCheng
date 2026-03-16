#!/bin/bash
# RLQAS Phase 2 Cluster Environment Check Script
# This script checks if the cluster environment is properly set up for Phase 2

echo "==============================================================="
echo " RLQAS Phase 2 Cluster Environment Check"
echo "==============================================================="
echo "Date: $(date)"
echo "User: $USER"
echo "Host: $(hostname)"
echo ""

# Check if we're on login node
echo "=== Node Type Check ==="
if [[ "$(hostname)" == *"login"* ]]; then
    echo "⚠ WARNING: Running on login node ($(hostname))"
    echo "  Login nodes are for light tasks only."
    echo "  Use salloc for interactive sessions or sbatch for batch jobs."
else
    echo "✓ Running on compute node ($(hostname))"
fi

# Check SLURM availability
echo ""
echo "=== SLURM Check ==="
if command -v sinfo &> /dev/null; then
    echo "✓ SLURM commands available"
    echo "  Available partitions:"
    sinfo --format="%P %.11l %.6D %.6t %N" | head -10
else
    echo "✗ SLURM not available or not in PATH"
fi

# Check Python environment
echo ""
echo "=== Python Environment Check ==="
if command -v python3 &> /dev/null; then
    PYTHON_PATH=$(which python3)
    echo "✓ Python3 found: $PYTHON_PATH"
    echo "  Python version: $(python3 --version)"

    # Check for conda
    if command -v conda &> /dev/null; then
        echo "✓ Conda available"
        if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
            source /software/devtools/anaconda3/etc/profile.d/conda.sh
            if conda activate llm 2>/dev/null; then
                echo "✓ Activated conda 'llm' environment"
                PYTHON_PATH=$(which python3)
                echo "  Python path in environment: $PYTHON_PATH"
                echo "  Python version: $(python3 --version)"
            else
                echo "⚠ Could not activate 'llm' conda environment"
            fi
        fi
    else
        echo "⚠ Conda not available"
    fi
else
    echo "✗ Python3 not found"
fi

# Check Phase 1 dependency
echo ""
echo "=== Phase 1 Dependency Check ==="
PHASE1_DIR="../../Phase1/006"
if [ -d "$PHASE1_DIR" ]; then
    echo "✓ Phase 1 directory exists: $PHASE1_DIR"

    # Check if Phase 1 is installed
    if $PYTHON_PATH -c "import rlqas.phase1; print('Phase 1 package import successful')" 2>/dev/null; then
        echo "✓ Phase 1 package importable"
    else
        echo "⚠ Phase 1 package not importable (may need installation)"
        echo "  To install: pip install -e $PHASE1_DIR"
    fi
else
    echo "✗ Phase 1 directory not found: $PHASE1_DIR"
    echo "  Phase 2 requires Phase 1 completion."
fi

# Check Phase 2 Task 001
echo ""
echo "=== Phase 2 Task 001 Check ==="
TASK001_DIR="../001"
if [ -d "$TASK001_DIR" ]; then
    echo "✓ Phase 2 Task 001 directory exists: $TASK001_DIR"

    if [ -f "$TASK001_DIR/src/rlqas/phase2/rl/dqn_agent.py" ]; then
        echo "✓ Task 001 DQN implementation exists"
    else
        echo "⚠ Task 001 DQN implementation not found"
    fi
else
    echo "⚠ Phase 2 Task 001 directory not found"
fi

# Check Phase 2 Full implementation directory
echo ""
echo "=== Phase 2 Full Implementation Check ==="
if [ -d "full" ]; then
    echo "✓ Phase 2 Full directory exists: full/"

    REQUIRED_FILES=("prd.json" "CLAUDE.md" "ralph.sh" "slurm_batch.sh")
    for file in "${REQUIRED_FILES[@]}"; do
        if [ -f "full/$file" ]; then
            echo "✓ Required file exists: full/$file"
        else
            echo "✗ Missing file: full/$file"
        fi
    done
else
    echo "✗ Phase 2 Full directory not found: full/"
fi

# Check Unified submission script
echo ""
echo "=== Unified Submission Script Check ==="
if [ -f "run_phase2_unified.sh" ]; then
    echo "✓ Unified submission script exists: run_phase2_unified.sh"
    if [ -x "run_phase2_unified.sh" ]; then
        echo "✓ Script is executable"
    else
        echo "⚠ Script is not executable (run: chmod +x run_phase2_unified.sh)"
    fi
else
    echo "✗ Unified submission script not found: run_phase2_unified.sh"
fi

# Check resource requirements
echo ""
echo "=== Resource Requirements Check ==="
echo "Phase 2 Full implementation requires:"
echo "  - CPU: Multi-core (quantum simulation + RL training)"
echo "  - GPU: Recommended (RL neural network acceleration)"
echo "  - Memory: 64GB (quantum statevectors + neural networks)"
echo "  - Time: 72 hours (all 6 Phase 2 tasks)"
echo "  - Disk space: ~1GB for code and results"

# Check disk space
echo ""
echo "=== Disk Space Check ==="
df -h . | head -2
echo "Current directory: $(pwd)"
echo "Available space: $(df -h . | tail -1 | awk '{print $4}')"

# Provide usage instructions
echo ""
echo "==============================================================="
echo " USAGE INSTRUCTIONS"
echo "==============================================================="
echo ""
echo "Option 1: Interactive Session (Recommended for development)"
echo "  salloc --job-name=ralph-phase2 --nodes=1 --ntasks=1 --gpus-per-task=1 \\"
echo "         --mem=64G --time=72:00:00 --partition=4V100"
echo "  cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2"
echo "  ./run_phase2_unified.sh"
echo ""
echo "Option 2: Batch Submission"
echo "  cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full"
echo "  sbatch slurm_batch.sh"
echo "  # Monitor with: squeue -u \$USER"
echo ""
echo "Option 3: Direct Ralph Execution (Only on compute nodes)"
echo "  cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full"
echo "  ./ralph.sh --tool claude 50"
echo ""
echo "==============================================================="
echo " Environment check completed."
echo " Fix any issues marked with ✗ before proceeding."
echo "==============================================================="