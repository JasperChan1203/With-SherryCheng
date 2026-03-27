#!/bin/bash
#SBATCH --job-name=ralph-phase4-full
#SBATCH --output=slurm_logs/ralph_phase4_full_%j.out
#SBATCH --error=slurm_logs/ralph_phase4_full_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00          # 24 hours sufficient for 3 wrapping tasks
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Ralph Phase 4 (Internal Research Tool) Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"

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
        echo "Warning: Failed to activate conda environment"
        if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
            PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        fi
    fi
fi
export PYTHON_PATH

echo "Python: $($PYTHON_PATH --version)"

# Install unified package (Phase 4 setup.py pulls in Phase 1/2/3 via file:// deps)
echo "Installing unified rlqas package (Phase 1-4)..."
$PYTHON_PATH -m pip install -e . -q
echo "✓ pip install complete"

# Verify all phases are importable
$PYTHON_PATH -c "
import sys

for phase, mod in [
    ('Phase 1', 'rlqas.phase1'),
    ('Phase 2', 'rlqas.phase2'),
    ('Phase 3', 'rlqas.phase3'),
    ('Phase 4 (rlqas top-level)', 'rlqas'),
]:
    try:
        __import__(mod)
        print(f'✓ {phase} ({mod}) available')
    except ImportError as e:
        print(f'✗ {phase} not found: {e}')
        sys.exit(1)

print('All phases satisfied — unified package ready.')
"

mkdir -p slurm_logs results

echo ""
echo "Starting Ralph for RLQAS Phase 4 at: $(date)"
echo "========================================"
echo "Tasks:"
echo "  001: Top-level Python API (rlqas.search / rlqas.Experiment)"
echo "  002: CLI Entry Point (rlqas search / rlqas experiment)"
echo "  003: Example Scripts (examples/01-04)"
echo "========================================"

./ralph.sh

echo ""
echo "Ralph agent completed for Phase 4."
echo "End time: $(date)"
echo ""
echo "Verification steps:"
echo "  1. python -m pytest tests/smoke/ -v"
echo "  2. rlqas search --molecule H2 --bond-length 0.74 --ansatz UCC --agent ppo --episodes 50"
echo "  3. python examples/01_ucc_search_lih.py"
