#!/bin/bash
#SBATCH --job-name=ralph-phase5-bugfix001
#SBATCH --output=slurm_logs/ralph_phase5_bugfix001_%j.out
#SBATCH --error=slurm_logs/ralph_phase5_bugfix001_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Phase 5 Bugfix 001 Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Add claude CLI to PATH
CLAUDE_BIN="/curie-home/jpchen/.vscode-server/extensions/anthropic.claude-code-2.1.77-linux-x64/resources/native-binary"
if [ -f "$CLAUDE_BIN/claude" ]; then
  export PATH="$CLAUDE_BIN:$PATH"
  echo "claude binary: $(which claude)"
else
  echo "Warning: claude binary not found at $CLAUDE_BIN"
fi

# Activate conda llm environment
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

# Verify unified rlqas package is importable (Phase 4 must be installed)
$PYTHON_PATH -c "
import rlqas, rlqas.phase1, rlqas.phase2, rlqas.phase3
print('All phases importable — ready for bugfix.')
" || {
  echo "Phases not importable. Installing Phase 4 unified package..."
  $PYTHON_PATH -m pip install -e ../../Phase4/full/ -q
}

mkdir -p slurm_logs

echo ""
echo "Starting Ralph for Phase 5 Bugfix 001 at: $(date)"
echo "  US-001: Fix Phase 1 UCCSearchController PPO training"
echo "  US-002: Fix Phase 2 HEASearchController best_energy=inf"

./ralph.sh

echo ""
echo "Ralph agent completed."
echo "End time: $(date)"
echo ""
echo "Verification:"
echo "  1. python -m pytest ../../Phase1/006/tests/ -v"
echo "  2. python -m pytest ../../Phase2/full/tests/ -v"
echo "  3. python -c \"import rlqas; r=rlqas.search('LiH',1.6,n_episodes=300); print(r['n_operators'])\""
