#!/bin/bash
#SBATCH --job-name=ralph-phase5-fix002
#SBATCH --output=slurm_logs/ralph_phase5_fix002_%j.out
#SBATCH --error=slurm_logs/ralph_phase5_fix002_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Phase 5 Fix 002: Genuine RL Circuit Search ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }
echo "Working directory: $(pwd)"

mkdir -p slurm_logs

# ── Claude CLI ──────────────────────────────────────────────────────────────
CLAUDE_BIN="/curie-home/jpchen/.vscode-server/extensions/anthropic.claude-code-2.1.77-linux-x64/resources/native-binary"
if [ -f "$CLAUDE_BIN/claude" ]; then
  export PATH="$CLAUDE_BIN:$PATH"
  echo "claude binary: $(which claude)"
else
  echo "Warning: claude binary not found at $CLAUDE_BIN"
fi

# ── Python environment ──────────────────────────────────────────────────────
PYTHON_PATH="python3"
if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
  source /software/devtools/anaconda3/etc/profile.d/conda.sh
  if conda activate llm 2>/dev/null; then
    echo "Conda llm activated: $(which python3)"
    PYTHON_PATH="$(which python3)"
  else
    echo "Warning: conda activate failed"
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
      PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
    fi
  fi
fi
export PYTHON_PATH
echo "Python: $($PYTHON_PATH --version)"

# ── Verify all phases importable ────────────────────────────────────────────
$PYTHON_PATH -c "
import rlqas, rlqas.phase1, rlqas.phase2, rlqas.phase3
print('All phases importable — ready for fix002.')
" || {
  echo "Phases not importable. Installing Phase 4 unified package..."
  $PYTHON_PATH -m pip install -e "$SCRIPT_DIR/../../Phase4/full/" -q
}

# ── Run Ralph ───────────────────────────────────────────────────────────────
echo ""
echo "Starting Ralph for Phase 5 Fix 002 at: $(date)"
echo "  US-001: Fix Phase 1 MDP — duplicate action terminates episode"
echo "  US-002: Fix Phase 1 ent_coef=0.0 default"
echo "  US-003: Verify Phase 1 LiH chemical accuracy"
echo "  US-004: Implement REINFORCE in Phase 3 HybridSearchController"
echo ""

"$SCRIPT_DIR/ralph.sh"
EXIT_CODE=$?

echo ""
echo "=== Ralph completed at: $(date) ==="
echo "Exit code: $EXIT_CODE"

# ── Post-run verification ────────────────────────────────────────────────────
echo ""
echo "=== Post-run Verification ==="

echo ""
echo "Phase 1 unit tests:"
cd "$SCRIPT_DIR/../../Phase1/006" && $PYTHON_PATH -m pytest tests/ -x -q 2>&1 | tail -5

echo ""
echo "Phase 3 integration tests:"
cd "$SCRIPT_DIR/../../Phase3/full" && $PYTHON_PATH -m pytest tests/ -x -q 2>&1 | tail -5

echo ""
echo "LiH chemical accuracy check (Phase 1 PPO):"
$PYTHON_PATH -c "
import rlqas
r = rlqas.search('LiH', 1.6, ansatz_type='UCC', agent_type='ppo', n_episodes=300)
status = 'PASS' if r['energy_error_mha'] < 1.6 else 'FAIL'
print(f'  [{status}] energy_error={r[\"energy_error_mha\"]:.3f} mHa  n_operators={r[\"n_operators\"]}')
"

exit $EXIT_CODE
