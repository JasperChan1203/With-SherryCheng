#!/bin/bash
#SBATCH --job-name=ralph-phase5-pack
#SBATCH --output=slurm_logs/ralph_phase5_pack_%j.out
#SBATCH --error=slurm_logs/ralph_phase5_pack_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Phase 5 Pack: RLQAS-CHEM Standalone Package ==="
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

# ── Verify Phase 1–4 source is available (read-only reference) ──────────────
$PYTHON_PATH -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/../../../Phase4/full/src')
import rlqas
print(f'Source packages OK: rlqas {getattr(rlqas, \"__version__\", \"installed\")}')
" || echo "Warning: Phase 1-4 source check failed — Ralph will install as needed"

# ── Run Ralph ───────────────────────────────────────────────────────────────
echo ""
echo "Starting Ralph for Phase 5 Pack at: $(date)"
echo "  US-001: Package scaffold"
echo "  US-002: Molecule + simulator modules"
echo "  US-003: RL agent modules (PPO/DQN/A2C/SAC + factory)"
echo "  US-004: UCC search with Phase5 fixes baked in"
echo "  US-005: HEA search module"
echo "  US-006: Hybrid search with REINFORCE"
echo "  US-007: Unified API + CLI"
echo "  US-008: Tests, examples, LiH benchmark"
echo ""

chmod +x "$SCRIPT_DIR/ralph.sh"
"$SCRIPT_DIR/ralph.sh"
EXIT_CODE=$?

echo ""
echo "=== Ralph completed at: $(date) ==="
echo "Exit code: $EXIT_CODE"

# ── Post-run verification ────────────────────────────────────────────────────
echo ""
echo "=== Post-run Verification ==="

PACKAGE_DIR="$SCRIPT_DIR/rlqas-chem"

echo ""
echo "Package installation check:"
if [ -f "$PACKAGE_DIR/pyproject.toml" ]; then
  $PYTHON_PATH -m pip install -e "$PACKAGE_DIR" -q 2>&1 | tail -3
  $PYTHON_PATH -c "import rlqas_chem; print(f'  rlqas_chem version: {rlqas_chem.__version__}')" || echo "  Import failed"
else
  echo "  rlqas-chem package not found at $PACKAGE_DIR"
fi

echo ""
echo "rlqas-chem unit tests:"
if [ -d "$PACKAGE_DIR/tests" ]; then
  cd "$PACKAGE_DIR" && $PYTHON_PATH -m pytest tests/ -x -q 2>&1 | tail -5
else
  echo "  No tests directory found"
fi

echo ""
echo "LiH benchmark (rlqas_chem):"
$PYTHON_PATH -c "
import rlqas_chem
r = rlqas_chem.search('LiH', 1.6, ansatz_type='UCC', agent_type='ppo',
                       active_space=(2,5), n_episodes=300)
status = 'PASS' if r['energy_error_mha'] < 1.6 else 'FAIL'
print(f'  [{status}] energy_error={r[\"energy_error_mha\"]:.3f} mHa  n_operators={r[\"n_operators\"]}')
" 2>&1 || echo "  LiH benchmark failed to run"

exit $EXIT_CODE
