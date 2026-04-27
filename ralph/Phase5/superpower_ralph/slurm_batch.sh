#!/bin/bash
#SBATCH --job-name=ralph-phase6-innovations
#SBATCH --output=slurm_logs/ralph_phase6_%j.out
#SBATCH --error=slurm_logs/ralph_phase6_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Phase 6: Innovation Extensions ==="
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

# ── Verify rlqas-chem is installed ──────────────────────────────────────────
$PYTHON_PATH -m pip install -e "$SCRIPT_DIR/../pack/rlqas-chem" -q 2>&1 | tail -2
$PYTHON_PATH -c "import rlqas_chem; print(f'rlqas_chem {rlqas_chem.__version__} ready')" \
  || { echo "ERROR: rlqas_chem import failed"; exit 1; }

# ── Run Ralph ───────────────────────────────────────────────────────────────
echo ""
echo "Starting Ralph for Phase 6 at: $(date)"
echo "Stories:"
echo "  US-009: QOP operator pool integration (HIGH)"
echo "  US-010: GRPO agent implementation (HIGH)"
echo "  US-011: Multi-objective Pareto reward / alpha parameter (MEDIUM)"
echo "  US-012: Hyperparameter optimization with Optuna (MEDIUM)"
echo "  US-013: Experiment runners E1/E2/E3 (MEDIUM)"
echo "  US-014: Cross-geometry transfer experiment (LOW)"
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

PACKAGE_DIR="$SCRIPT_DIR/../pack/rlqas-chem"

echo "QOP pool check:"
$PYTHON_PATH -c "
from rlqas_chem.molecule import process_molecule
from rlqas_chem.search.qop import QubitOperatorPool
mol = process_molecule('LiH', 1.6, 'UCC', active_space=(2,5))
pool = QubitOperatorPool(mol)
size = pool.get_pool_size()
status = 'PASS' if size >= 2 else 'FAIL'
print(f'  [{status}] QOP pool size: {size}')
" 2>&1 || echo "  QOP check failed (US-009 may not be complete)"

echo "GRPO agent check:"
$PYTHON_PATH -c "
import rlqas_chem
r = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type='grpo', n_episodes=20)
status = 'PASS' if r['best_energy'] < -1.0 else 'FAIL'
print(f'  [{status}] GRPO best_energy={r[\"best_energy\"]:.4f} Ha')
" 2>&1 || echo "  GRPO check failed (US-010 may not be complete)"

echo "Alpha backward-compat check:"
$PYTHON_PATH -c "
from rlqas_chem.search.ucc.reward_function import UCCRewardFunction
rf_default = UCCRewardFunction()
rf_alpha1  = UCCRewardFunction({'alpha': 1.0})
e1 = rf_default.compute_reward(-1.1, 3)
e2 = rf_alpha1.compute_reward(-1.1, 3)
status = 'PASS' if abs(e1 - e2) < 1e-10 else 'FAIL'
print(f'  [{status}] alpha=1.0 compat: e1={e1:.6f} e2={e2:.6f}')
" 2>&1 || echo "  Alpha check failed (US-011 may not be complete)"

echo ""
echo "Progress log:"
[ -f "$SCRIPT_DIR/progress.txt" ] && tail -20 "$SCRIPT_DIR/progress.txt" || echo "  No progress.txt found"

exit $EXIT_CODE
