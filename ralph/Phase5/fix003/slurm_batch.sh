#!/bin/bash
#SBATCH --job-name=ralph-p5-fix003
#SBATCH --output=slurm_logs/ralph_p5_fix003_%j.out
#SBATCH --error=slurm_logs/ralph_p5_fix003_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Phase 5 Fix 003: QOP + Hybrid GRPO + HEA Cleanup ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }
echo "Script dir: $SCRIPT_DIR"
echo "Repo root:  $REPO_ROOT"

mkdir -p slurm_logs

# ── Claude CLI ──────────────────────────────────────────────────────────────
CLAUDE_BIN="/curie-home/jpchen/.vscode-server/extensions/anthropic.claude-code-2.1.77-linux-x64/resources/native-binary"
if [ -f "$CLAUDE_BIN/claude" ]; then
  export PATH="$CLAUDE_BIN:$PATH"
  echo "claude binary: $(which claude)"
else
  echo "Warning: claude binary not found at $CLAUDE_BIN — checking PATH"
  which claude 2>/dev/null || echo "ERROR: claude not found in PATH"
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

# ── Verify rlqas_chem importable ────────────────────────────────────────────
$PYTHON_PATH -c "
import rlqas_chem
from rlqas_chem.search.ucc.circuit_builder import UCCCircuitBuilder
from rlqas_chem.search.hybrid.controller import HybridSearchController
from rlqas_chem.search.hea.controller import HEASearchController
print('rlqas_chem importable — ready for fix003.')
" || {
  echo "rlqas_chem not importable. Installing..."
  $PYTHON_PATH -m pip install -e "$REPO_ROOT/rlqas-chem/" -q
  $PYTHON_PATH -c "import rlqas_chem; print('Install succeeded.')"
}

# ── Pre-run state: show which acceptance tests currently fail ────────────────
echo ""
echo "=== Pre-run acceptance state (expect 3 failures) ==="
cd "$REPO_ROOT"
$PYTHON_PATH -m pytest rlqas_acceptance_system/rl_algorithms/test_acceptance_level1_5.py \
  -k 'qop_search_basic or qop_cnot or chemical_accuracy_qop' \
  --tb=no -q 2>&1 | tail -10 || true

# ── Run Ralph ───────────────────────────────────────────────────────────────
echo ""
echo "Starting Ralph for Phase 5 Fix 003 at: $(date)"
echo "  US-001: QOP — pass mode='qubit' to TenCirChem + cnot_count field"
echo "  US-002: Hybrid GRPO-family — add _run_grpo_loop"
echo "  US-003: Block Tree-GRPO from HEA + update EXISTING_AGENTS"
echo "  US-004: Verify all QOP acceptance tests pass"
echo ""

cd "$SCRIPT_DIR"
chmod +x ralph.sh
./ralph.sh
EXIT_CODE=$?

# ── Post-run verification ────────────────────────────────────────────────────
echo ""
echo "=== Post-run Verification ==="
cd "$REPO_ROOT"

echo ""
echo "QOP acceptance tests (all 4 must PASS):"
$PYTHON_PATH -m pytest rlqas_acceptance_system/rl_algorithms/test_acceptance_level1_5.py \
  -k 'qop' -v --tb=short 2>&1 | tail -20 || true

echo ""
echo "QOP quick sanity (H2, 20 episodes):"
$PYTHON_PATH -c "
import rlqas_chem
r_f = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', operator_pool='fop', n_episodes=5)
r_q = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', operator_pool='qop', n_episodes=20)
fop_c = r_f.get('cnot_count', 'MISSING')
qop_c = r_q.get('cnot_count', 'MISSING')
status = 'PASS' if (isinstance(fop_c, int) and isinstance(qop_c, int) and qop_c < fop_c) else 'FAIL'
print(f'  [{status}] FOP cnot={fop_c}  QOP cnot={qop_c}  QOP error={r_q[\"energy_error_mha\"]:.2f} mHa')
" || true

echo ""
echo "Hybrid GiGPO smoke test:"
$PYTHON_PATH -c "
import rlqas_chem
r = rlqas_chem.search('H2', 0.74, ansatz_type='HYBRID', agent_type='gigppo', n_episodes=40)
status = 'PASS' if r.get('best_energy') is not None else 'FAIL'
print(f'  [{status}] best_energy={r.get(\"best_energy\")}')
" || true

echo ""
echo "HEA Tree-GRPO block test:"
$PYTHON_PATH -c "
import rlqas_chem
try:
    rlqas_chem.search('H2', 0.74, ansatz_type='HEA', agent_type='tree_grpo', n_episodes=5)
    print('  [FAIL] Expected ValueError but got none')
except ValueError as e:
    print(f'  [PASS] Correctly raised ValueError: {e}')
" || true

echo ""
echo "=== Ralph completed at: $(date) ==="
echo "Exit code: $EXIT_CODE"
exit $EXIT_CODE
