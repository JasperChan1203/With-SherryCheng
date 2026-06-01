#!/bin/bash
#SBATCH --job-name=ralph-gigppo-treegrpo-ddqn
#SBATCH --output=slurm_logs/ralph_new_algorithms_%j.out
#SBATCH --error=slurm_logs/ralph_new_algorithms_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Phase 5: New Algorithms (GiGPO, Tree-GRPO, Double-DQN) ==="
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

# ── Verify rlqas-chem is installed ─────────────────────────────────────────
RLQAS_CHEM_DIR="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem"
$PYTHON_PATH -c "
import rlqas_chem
print(f'rlqas_chem OK: {getattr(rlqas_chem, \"__version__\", \"installed\")}')
" 2>/dev/null || {
  echo "rlqas_chem not importable, installing..."
  $PYTHON_PATH -m pip install -e "$RLQAS_CHEM_DIR" -q
}

# ── Run Ralph ───────────────────────────────────────────────────────────────
echo ""
echo "Starting Ralph at: $(date)"
echo "  US-ALG-001: GiGPO  — step-level credit assignment"
echo "  US-ALG-002: Tree-GRPO — prefix sharing + VQE caching"
echo "  US-ALG-003: Double-DQN — decoupled Q-value + replay buffer"
echo ""

chmod +x "$SCRIPT_DIR/ralph.sh"
"$SCRIPT_DIR/ralph.sh"
EXIT_CODE=$?

echo ""
echo "=== Ralph completed at: $(date) ==="
echo "Exit code: $EXIT_CODE"

# ── Post-run: 快速冒烟验证三个新算法 ────────────────────────────────────────
echo ""
echo "=== Post-run Smoke Check ==="

for AGENT in gigppo tree_grpo double_dqn; do
  echo ""
  echo "--- $AGENT ---"
  $PYTHON_PATH -c "
import rlqas_chem
r = rlqas_chem.search('H2', 0.74, ansatz_type='UCC', agent_type='$AGENT', n_episodes=10)
status = 'PASS' if r.get('best_energy') and r['best_energy'] < -1.0 else 'FAIL'
print(f'  [$status] best_energy={r.get(\"best_energy\")}')
" 2>&1 || echo "  [$AGENT] import or search failed"
done

echo ""
echo "=== Acceptance System Final Check ==="
REPO_ROOT="/curie-home/jpchen/scratch/LLM/code/RLQAS"
for AGENT in gigppo tree_grpo double_dqn; do
  echo ""
  echo "--- run_acceptance.sh --agent $AGENT ---"
  bash "$REPO_ROOT/rlqas_acceptance_system/run_acceptance.sh" --agent "$AGENT" && \
    echo "  ✅ $AGENT PASS" || echo "  ❌ $AGENT FAIL"
done

exit $EXIT_CODE
