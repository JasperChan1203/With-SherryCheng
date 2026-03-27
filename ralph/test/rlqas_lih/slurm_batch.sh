#!/bin/bash
#SBATCH --job-name=rlqas-lih-test
#SBATCH --output=slurm_logs/rlqas_lih_%j.out
#SBATCH --error=slurm_logs/rlqas_lih_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS LiH Benchmark Test ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Use submit directory so paths work correctly when SLURM copies the script
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$SCRIPT_DIR" || { echo "ERROR: cannot cd to $SCRIPT_DIR"; exit 1; }
echo "Working directory: $(pwd)"

mkdir -p slurm_logs

# ── Python environment ─────────────────────────────────────────────────────────
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
echo "Python: $($PYTHON_PATH --version)"

# ── Verify rlqas is importable ─────────────────────────────────────────────────
$PYTHON_PATH -c "import rlqas; print(f'rlqas OK: {rlqas.__version__ if hasattr(rlqas, \"__version__\") else \"installed\"}')" || {
  echo "rlqas not importable. Installing Phase 4 package..."
  PHASE4_DIR="$(cd "$SCRIPT_DIR/../../Phase4/full" && pwd)"
  $PYTHON_PATH -m pip install -e "$PHASE4_DIR" -q
}

# ── Run tests ──────────────────────────────────────────────────────────────────
echo ""
echo "Running: $SCRIPT_DIR/test_lih.py"
echo "Tests: 2 baseline + 3 tuned (hartree_fock baseline, ent_coef=0.05/0.1, energy_weight=100)"
echo ""

$PYTHON_PATH "$SCRIPT_DIR/test_lih.py"
EXIT_CODE=$?

echo ""
echo "=== Job finished at: $(date) ==="
echo "Exit code: $EXIT_CODE"

if [ -f "$SCRIPT_DIR/results/summary.json" ]; then
  echo ""
  echo "=== Result Summary ==="
  $PYTHON_PATH -c "
import json
with open('$SCRIPT_DIR/results/summary.json') as f:
    records = json.load(f)
for r in records:
    status = 'PASS' if r.get('chemical_accuracy_pass') else 'FAIL'
    err = r.get('energy_error_mha', r.get('error', 'N/A'))
    err_str = f'{err:.3f} mHa' if isinstance(err, float) else str(err)
    print(f'  [{status}] {r[\"name\"]:45s}  {err_str}')
"
fi

exit $EXIT_CODE
