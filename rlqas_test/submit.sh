#!/bin/bash
#SBATCH --job-name=rlqas-mol-test
#SBATCH --output=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/slurm_logs/test_%j.out
#SBATCH --error=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/slurm_logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=4V100
#SBATCH --qos=normal

echo "=== RLQAS Molecule Test ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started : $(date)"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

SCRIPT_DIR="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test"
cd "$SCRIPT_DIR"

mkdir -p "$SCRIPT_DIR/slurm_logs" "$SCRIPT_DIR/results"

# ── Python environment ───────────────────────────────────────────────────────
PYTHON="/curie-home/jpchen/.conda/envs/llm/bin/python3"
if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    conda activate llm 2>/dev/null && PYTHON="$(which python3)"
fi
echo "Python  : $($PYTHON --version)"

# ── Verify rlqas-chem ────────────────────────────────────────────────────────
RLQAS_CHEM_DIR="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem"
$PYTHON -m pip install -e "$RLQAS_CHEM_DIR" -q 2>&1 | tail -1
$PYTHON -c "import rlqas_chem; print('rlqas_chem ready')" \
    || { echo "ERROR: rlqas_chem import failed"; exit 1; }

# ── Run test ─────────────────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="results/results_${TIMESTAMP}.json"

EPISODES="${RLQAS_N_EPISODES:-500}"
echo "Episodes: $EPISODES"
echo ""

$PYTHON test_molecules.py --episodes "$EPISODES" --output "$OUTPUT"
EXIT_CODE=$?

echo ""
echo "=== Finished at $(date) | Exit code: $EXIT_CODE ==="
exit $EXIT_CODE
