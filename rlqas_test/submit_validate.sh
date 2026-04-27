#!/bin/bash
#SBATCH --job-name=rlqas-validate
#SBATCH --output=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/slurm_logs/validate_%j.out
#SBATCH --error=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/slurm_logs/validate_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=4V100
#SBATCH --qos=normal

WORKDIR="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test"
RLQAS_CHEM="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem"
PYTHON="/curie-home/jpchen/.conda/envs/llm/bin/python3"

cd "$WORKDIR"
mkdir -p "$WORKDIR/slurm_logs" "$WORKDIR/results"

echo "=== RLQAS LiH Learning Validation ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started : $(date)"

export PYTHONUNBUFFERED=1

# Install latest rlqas-chem
$PYTHON -m pip install -e "$RLQAS_CHEM" -q 2>&1 | tail -1
$PYTHON -c "import rlqas_chem; print('rlqas_chem ready')" || { echo "ERROR: import failed"; exit 1; }

EPISODES="${RLQAS_N_EPISODES:-2000}"

$PYTHON validate_lih_learning.py \
    --episodes "$EPISODES" \
    --output "results/lih_validation_${SLURM_JOB_ID}.json" \
    --diag   "results/lih_diagnostics_${SLURM_JOB_ID}.json"

EXIT_CODE=$?
echo ""
echo "=== Finished at $(date) — exit code: $EXIT_CODE ==="
exit $EXIT_CODE
