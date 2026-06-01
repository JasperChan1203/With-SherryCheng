#!/bin/bash
#SBATCH --job-name=rlqas-hea-val-grpo
#SBATCH --output=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/hea_search/slurm_logs/val_grpo_%j.out
#SBATCH --error=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/hea_search/slurm_logs/val_grpo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=4V100
#SBATCH --qos=normal

WORKDIR="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/hea_search"
RLQAS_CHEM="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem"
PYTHON="/curie-home/jpchen/.conda/envs/llm/bin/python3"

cd "$WORKDIR"
mkdir -p "$WORKDIR/slurm_logs" "$WORKDIR/results"

echo "=== RLQAS LiH HEA GRPO Learning Validation ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started : $(date)"

export PYTHONUNBUFFERED=1

if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    conda activate llm 2>/dev/null && PYTHON="$(which python3)"
fi

$PYTHON -m pip install -e "$RLQAS_CHEM" --no-deps -q 2>&1 | tail -1
$PYTHON -c "import rlqas_chem; print('rlqas_chem ready')" || { echo "ERROR: import failed"; exit 1; }

EPISODES="${RLQAS_N_EPISODES:-500}"

$PYTHON validate_lih_grpo.py \
    --episodes "$EPISODES" \
    --output "results/grpo_lih_hea_validation_${SLURM_JOB_ID}.json"

EXIT_CODE=$?
echo ""
echo "=== Finished at $(date) -- exit code: $EXIT_CODE ==="
exit $EXIT_CODE
