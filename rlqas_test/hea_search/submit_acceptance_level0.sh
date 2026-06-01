#!/bin/bash
#SBATCH --job-name=rlqas-hea-level0
#SBATCH --output=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/hea_search/slurm_logs/hea_level0_%j.out
#SBATCH --error=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/hea_search/slurm_logs/hea_level0_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=4V100
#SBATCH --qos=normal

REPO="/curie-home/jpchen/scratch/LLM/code/RLQAS"
WORKDIR="$REPO/rlqas_test/hea_search"
PYTHON="/curie-home/jpchen/.conda/envs/llm/bin/python3"

cd "$REPO"
mkdir -p "$WORKDIR/slurm_logs" "$WORKDIR/results"

echo "=== RLQAS HEA Acceptance Test - Level 0 ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Agent   : ${RLQAS_AGENT:-ppo}"
echo "Started : $(date)"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    conda activate llm 2>/dev/null && PYTHON="$(which python3)"
fi
echo "Python  : $($PYTHON --version)"

$PYTHON -m pip install -e "$REPO/rlqas-chem" -q 2>&1 | tail -1
$PYTHON -c "import rlqas_chem; print('rlqas_chem ready')" \
    || { echo "ERROR: rlqas_chem import failed"; exit 1; }

AGENT="${RLQAS_AGENT:-ppo}"

$PYTHON rlqas_acceptance_system/hea_algorithms/test_acceptance_hea_level0.py \
    --agent "$AGENT"

EXIT_CODE=$?
echo ""
echo "=== Finished at $(date) — exit code: $EXIT_CODE ==="
exit $EXIT_CODE
