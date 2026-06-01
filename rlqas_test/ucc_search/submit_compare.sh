#!/bin/bash
#SBATCH --job-name=rlqas-vs-adapt
#SBATCH --output=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/ucc_search/slurm_logs/compare_%j.out
#SBATCH --error=/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/ucc_search/slurm_logs/compare_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=4V100
#SBATCH --qos=normal

WORKDIR="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas_test/ucc_search"
RLQAS_CHEM="/curie-home/jpchen/scratch/LLM/code/RLQAS/rlqas-chem"
PYTHON="/curie-home/jpchen/.conda/envs/llm/bin/python3"

cd "$WORKDIR"
mkdir -p "$WORKDIR/slurm_logs" "$WORKDIR/results"

echo "=== RLQAS vs ADAPT-VQE Comparison ==="
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $(hostname)"
echo "Started : $(date)"
echo ""

export PYTHONUNBUFFERED=1

# Verify package
$PYTHON -m pip install -e "$RLQAS_CHEM" -q 2>&1 | tail -1
$PYTHON -c "import rlqas_chem; print('rlqas_chem ready')" || { echo "ERROR: import failed"; exit 1; }

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Step 1: ADAPT-VQE baseline
echo ""
echo ">>> Step 1: ADAPT-VQE"
echo "Started : $(date)"
$PYTHON adapt_vqe_test.py
echo "Finished: $(date)"

# Step 2: RLQAS efficiency test
echo ""
echo ">>> Step 2: RLQAS (PPO + GRPO, early_stop=True, max_ops=30)"
echo "Started : $(date)"
$PYTHON rlqas_efficiency_test.py \
    --episodes "${RLQAS_N_EPISODES:-1000}" \
    --output "results/rlqas_efficiency_${TIMESTAMP}.json"
echo "Finished: $(date)"

echo ""
echo "=== All done at $(date) ==="
