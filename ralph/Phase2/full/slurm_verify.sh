#!/bin/bash
#SBATCH --job-name=verify-single-op
#SBATCH --output=slurm_logs/verify_single_op_%j.out
#SBATCH --error=slurm_logs/verify_single_op_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=4V100
#SBATCH --qos=normal

echo "=== Verify Single Operator Chemical Accuracy ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $(hostname)"

export PYTHONUNBUFFERED=1

# Activate conda env
if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    conda activate llm 2>/dev/null
fi

PYTHON_PATH="$(which python3)"
echo "Python: $PYTHON_PATH"

cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full

$PYTHON_PATH scripts/verify_single_operator.py

echo "Finished at: $(date)"
