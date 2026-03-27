#!/bin/bash
#SBATCH --job-name=ralph-phase2-fix001
#SBATCH --output=slurm_logs/ralph_phase2_fix001_%j.out
#SBATCH --error=slurm_logs/ralph_phase2_fix001_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== RLQAS Phase 2 Fix001: Correct Partial-Circuit Architecture Search ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Activate conda environment
PYTHON_PATH="python3"
if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    if conda activate llm 2>/dev/null; then
        echo "Conda environment activated: $(which python3)"
        PYTHON_PATH="$(which python3)"
    else
        if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
            PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        fi
    fi
else
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
        PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
    fi
fi
export PYTHON_PATH
echo "Python: $($PYTHON_PATH --version)"

# Quick dependency check (packages are installed as editable, no sys.path manipulation needed)
$PYTHON_PATH -c "
from rlqas.phase1.search.environment import UCCSearchEnv
from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
from rlqas.phase1.simulator.tencirchem import TencirchemCISimulator
from tencirchem import UCCSD
import stable_baselines3, scipy, numpy, torch
print('✓ rlqas.phase1 (environment, circuit_builder, simulator) OK')
print('✓ tencirchem:', __import__('tencirchem').__version__)
print('✓ stable_baselines3:', stable_baselines3.__version__)
print('✓ scipy:', scipy.__version__, '| numpy:', numpy.__version__, '| torch:', torch.__version__)
print('✓ Phase2/full src path:', '/curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full/src')
print('  (Phase2/full not installed as package; Ralph will add to sys.path as needed)')
" || { echo "Dependency check failed"; exit 1; }

# Verify required files
for f in prd.json CLAUDE.md ralph.sh progress.txt; do
    [ -f "$f" ] && echo "✓ $f" || { echo "✗ Missing: $f"; exit 1; }
done

mkdir -p slurm_logs

echo ""
echo "========================================"
echo "Starting Ralph for Fix001 at: $(date)"
echo "Tasks:"
echo "  1. Fix Bug A: constrain optimizer to active params (environment.py)"
echo "  2. Fix Bug B: remove/guard simulator shortcut (tencirchem.py)"
echo "  3. Write tests confirming the fix"
echo "  4. Re-run honest benchmarks (PPO/DQN/A2C/SAC-Discrete on LiH)"
echo "  5. Update integration tests"
echo "  6. Document in progress.txt"
echo "Target files:"
echo "  ../../Phase1/006/src/rlqas/phase1/search/environment.py"
echo "  ../../Phase1/006/src/rlqas/phase1/simulator/tencirchem.py"
echo "  ../full/tests/integration/"
echo "  ../full/results/algorithm_comparison/"
echo "========================================"

./ralph.sh --tool claude 20

echo ""
echo "Ralph completed Fix001 at: $(date)"
echo ""
echo "Check:"
echo "  1. Phase1/006/src/rlqas/phase1/search/environment.py — optimizer constrained"
echo "  2. Phase1/006/src/rlqas/phase1/simulator/tencirchem.py — shortcut guarded"
echo "  3. Phase2/full/results/algorithm_comparison/honest_ppo_dqn_a2c_sacd_lih_10q.json"
echo "  4. progress.txt — Fix001 section added"
