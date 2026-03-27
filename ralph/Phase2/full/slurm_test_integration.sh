#!/bin/bash
#SBATCH --job-name=phase2-integration-test
#SBATCH --output=slurm_logs/integration_test_%j.out
#SBATCH --error=slurm_logs/integration_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== Phase 2 Integration Tests (post-fix001 verification) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $(hostname)"

export PYTHONUNBUFFERED=1

# Activate conda
if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    conda activate llm 2>/dev/null || true
fi
PYTHON="$(which python3)"
echo "Python: $($PYTHON --version)"

cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full

mkdir -p slurm_logs

# ── 1. 回归测试：单个算符不能再作弊达到化学精度 ──────────────────
echo ""
echo "================================================================"
echo "[Step 1] 回归验证：Bug A/B 修复确认（单算符不能达到化学精度）"
echo "================================================================"
$PYTHON -c "
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase1.search.environment import UCCSearchEnv
mol = process_molecule('LiH', 1.6, 'UCC',
    active_space=(2, 5), basis_set='sto-3g', transform='jordan_wigner')
env = UCCSearchEnv(mol, {
    'run_classical_opt': True, 'complexity_penalty': 0.0,
    'param_init_strategy': 'zeros', 'max_depth': 10,
})
obs, _ = env.reset()
obs, r, term, trunc, info = env.step(0)
err = abs(env.current_energy - mol.fci_energy)
nz  = sum(1 for x in env.current_params if abs(x) > 1e-10)
print(f'  1-operator error : {err*1000:.4f} mHa  (must be > 1.6 mHa)')
print(f'  Non-zero params  : {nz}             (must be == 1)')
assert err > 1.6e-3, f'BUG STILL PRESENT: 1 op error={err*1000:.4f} mHa < 1.6 mHa'
assert nz == 1,      f'BUG STILL PRESENT: {nz} non-zero params (expected 1)'
print('  [PASS] Bug A/B are fixed.')
"
if [ $? -ne 0 ]; then
    echo "[FAIL] Bug A/B regression check failed — environment.py fix may have been reverted!"
    exit 1
fi

# ── 2. LiH 10-qubit 化学精度集成测试 ─────────────────────────────
echo ""
echo "================================================================"
echo "[Step 2] LiH active_space=(2,5) 10-qubit 化学精度测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_lih_molecule.py::TestLiH10Qubits::test_lih_10q_chemical_accuracy_with_full_training \
    -v --tb=short --no-header 2>&1
LIH10_STATUS=$?

# ── 3. LiH 12-qubit 化学精度集成测试 ─────────────────────────────
echo ""
echo "================================================================"
echo "[Step 3] LiH active_space=(2,6) 12-qubit 化学精度测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_lih_molecule.py::TestLiH12Qubits::test_lih_12q_chemical_accuracy_with_full_training \
    -v --tb=short --no-header 2>&1
LIH12_STATUS=$?

# ── 4. 多算法比较测试（含 does_not_cheat 回归测试）────────────────
echo ""
echo "================================================================"
echo "[Step 4] 多算法比较 + does_not_cheat 回归测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_multi_algorithm.py \
    -v --tb=short --no-header 2>&1
MULTI_STATUS=$?

# ── 5. HEA 集成测试 ───────────────────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 5] HEA 集成测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_hea_integration.py \
    -v --tb=short --no-header 2>&1
HEA_STATUS=$?

# ── 6. Phase 2 全量集成测试 ───────────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 6] Phase 2 全量集成测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_phase2_integration.py \
    -v --tb=short --no-header 2>&1
PHASE2_STATUS=$?

# ── 7. fix001 新增测试（Phase1 circuit fix tests）─────────────────
echo ""
echo "================================================================"
echo "[Step 7] fix001 新增测试（Phase1/006/tests/integration/test_circuit_fix.py）"
echo "================================================================"
$PYTHON -m pytest ../../Phase1/006/tests/integration/test_circuit_fix.py \
    -v --tb=short --no-header 2>&1
CIRCUIT_FIX_STATUS=$?

# ── 汇总 ─────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "测试结果汇总"
echo "================================================================"
echo "  Bug A/B 回归检查    : PASS（已在 Step 1 确认）"
[ $LIH10_STATUS   -eq 0 ] && echo "  LiH 10-qubit        : PASS" || echo "  LiH 10-qubit        : FAIL ❌"
[ $LIH12_STATUS   -eq 0 ] && echo "  LiH 12-qubit        : PASS" || echo "  LiH 12-qubit        : FAIL ❌"
[ $MULTI_STATUS   -eq 0 ] && echo "  多算法比较          : PASS" || echo "  多算法比较          : FAIL ❌"
[ $HEA_STATUS     -eq 0 ] && echo "  HEA 集成            : PASS" || echo "  HEA 集成            : FAIL ❌"
[ $PHASE2_STATUS  -eq 0 ] && echo "  Phase2 全量集成     : PASS" || echo "  Phase2 全量集成     : FAIL ❌"
[ $CIRCUIT_FIX_STATUS -eq 0 ] && echo "  fix001 circuit tests: PASS" || echo "  fix001 circuit tests: FAIL ❌"

OVERALL=$(( LIH10_STATUS + LIH12_STATUS + MULTI_STATUS + HEA_STATUS + PHASE2_STATUS + CIRCUIT_FIX_STATUS ))
echo ""
echo "Finished at: $(date)"
if [ $OVERALL -eq 0 ]; then
    echo "✅ 所有测试通过 — Phase 2 完全验证完成"
else
    echo "❌ 有测试失败，请查看上方详细输出"
fi

exit $OVERALL
