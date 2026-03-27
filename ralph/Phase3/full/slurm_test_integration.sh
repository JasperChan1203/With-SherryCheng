#!/bin/bash
#SBATCH --job-name=phase3-integration-test
#SBATCH --output=slurm_logs/integration_test_%j.out
#SBATCH --error=slurm_logs/integration_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --partition=4V100
#SBATCH --mail-type=NONE
#SBATCH --qos=normal

echo "=== Phase 3 Integration Tests ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $(hostname)"

export PYTHONUNBUFFERED=1

if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    conda activate llm 2>/dev/null || true
fi
PYTHON="$(which python3)"
echo "Python: $($PYTHON --version)"

cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase3/full

mkdir -p slurm_logs results/phase3_integration results/phase3_integration/hydrogen_chain benchmarks

# ── 0. 反空壳回归检查 (最重要) ───────────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 0] 反空壳回归检查 — HybridSearchEnv 单步不能作弊到化学精度"
echo "================================================================"
$PYTHON -c "
import sys
sys.path.insert(0, '../../Phase1/006/src')
sys.path.insert(0, '../../Phase2/full/src')
sys.path.insert(0, 'src')
from rlqas.phase1.molecule.processor import process_molecule
from rlqas.phase3.hybrid_search.environment import HybridSearchEnv
from rlqas.phase3.hybrid_search.circuit_builder import HybridFusionStrategy

mol = process_molecule('LiH', 1.6, 'UCC',
    active_space=(2, 5), basis_set='sto-3g', transform='jordan_wigner')
env = HybridSearchEnv(mol, HybridFusionStrategy({'fusion_mode': 'sequential'}),
    {'run_classical_opt': True, 'complexity_penalty': 0.0,
     'max_depth': 10, 'max_blocks': 10})
obs, _ = env.reset()
obs, reward, done, trunc, info = env.step(0)
energy = info['energy']
error = abs(energy - mol.fci_energy)
print(f'  Single-step energy  : {energy:.6f} Ha')
print(f'  FCI energy          : {mol.fci_energy:.6f} Ha')
print(f'  Energy error        : {error*1000:.4f} mHa  (must be > 1.6 mHa)')

assert error > 1.6e-3, (
    f'HOLLOW IMPL DETECTED: Single-step error {error*1000:.4f} mHa < 1.6 mHa. '
    f'Possible causes: (1) run_classical_opt disabled, (2) energy returned from FCI directly, '
    f'(3) circuit not actually built'
)
# Also check energy is below HF (classical opt must be running)
hf_approx = -7.862
assert energy < hf_approx, (
    f'HOLLOW IMPL DETECTED: Energy {energy:.6f} Ha above HF level {hf_approx} Ha. '
    f'Classical optimization (scipy.minimize) is not running.'
)
print(f'  Energy below HF     : YES ({energy:.6f} < {hf_approx})')
print('  [PASS] Anti-hollow checks passed — HybridSearchEnv uses real quantum energy evaluation')
"
ANTIHOL_STATUS=$?
if [ $ANTIHOL_STATUS -ne 0 ]; then
    echo "[FAIL] 反空壳检查失败 — HybridSearchEnv 能量评估存在空壳实现!"
    echo "       这是 Phase 2 BUG A/B 类型的错误，必须修复后才能继续其他测试。"
    exit 1
fi

# ── 1. Module importability ──────────────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 1] Phase 3 全模块可导入性检查"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_phase3_integration.py::test_all_phase3_modules_importable \
    -v --tb=short --no-header 2>&1
IMPORT_STATUS=$?

# ── 2. Hybrid search smoke test (LiH) ───────────────────────────────
echo ""
echo "================================================================"
echo "[Step 2] Hybrid Search 冒烟测试 (LiH 4-qubit)"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_phase3_integration.py::test_hybrid_search_lih_smoke_test \
    -v --tb=short --no-header 2>&1
SMOKE_STATUS=$?

# ── 3. Hybrid search on BeH2 8q ─────────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 3] Hybrid Search BeH2 8-qubit 化学精度测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_hybrid_search.py \
    -v --tb=short --no-header 2>&1
HYBRID_STATUS=$?

# ── 4. Batch evaluation performance ─────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 4] 批量评估性能测试 (>=1.5x speedup)"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_batch_performance.py \
    -v --tb=short --no-header 2>&1
BATCH_STATUS=$?

# ── 5. Circuit encoding comparison ──────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 5] 电路编码方法对比测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_encoding_comparison.py \
    -v --tb=short --no-header 2>&1
ENCODING_STATUS=$?

# ── 6. BeH2 scalability tests ───────────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 6] BeH2 可扩展性测试 (10q, 12q)"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_beh2_scalability.py \
    -v --tb=short --no-header -m "not slow" 2>&1
BEH2_STATUS=$?

# ── 7. Hydrogen chain tests ─────────────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 7] 氢链关联效应测试 (H4 8q, H6 12q 全空间化学精度)"
echo "        H6: 不控制active space，STO-3G全空间 (6e, 6orb → 12q)"
echo "        H6 同样要求 energy_error < 1.6e-3 Ha"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_hydrogen_chain.py \
    -v --tb=short --no-header 2>&1
H_CHAIN_STATUS=$?

# ── 8. ExperimentManager HYBRID end-to-end ──────────────────────────
echo ""
echo "================================================================"
echo "[Step 8] ExperimentManager HYBRID ansatz 端到端测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_phase3_integration.py::test_experiment_manager_hybrid_dispatch \
    -v --tb=short --no-header 2>&1
EXP_MGR_STATUS=$?

# ── 9. Qubit operator comparison ────────────────────────────────────
echo ""
echo "================================================================"
echo "[Step 9] Qubit算符 vs Fermion算符对比测试"
echo "================================================================"
$PYTHON -m pytest tests/integration/test_qubit_ucc_search.py \
    -v --tb=short --no-header 2>&1
QUBIT_STATUS=$?

# ── Summary ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "测试结果汇总"
echo "================================================================"
[ $ANTIHOL_STATUS  -eq 0 ] && echo "  反空壳回归检查      : PASS" || echo "  反空壳回归检查      : FAIL ❌ (空壳实现!)"
[ $IMPORT_STATUS   -eq 0 ] && echo "  模块可导入性        : PASS" || echo "  模块可导入性        : FAIL ❌"
[ $SMOKE_STATUS    -eq 0 ] && echo "  Hybrid LiH冒烟测试  : PASS" || echo "  Hybrid LiH冒烟测试  : FAIL ❌"
[ $HYBRID_STATUS   -eq 0 ] && echo "  BeH2 8q Hybrid      : PASS" || echo "  BeH2 8q Hybrid      : FAIL ❌"
[ $BATCH_STATUS    -eq 0 ] && echo "  批量评估性能        : PASS" || echo "  批量评估性能        : FAIL ❌"
[ $ENCODING_STATUS -eq 0 ] && echo "  电路编码对比        : PASS" || echo "  电路编码对比        : FAIL ❌"
[ $BEH2_STATUS     -eq 0 ] && echo "  BeH2 可扩展性       : PASS" || echo "  BeH2 可扩展性       : FAIL ❌"
[ $H_CHAIN_STATUS  -eq 0 ] && echo "  氢链 H4/H6 化学精度 : PASS" || echo "  氢链 H4/H6 化学精度 : FAIL ❌"
[ $EXP_MGR_STATUS  -eq 0 ] && echo "  ExperimentManager   : PASS" || echo "  ExperimentManager   : FAIL ❌"
[ $QUBIT_STATUS    -eq 0 ] && echo "  Qubit算符对比       : PASS" || echo "  Qubit算符对比       : FAIL ❌"

OVERALL=$(( ANTIHOL_STATUS + IMPORT_STATUS + SMOKE_STATUS + HYBRID_STATUS + BATCH_STATUS + ENCODING_STATUS + BEH2_STATUS + H_CHAIN_STATUS + EXP_MGR_STATUS + QUBIT_STATUS ))

echo ""
echo "Finished at: $(date)"
if [ $OVERALL -eq 0 ]; then
    echo "✅ 所有测试通过 — Phase 3 完全验证完成"
else
    echo "❌ 有测试失败，请查看上方详细输出"
fi

exit $OVERALL
