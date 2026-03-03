#!/bin/bash
#SBATCH --job-name=ralph-phase1-002
#SBATCH --output=ralph_phase1_002_%j.out
#SBATCH --error=ralph_phase1_002_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00          # 48小时，量子模拟可能耗时
#SBATCH --partition=CPU          # 根据集群调整
#SBATCH --mail-type=NONE

echo "=== RLQAS Ralph Phase 1 Task 002 Batch Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Running on node: $(hostname)"
echo "Current directory: $(pwd)"

# 设置Python无缓冲输出
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# 激活Python环境（参考001脚本的路径）
PYTHON_PATH="python3"
if [ -f "/software/devtools/anaconda3/etc/profile.d/conda.sh" ]; then
    source /software/devtools/anaconda3/etc/profile.d/conda.sh
    if conda activate llm 2>/dev/null; then
        echo "Conda环境已激活: $(which python3)"
        PYTHON_PATH="$(which python3)"
    else
        echo "警告: 无法激活conda环境，使用默认python3"
        if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
            PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
            echo "使用直接路径: $PYTHON_PATH"
        fi
    fi
else
    echo "警告: 未找到conda"
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
        PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        echo "使用直接路径: $PYTHON_PATH"
    fi
fi
export PYTHON_PATH

# 检查Python环境
echo "Python版本: $($PYTHON_PATH --version)"
echo "检查RLQAS Phase 1 Task 002的关键依赖..."
$PYTHON_PATH -c "
import sys
import traceback
print(f'Python路径: {sys.executable}')
try:
    import tencirchem
    print(f'tencirchem版本: {tencirchem.__version__}')
    # 检查CI矢量引擎特定功能
    from tencirchem import UCC
    print('✓ tencirchem导入成功，CI矢量引擎可用')
except ImportError as e:
    print(f'✗ tencirchem导入错误: {e}')
    traceback.print_exc()
try:
    import openfermion
    print(f'openfermion版本: {openfermion.__version__}')
    from openfermion import QubitOperator
    print('✓ openfermion导入成功')
except ImportError as e:
    print(f'✗ openfermion导入错误: {e}')
    traceback.print_exc()
try:
    import numpy
    print(f'numpy版本: {numpy.__version__}')
except ImportError as e:
    print(f'✗ numpy导入错误: {e}')
    traceback.print_exc()
try:
    import scipy
    print(f'scipy版本: {scipy.__version__}')
except ImportError as e:
    print(f'✗ scipy导入错误: {e}')
    traceback.print_exc()
try:
    import torch
    print(f'PyTorch版本: {torch.__version__}')
    print('✓ PyTorch导入成功 (GPU加速可选)')
except ImportError as e:
    print(f'PyTorch导入警告: {e} (可选依赖)')
"

# 检查Task 001依赖（002需要导入001的模块）
echo "检查Task 001模块导入..."
$PYTHON_PATH -c "
import sys
import os

# 添加Task 001路径
task001_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../001')
sys.path.append(task001_dir)

print(f'尝试从以下路径导入Task 001模块: {task001_dir}')
print(f'Task 001目录内容:')
import os
if os.path.exists(task001_dir):
    for item in os.listdir(task001_dir):
        print(f'  - {item}')
else:
    print(f'  Task 001目录不存在: {task001_dir}')

try:
    from src.modules.molecule_processor import MoleculeData, process_molecule
    print('✓ 成功导入Task 001模块: MoleculeData, process_molecule')

    # 测试导入的模块是否可用
    print('测试Task 001模块功能...')
    import numpy as np
    from openfermion import QubitOperator

    # 创建简单的测试数据
    test_hamiltonian = QubitOperator('Z0', 1.0)
    test_reference = np.array([1, 0])

    print('✓ Task 001模块导入测试通过')

except ImportError as e:
    print(f'✗ 无法导入Task 001模块: {e}')
    import traceback
    traceback.print_exc()
    print('请确保Task 001已完成且可访问')
"

# 检查必需文件
REQUIRED_FILES=("prd.json" "CLAUDE.md" "ralph.sh")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "✗ 必需文件不存在: $file"
        exit 1
    else
        echo "✓ 必需文件存在: $file"
    fi
done

# 运行Ralph（20次迭代，可根据需要调整）
echo "开始运行Ralph for RLQAS Phase 1 Task 002: $(date)"
echo "========================================"
echo "最大迭代次数: 20（48小时时间限制）"
echo "注意: Task 002依赖于Task 001的模块"
echo "Ralph将实现量子模拟器模块，包括："
echo "  - QuantumSimulator抽象基类"
echo "  - TencirchemCISimulator (CI矢量引擎)"
echo "  - SimulatorFactory工厂类"
echo "  - 性能测试 (8量子比特电路 <500ms)"
echo "========================================"
./ralph.sh --tool claude 20
RALPH_EXIT_CODE=$?
echo "========================================"
echo "Ralph退出代码: $RALPH_EXIT_CODE"

# 检查结果
echo "检查进度..."
if [ -f "progress.txt" ]; then
    echo "progress.txt最后30行:"
    tail -30 progress.txt
else
    echo "progress.txt未找到"
fi

# 检查学习日志
if [ -f "ralph_learning_log.txt" ]; then
    echo ""
    echo "ralph_learning_log.txt最后20行:"
    tail -20 ralph_learning_log.txt
fi

# 检查AGENTS.md知识库
if [ -f "AGENTS.md" ]; then
    echo ""
    echo "知识库AGENTS.md已更新"
    echo "最后10行内容:"
    tail -10 AGENTS.md
fi

# 检查生成的代码文件
echo ""
echo "检查生成的实现文件:"
if [ -d "src" ]; then
    echo "✓ src/目录存在"
    find src -name "*.py" -type f | head -10
else
    echo "✗ src/目录未找到"
fi

if [ -d "tests" ]; then
    echo "✓ tests/目录存在"
    find tests -name "*.py" -type f | head -10
else
    echo "✗ tests/目录未找到"
fi

echo "作业完成于: $(date)"
exit $RALPH_EXIT_CODE