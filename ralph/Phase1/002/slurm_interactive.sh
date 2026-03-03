#!/bin/bash
# SLURM交互式作业脚本 for Phase 1 Task 002
# 使用方法:
# 1. 先申请交互式资源: salloc --job-name=ralph-002-interactive --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=24:00:00 --partition=CPU
# 2. 然后运行此脚本: ./slurm_interactive.sh

echo "=== RLQAS Phase 1 Task 002 交互式测试 ==="
echo "开始时间: $(date)"
echo "运行在主机: $(hostname)"
echo "当前目录: $(pwd)"

# 设置Python无缓冲输出
export PYTHONUNBUFFERED=1

# 激活Python环境
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
    if [ -f "/curie-home/jpchen/.conda/envs/llm/bin/python3" ]; then
        PYTHON_PATH="/curie-home/jpchen/.conda/envs/llm/bin/python3"
        echo "使用直接路径: $PYTHON_PATH"
    fi
fi

# 检查Python环境
echo "Python版本: $($PYTHON_PATH --version)"
echo "检查Task 002依赖..."

$PYTHON_PATH -c "
import sys
print(f'Python路径: {sys.executable}')

# 检查核心依赖
deps_ok = True
try:
    import tencirchem
    print(f'✓ tencirchem版本: {tencirchem.__version__}')
except ImportError as e:
    print(f'✗ tencirchem导入错误: {e}')
    deps_ok = False

try:
    import openfermion
    print(f'✓ openfermion版本: {openfermion.__version__}')
except ImportError as e:
    print(f'✗ openfermion导入错误: {e}')
    deps_ok = False

try:
    import numpy
    print(f'✓ numpy版本: {numpy.__version__}')
except ImportError as e:
    print(f'✗ numpy导入错误: {e}')
    deps_ok = False

# 检查Task 001导入
import os
task001_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../001')
sys.path.append(task001_dir)

try:
    from src.modules.molecule_processor import MoleculeData, process_molecule
    print('✓ 成功导入Task 001模块')
except ImportError as e:
    print(f'✗ Task 001模块导入错误: {e}')
    print(f'Task 001路径: {task001_dir}')
    deps_ok = False

if deps_ok:
    print('所有依赖检查通过，可以运行Ralph')
else:
    print('依赖检查失败，请先解决依赖问题')
    sys.exit(1)
"

# 检查必需文件
echo "检查必需文件..."
if [ ! -f "prd.json" ]; then
    echo "✗ prd.json不存在"
    exit 1
else
    echo "✓ prd.json存在"
fi

if [ ! -f "CLAUDE.md" ]; then
    echo "✗ CLAUDE.md不存在"
    exit 1
else
    echo "✓ CLAUDE.md存在"
fi

if [ ! -f "ralph.sh" ]; then
    echo "✗ ralph.sh不存在"
    exit 1
else
    echo "✓ ralph.sh存在"
fi

echo "所有检查通过，开始运行Ralph..."

# 运行Ralph（可以先试运行少量迭代）
echo "========================================"
echo "运行Ralph (10次迭代测试)..."
echo "任务: RLQAS Phase 1 Task 002 - 量子模拟器模块"
echo "目标: 实现Tencirchem CI矢量引擎量子模拟器"
echo "========================================"

./ralph.sh --tool claude 10

echo "========================================"
echo "Ralph运行完成"
echo "结束时间: $(date)"
echo ""
echo "检查以下文件查看进度:"
echo "  - progress.txt - Ralph进度日志"
echo "  - ralph_learning_log.txt - 学习过程记录"
echo "  - AGENTS.md - 知识库"
echo ""
echo "如果需要继续运行，可以再次执行: ./ralph.sh --tool claude 10"