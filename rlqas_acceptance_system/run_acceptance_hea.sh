#!/bin/bash
# RLQAS HEA 验收系统 — 统一入口
#
# 用途：ralph 加入新算法后，运行此脚本作为 HEA 搜索的合并门控
#
# 用法：
#   ./run_acceptance_hea.sh                  # PPO 回归测试（确保现有功能未被破坏）
#   ./run_acceptance_hea.sh --agent gigppo   # 验收新算法 gigppo
#
# 测试分子：LiH，active space 2e 3o，Jordan-Wigner 变换，max_layers=3
# 通过标准：Level 0 和 Level 1-5 全部 PASS，脚本退出码为 0
# 失败行为：任意 Level 失败立即停止，打印失败原因，退出码为 1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/curie-home/jpchen/.conda/envs/llm/bin/python3"
AGENT="ppo"

# 解析 --agent 参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --agent)
      AGENT="$2"; shift 2 ;;
    --agent=*)
      AGENT="${1#*=}"; shift ;;
    *)
      echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "$AGENT" == "ppo" ]]; then
  LABEL="回归测试 (PPO)"
else
  LABEL="新算法验收: $AGENT"
fi

echo "========================================"
echo "  RLQAS HEA 验收系统"
echo "  $LABEL"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# ---------- Level 0 ----------
echo ">>> 运行 Level 0: HEA 接口契约测试..."
echo ""
if ! $PYTHON "$SCRIPT_DIR/hea_algorithms/test_acceptance_hea_level0.py" --agent "$AGENT"; then
  echo ""
  echo "========================================"
  echo "  ❌ HEA 验收未通过（Level 0 失败）"
  echo "  $LABEL"
  echo "========================================"
  exit 1
fi

echo ""
echo ">>> Level 0 通过，继续 Level 1-5..."
echo ""

# ---------- Level 1-5 ----------
echo ">>> 运行 Level 1-5: 环境稳定性 / 搜索功能 / 化学精度测试..."
echo ""
if ! $PYTHON "$SCRIPT_DIR/hea_algorithms/test_acceptance_hea_level1_5.py" --agent "$AGENT"; then
  echo ""
  echo "========================================"
  echo "  ❌ HEA 验收未通过（Level 1-5 失败）"
  echo "  $LABEL"
  echo "========================================"
  exit 1
fi

echo ""
echo "========================================"
echo "  ✅ HEA 验收通过"
echo "  $LABEL"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
exit 0
