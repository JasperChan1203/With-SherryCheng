#!/bin/bash
# Ralph — RLQAS Phase 5: New Algorithms (GiGPO, Tree-GRPO, Double-DQN)
# Adds three new RL algorithms to rlqas-chem and validates each through the acceptance system
#
# Usage:
#   ./ralph.sh                      # run with default settings
#   ./ralph.sh --tool amp           # use amp instead of claude
#   ./ralph.sh 20                   # set max iterations to 20

set -e

TOOL="claude"
MAX_ITERATIONS=30  # 3 stories, each may need multiple iterations

while [[ $# -gt 0 ]]; do
  case $1 in
    --tool)
      TOOL="$2"; shift 2 ;;
    --tool=*)
      TOOL="${1#*=}"; shift ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then MAX_ITERATIONS="$1"; fi
      shift ;;
  esac
done

if [[ "$TOOL" != "amp" && "$TOOL" != "claude" ]]; then
  echo "Error: Invalid tool '$TOOL'. Must be 'amp' or 'claude'."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
COST_LOG_FILE="$SCRIPT_DIR/cost_log.txt"

if [ ! -f "$COST_LOG_FILE" ]; then
  echo "# RLQAS Phase 5 New Algorithms - Cost Log" > "$COST_LOG_FILE"
  echo "# Tool: $TOOL | Max iterations: $MAX_ITERATIONS" >> "$COST_LOG_FILE"
  echo "# Started: $(date)" >> "$COST_LOG_FILE"
  echo "---" >> "$COST_LOG_FILE"
fi

echo "Starting Ralph — Phase 5: New Algorithms"
echo "Tool: $TOOL | Max iterations: $MAX_ITERATIONS"
echo "Stories:"
echo "  US-ALG-001: GiGPO (P1) — Step-level credit assignment"
echo "  US-ALG-002: Tree-GRPO (P2) — Prefix sharing + VQE caching"
echo "  US-ALG-003: Double-DQN (P3) — Decoupled Q-value estimation + replay buffer"

SESSION_COST_USD=0

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "================================================================"
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "  Phase 5 New Algorithms"
  echo "================================================================"

  ITER_COST="N/A"
  ITER_DURATION_MS="N/A"
  ITER_TURNS="N/A"

  if [[ "$TOOL" == "amp" ]]; then
    OUTPUT=$(cat "$SCRIPT_DIR/CLAUDE.md" | amp --dangerously-allow-all 2>&1 | tee /dev/stderr) || true
  else
    TEMP_JSON=$(mktemp)
    claude --dangerously-skip-permissions --print --output-format json \
      < "$SCRIPT_DIR/CLAUDE.md" > "$TEMP_JSON" 2>/dev/null || true

    RAW_JSON=$(cat "$TEMP_JSON")
    rm -f "$TEMP_JSON"

    if echo "$RAW_JSON" | jq -e '.type == "result"' > /dev/null 2>&1; then
      OUTPUT=$(echo "$RAW_JSON" | jq -r '.result // ""')
      ITER_COST=$(echo "$RAW_JSON" | jq -r '.total_cost_usd // "N/A"')
      ITER_DURATION_MS=$(echo "$RAW_JSON" | jq -r '.duration_ms // "N/A"')
      ITER_TURNS=$(echo "$RAW_JSON" | jq -r '.num_turns // "N/A"')
    else
      OUTPUT=$(claude --dangerously-skip-permissions --print \
        < "$SCRIPT_DIR/CLAUDE.md" 2>&1 | tee /dev/stderr) || true
    fi
    echo "$OUTPUT" >&2
  fi

  if [[ "$ITER_COST" != "N/A" ]]; then
    SESSION_COST_USD=$(awk "BEGIN {printf \"%.6f\", $SESSION_COST_USD + $ITER_COST}" 2>/dev/null \
      || echo "$SESSION_COST_USD")
  fi

  COST_LINE="[Iter $i | $(date '+%Y-%m-%d %H:%M')] cost=\$${ITER_COST}  session_total=\$${SESSION_COST_USD}  duration=${ITER_DURATION_MS}ms  turns=${ITER_TURNS}"
  echo "$COST_LINE" | tee -a "$COST_LOG_FILE"

  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "Ralph completed all new algorithm stories!"
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    echo ""
    echo "=== Session Cost Summary ==="
    echo "  Iterations: $i  |  Cost: \$$SESSION_COST_USD"
    {
      echo "--- Session Complete ---"
      echo "Finished: $(date) | Iterations: $i | Session cost: \$$SESSION_COST_USD"
    } >> "$COST_LOG_FILE"
    exit 0
  fi

  echo "Iteration $i complete. Continuing..."
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS). Check $PROGRESS_FILE for status."
{
  echo "--- Session Ended (max iterations) ---"
  echo "Finished: $(date) | Iterations: $MAX_ITERATIONS | Session cost: \$$SESSION_COST_USD"
} >> "$COST_LOG_FILE"
echo "To continue: ./ralph.sh $((MAX_ITERATIONS + 10))"
exit 1
