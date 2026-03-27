#!/bin/bash
# Ralph — RLQAS Phase 5 Pack: RLQAS-CHEM Standalone Package
# Builds a self-contained pip-installable library from Phase 1–5 source
# Usage: ./ralph.sh [--tool amp|claude] [max_iterations]

set -e

TOOL="claude"
MAX_ITERATIONS=25  # 8 stories, some need multiple iterations

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
  echo "# RLQAS Phase 5 Pack - Cost Log" > "$COST_LOG_FILE"
  echo "# Tool: $TOOL | Max iterations: $MAX_ITERATIONS" >> "$COST_LOG_FILE"
  echo "# Started: $(date)" >> "$COST_LOG_FILE"
  echo "---" >> "$COST_LOG_FILE"
fi

echo "Starting Ralph — Phase 5 Pack: RLQAS-CHEM Standalone Package"
echo "Tool: $TOOL | Max iterations: $MAX_ITERATIONS"
echo "Stories:"
echo "  US-001: Package scaffold (HIGH)"
echo "  US-002: Molecule + simulator modules (HIGH)"
echo "  US-003: RL agent modules — PPO/DQN/A2C/SAC + factory (HIGH)"
echo "  US-004: UCC search with Phase5 fixes (HIGH)"
echo "  US-005: HEA search module (MEDIUM)"
echo "  US-006: Hybrid search with REINFORCE (MEDIUM)"
echo "  US-007: Unified API + CLI (MEDIUM)"
echo "  US-008: Tests, examples, LiH benchmark (MEDIUM)"

SESSION_COST_USD=0

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "================================================================"
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "  Phase 5 Pack: RLQAS-CHEM"
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
    echo "Ralph completed all Phase 5 Pack stories!"
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
echo "To continue: ./ralph.sh $((MAX_ITERATIONS + 5))"
exit 1
