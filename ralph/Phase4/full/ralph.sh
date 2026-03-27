#!/bin/bash
# Ralph Wiggum - Long-running AI agent loop for RLQAS Phase 4
# Usage: ./ralph.sh [--tool amp|claude] [max_iterations]

set -e

TOOL="claude"
MAX_ITERATIONS=30  # Phase 4 has 3 simple tasks

while [[ $# -gt 0 ]]; do
  case $1 in
    --tool)
      TOOL="$2"
      shift 2
      ;;
    --tool=*)
      TOOL="${1#*=}"
      shift
      ;;
    *)
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
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
ARCHIVE_DIR="$SCRIPT_DIR/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")

  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    DATE=$(date +%Y-%m-%d)
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"

    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$COST_LOG_FILE" ] && cp "$COST_LOG_FILE" "$ARCHIVE_FOLDER/"
    echo "   Archived to: $ARCHIVE_FOLDER"

    echo "# RLQAS Phase 4 - Progress Log" > "$PROGRESS_FILE"
    echo "Started: $(date)" >> "$PROGRESS_FILE"
    echo "---" >> "$PROGRESS_FILE"
  fi
fi

# Track current branch
if [ -f "$PRD_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  if [ -n "$CURRENT_BRANCH" ]; then
    echo "$CURRENT_BRANCH" > "$LAST_BRANCH_FILE"
  fi
fi

# Initialize progress file if not exists
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# RLQAS Phase 4 - Progress Log" > "$PROGRESS_FILE"
  echo "# Task: Internal Research Tool (Python API + CLI + Examples)" >> "$PROGRESS_FILE"
  echo "" >> "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

# Initialize cost log if not exists
if [ ! -f "$COST_LOG_FILE" ]; then
  echo "# RLQAS Phase 4 - Token & Cost Log" > "$COST_LOG_FILE"
  echo "# Tool: $TOOL | Max iterations: $MAX_ITERATIONS" >> "$COST_LOG_FILE"
  echo "# Started: $(date)" >> "$COST_LOG_FILE"
  echo "---" >> "$COST_LOG_FILE"
fi

echo "Starting Ralph - Tool: $TOOL - Max iterations: $MAX_ITERATIONS"
echo "Task: RLQAS Phase 4 Internal Research Tool (3 tasks)"
echo "Dependencies:"
echo "  - Phase 1 (../../Phase1/006)"
echo "  - Phase 2 (../../Phase2/full)"
echo "  - Phase 3 (../../Phase3/full)"
echo "Tasks:"
echo "  001: Top-level Python API (rlqas.search / rlqas.Experiment)"
echo "  002: CLI Entry Point (rlqas search / rlqas experiment)"
echo "  003: Example Scripts (examples/01-04)"

SESSION_COST_USD=0

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "  Phase 4: Internal Research Tool (3 tasks)"
  echo "==============================================================="

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
    echo "Ralph completed all Phase 4 tasks!"
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    echo ""
    echo "=== Session Cost Summary ==="
    echo "  Iterations completed : $i"
    echo "  Session cost (USD)   : \$$SESSION_COST_USD"
    echo "  Cost log             : $COST_LOG_FILE"
    echo "  (For historical runs : check Anthropic Console at console.anthropic.com)"
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
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
echo "Check $PROGRESS_FILE for status."
echo ""
echo "=== Session Cost Summary ==="
echo "  Iterations completed : $MAX_ITERATIONS"
echo "  Session cost (USD)   : \$$SESSION_COST_USD"
echo "  Cost log             : $COST_LOG_FILE"
echo "  (For historical runs : check Anthropic Console at console.anthropic.com)"
{
  echo "--- Session Ended (max iterations reached) ---"
  echo "Finished: $(date) | Iterations: $MAX_ITERATIONS | Session cost: \$$SESSION_COST_USD"
} >> "$COST_LOG_FILE"
echo "To continue, run: ./ralph.sh $((MAX_ITERATIONS + 10))"
exit 1
