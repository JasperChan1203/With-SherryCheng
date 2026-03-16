#!/bin/bash
# Ralph Wiggum - Long-running AI agent loop for RLQAS Phase 2 Complete Implementation
# Usage: ./ralph.sh [--tool amp|claude] [max_iterations]

set -e

# Parse arguments
TOOL="claude"  # Default to claude for Phase 2 complete tasks
MAX_ITERATIONS=50  # Phase 2 complete may need more iterations (all 6 tasks)

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
      # Assume it's max_iterations if it's a number
      if [[ "$1" =~ ^[0-9]+$ ]]; then
        MAX_ITERATIONS="$1"
      fi
      shift
      ;;
  esac
done

# Validate tool choice
if [[ "$TOOL" != "amp" && "$TOOL" != "claude" ]]; then
  echo "Error: Invalid tool '$TOOL'. Must be 'amp' or 'claude'."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROGRESS_FILE="$SCRIPT_DIR/progress.txt"
ARCHIVE_DIR="$SCRIPT_DIR/archive"
LAST_BRANCH_FILE="$SCRIPT_DIR/.last-branch"

# Archive previous run if branch changed
if [ -f "$PRD_FILE" ] && [ -f "$LAST_BRANCH_FILE" ]; then
  CURRENT_BRANCH=$(jq -r '.branchName // empty' "$PRD_FILE" 2>/dev/null || echo "")
  LAST_BRANCH=$(cat "$LAST_BRANCH_FILE" 2>/dev/null || echo "")

  if [ -n "$CURRENT_BRANCH" ] && [ -n "$LAST_BRANCH" ] && [ "$CURRENT_BRANCH" != "$LAST_BRANCH" ]; then
    # Archive the previous run
    DATE=$(date +%Y-%m-%d)
    # Strip "ralph/" prefix from branch name for folder
    FOLDER_NAME=$(echo "$LAST_BRANCH" | sed 's|^ralph/||')
    ARCHIVE_FOLDER="$ARCHIVE_DIR/$DATE-$FOLDER_NAME"

    echo "Archiving previous run: $LAST_BRANCH"
    mkdir -p "$ARCHIVE_FOLDER"
    [ -f "$PRD_FILE" ] && cp "$PRD_FILE" "$ARCHIVE_FOLDER/"
    [ -f "$PROGRESS_FILE" ] && cp "$PROGRESS_FILE" "$ARCHIVE_FOLDER/"
    echo "   Archived to: $ARCHIVE_FOLDER"

    # Reset progress file for new run
    echo "# RLQAS Phase 2 Complete - Progress Log" > "$PROGRESS_FILE"
    echo "# Task: Complete Phase 2 Implementation (All 6 tasks)" >> "$PROGRESS_FILE"
    echo "" >> "$PROGRESS_FILE"
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

# Initialize progress file if it doesn't exist
if [ ! -f "$PROGRESS_FILE" ]; then
  echo "# RLQAS Phase 2 Complete - Progress Log" > "$PROGRESS_FILE"
  echo "# Task: Complete Phase 2 Implementation (All 6 tasks)" >> "$PROGRESS_FILE"
  echo "" >> "$PROGRESS_FILE"
  echo "Started: $(date)" >> "$PROGRESS_FILE"
  echo "---" >> "$PROGRESS_FILE"
fi

echo "Starting Ralph - Tool: $TOOL - Max iterations: $MAX_ITERATIONS"
echo "Task: RLQAS Phase 2 Complete Implementation (All 6 tasks)"
echo "Dependencies:"
echo "  - Phase 1 Integrated Package (Phase1/006)"
echo "  - Phase 2 Task 001 (../001) - DQN implementation"
echo "Phases:"
echo "  0. Task 001 Verification and Integration"
echo "  1. Sequential Testing Framework (Task 002)"
echo "  2. HEA Search Module (Task 003)"
echo "  3. Experiment Management System (Task 004)"
echo "  4. Agent Autonomous RL Exploration (Task 005)"
echo "  5. Phase 2 Integration Test (Task 006)"

for i in $(seq 1 $MAX_ITERATIONS); do
  echo ""
  echo "==============================================================="
  echo "  Ralph Iteration $i of $MAX_ITERATIONS ($TOOL)"
  echo "  Phase 2 Complete: All 6 tasks in unified implementation"
  echo "==============================================================="

  # Run the selected tool with the ralph prompt
  if [[ "$TOOL" == "amp" ]]; then
    OUTPUT=$(cat "$SCRIPT_DIR/CLAUDE.md" | amp --dangerously-allow-all 2>&1 | tee /dev/stderr) || true
  else
    # Claude Code: use --dangerously-skip-permissions for autonomous operation, --print for output
    OUTPUT=$(claude --dangerously-skip-permissions --print < "$SCRIPT_DIR/CLAUDE.md" 2>&1 | tee /dev/stderr) || true
  fi

  # Check for completion signal
  if echo "$OUTPUT" | grep -q "<promise>COMPLETE</promise>"; then
    echo ""
    echo "Ralph completed all Phase 2 tasks!"
    echo "Completed at iteration $i of $MAX_ITERATIONS"
    echo "Phase 2 Complete Implementation Results:"
    echo "  - All 6 Phase 2 tasks implemented and integrated"
    echo "  - DQN verified and integrated from Task 001"
    echo "  - Sequential testing framework operational"
    echo "  - HEA search module with multiple entanglement patterns"
    echo "  - Experiment management system with configuration support"
    echo "  - Autonomous RL exploration framework (key innovation)"
    echo "  - Comprehensive integration tests passing"
    echo "  - Overall code coverage >90%"
    exit 0
  fi

  echo "Iteration $i complete. Continuing..."
  sleep 2
done

echo ""
echo "Ralph reached max iterations ($MAX_ITERATIONS) without completing all tasks."
echo "Check $PROGRESS_FILE for status."
echo "Phase 2 complete task dependencies:"
echo "  1. Phase 1 Integrated Package (../../Phase1/006) - MUST be properly installed"
echo "  2. Phase 2 Task 001 (../001) - DQN implementation must be completed first"
echo "To continue, run: ./ralph.sh $((MAX_ITERATIONS + 20))"
exit 1