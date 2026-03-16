"""
Test configuration for RLQAS Phase 2 tests.

This file ensures Phase 1 is importable during tests.
IMPORTANT: Phase 1 must be inserted FIRST in sys.path before Phase 2.
"""

import sys
import os

# Add Phase 1 src to path FIRST (before Phase 2)
# From tests/ directory: ../../Phase1/006/src
phase1_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Phase1', '006', 'src'))
phase2_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Phase 1 must be first!
if phase1_src not in sys.path:
    sys.path.insert(0, phase1_src)
if phase2_src not in sys.path:
    sys.path.insert(0, phase2_src)

# Pre-import Phase 1 to ensure it's loaded first
import rlqas.phase1.rl
