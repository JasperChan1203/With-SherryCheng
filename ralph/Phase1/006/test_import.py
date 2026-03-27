#!/usr/bin/env python3
"""Test basic import of rlqas.phase1 package."""

import sys
sys.path.insert(0, 'src')

try:
    import rlqas.phase1
    print("SUCCESS: imported rlqas.phase1")
    print("Available submodules:", dir(rlqas.phase1))

    # Try importing key components
    from rlqas.phase1.molecule.processor import process_molecule, MoleculeData
    print("SUCCESS: imported molecule.processor")

    from rlqas.phase1.simulator.factory import SimulatorFactory
    print("SUCCESS: imported simulator.factory")

    from rlqas.phase1.rl.ppo_agent import PPOAgent
    print("SUCCESS: imported rl.ppo_agent")

    from rlqas.phase1.search.environment import UCCSearchEnv
    print("SUCCESS: imported search.environment")

    from rlqas.phase1.search.controller import UCCSearchController
    print("SUCCESS: imported search.controller")

    from rlqas.phase1.validation import *
    print("SUCCESS: imported validation")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)