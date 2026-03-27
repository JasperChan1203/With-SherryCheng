#!/usr/bin/env python3
import sys
import os
os.environ['GYM_NO_DEPRECATION_WARNING'] = '1'
sys.path.insert(0, 'src')
import warnings
warnings.filterwarnings('ignore')

import rlqas.phase1
print("Imported rlqas.phase1")
print("Modules containing 'gym':")
for mod in sys.modules:
    if 'gym' in mod:
        print(f"  {mod}")
        # Check if it's gym or gymnasium
        try:
            m = sys.modules[mod]
            print(f"    file: {getattr(m, '__file__', 'unknown')}")
        except:
            pass
print("\nChecking gymnasium:")
if 'gymnasium' in sys.modules:
    print("  gymnasium imported")
if 'gym' in sys.modules:
    print("  gym imported (direct)")
    # Check if it's actually gymnasium aliased
    import gym
    print(f"  gym.__file__: {gym.__file__}")
    print(f"  gym.__version__: {getattr(gym, '__version__', 'N/A')}")