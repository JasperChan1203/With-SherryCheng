#!/usr/bin/env python3
"""Run all experiments E1, E2, E3 in sequence."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Ensure experiment scripts are importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))


def main():
    print("=" * 60)
    print("RLQAS-CHEM: Running all innovation experiments")
    print("=" * 60)

    # E1: Pool comparison
    print("\n" + "=" * 60)
    print("Running E1: FOP vs QOP Pool Comparison")
    print("=" * 60)
    from run_e1_pool_comparison import run_e1
    run_e1()

    # E2: GRPO vs PPO
    print("\n" + "=" * 60)
    print("Running E2: GRPO vs PPO Agent Comparison")
    print("=" * 60)
    from run_e2_grpo_vs_ppo import run_e2
    run_e2()

    # E3: Pareto frontier
    print("\n" + "=" * 60)
    print("Running E3: Pareto Frontier")
    print("=" * 60)
    from run_e3_pareto import run_e3
    run_e3()

    print("\n" + "=" * 60)
    print("All experiments complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
    sys.exit(0)
