"""RLQAS command-line interface."""
import argparse
import json
import sys
import rlqas


def _print_result(result: dict):
    acc = "✓ Chemical accuracy achieved" if result["chemical_accuracy"] \
          else "✗ Chemical accuracy NOT achieved"
    ft = f"\nFusion template : {result['fusion_template']}" \
         if result.get("fusion_template") else ""
    print(f"""
=== RLQAS Result ===
Molecule    : {result['molecule']}  (bond={result['bond_length']:.3f} Å, {result['n_qubits']} qubits)
Ansatz      : {result['ansatz_type']}  |  Agent: {result['agent_type']}  |  Episodes: {result['n_episodes_run']}
Best energy : {result['best_energy']:.6f} Ha
FCI energy  : {result['fci_energy']:.6f} Ha
Error       : {result['energy_error_mha']:.3f} mHa  {acc}{ft}
""")


def cmd_search(args):
    active_space = tuple(args.active_space) if args.active_space else None
    result = rlqas.search(
        molecule=args.molecule,
        bond_length=args.bond_length,
        ansatz_type=args.ansatz,
        agent_type=args.agent,
        n_episodes=args.episodes,
        active_space=active_space,
    )
    _print_result(result)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to: {args.output}")


def cmd_experiment(args):
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    exp = rlqas.Experiment(
        molecule_config=cfg["molecule"],
        search_config=cfg.get("search", {}),
        rl_config=cfg.get("rl", {}),
    )
    result = exp.run()
    _print_result(result)
    if args.output:
        exp.save(args.output)
        print(f"Result saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        prog="rlqas",
        description="RLQAS: Reinforcement Learning Quantum Architecture Search"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # rlqas search
    p_search = sub.add_parser("search", help="Run architecture search with inline args")
    p_search.add_argument("--molecule", required=True, help="Molecular formula, e.g. LiH")
    p_search.add_argument("--bond-length", type=float, required=True, dest="bond_length")
    p_search.add_argument("--ansatz", default="UCC", choices=["UCC", "HEA", "HYBRID"])
    p_search.add_argument("--agent", default="ppo", choices=["ppo", "dqn", "a2c", "sac_discrete"])
    p_search.add_argument("--episodes", type=int, default=500)
    p_search.add_argument("--active-space", type=int, nargs=2, metavar=("N_ELEC", "N_ORB"),
                          dest="active_space", help="e.g. --active-space 2 5")
    p_search.add_argument("--output", help="Save result JSON to this path")

    # rlqas experiment
    p_exp = sub.add_parser("experiment", help="Run experiment from YAML config file")
    p_exp.add_argument("--config", required=True, help="Path to YAML config file")
    p_exp.add_argument("--output", help="Save result JSON to this path")

    args = parser.parse_args()
    try:
        if args.command == "search":
            cmd_search(args)
        elif args.command == "experiment":
            cmd_experiment(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
