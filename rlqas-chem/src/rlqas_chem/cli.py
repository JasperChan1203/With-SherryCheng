"""CLI for rlqas-chem."""
import sys
import json
import click

from rlqas_chem.api import search as _search


@click.group()
def main():
    """rlqas-chem: RL-based Quantum Architecture Search."""


@main.command()
@click.option('--molecule', '-m', required=True, help='Molecular formula (e.g. H2, LiH)')
@click.option('--bond-length', '-b', type=float, required=True, help='Bond length in Angstroms')
@click.option('--ansatz', '-a', default='UCC', show_default=True, help='Ansatz type: UCC, HEA, HYBRID')
@click.option('--agent', default='ppo', show_default=True, help='RL agent: ppo, dqn, a2c, sac_discrete')
@click.option('--episodes', '-e', type=int, default=500, show_default=True, help='Number of episodes')
@click.option('--active-space', default=None, help='Active space as "n_elec,n_orb" e.g. "2,5"')
@click.option('--basis', default='sto-3g', show_default=True, help='Basis set')
@click.option('--transform', default='jordan_wigner', show_default=True, help='Fermion-qubit transform')
@click.option('--output', '-o', default=None, help='Save results as JSON to this path')
def search(molecule, bond_length, ansatz, agent, episodes, active_space, basis, transform, output):
    """Run quantum architecture search."""
    active = None
    if active_space:
        parts = active_space.split(',')
        active = (int(parts[0]), int(parts[1]))

    click.echo(f"Running RLQAS: {molecule} @ {bond_length} Å, {ansatz}/{agent}, {episodes} episodes")
    result = _search(
        molecule=molecule,
        bond_length=bond_length,
        ansatz_type=ansatz,
        agent_type=agent,
        n_episodes=episodes,
        active_space=active,
        basis_set=basis,
        transform=transform,
    )

    click.echo(f"Best energy:      {result['best_energy']:.6f} Ha")
    click.echo(f"FCI energy:       {result['fci_energy']:.6f} Ha")
    click.echo(f"Error:            {result['energy_error_mha']:.3f} mHa")
    click.echo(f"Chemical accuracy: {'YES' if result['chemical_accuracy'] else 'NO'}")

    if output:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        click.echo(f"Results saved to {output}")

    sys.exit(0)


@main.command()
@click.option('--config', '-c', required=True, help='Experiment config YAML file')
@click.option('--output', '-o', default=None, help='Save results as JSON to this path')
def experiment(config, output):
    """Run experiment from YAML config file."""
    try:
        import yaml
        with open(config) as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        import json
        with open(config) as f:
            cfg = json.load(f)

    from rlqas_chem.experiment.manager import Experiment
    exp = Experiment(
        molecule_config=cfg.get('molecule', {}),
        search_config=cfg.get('search', {}),
        rl_config=cfg.get('rl', {}),
    )
    result = exp.run()

    click.echo(f"Best energy:      {result['best_energy']:.6f} Ha")
    click.echo(f"FCI energy:       {result['fci_energy']:.6f} Ha")
    click.echo(f"Error:            {result['energy_error_mha']:.3f} mHa")

    if output:
        exp.save(output)
        click.echo(f"Results saved to {output}")

    sys.exit(0)


@main.command()
@click.option('--molecule', '-m', required=True, help='Molecular formula (e.g. H2, LiH)')
@click.option('--bond-length', '-b', type=float, required=True, help='Bond length in Angstroms')
@click.option('--agent', default='ppo', show_default=True, help='RL agent type')
@click.option('--operator-pool', default='fop', show_default=True, help='Operator pool (fop, qop)')
@click.option('--trials', '-t', type=int, default=50, show_default=True, help='Number of Optuna trials')
@click.option('--episodes', '-e', type=int, default=150, show_default=True, help='Episodes per trial')
@click.option('--output', '-o', default=None, help='Save results as JSON to this path')
def optimize(molecule, bond_length, agent, operator_pool, trials, episodes, output):
    """Optimize hyperparameters using Optuna."""
    click.echo(f"Optimizing hyperparams: {molecule} @ {bond_length} Å, {agent}, {trials} trials")

    from rlqas_chem.experiment.hpo import optimize_hyperparams
    result = optimize_hyperparams(
        molecule=molecule,
        bond_length=bond_length,
        agent_type=agent,
        operator_pool=operator_pool,
        n_trials=trials,
        n_episodes_per_trial=episodes,
    )

    click.echo(f"Best energy: {result['best_energy']:.6f} Ha")
    click.echo(f"Best params: {result['best_params']}")

    if output:
        import os
        serializable = {k: v for k, v in result.items() if k != 'study'}
        os.makedirs(os.path.dirname(os.path.abspath(output)) if os.path.dirname(output) else '.', exist_ok=True)
        with open(output, 'w') as f:
            json.dump(serializable, f, indent=2)
        click.echo(f"Results saved to {output}")

    sys.exit(0)
