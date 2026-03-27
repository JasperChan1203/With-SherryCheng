"""CLI for rlqas-chem."""
import click


@click.group()
def main():
    """rlqas-chem: RL-based Quantum Architecture Search."""


@main.command()
@click.option('--molecule', required=True)
@click.option('--bond-length', type=float, required=True)
@click.option('--ansatz', default='UCC')
@click.option('--agent', default='ppo')
@click.option('--episodes', type=int, default=500)
def search(molecule, bond_length, ansatz, agent, episodes):
    """Run quantum architecture search."""
    raise NotImplementedError("CLI not yet implemented — fill in US-007")


@main.command()
@click.option('--config', required=True)
def experiment(config):
    """Run experiment from config file."""
    raise NotImplementedError("CLI not yet implemented — fill in US-007")
