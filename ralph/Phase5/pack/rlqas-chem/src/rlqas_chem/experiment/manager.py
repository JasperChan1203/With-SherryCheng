"""Experiment manager for rlqas-chem."""


class Experiment:
    """Manages an RLQAS experiment with molecule, search, and RL configs."""

    def __init__(self, molecule_config, search_config, rl_config):
        self.molecule_config = molecule_config
        self.search_config = search_config
        self.rl_config = rl_config

    def run(self):
        """Run the experiment and return results dict."""
        raise NotImplementedError("Experiment.run() not yet implemented — fill in US-007")

    def save(self, path: str):
        """Save results to path."""
        raise NotImplementedError("Experiment.save() not yet implemented — fill in US-007")
