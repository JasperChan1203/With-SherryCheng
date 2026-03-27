"""Top-level API for rlqas-chem."""


def search(molecule, bond_length, ansatz_type='UCC', agent_type='ppo', n_episodes=500,
           active_space=None, basis_set='sto-3g', transform='jordan_wigner',
           early_stop_threshold=1.6e-3, config=None):
    """Search for optimal quantum circuit architecture.

    Returns dict with best_energy, fci_energy, energy_error_mha, chemical_accuracy, etc.
    """
    raise NotImplementedError("API not yet implemented — fill in US-007")
