import sys
sys.path.append("../001")
sys.path.append("../002")
sys.path.append("../003")
from unittest.mock import Mock, patch
import numpy as np
from src.modules.molecule_processor import MoleculeData
from src.modules.ucc_search.environment import UCCSearchEnv

# Create mock molecule data
molecule_data = Mock(spec=MoleculeData)
molecule_data.n_qubits = 2
molecule_data.fci_energy = -1.1372838344885023
molecule_data.hamiltonian = Mock()
molecule_data.reference_state = np.array([1, 0, 0, 0], dtype=complex)
molecule_data.molecular_info = {
    "hf_energy": -1.1167593073964255,
    "formula": "H2",
    "bond_length_angstrom": 0.74,
    "basis_set": "sto-3g",
    "transform": "parity",
    "ansatz_type": "UCC"
}
mock_builder = Mock()
mock_builder.n_params = 2
mock_builder.get_available_excitations.return_value = [(3, 2), (1, 0), (1, 3, 2, 0)]
mock_builder.get_parameter_indices_for_excitation.return_value = [0]
mock_builder.initialize_parameters.return_value = np.array([0.1, 0.2])
mock_builder.build_circuit.return_value = Mock()
mock_builder.evaluate_energy.return_value = -1.1
mock_builder.ucc = Mock()
mock_builder.ucc.hamiltonian = Mock()
mock_reward = Mock()
mock_reward.compute_reward.return_value = 0.0
mock_reward.update_baseline = Mock()
mock_simulator = Mock()
mock_simulator.compute_energy.return_value = -1.1

with patch('src.modules.ucc_search.environment.UCCCircuitBuilder',
           return_value=mock_builder), \
     patch('src.modules.ucc_search.environment.UCCRewardFunction',
           return_value=mock_reward), \
     patch('src.modules.ucc_search.environment.SimulatorFactory.create_simulator',
           return_value=mock_simulator):
    env = UCCSearchEnv(molecule_data, config={"environment": {"max_depth": 1}})
    env.reset()
    print("current_excitations length before step:", len(env.current_excitations))
    print("max_depth:", env.config.get("max_depth"))
    obs, reward, done, info = env.step(0)
    print("reward:", reward)
    print("done:", done)
    print("termination_reason:", info.get('termination_reason'))
    print("current_excitations length after step:", len(env.current_excitations))
    print("mock reward compute_reward called:", mock_reward.compute_reward.called)
    if mock_reward.compute_reward.called:
        print("compute_reward args:", mock_reward.compute_reward.call_args)