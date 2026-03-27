"""
UCC compatibility tests for RL agents.

These tests validate that RL agents can interface with UCC quantum chemistry
data from Phase 1 Task 001.
"""

import sys
import pytest
import numpy as np

# Add Task 001 to path for import
sys.path.append("../001")


class TestUCCCompatibility:
    """Test UCC compatibility of RL agents."""

    def test_import_task001_modules(self):
        """Test that Task 001 modules can be imported."""
        from src.modules.molecule_processor import MoleculeData, process_molecule

        assert MoleculeData is not None
        assert process_molecule is not None

    def test_molecule_data_structure(self):
        """Test basic structure of MoleculeData."""
        from src.modules.molecule_processor import MoleculeData

        # Create a mock MoleculeData object to understand structure
        # This test doesn't require actual molecule processing
        mock_hamiltonian = None  # Would be QubitOperator
        mock_data = MoleculeData(
            hamiltonian=mock_hamiltonian,
            n_qubits=4,
            reference_state=np.array([1.0, 0.0, 0.0, 0.0]),
            fci_energy=-1.5,
            molecular_info={"molecule": "H2", "bond_length": 0.74}
        )

        assert mock_data.n_qubits == 4
        assert mock_data.fci_energy == -1.5
        assert isinstance(mock_data.reference_state, np.ndarray)
        assert mock_data.reference_state.shape == (4,)

    @pytest.mark.slow
    def test_h2_molecule_processing(self):
        """Test processing H2 molecule (slow test)."""
        from src.modules.molecule_processor import process_molecule

        # Process H2 molecule - this is relatively fast
        molecule_data = process_molecule(
            molecule="H2",
            bond_length=0.74,
            ansatz_type="UCC"
        )

        # Verify structure
        assert molecule_data.n_qubits > 0
        assert molecule_data.fci_energy < 0  # Energy should be negative
        assert isinstance(molecule_data.reference_state, np.ndarray)
        assert molecule_data.reference_state.shape == (2**molecule_data.n_qubits,)
        assert molecule_data.hamiltonian is not None

        print(f"H2 processed: {molecule_data.n_qubits} qubits, "
              f"FCI energy: {molecule_data.fci_energy}")

    def test_rlagent_ucc_helper_methods(self):
        """Test RLAgent UCC helper methods with mock molecule data."""
        from src.modules.rl_agents.base_agent import RLAgent

        # Create concrete agent for testing
        class TestAgent(RLAgent):
            def act(self, state):
                return 0, {}

            def learn(self, experience):
                return {}

            def save(self, path):
                pass

            def load(self, path):
                pass

        agent = TestAgent()

        # Test format_ucc_state with mock molecule data
        energy = -1.5  # Mock energy
        circuit_params = np.array([0.1, 0.2, 0.3, 0.4])  # Mock circuit parameters

        state = agent.format_ucc_state(energy, circuit_params)
        expected = np.concatenate([[energy], circuit_params])
        assert np.array_equal(state, expected)

        # Test parse_ucc_action
        action_idx = 2
        action_info = agent.parse_ucc_action(action_idx)
        assert action_info["excitation_idx"] == action_idx

    @pytest.mark.slow
    def test_agent_with_real_molecule_data(self):
        """Test agent interface with real H2 molecule data (slow)."""
        from src.modules.molecule_processor import process_molecule
        from src.modules.rl_agents.base_agent import RLAgent

        # Process H2 molecule
        molecule_data = process_molecule(
            molecule="H2",
            bond_length=0.74,
            ansatz_type="UCC"
        )

        # Create concrete agent
        class TestAgent(RLAgent):
            def act(self, state):
                # Simple deterministic action for testing
                return 0, {"state": state}

            def learn(self, experience):
                return {"loss": 0.0}

            def save(self, path):
                pass

            def load(self, path):
                pass

        agent = TestAgent()

        # Test format_ucc_state with real data
        # Use FCI energy as mock energy, and some random circuit parameters
        energy = molecule_data.fci_energy
        circuit_params = np.random.randn(10)  # Mock circuit parameters

        state = agent.format_ucc_state(energy, circuit_params)
        assert state[0] == energy
        assert np.array_equal(state[1:], circuit_params)

        # Test that state can be passed to act()
        action, info = agent.act(state)
        assert isinstance(action, int)
        assert isinstance(info, dict)

    def test_ucc_state_action_integration(self):
        """Test integration of UCC state formatting and action parsing."""
        from src.modules.rl_agents.base_agent import RLAgent

        class TestAgent(RLAgent):
            def act(self, state):
                # Return action based on state (simple threshold)
                return 0 if state[0] < -1.0 else 1, {}

            def learn(self, experience):
                return {}

            def save(self, path):
                pass

            def load(self, path):
                pass

        agent = TestAgent()

        # Simulate UCC optimization loop
        energies = [-1.5, -1.2, -1.0, -0.8]
        circuit_params_list = [np.random.randn(5) for _ in range(4)]

        actions = []
        for energy, params in zip(energies, circuit_params_list):
            state = agent.format_ucc_state(energy, params)
            action, _ = agent.act(state)
            action_info = agent.parse_ucc_action(action)
            actions.append((action, action_info))

        # Verify actions were generated
        assert len(actions) == 4
        for action, info in actions:
            assert isinstance(action, int)
            assert isinstance(info, dict)
            assert info["excitation_idx"] == action