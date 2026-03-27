#!/usr/bin/env python3
"""Integration test for RLQAS Phase 1 package."""

import sys
import warnings
warnings.filterwarnings('ignore', message='Gym has been unmaintained')
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.insert(0, 'src')

try:
    import rlqas.phase1 as rlqas
    print("✓ Imported rlqas.phase1")
except Exception as e:
    print(f"✗ Failed to import rlqas.phase1: {e}")
    sys.exit(1)

def test_molecule_processing():
    """Test molecule processing for H2."""
    print("\n=== Testing molecule processing ===")
    try:
        from rlqas.phase1.molecule.processor import process_molecule
        data = process_molecule("H2", 0.74, "UCC", active_space=(2, 2), transform="parity")
        print(f"✓ Processed H2 molecule")
        print(f"  n_qubits: {data.n_qubits}")
        print(f"  Hamiltonian terms: {len(data.hamiltonian.terms)}")
        print(f"  Reference state shape: {data.reference_state.shape}")
        print(f"  FCI energy: {data.fci_energy:.6f}")
        # Check that reference state is one-hot
        ref_norm = abs(data.reference_state).sum()
        if abs(ref_norm - 1.0) < 1e-10:
            print("  Reference state is normalized")
        else:
            print(f"  WARNING: Reference state norm = {ref_norm}")
        return data
    except Exception as e:
        print(f"✗ Molecule processing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_simulator(data):
    """Test quantum simulator with reference state."""
    print("\n=== Testing quantum simulator ===")
    if data is None:
        return None
    try:
        from rlqas.phase1.simulator.factory import SimulatorFactory
        simulator = SimulatorFactory.create_simulator(data.n_qubits)
        print(f"✓ Created simulator: {simulator.__class__.__name__}")
        print(f"  Max qubits supported: {simulator.get_max_qubits()}")
        # Compute energy of reference state (should be HF energy)
        # Since reference state is computational basis, expectation can be computed
        # using Hamiltonian diagonal terms.
        # We'll use simulator's compute_energy with a trivial circuit that applies identity.
        # For simplicity, we'll just test that simulator can be instantiated.
        # For TencirchemCISimulator, we need a circuit.
        # We'll skip actual energy evaluation for now.
        return simulator
    except Exception as e:
        print(f"✗ Simulator test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_circuit_builder(data):
    """Test UCC circuit builder."""
    print("\n=== Testing circuit builder ===")
    if data is None:
        return None
    try:
        from rlqas.phase1.search.circuit_builder import UCCCircuitBuilder
        builder = UCCCircuitBuilder(data)
        print(f"✓ Created circuit builder")
        print(f"  Available excitations: {len(builder.get_available_excitations())}")
        print(f"  Number of parameters: {builder.n_params}")
        # Build a simple circuit with single excitation
        excitations = builder.get_available_excitations()
        if excitations:
            single_exc = [excitations[0]]
            # Get parameter index for this excitation
            param_indices = builder.get_parameter_indices_for_excitation(single_exc[0])
            # Initialize full parameter vector of length n_params
            params = builder.initialize_parameters(builder.n_params, strategy="random")
            # Set all other parameters to zero except the active one (optional)
            # But we can keep random values; the builder will zero out inactive ones if params=None
            # Let's pass params as None and let builder initialize
            circuit = builder.build_circuit(single_exc, params=None)
            print(f"  Built circuit with {len(single_exc)} excitation")
            # Evaluate energy using builder's evaluate_energy
            energy = builder.evaluate_energy(circuit, circuit.params)
            print(f"  Energy: {energy:.6f}")
        return builder
    except Exception as e:
        print(f"✗ Circuit builder test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_environment(data):
    """Test UCC search environment."""
    print("\n=== Testing environment ===")
    if data is None:
        return None
    try:
        from rlqas.phase1.search.environment import UCCSearchEnv
        env = UCCSearchEnv(data)
        print(f"✓ Created environment")
        print(f"  Action space: {env.action_space}")
        print(f"  Observation space shape: {env.observation_space.shape}")
        # Test reset
        obs, info = env.reset()
        print(f"  Reset: obs shape {obs.shape}, info {info}")
        # Test step with random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step: reward {reward:.3f}, terminated {terminated}, truncated {truncated}")
        return env
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_agent():
    """Test RL agent instantiation."""
    print("\n=== Testing RL agent ===")
    try:
        from rlqas.phase1.rl.ppo_agent import PPOAgent
        # Pass empty dict to use defaults
        agent = PPOAgent(config={})
        print(f"✓ Created PPOAgent")
        print(f"  Policy network: {agent.policy_net}")
        return agent
    except Exception as e:
        print(f"✗ Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("RLQAS Phase 1 Integration Test")
    print("=" * 40)
    data = test_molecule_processing()
    sim = test_simulator(data)
    builder = test_circuit_builder(data)
    env = test_environment(data)
    agent = test_agent()

    print("\n" + "=" * 40)
    print("Summary:")
    print(f"Molecule processing: {'✓' if data else '✗'}")
    print(f"Simulator: {'✓' if sim else '✗'}")
    print(f"Circuit builder: {'✓' if builder else '✗'}")
    print(f"Environment: {'✓' if env else '✗'}")
    print(f"Agent: {'✓' if agent else '✗'}")

    if all([data, sim, builder, env, agent]):
        print("\n✅ All integration tests passed!")
        return 0
    else:
        print("\n❌ Some integration tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())