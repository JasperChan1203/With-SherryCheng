# Ralph Agents Knowledge Base

This file contains patterns, learnings, and best practices discovered during Ralph runs.

## RLQAS Project Patterns

### Quantum Computing Environment Setup
- **Molecular Hamiltonian Generation**: For LiH with 4 qubits, use tencirchem library which provides built-in methods for molecular Hamiltonians with parity transformation and active space selection.
- **Tencirchem Usage**: Tencirchem specializes in quantum computational chemistry. Use `tencirchem` for Hamiltonian generation, FCI energy calculation, and potentially circuit simulation.
- **Qubit Mapping**: Parity transformation maps fermionic operators to qubit operators, reducing qubit count but potentially increasing gate complexity. Tencirchem handles this automatically.
- **Exact Energy Reference**: Always compute Full Configuration Interaction (FCI) energy for reference to calculate accuracy rewards. Tencirchem provides efficient FCI calculations.

### Reinforcement Learning for Quantum Architecture Search
- **State Representation**: Encode current circuit as graph or sequence, include depth, gate counts, and estimated energy.
- **Action Space Design**: Keep action space manageable: single-qubit rotations (Rx, Ry, Rz), two-qubit gates (CNOT), parameter adjustments, and termination.
- **Reward Function Balance**: Multi-objective reward should balance accuracy (energy error), circuit depth, and gate count. Start with simple weighted sum.
- **Training Stability**: Quantum architecture search has sparse rewards. Consider reward shaping, curriculum learning, or intermediate rewards.

### Code Organization Best Practices
- **Modular Design**: Separate quantum simulation, RL environment, agent training, and analysis.
- **Reproducibility**: Seed random number generators, log configurations, and save model checkpoints.
- **Visualization**: Plot training curves, circuit structures, and energy convergence.
- **Documentation**: Comment code thoroughly, especially for quantum-specific operations.

### Hardware and Computational Resources
- **GPU Acceleration**: RL training (neural networks) benefits from GPU acceleration. Quantum simulation with tencirchem typically runs on CPU.
- **CPU vs GPU Decision**: For 4-qubit LiH test, CPU-only is sufficient. Design code to optionally use GPU for future scalability.
- **Cluster Environment**: If running on a cluster with job submission systems:
  - Provide example submission scripts (SLURM, PBS, etc.)
  - Design code to checkpoint training progress for preemptible jobs
  - Log resource usage (CPU, GPU, memory) for optimization
  - Consider using distributed training for larger experiments
- **Resource Management**: Monitor memory usage during quantum simulations. Larger qubit counts require exponential memory.
- **Cluster Job Submission**: This cluster uses SLURM. Never run computationally intensive tasks on login nodes.
  - **Interactive jobs**: Use `salloc` to get interactive shell on compute node: `salloc --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=24:00:00 --partition=CPU`
  - **Batch jobs**: Use `sbatch slurm_batch.sh` for non-interactive execution
  - **Check partitions**: Use `sinfo` to see available partitions and resources
  - **Monitor jobs**: Use `squeue -u $USER` and `sacct` for job status and accounting
  - **Files created**: `slurm_interactive.sh`, `slurm_batch.sh`, `cluster_usage_guide.md` provide templates and guidance

### Reference Code Insights (from existing DQAS implementation)
- **Existing DQAS Implementation**: Reference code shows differentiable quantum architecture search using tencirchem and TensorCircuit.
- **Key Components from Reference**:
  - Hamiltonian generation with tencirchem's `UCCSD` and `HEA` classes
  - FCI energy calculation for reward reference
  - Circuit building with TensorCircuit's `Circuit` class
  - Parameterized gate operations (Rx, Ry, Rz, CNOT)
  - Structure and parameter optimization with gradient methods
- **Adaptation to RLQAS**:
  - Replace gradient-based optimization with reinforcement learning (PPO)
  - Design state representation from circuit structure and current metrics
  - Define action space based on available quantum gates
  - Implement reward function using energy accuracy and circuit efficiency
  - Maintain compatibility with tencirchem for quantum computations
- **Technical Stack Compatibility**: Reference code uses tencirchem, TensorCircuit, JAX - align with PRD requirements.

### Common Pitfalls and Solutions
- **Exploration vs Exploitation**: Quantum search spaces are huge. Start with high exploration, gradually reduce.
- **Computational Cost**: Quantum simulations are expensive. Use statevector simulation for small systems (≤10 qubits), consider approximations for larger.
- **Reward Scaling**: Energy differences can be very small. Use logarithmic or normalized rewards.
- **Circuit Validity**: Ensure circuits are physically realizable (proper connectivity, valid parameters).

## Test Project Patterns
- Simple test scripts should include main guards and be executable
- Progress logging should include timestamps and clear descriptions
- Each objective completion should be verifiable against acceptance criteria

## PySCF Hamiltonian Generation Patterns
- **Molecule Definition**: Use `pyscf.gto.M` to define molecule geometry. Specify atom positions, charge, spin, basis set. For H2 at 0.5 Å bond length, use `atom=[["H", 0, 0, 0], ["H", bond_length, 0, 0]]`, `basis="sto-3g"`.
- **Hartree-Fock Calculation**: Use `pyscf.scf.RHF` for restricted Hartree-Fock. Call `kernel()` method to compute HF energy and molecular orbitals.
- **Integral Extraction**: Core Hamiltonian (`get_hcore()`) and two-electron integrals (`intor("int2e")`) are needed for post-HF methods and quantum simulation.
- **Full Configuration Interaction (FCI)**: Use `pyscf.fci.FCI` solver to compute exact energy within the basis set. Requires core Hamiltonian, two-electron integrals, number of electrons.
- **Qubit Mapping**: In Jordan-Wigner mapping, each spin orbital maps to one qubit. Number of qubits = number of spatial orbitals × 2 (for spin). For H2/sto-3g, 2 spatial orbitals → 4 qubits.
- **Hamiltonian Properties**: Core Hamiltonian should be Hermitian. Two-electron integrals have 8-fold symmetry. Save integrals to disk for later use with quantum simulation libraries.
- **Verification**: Compare FCI energy with literature values. Check Hermiticity of core Hamiltonian. Ensure qubit count matches expected mapping.
- **FCI Calculation Correctness**: When computing FCI energy with PySCF, use `fci.FCI(mf)` where `mf` is the RHF object rather than manually transforming integrals. Manual transformation can lead to double-counting of nuclear repulsion or incorrect integral scaling. The built-in method ensures correct Hamiltonian construction.

## Environment Configuration for RLQAS Project
- **PySCF Installation**: Successfully installed PySCF 2.12.0 via pip in base Python environment
- **Dependency Management**: For current H2 Hamiltonian task, only PySCF and numpy are required
- **Conda Environment Issues**: Encountered corrupted setuptools package in conda cache; pip installation worked as alternative
- **Future Environment Setup**: For full RLQAS project, consider:
  - Creating fresh conda environment after cleaning cache: `conda clean --all`
  - Using pip for PySCF if conda issues persist
  - Installing additional packages: tencirchem, stable-baselines3, torch, tensorcircuit, jax, optax, openfermion
- **Current Working Environment**: Base Python with PySCF 2.12.0 is sufficient for H2 Hamiltonian generation task

## Ralph Learning Log Best Practices
- **Purpose**: Document iterative learning progress, thought process, challenges, and solutions
- **Format**: Create a dedicated learning log file (e.g., `ralph_learning_log.txt` or `ralph_learning_log.md`)
- **Content**: Include for each iteration:
  - Iteration number and timestamp
  - Observations about the current state
  - Decisions made and reasoning
  - Problems encountered and solutions attempted
  - Lessons learned
- **Integration**: Reference the learning log in progress.txt updates
- **Value**: Demonstrates Ralph's learning capability and provides audit trail for debugging