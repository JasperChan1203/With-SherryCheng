# RLQAS Phase 2 Cluster Usage Guide

## Overview
This guide explains how to run the complete RLQAS Phase 2 implementation on a SLURM cluster. Phase 2 includes all 6 tasks implemented in a unified manner.

## Phase 2 Components
The unified Phase 2 implementation includes:
1. **Task 001**: DQN implementation (pre-existing, will be integrated)
2. **Task 002**: Sequential Testing Framework
3. **Task 003**: HEA Search Module
4. **Task 004**: Experiment Management System
5. **Task 005**: Agent Autonomous RL Exploration (Key Innovation)
6. **Task 006**: Phase 2 Integration Test

## Resource Requirements

### Minimum Requirements
- **CPU**: 8+ cores (quantum simulation + RL training)
- **GPU**: 1x V100/A100 (recommended for RL neural network acceleration)
- **Memory**: 64 GB (quantum statevectors + neural networks)
- **Storage**: 1 GB for code and results
- **Time**: 72 hours (for all 6 tasks completion)

### Recommended Configuration
```
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --partition=4V100  # Adjust based on cluster
#SBATCH --cpus-per-task=8
```

## Dependencies

### Required Software
1. **Phase 1 Integrated Package** (`../../Phase1/006/`): Must be completed and installed
2. **Python Environment**: With all Phase 1 dependencies
3. **Phase 2 Task 001** (`../001/`): DQN implementation (will be integrated)

### Python Dependencies
- `tencirchem-ng>=2024.10` (quantum chemistry)
- `openfermion>=1.5` (fermion-to-qubit transformations)
- `gymnasium>=1.0.0` (RL environments)
- `stable-baselines3>=2.0.0` (RL algorithms)
- `torch>=1.9` (neural networks)
- `PyYAML>=6.0` (configuration files)
- `pyscf>=2.0` (quantum chemistry integrals)

## Execution Methods

### Method 1: Unified Script (Recommended)
Use the unified submission script that handles all dependencies and execution:

```bash
# Navigate to Phase 2 directory
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2

# Check cluster environment
./check_cluster.sh

# Start interactive session
salloc --job-name=ralph-phase2 --nodes=1 --ntasks=1 --gpus-per-task=1 \
       --mem=64G --time=72:00:00 --partition=4V100 --cpus-per-task=8

# Run unified implementation
./run_phase2_unified.sh
```

### Method 2: Direct Batch Submission
Submit directly to the batch system:

```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full
sbatch slurm_batch.sh

# Monitor job
squeue -u $USER
tail -f ralph_phase2_full_<jobid>.out
```

### Method 3: Interactive Development
For step-by-step development and debugging:

```bash
# Start interactive session
salloc --job-name=ralph-phase2-dev --nodes=1 --ntasks=1 --gpus-per-task=1 \
       --mem=64G --time=24:00:00 --partition=4V100

# Navigate and run manually
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full
./ralph.sh --tool claude 50
```

## Directory Structure

```
ralph/Phase2/
├── check_cluster.sh                    # Environment verification script
├── run_phase2_unified.sh               # Unified execution script
├── full/                               # Complete Phase 2 implementation
│   ├── CLAUDE.md                       # Ralph prompt (all 6 tasks)
│   ├── prd.json                        # Requirements document
│   ├── ralph.sh                        # Ralph execution script
│   ├── slurm_batch.sh                  # SLURM batch script
│   ├── progress.txt                    # Progress log (created during run)
│   └── slurm_logs/                     # SLURM logs directory (created)
├── 001/                                # Phase 2 Task 001 (DQN implementation)
│   ├── src/rlqas/phase2/rl/            # DQN agent and factory
│   └── tests/                          # Task 001 tests
└── results/                            # Phase 2 results (created during run)
```

## Execution Phases

Ralph will execute Phase 2 in the following sequence:

### Phase 0: Task 001 Verification and Integration
- Verify existing DQN implementation from `../001/`
- Integrate into unified Phase 2 package structure

### Phase 1: Task 002 - Sequential Testing Framework
- Implement `SequentialRLTester` class
- Create algorithm comparison utilities
- Add metrics collection for excitation operator counts

### Phase 2: Task 003 - HEA Search Module
- Implement `HEASearchEnv` environment
- Create `HEACircuitBuilder` with entanglement patterns
- Add HEA configuration system

### Phase 3: Task 004 - Experiment Management System
- Implement `ExperimentManager` class
- Create YAML/JSON configuration support
- Build results database

### Phase 4: Task 005 - Agent Autonomous RL Exploration
- Implement RL algorithm exploration framework
- Create runtime adaptive environment enhancement
- Add capability detection and implementation system

### Phase 5: Task 006 - Phase 2 Integration Test
- Run comprehensive tests on LiH molecules
- Test Jordan-Wigner transformation
- Validate chemical accuracy (<1.6 mHa)
- Generate performance benchmarks

## Monitoring Progress

### During Execution
1. **SLURM Output**: `tail -f ralph_phase2_full_<jobid>.out`
2. **Progress Log**: Check `full/progress.txt` for detailed phase completion
3. **Error Log**: Check `full/slurm_logs/` for error details

### Completion Signals
Ralph signals completion with `<promise>COMPLETE</promise>` in output. Successful completion includes:

1. All 6 tasks implemented and integrated
2. HEA search module with multiple entanglement patterns
3. Experiment management system operational
4. Autonomous RL exploration framework functional
5. Integration tests passing (>90% coverage)
6. Chemical accuracy achieved on LiH test cases

## Troubleshooting

### Common Issues

#### 1. Dependencies Not Found
```
Error: Phase 1 package not found
```
**Solution**: Install Phase 1 first:
```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase1/006
pip install -e .
```

#### 2. GPU Not Available
```
CUDA not available (CPU only)
```
**Solution**: This is acceptable but slower. Consider:
- Requesting GPU partition: `--partition=GPU` or `--partition=4V100`
- Adjusting RL training parameters for CPU-only execution

#### 3. Memory Issues
```
Out of memory error
```
**Solution**: Request more memory:
- Increase `--mem=128G` in SLURM request
- Reduce quantum simulator memory usage in configuration

#### 4. Time Limit Exceeded
```
Job killed due to time limit
```
**Solution**: Request more time or checkpoint progress:
- Increase `--time=96:00:00`
- Ralph automatically checkpoints in `progress.txt`

### Debug Steps

1. **Check Environment**: Run `./check_cluster.sh`
2. **Test Phase 1**: Verify Phase 1 installation works
3. **Test Task 001**: Ensure DQN implementation exists
4. **Monitor Resources**: Use `sacct -j <jobid>` to check resource usage
5. **Review Logs**: Check output and error files in `slurm_logs/`

## Expected Output

### Successful Completion
- Complete `src/rlqas/phase2/` package structure
- All tests passing with >90% coverage
- `results/` directory with test outputs
- Chemical accuracy (<1.6 mHa) on LiH test cases
- Algorithm comparison results

### Files Generated
```
full/
├── progress.txt                    # Complete implementation log
├── src/rlqas/phase2/               # Complete Phase 2 package
│   ├── rl/                         # RL agents (including DQN)
│   ├── sequential_tester/          # Testing framework
│   ├── hea_search/                 # HEA module
│   ├── experiment/                 # Experiment management
│   └── adaptation/                 # Autonomous exploration
├── tests/                          # Test suite
├── config/                         # Configuration files
├── results/phase2_integration/     # Test results
│   ├── algorithm_comparison.json
│   ├── lih_10qubits_results/
│   ├── lih_12qubits_results/
│   └── hea_results/
└── docs/                           # Documentation
```

## Post-Completion Validation

After Ralph completes, validate the implementation:

```bash
# Navigate to Phase 2 directory
cd /curie-home/jpchen/scratch/LLM/code/RLQAS/ralph/Phase2/full

# Run integration tests
python -m pytest tests/integration/ -v

# Check test coverage
python -m pytest --cov=rlqas.phase2 --cov-report=html

# Run example tests
python scripts/run_phase2_tests.py --quick-test

# Verify chemical accuracy
python scripts/validate_lih_10qubits.py
```

## Performance Optimization Tips

1. **GPU Utilization**: Ensure RL neural networks use GPU
2. **Memory Efficiency**: Use CI vector engine for quantum simulation
3. **Checkpointing**: Ralph automatically saves progress
4. **Parallel Testing**: Configure experiment manager for parallel execution
5. **Caching**: Quantum integrals are cached by tencirchem

## Support

For issues:
1. Check `slurm_logs/` for error messages
2. Review `progress.txt` for Ralph's implementation steps
3. Verify Phase 1 installation
4. Ensure sufficient resources are requested

## Notes

- **Login Nodes**: Never run Ralph directly on login nodes
- **Time Estimates**: Phase 2 may take 48-72 hours to complete
- **Checkpointing**: Ralph saves progress after each phase
- **Resume Capability**: Can resume execution if interrupted