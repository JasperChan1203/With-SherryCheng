# RLQAS LiH VQE Circuit Generation Test

A comprehensive test suite for validating Ralph AI agent's ability to generate variational quantum eigensolver (VQE) circuits for LiH molecule. This test extends the H2 Hamiltonian test to more complex quantum computational chemistry tasks.

## 📋 Project Overview

This project tests Ralph (an autonomous AI agent) on an advanced quantum chemistry task: generating a VQE circuit for LiH molecule at 2.0 Å bond length with specific active space settings. The test includes:

- **Active space selection**: (2 electrons, 3 orbitals) with orbital selection `aslst=[1, 2, 5]`
- **Parity transformation**: 4-qubit Hamiltonian reduction
- **VQE circuit design**: Parameterized ansatz with Scipy BFGS optimization
- **Chemical accuracy target**: Final energy within 1.6 mHa of FCI energy
- **Iterative learning**: With benchmark validation feedback
- **Cluster-ready execution**: Can be adapted for SLURM

## 🎯 Test Objective

Validate that Ralph can correctly:

1. Use PySCF to define LiH molecule with 2.0 Å bond length and specified active space
2. Perform parity transformation to obtain 4-qubit Hamiltonian
3. Design VQE circuit using Tencirchem with parameter optimization
4. Track energy convergence during optimization
5. Achieve chemical accuracy (within 1.6 mHa of FCI energy)
6. Format results according to specification (circuit gates, parameters, energy curve)

## 📁 File Structure

```
lih_test/
├── README.md                          # This file
├── generate_lih_benchmark.py          # Generate reference FCI values (user-provided)
├── lih_benchmark.json                 # Reference values (not accessible to Ralph)
├── validate_lih.py                    # Validation script (compares Ralph's results)
├── prd.json                           # Full project requirements document
└── Ralph_Test_LiH_VQE/                # Ralph test directory
    ├── prd.json                       # Test requirements document
    ├── CLAUDE.md                      # Ralph agent prompt instructions
    ├── AGENTS.md                      # Ralph's knowledge base (updated during runs)
    ├── ralph.sh                       # Ralph execution script
    ├── progress.txt                   # Ralph's progress log
    ├── ralph_learning_log.txt         # Detailed learning log
    └── (Ralph will create):
        ├── generate_lih_vqe.py        # Ralph-generated implementation
        └── lih_results.json           # Ralph's implementation results
```

## 🔧 Key Components

### 1. Benchmark Generation (`generate_lih_benchmark.py`)
Creates benchmark file with user-provided FCI energy reference (-7.844879 Hartree):
- Uses user-provided FCI energy for LiH at 2.0 Å
- Molecular settings for validation (bond length, active space, orbitals)
- Validation tolerances (1.6 mHa chemical accuracy)
- Saves to `lih_benchmark.json`

### 2. Validation System (`validate_lih.py`)
Compares Ralph's results against benchmark:
- Checks molecular definition (bond length, active space, orbitals)
- Validates VQE energy within 1.6 mHa of FCI
- Verifies circuit properties (gates, parameters)
- Checks energy convergence curve
- Returns exit code 0 (success) or 1 (failure)
- Provides detailed error feedback for iterative improvement

### 3. Ralph Test Environment (`Ralph_Test_LiH_VQE/`)
Contains all files Ralph needs:
- `prd.json`: Task requirements and validation procedure
- `CLAUDE.md`: Ralph's instructions including validation workflow and H2 VQE reference
- `ralph.sh`: Execution script with Claude Code agent support
- Ralph generates `generate_lih_vqe.py` and `lih_results.json`

## 🚀 Test Execution Workflow

### Step 1: Environment Verification
```bash
cd lih_test
# Verify required packages are installed
python3 -c "
import pyscf, tencirchem, numpy, scipy
print(f'PySCF: {pyscf.__version__}')
print(f'tencirchem: {tencirchem.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'SciPy: {scipy.__version__}')
print('✓ All dependencies available')
"
```

### Step 2: Generate Benchmark Values
```bash
cd lih_test
python generate_lih_benchmark.py
```
Creates `lih_benchmark.json` with user-provided FCI energy reference (-7.844879 Hartree).

### Step 3: Run Ralph Test
```bash
cd lih_test/Ralph_Test_LiH_VQE
./ralph.sh --tool claude 10
```

### Step 4: Review Results
Check these files after Ralph completes:

1. **`validation_summary.txt`** - Validation results
2. **`progress.txt`** - Ralph's progress log
3. **`ralph_learning_log.txt`** - Detailed thought process
4. **`lih_results.json`** - Ralph's implementation results
5. **`generate_lih_vqe.py`** - Ralph-generated Python script

## 🔄 Iterative Feedback Mechanism

Ralph follows this iterative learning process:

```
1. Read PRD & CLAUDE.md → Understand task requirements and H2 reference
2. Implement → Write Python script + generate lih_results.json
3. Validate → Run: python ../validate_lih.py lih_results.json
   ├─ ✅ Exit code 0 → All checks pass → Output <promise>COMPLETE</promise>
   └─ ❌ Exit code 1 → Read validation errors → Improve implementation
4. Repeat → Maximum 10 iterations (configurable)
```

**Key Features:**
- **Feedback-driven**: Ralph receives specific error messages from validation script
- **Benchmark-hidden**: Ralph cannot directly access benchmark values or generation code
- **Reference-guided**: H2 VQE example provided for circuit structure patterns
- **Knowledge accumulation**: Updates `AGENTS.md` with learned patterns

## 📊 Validation Criteria

Ralph's implementation must satisfy:

| Validation Check | Target Value | Tolerance |
|------------------|--------------|-----------|
| **Bond Length** | 2.0 Å | ±0.01 Å |
| **Active Space** | (2, 3) | Exact match |
| **Orbital Selection** | [1, 2, 5] | Exact match (order matters) |
| **Qubit Count** | 4 | Exact match |
| **VQE Energy** | FCI energy | ±0.0016 Hartree (1.6 mHa) |
| **Circuit** | Valid gates/parameters | Must produce |

## 🧪 Expected Challenges

This test is significantly more complex than H2 Hamiltonian generation:

1. **Active space implementation**: Correct PySCF usage for orbital selection
2. **Parity transformation**: Reducing to 4-qubit Hamiltonian
3. **VQE circuit design**: Designing expressive ansatz for 4-qubit system
4. **Optimization convergence**: Achieving chemical accuracy with BFGS
5. **Energy tracking**: Recording convergence curve during optimization

## 🎯 Success Metrics

- **Technical success**: VQE energy within 1.6 mHa of FCI energy
- **Process success**: Ralph completes within 10 iterations with clear learning progression
- **Learning success**: Knowledge captured in AGENTS.md for future quantum tasks
- **Reproducibility**: Code generates consistent results

## 📈 Extending the Framework

This test framework can be extended for:

1. **More complex molecules** (H2O, NH3, etc.)
2. **Different active spaces** and orbital selections
3. **Alternative transformations** (Jordan-Wigner, Bravyi-Kitaev)
4. **Advanced VQE ansätze** (UCCSD, hardware-efficient, etc.)
5. **Different optimizers** (COBYLA, SPSA, Adam)
6. **Noise-aware simulations** (with device noise models)

## 📝 License & Attribution

This test framework is part of the RLQAS (Reinforcement Learning for Quantum Architecture Search) project. Developed for testing autonomous AI agents in advanced quantum computational chemistry tasks.

**Author**: RLQAS Test Team
**Environment**: Curie cluster compatible
**Date**: January 2026
**Status**: Test framework ready for Ralph execution

---

*For questions or issues, refer to the test documentation or contact the RLQAS project maintainers.*