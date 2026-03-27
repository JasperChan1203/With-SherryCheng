# RLQAS H2 Hamiltonian Generation Test

A comprehensive test suite for validating Ralph AI agent's ability to generate molecular Hamiltonians for quantum computational chemistry tasks. This test focuses on the H2 molecule at 0.5 Å bond length using PySCF, with iterative feedback-based learning and automated validation.

## 📋 Project Overview

This project tests Ralph (an autonomous AI agent) on a foundational quantum chemistry task: generating the Hamiltonian for an H2 molecule at 0.5 Å bond length. The test includes:
- **Autonomous implementation** by Ralph using PySCF
- **Iterative learning** with benchmark validation feedback
- **Cluster-ready execution** with SLURM support
- **Comprehensive logging** of Ralph's learning process

## 🎯 Test Objective

Validate that Ralph can correctly:
1. Use PySCF to define an H2 molecule with 0.5 Å bond length (STO-3G basis)
2. Perform Hartree-Fock and Full Configuration Interaction (FCI) calculations
3. Extract Hamiltonian information (integrals, Hermiticity, qubit count)
4. Format results according to specification
5. Iteratively improve implementation based on validation feedback

## 📁 File Structure

```
With-SherryCheng/ralph/                # Ralph test suite directory
├── README.md                          # This file (test documentation)
├── generate_h2_benchmark.py           # Generate reference values using PySCF
├── h2_benchmark.json                  # Reference values (not accessible to Ralph)
├── validate_h2.py                     # Validation script (compares Ralph's results)
├── check_cluster.sh                   # Cluster status checker
├── slurm_batch.sh                     # SLURM batch job script
├── slurm_interactive.sh               # SLURM interactive job script
├── cluster_usage_guide.md             # Cluster usage documentation
├── prd_full_backup.json               # Full RLQAS project requirements (backup)
└── Ralph_Test_H2_Hamiltonian/         # Ralph test directory
    ├── prd.json                       # Test requirements document
    ├── CLAUDE.md                      # Ralph agent prompt instructions
    ├── AGENTS.md                      # Ralph's knowledge base (updated during runs)
    ├── ralph.sh                       # Ralph execution script
    ├── generate_h2_hamiltonian.py     # Ralph-generated implementation
    ├── h2_results.json                # Ralph's implementation results
    ├── validation_summary.txt         # Validation results summary
    ├── progress.txt                   # Ralph's progress log
    ├── ralph_learning_log.txt         # Detailed learning log
    └── debug_fci.py                   # Debug script created by Ralph
```

## 🔧 Key Components

### 1. Benchmark Generation (`generate_h2_benchmark.py`)
Generates reference values for H2 at 0.5 Å bond length:
- HF energy: -1.042996274540095 Hartree
- FCI energy: -1.0551597944706257 Hartree
- Hamiltonian properties (Hermiticity, 2-4 qubits)
- Saves to `h2_benchmark.json`

### 2. Validation System (`validate_h2.py`)
Compares Ralph's results against benchmark:
- Checks molecular definition (bond length, basis set)
- Validates energy calculations within tolerances
- Verifies Hamiltonian properties
- Returns exit code 0 (success) or 1 (failure)
- Provides detailed error feedback for iterative improvement

### 3. Ralph Test Environment (`Ralph_Test_H2_Hamiltonian/`)
Contains all files Ralph needs:
- `prd.json`: Task requirements and validation procedure
- `CLAUDE.md`: Ralph's instructions including validation workflow
- `ralph.sh`: Execution script with Claude Code agent support
- Ralph generates `generate_h2_hamiltonian.py` and `h2_results.json`

## 🚀 Test Execution Workflow

### Step 1: Generate Benchmark Values
```bash
cd ralph  # Navigate to ralph directory
python generate_h2_benchmark.py
```
Creates `h2_benchmark.json` with reference values.

### Step 2: Run Ralph Test (Three Methods)

#### Method A: SLURM Batch Job (Recommended)
```bash
cd ralph  # Navigate to ralph directory
sbatch slurm_batch.sh
```
Output files: `ralph_h2_test_<jobid>.out` and `.err`

#### Method B: SLURM Interactive Session
```bash
# Request compute node
salloc --job-name=ralph-h2-test --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=2:00:00 --partition=CPU

# Run test
cd ralph  # Navigate to ralph directory
./slurm_interactive.sh

# Or manually:
cd Ralph_Test_H2_Hamiltonian
./ralph.sh --tool claude 5
exit  # After completion
```

#### Method C: Direct Execution (Login Node - Lightweight Only)
```bash
cd ralph/Ralph_Test_H2_Hamiltonian  # Navigate to test directory
./ralph.sh --tool claude 5
```

### Step 3: Review Results
Check these files after Ralph completes:

1. **`validation_summary.txt`** - Validation results
   ```
   Overall Status: PASS - All checks passed
   MOLECULE     PASS   - OK
   ENERGIES     PASS   - OK
   HAMILTONIAN  PASS   - OK
   ```

2. **`progress.txt`** - Ralph's progress log
3. **`ralph_learning_log.txt`** - Detailed thought process
4. **`h2_results.json`** - Ralph's implementation results
5. **`generate_h2_hamiltonian.py`** - Ralph-generated Python script

## 🔄 Iterative Feedback Mechanism

Ralph follows this iterative learning process:

```
1. Read PRD → Understand task requirements
2. Implement → Write Python script + generate h2_results.json
3. Validate → Run: python ../validate_h2.py h2_results.json
   ├─ ✅ Exit code 0 → All checks pass → Output <promise>COMPLETE</promise>
   └─ ❌ Exit code 1 → Read validation errors → Improve implementation
4. Repeat → Maximum 5 iterations (configurable)
```

**Key Features:**
- **Feedback-driven**: Ralph receives specific error messages from validation script
- **Benchmark-hidden**: Ralph cannot directly access benchmark values or generation code
- **Self-debugging**: Ralph creates debug scripts (e.g., `debug_fci.py`) to solve problems
- **Knowledge accumulation**: Updates `AGENTS.md` with learned patterns

## 📊 Validation Criteria

Ralph's implementation must satisfy:

| Validation Check | Benchmark Value | Tolerance |
|------------------|----------------|-----------|
| **Bond Length** | 0.5 Å | ±0.01 Å |
| **HF Energy** | -1.042996 Hartree | ±0.001 Hartree |
| **FCI Energy** | -1.055160 Hartree | ±0.001 Hartree |
| **Hamiltonian** | Hermitian, 2-4 qubits | Must satisfy |

## 🧪 Test Results Summary

### Successful Execution (Observed)
- **Completion**: Ralph completed task in 1 iteration (of maximum 5)
- **Validation**: All checks passed (`validation_summary.txt` shows PASS)
- **Accuracy**: Energy values match benchmark within tolerance
- **Learning**: Detailed logs show Ralph's debugging process

### Ralph's Implementation (`generate_h2_hamiltonian.py`)
Key features of Ralph's solution:
- Correct H2 molecule definition with 0.5 Å bond length
- Proper Hartree-Fock calculation using `pyscf.scf.RHF`
- Accurate FCI calculation using `pyscf.fci.FCI`
- Hamiltonian information extraction (integrals, Hermiticity check)
- JSON output formatting matching specification

### Learning Process Highlights
1. **Initial Challenge**: Incorrect FCI energy due to integral transformation issues
2. **Debugging**: Created `debug_fci.py` to test multiple FCI calculation methods
3. **Solution**: Used `fci.FCI(mf)` method which correctly handles integral transformation
4. **Validation**: Ran validation script, received feedback, corrected implementation
5. **Completion**: Generated correct `h2_results.json` that passed all validation checks

## 🖥️ Cluster Usage Notes

### Environment Requirements
- **Python 3.8+** with PySCF 2.12.0 and NumPy
- **SLURM scheduler** for cluster execution
- **Compute nodes** for running PySCF calculations (not login nodes)

### Dependencies Check
```bash
# Check if PySCF is available
python3 -c "import pyscf; print(f'PySCF {pyscf.__version__} available')"

# Install if missing
pip install pyscf numpy
```

### Cluster Safety
- **Never run computationally intensive tasks on login nodes**
- **Use SLURM** (`salloc` or `sbatch`) to request compute resources
- **Check cluster status** with `./check_cluster.sh`

## 🎓 Learning Outcomes

This test demonstrates Ralph's capabilities for:

### Technical Implementation
- **PySCF Usage**: Correct molecular definition and quantum chemistry calculations
- **Hamiltonian Extraction**: Proper handling of molecular integrals and properties
- **Validation Integration**: Automated testing and feedback incorporation

### Problem-Solving Skills
- **Debugging**: Systematic testing of alternative approaches (`debug_fci.py`)
- **Iterative Improvement**: Using validation feedback to correct implementation
- **Documentation**: Maintaining detailed logs of thought process and decisions

### Autonomous Learning
- **Knowledge Capture**: Updating `AGENTS.md` with PySCF patterns
- **Process Documentation**: Creating `progress.txt` and `ralph_learning_log.txt`
- **Quality Assurance**: Self-validation against benchmark standards

## 📤 Upload to GitHub

To share this test framework on GitHub:

### Option 1: Initialize New Repository
```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS

# Initialize git repository
git init

# Add all test files
git add README.md
git add generate_h2_benchmark.py
git add validate_h2.py
git add h2_benchmark.json
git add slurm_batch.sh
git add slurm_interactive.sh
git add check_cluster.sh
git add cluster_usage_guide.md
git add Ralph_Test_H2_Hamiltonian/

# Commit with descriptive message
git commit -m "Initial commit: RLQAS H2 Hamiltonian Generation Test

Complete test framework for validating Ralph AI agent on quantum chemistry tasks.
Includes benchmark generation, validation system, SLURM support, and test results."

# Add remote repository (replace with your GitHub URL)
git remote add origin https://github.com/yourusername/RLQAS-H2-Test.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Option 2: Add to Existing Repository
If you already have a RLQAS repository:
```bash
cd /curie-home/jpchen/scratch/LLM/code/RLQAS
git add .
git commit -m "Add H2 Hamiltonian generation test suite"
git push
```

### Recommended .gitignore
Create `.gitignore` to exclude generated files:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# SLURM output files
*.out
*.err

# Ralph generated files (optional to keep)
# Ralph_Test_H2_Hamiltonian/h2_results.json
# Ralph_Test_H2_Hamiltonian/validation_summary.txt
# Ralph_Test_H2_Hamiltonian/progress.txt
# Ralph_Test_H2_Hamiltonian/ralph_learning_log.txt

# Benchmark file (sensitive - consider keeping private)
# h2_benchmark.json
```

### Repository Structure Recommendations
- **Public Repository**: Include all files except `h2_benchmark.json` (keep private)
- **Private Repository**: Include all files for full reproducibility
- **Documentation**: This README provides complete setup and execution instructions
- **License**: Add appropriate open-source license (MIT, Apache 2.0, etc.)

## 📈 Future Test Extensions

This test framework can be extended for:

1. **More Complex Molecules** (LiH, H2O, etc.)
2. **Advanced Quantum Methods** (UCCSD, VQE, etc.)
3. **Reinforcement Learning Integration** (RLQAS project)
4. **Performance Benchmarking** (Execution time, accuracy trade-offs)
5. **Multi-Agent Testing** (Collaborative quantum chemistry tasks)

## 📝 License & Attribution

This test framework is part of the RLQAS (Reinforcement Learning for Quantum Architecture Search) project. Developed for testing autonomous AI agents in quantum computational chemistry tasks.

**Author**: RLQAS Test Team
**Environment**: Curie cluster with SLURM scheduler
**Date**: January 2026
**Status**: Test completed successfully - Ralph passed all validation checks

---

*For questions or issues, refer to the test documentation or contact the RLQAS project maintainers.*