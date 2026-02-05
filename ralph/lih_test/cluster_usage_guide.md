# LiH VQE Test Cluster Usage Guide

## 🖥️ Cluster Environment

This test is designed for HPC clusters with SLURM scheduler. Key requirements:

- **Python 3.8+** with PySCF, Tencirchem, NumPy, SciPy
- **SLURM scheduler** for resource allocation
- **Compute nodes** for running quantum chemistry calculations
- **Adequate memory**: 16GB recommended for LiH VQE calculations

## 🔧 Environment Check

The test assumes Python environment with PySCF, Tencirchem, NumPy, and SciPy is already configured.

### Verify Environment
```bash
cd lih_test
python3 -c "
import sys
try:
    import pyscf
    print(f'✓ PySCF version: {pyscf.__version__}')
except ImportError as e:
    print(f'✗ PySCF import error: {e}')
try:
    import tencirchem
    print(f'✓ tencirchem version: {tencirchem.__version__}')
except ImportError as e:
    print(f'✗ tencirchem import error: {e}')
try:
    import numpy
    print(f'✓ NumPy version: {numpy.__version__}')
except ImportError as e:
    print(f'✗ NumPy import error: {e}')
try:
    import scipy
    print(f'✓ SciPy version: {scipy.__version__}')
except ImportError as e:
    print(f'✗ SciPy import error: {e}')
"
```

## 🚀 Running the Test

### Step 1: Generate Benchmark
```bash
cd lih_test
python generate_lih_benchmark.py
```
This creates `lih_benchmark.json` with reference FCI energy.

### Step 2: Run Ralph Test

#### Method A: SLURM Batch Job (Recommended)
```bash
cd lih_test
sbatch slurm_batch.sh
```
Output files: `ralph_lih_vqe_<jobid>.out` and `.err`

#### Method B: SLURM Interactive Session
```bash
# Request compute node
salloc --job-name=ralph-lih-vqe --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=6:00:00 --partition=CPU

# Run test
cd lih_test
./slurm_interactive.sh

# Or manually:
cd Ralph_Test_LiH_VQE
./ralph.sh --tool claude 5
exit  # After completion
```

#### Method C: Direct Execution (Login Node - Lightweight Only)
```bash
cd lih_test/Ralph_Test_LiH_VQE
./ralph.sh --tool claude 5
```
**Warning**: Only run direct execution for testing script logic, not for actual VQE calculations.

## 📊 Expected Resource Usage

- **Memory**: 8-16 GB (PySCF integrals + VQE optimization)
- **CPU**: 1-4 cores (BFGS optimization can use multiple threads)
- **Time**: 1-6 hours depending on convergence
- **Storage**: Minimal (<100MB)

## 🔍 Monitoring Progress

### During Execution
```bash
# Check job status
squeue -u $USER

# Check output files
tail -f ralph_lih_vqe_<jobid>.out
tail -f ralph_lih_vqe_<jobid>.err

# Check Ralph progress (if job is running)
ls -la Ralph_Test_LiH_VQE/progress.txt
tail -f Ralph_Test_LiH_VQE/progress.txt
```

### After Completion
Check these files:
1. **`Ralph_Test_LiH_VQE/validation_summary.txt`** - Validation results
2. **`Ralph_Test_LiH_VQE/progress.txt`** - Ralph's progress log
3. **`Ralph_Test_LiH_VQE/ralph_learning_log.txt`** - Detailed thought process
4. **`Ralph_Test_LiH_VQE/lih_results.json`** - Circuit, parameters, energy results
5. **`Ralph_Test_LiH_VQE/generate_lih_vqe.py`** - Ralph-generated implementation

## 🐛 Troubleshooting

### Issue: Import errors
**Solution**: Ensure dependencies are installed
```bash
python3 -c "import pyscf, tencirchem, numpy, scipy; print('All imports successful')"
```

### Issue: Benchmark generation fails
**Solution**: Check PySCF installation and active space settings

### Issue: VQE not converging
**Solution**: Ralph will iterate and improve based on validation feedback

### Issue: Memory errors
**Solution**: Increase memory request in SLURM scripts (change `--mem=16G` to `--mem=32G`)

### Issue: Timeout
**Solution**: Increase time limit in SLURM scripts (change `--time=6:00:00` to `--time=12:00:00`)

## 🔧 Customization

### Adjusting SLURM Parameters
Edit `slurm_batch.sh` and `slurm_interactive.sh`:
- `--mem`: Memory allocation (increase if out of memory)
- `--time`: Time limit (increase if jobs timeout)
- `--cpus-per-task`: CPU cores (BFGS can use multiple threads)
- `--partition`: Cluster partition name

### Modifying Validation Tolerance
Edit `validate_lih.py`:
- Change `vqe_energy_tolerance_hartree` for different accuracy targets
- Adjust `bond_length_tolerance_angstrom` for molecule validation

### Changing Active Space
Edit `prd.json`:
- Update `active_space` and `orbital_selection`
- Update benchmark generation script accordingly

## 📈 Performance Tips

1. **Start small**: Test on login node with minimal iterations
2. **Monitor memory**: Use `top` or `htop` during execution
3. **Check convergence**: Look at energy curve in `lih_results.json`
4. **Iterate**: Ralph has 10 iterations to improve based on feedback
5. **Document**: Check `AGENTS.md` for Ralph's learned patterns

## 🆘 Getting Help

1. Check cluster status: `./check_cluster.sh`
2. Check SLURM documentation: `man sbatch`, `man salloc`
3. Check Python dependencies: `python3 -c "import pyscf; print(pyscf.__version__)"`
4. Review Ralph logs: `Ralph_Test_LiH_VQE/ralph_learning_log.txt`

---

*For cluster-specific issues, contact your system administrator. For test-specific issues, check the RLQAS project documentation.*