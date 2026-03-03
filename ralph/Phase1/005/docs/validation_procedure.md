# RLQAS Phase 1 Validation Procedure

This document describes the validation procedure for RLQAS Phase 1, focusing on the LiH molecule test case.

## Overview

The validation test verifies that all Phase 1 components work together correctly and achieve chemical accuracy (<1.6 mHa error) for the LiH molecule within reasonable time (<2 hours).

## Prerequisites

### Software Dependencies
- Python 3.8+
- RLQAS Phase 1 Tasks 001-004 completed and accessible
- Core libraries: tencirchem-ng (>=2024.10), openfermion (>=1.5), PySCF (>=2.0.0)
- RL libraries: Stable-Baselines3 (>=2.0.0), Gym (>=0.21.0), PyTorch (>=1.9.0)
- Analysis tools: pandas (>=1.3), matplotlib (>=3.5)
- Testing: pytest (>=7.0), pytest-cov (>=4.0)

### System Requirements
- Moderate compute resources (CPU/GPU optional)
- 8GB+ RAM recommended for simulator
- Disk space for saving results

## Validation Steps

### Step 1: Health Check
Run the module health check script to verify all Phase 1 modules are accessible:

```bash
python scripts/health_check.py
```

Expected output: "All Phase 1 modules pass basic health checks!"

### Step 2: Fast Validation (Debugging)
Run a fast validation with minimal parameters to verify basic functionality:

```bash
python scripts/validate_lih.py --fast
```

Or using the convenience script:

```bash
python scripts/run_phase1_test.py --config fast
```

This runs with:
- 50 episodes maximum
- Loose convergence threshold (0.01 Hartree)
- Reduced circuit depth and excitations

### Step 3: Full Validation (Default Parameters)
Run the full validation with specification-compliant parameters:

```bash
python scripts/validate_lih.py
```

Or:

```bash
python scripts/run_phase1_test.py --config default
```

This runs with:
- 500 episodes maximum
- Chemical accuracy threshold (1.6e-3 Hartree)
- Standard circuit constraints (max_depth=12, max_excitations=15)

### Step 4: Integration Tests
Run the integration test suite:

```bash
pytest tests/integration/test_lih_validation.py -v
pytest tests/integration/test_phase1_integration.py -v
```

Skip slow tests:

```bash
pytest tests/integration/ -v -m "not slow"
```

### Step 5: Report Generation
After successful validation, generate comprehensive reports:

```python
from src.evaluation.report_generator import ReportGenerator
from src.evaluation.metrics_collector import MetricsCollector

# Load results
import json
with open('results/lih_test_results/validation_results.json') as f:
    results = json.load(f)

# Load metrics
metrics_collector = MetricsCollector('results/lih_test_results')
metrics = metrics_collector.load_metrics('results/lih_test_results/metrics.json')

# Generate report
report_generator = ReportGenerator(metrics, results)
report_generator.generate_markdown_report('results/lih_test_results/validation_report.md')
report_generator.generate_visualizations('results/lih_test_results/visualizations/')
```

## Configuration Options

### Default Validation Configuration
- **Molecule**: LiH
- **Bond length**: 1.6 Å
- **Active space**: (2 electrons, 2 orbitals)
- **Basis set**: sto-3g
- **Transform**: parity (yields 2 qubits)
- **RL episodes**: 500 maximum
- **Convergence threshold**: 1.6e-3 Hartree (1.6 mHa)
- **Circuit constraints**: max_depth=12, max_excitations=15
- **Output directory**: `results/lih_test_results/`

### Fast Configuration (for debugging)
- RL episodes: 50
- Convergence threshold: 0.01 Hartree
- Circuit constraints: max_depth=8, max_excitations=10
- Output directory: `results/lih_test_fast/`

## Expected Output

### Generated Files
1. `validation_results.json`: Complete validation results dictionary
2. `metrics.json`: Detailed performance metrics
3. CSV files: `energy_metrics.csv`, `training_metrics.csv`, etc.
4. `validation_report.md`: Comprehensive validation report
5. Visualization plots (PNG): energy convergence, training rewards

### Success Criteria
1. **Technical Success**: End-to-end test runs without errors
2. **Accuracy Success**: System achieves chemical accuracy (<1.6 mHa error) on LiH
3. **Performance Success**: Test completes within reasonable time (<2 hours goal)
4. **Integration Success**: All Phase 1 modules work together correctly
5. **Documentation Success**: Comprehensive validation report and metrics analysis
6. **Reproducibility**: Fixed random seeds ensure reproducible results

## Troubleshooting

### Common Issues

#### 1. Module Import Failures
**Symptoms**: ImportError when running validation script
**Solutions**:
- Verify Task 001-004 directories exist and contain `src/modules/`
- Check Python path modifications in validation scripts
- Run health check script to identify specific module failures

#### 2. Chemical Accuracy Not Achieved
**Symptoms**: Energy error exceeds 1.6 mHa
**Debugging steps**:
1. Verify FCI reference energy from Task 001 is correct
2. Check circuit expressiveness (should have ≥8 parameters for 4-qubit LiH)
3. Monitor energy progression during search
4. Adjust RL hyperparameters (learning rate, entropy coefficient)

#### 3. Performance Issues
**Symptoms**: Validation takes too long (>2 hours)
**Optimizations**:
1. Use fast configuration for development
2. Monitor timing breakdown between stages
3. Adjust simulator memory settings
4. Reduce maximum episodes for debugging

#### 4. Resource Exhaustion
**Symptoms**: Memory errors or process killed
**Mitigation**:
1. Reduce simulator `max_memory_gb` setting
2. Limit maximum circuit depth
3. Monitor memory usage with `psutil`
4. Implement garbage collection after large computations

## Validation Report Interpretation

### Key Metrics to Examine
1. **Energy Error**: Should be <1.6 mHa for chemical accuracy
2. **Convergence Status**: Whether search converged before episode limit
3. **Circuit Complexity**: Depth, number of excitations, parameters
4. **Training Performance**: Episode rewards, learning progression
5. **Resource Usage**: Memory, CPU, total time

### Decision Criteria
- **PASS**: All success criteria met (chemical accuracy achieved, no errors)
- **CONDITIONAL PASS**: System functional but chemical accuracy not achieved
- **FAIL**: System integration failures or runtime errors

## Reproducibility Notes

- Random seeds are fixed (seed=42) for all stochastic components
- Software versions should be recorded for reproducibility
- Configuration files should be saved with results
- Raw data should be preserved for future analysis

## Next Steps After Validation

1. **Success**: Proceed to Phase 2 (multi-algorithm support)
2. **Conditional**: Address accuracy issues, then proceed
3. **Failure**: Debug integration issues, rerun validation

## References

- RLQAS Phase 1 Specification (Sections 5.1, 6.1)
- RLQAS Phase 1 Tasks Document
- Tencirchem Documentation: https://tencirchem.readthedocs.io/
- OpenFermion Documentation: https://quantumai.google/openfermion