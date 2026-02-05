#!/bin/bash
# Check cluster status and available resources for LiH VQE test

echo "=== Cluster Status Check for LiH VQE Test ==="
echo "Date: $(date)"
echo "User: $USER"
echo "Host: $(hostname)"
echo ""

# Check if we're on login node
if [[ "$(hostname)" == *"login"* ]]; then
    echo "⚠️  WARNING: You are on a login node."
    echo "   Do not run computationally intensive tasks here."
    echo "   Use SLURM job submission (salloc or sbatch)."
    echo ""
fi

# Check SLURM availability
echo "1. Checking SLURM commands..."
which sbatch salloc sinfo squeue 2>/dev/null
if [ $? -eq 0 ]; then
    echo "   ✓ SLURM commands available"
else
    echo "   ✗ SLURM commands not found"
    exit 1
fi

echo ""
echo "2. Available partitions (sinfo):"
echo "--------------------------------"
sinfo -o "%20P %10l %8D %6t %10G %10c %10m" | head -20

echo ""
echo "3. Your current jobs (squeue -u $USER):"
echo "---------------------------------------"
squeue -u $USER -o "%.10i %.20j %.10P %.5C %.10m %.10l %.20S %.10T %.20R" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "   No jobs found or error running squeue"
fi

echo ""
echo "4. Recent job history (last 5 jobs):"
echo "------------------------------------"
sacct -u $USER --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode,Elapsed -X 2>/dev/null | head -6

echo ""
echo "5. Recommended SLURM commands for LiH VQE:"
echo "------------------------------------------"
echo "   Interactive job (6 hours, 4 CPUs, 16GB):"
echo "   salloc --job-name=ralph-lih-vqe --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=6:00:00 --partition=CPU"
echo ""
echo "   Batch job:"
echo "   sbatch slurm_batch.sh"
echo ""
echo "   Cancel a job:"
echo "   scancel <jobid>"
echo ""
echo "   Detailed job info:"
echo "   scontrol show job <jobid>"

echo ""
echo "6. LiH VQE directory check:"
echo "---------------------------"
if [ -f "prd.json" ] || [ -f "generate_lih_benchmark.py" ]; then
    echo "   ✓ LiH VQE test files found"
else
    echo "   ✗ LiH VQE files not found - are you in the right directory?"
fi

if [ -f "slurm_batch.sh" ]; then
    echo "   ✓ slurm_batch.sh found"
else
    echo "   ✗ slurm_batch.sh not found"
fi

if [ -d "Ralph_Test_LiH_VQE" ]; then
    echo "   ✓ Ralph test directory found"
else
    echo "   ✗ Ralph test directory not found"
fi

echo ""
echo "=== LiH VQE Test Setup ==="
echo "1. Install dependencies (if not already):"
echo "   pip install pyscf tencirchem-ng numpy scipy"
echo ""
echo "2. Generate benchmark (if not exists):"
echo "   python generate_lih_benchmark.py"
echo ""
echo "3. Run test:"
echo "   - Interactive: ./slurm_interactive.sh"
echo "   - Batch: sbatch slurm_batch.sh"
echo "   - Direct: cd Ralph_Test_LiH_VQE && ./ralph.sh --tool claude 10"
echo ""
echo "4. Monitor with: squeue -u $USER"
echo "5. Check results: validation_summary.txt, lih_results.json"
echo ""