#!/bin/bash
# Check cluster status and available resources

echo "=== Cluster Status Check ==="
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
echo "5. Recommended SLURM commands for RLQAS:"
echo "----------------------------------------"
echo "   Interactive job (2 hours, 4 CPUs, 8GB):"
echo "   salloc --job-name=rlqas-ralph --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=24:00:00 --partition=CPU"
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
echo "6. RLQAS directory check:"
echo "-------------------------"
if [ -f "prd.json" ]; then
    echo "   ✓ prd.json found"
else
    echo "   ✗ prd.json not found - are you in the right directory?"
fi

if [ -f "slurm_batch.sh" ]; then
    echo "   ✓ slurm_batch.sh found"
else
    echo "   ✗ slurm_batch.sh not found"
fi

echo ""
echo "=== Next Steps ==="
echo "1. Review available partitions above"
echo "2. Update partition in slurm_batch.sh if needed (currently 'debug')"
echo "3. Request interactive job: salloc ..."
echo "4. Or submit batch job: sbatch slurm_batch.sh"
echo "5. Monitor with: squeue -u $USER"
echo ""