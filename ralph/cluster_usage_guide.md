# RLQAS Cluster Usage Guide

## Problem Statement
Running `./ralph.sh` directly on the login node (curie-login) may:
1. Impact cluster performance for other users
2. Be against cluster policies (login nodes are typically for light tasks only)
3. Risk process termination by system administrators

## Solution: Use SLURM Job Scheduler

### Option 1: Interactive Job (Recommended for Ralph Development)
Allocates resources and gives you an interactive shell on a compute node.

```bash
# Request resources (adjust parameters as needed)
salloc --job-name=rlqas-ralph --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=24:00:00 --partition=CPU

# Once resources are allocated, you'll get a shell on a compute node
# Navigate to your RLQAS directory
cd /curie-home/jpchen/scratch/LLM/code/RLQAS

# Run Ralph
./ralph.sh

# When finished, exit the interactive session
exit
```

**Interactive script alternative:**
```bash
chmod +x slurm_interactive.sh
./slurm_interactive.sh  # Note: This script is meant to be run AFTER salloc
```

### Option 2: Batch Job
Submits a job that runs independently.

```bash
# First, check available partitions
sinfo

# Edit slurm_batch.sh to match your cluster configuration
# Update: --partition, --mail-user, resource requests

# Submit the job
sbatch slurm_batch.sh

# Check job status
squeue -u $USER

# Monitor output
tail -f rlqas_ralph_<jobid>.out
```

## Resource Recommendations

### For LiH 4-qubit Test
- **CPU**: 4 cores (quantum simulation + RL training)
- **Memory**: 8 GB (statevector for 4 qubits = 16 complex numbers)
- **Time**: 24 hours (for Ralph iterative development and testing)
- **Partition**: Use CPU partition (or other available partitions)

### For Future Scaling
- **GPU**: Optional for RL neural network acceleration
- **More memory**: For larger qubit counts (n qubits → 2^n statevector)
- **Checkpointing**: Design code to save progress for longer runs

## Important Considerations

### 1. Environment Setup
Ensure your Python environment (with tencirchem, PyTorch, stable-baselines3) is available on compute nodes:
- Use conda environments with `conda activate`
- Or install dependencies in home directory
- Or use cluster-wide modules: `module load python/3.9`

### 2. File System Access
Your home directory (`/curie-home/jpchen/`) should be accessible from compute nodes.

### 3. Monitoring
- Use `squeue -u $USER` to check job status
- Use `sacct -j <jobid>` to see detailed accounting
- Monitor output files: `rlqas_ralph_<jobid>.out` and `.err`

### 4. Common SLURM Commands
```bash
sinfo                          # Show partition status
squeue -u $USER               # Show your jobs
scancel <jobid>               # Cancel a job
scontrol show job <jobid>     # Detailed job info
sacct -j <jobid> --format=JobID,JobName,Partition,AllocCPUS,State,ExitCode  # Accounting
```

## Troubleshooting

### Job Pending
- Check partition availability: `sinfo`
- Check queue limits: `squeue -u $USER`
- Adjust resource requests (reduce CPU/memory/time)

### Module/Environment Issues
- Check if modules load correctly in job script
- Test environment in interactive session first
- Consider using absolute paths

### Ralph-Specific Issues
- Monitor `progress.txt` for Ralph's activities
- Check output files for errors
- Ralph may need adjustments for non-interactive batch execution

## Next Steps

1. **Test with interactive job** to ensure Ralph works in cluster environment
2. **Adjust resource requests** based on actual usage (monitor with `sacct`)
3. **Implement checkpointing** in Ralph's code for longer runs
4. **Consider GPU** for larger-scale RL training

Remember: Always use compute nodes for computational work, not login nodes.