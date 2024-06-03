#!/bin/bash
#SBATCH --job-name="Milestone1"
#SBATCH --output="Milestone1.out"
#SBATCH --error="Milestone1.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# Generate running time information
{ time srun ./m1 100 > m1_100.out; } 2> m1_100_time.out
{ time srun ./m1 1000 > m1_1000.out; } 2> m1_1000_time.out
{ time srun ./m1 10000 > m1_10000.out; } 2> m1_10000_time.out

# Generate gprof output
# srun ./m1 100 && gprof -Q ./m1 gmon.out > outfile
