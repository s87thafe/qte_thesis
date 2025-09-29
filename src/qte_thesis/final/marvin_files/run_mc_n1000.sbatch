#!/bin/bash
#SBATCH --job-name=mc_1000
#SBATCH --partition=intelsr_medium
#SBATCH --array=0-499
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mc_v5_%A_%a.out
#SBATCH --error=logs/mc_v5_%A_%a.err

mkdir -p logs BLD_data

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate qte_thesis
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
srun python -m simulation_n1000 \
  --sim-id="${SLURM_ARRAY_TASK_ID}"