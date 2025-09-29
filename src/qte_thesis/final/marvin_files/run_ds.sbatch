#!/bin/bash
#SBATCH --job-name=real_data_ds_sub
#SBATCH --partition=lm_long
#SBATCH --array=0-12
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000G
#SBATCH --time=120:00:00
#SBATCH --output=logs/ds_%A_%a.out
#SBATCH --error=logs/ds_%A_%a.err

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate qte_thesis
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TAUS=(0.01 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99)
idx="${SLURM_ARRAY_TASK_ID}"
tau="${TAUS[$idx]}"

srun python -m run_real_data_ds \
  --tau "$tau"