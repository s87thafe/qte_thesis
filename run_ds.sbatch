#!/bin/bash
#SBATCH --job-name=real_data_ds_sub
#SBATCH --account=ag_ifs_noack
#SBATCH --partition=intelsr_long
#SBATCH --array=0-1299
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=96:00:00
#SBATCH --output=logs/ds_%A_%a.out
#SBATCH --error=logs/ds_%A_%a.err

source ~/.bashrc 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate qte_thesis
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

TAUS=(0.01 0.05 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90 0.95 0.99)
M=100                                  # subsamples per tau
tau_idx=$(( SLURM_ARRAY_TASK_ID / M ))
sub_id=$(( SLURM_ARRAY_TASK_ID % M ))
tau="${TAUS[$tau_idx]}"

SEED=$(( 12345 + sub_id + 1000 * tau_idx ))
OUT=".../real_data_${tau}_ds_sub${sub_id}.csv"

srun python -m qte_thesis.final.run_real_data_ds \
  --tau "$tau" \
  --seed "$SEED" \
  --n_subsample 2500 \
  --subsample_id "$sub_id"
