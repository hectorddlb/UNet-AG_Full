#!/bin/bash
#SBATCH --job-name=teacher_uty
#SBATCH --partition=C0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=126000M
#SBATCH --time=03:00:00
#SBATCH --output=slurm_logs/teacher_uty_%j.out
#SBATCH --error=slurm_logs/teacher_uty_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us
cd "$SLURM_SUBMIT_DIR"

DATA_ROOT="${DATA_ROOT:-datasets/Field II/field_ii_data_multiangle}"
PHASE_A_DIR="${PHASE_A_DIR:-zhang_drus/results_drus/fieldii_phaseA}"
SVD_DIR="${SVD_DIR:-zhang_drus/results_drus/fieldii_phaseB}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "== teacher_uty: node $(hostname) — $(nproc) cpus — $DATA_ROOT =="
python zhang_drus/scripts/make_uty_fieldii.py \
    --data_root "$DATA_ROOT" \
    --phase_a_dir "$PHASE_A_DIR" \
    --svd_dir "$SVD_DIR"
echo "== teacher_uty DONE =="
