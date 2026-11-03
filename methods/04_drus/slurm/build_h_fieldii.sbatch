#!/bin/bash
#SBATCH --job-name=drus_hfii
#SBATCH --partition=C0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/drus_hfii_%j.out
#SBATCH --error=slurm_logs/drus_hfii_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# Phase F.1-A — H/B of the multi-angle Field II geometry + calibration gates
# (t_start pad) and sanity check vs DAS 1PW. CPU-only. See build_h_fieldii.py.

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"
source env.sh

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
OUT_DIR="${OUT_DIR:-zhang_drus/results_drus/fieldii_phaseA}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "== drus_hfii: DATA_ROOT=$DATA_ROOT OUT_DIR=$OUT_DIR =="

python -u -m zhang_drus.scripts.build_h_fieldii \
    --data_root "$DATA_ROOT" \
    --out "$OUT_DIR"

echo "== drus_hfii DONE =="
