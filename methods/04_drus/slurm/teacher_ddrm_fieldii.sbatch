#!/bin/bash
#SBATCH --job-name=teacher_ddrm
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48000M
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/teacher_ddrm_%j.out
#SBATCH --error=slurm_logs/teacher_ddrm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us
cd "$SLURM_SUBMIT_DIR"

# Teacher DRUS-noFT: unconditional base prior (no Schiffner fine-tuning), one γ.
SVD_DIR="${SVD_DIR:-zhang_drus/results_drus/fieldii_phaseB}"
CKPT="${CKPT:-zhang_drus/results_drus/ckpt/256x256_diffusion_uncond.pt}"
OUT="${OUT:-zhang_drus/results_drus/fieldii_teacher_noft}"
GAMMA="${GAMMA:-25}"
SEED="${SEED:-0}"
TIMESTEPS="${TIMESTEPS:-50}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "== teacher_ddrm: svd=$SVD_DIR ckpt=$(basename $CKPT) → $OUT | γ=$GAMMA seed=$SEED =="
python zhang_drus/scripts/sample_ddrm.py \
    --svd_dir "$SVD_DIR" \
    --ckpt "$CKPT" \
    --out "$OUT" \
    --phantoms all \
    --gammas "$GAMMA" \
    --seeds "$SEED" \
    --timesteps "$TIMESTEPS" \
    --no_png \
    --skip_existing
echo "== teacher_ddrm DONE =="
