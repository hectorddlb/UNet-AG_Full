#!/bin/bash
#SBATCH --job-name=drus_ft
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=17:30:00
#SBATCH --output=slurm_logs/drus_ft_%j.out
#SBATCH --error=slurm_logs/drus_ft_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"

DATA_DIR="${DATA_DIR:-zhang_drus/results_drus/ft_data_schiffner287_signed}"
OUT_DIR="${OUT_DIR:-zhang_drus/results_drus/ft_prior_s287_signed}"
BASE_CKPT="${BASE_CKPT:-zhang_drus/results_drus/ckpt/256x256_diffusion_uncond.pt}"
MAX_STEPS="${MAX_STEPS:-6000}"
BATCH="${BATCH:-4}"
MICROBATCH="${MICROBATCH:-1}"
LR="${LR:-1e-5}"
DOMAIN="${DOMAIN:-signed}"

echo "== drus_ft: $DATA_DIR → $OUT_DIR | steps=$MAX_STEPS batch=$BATCH lr=$LR domain=$DOMAIN =="
nvidia-smi --query-gpu=name,memory.total --format=csv || true

# Resumable: detects the last model??????.pt in OUT_DIR and continues.
python zhang_drus/scripts/finetune_prior.py \
    --data_dir "$DATA_DIR" \
    --base_ckpt "$BASE_CKPT" \
    --out_dir "$OUT_DIR" \
    --max_steps "$MAX_STEPS" \
    --batch_size "$BATCH" \
    --microbatch "$MICROBATCH" \
    --lr "$LR" \
    --domain "$DOMAIN"

echo "== drus_ft DONE =="
