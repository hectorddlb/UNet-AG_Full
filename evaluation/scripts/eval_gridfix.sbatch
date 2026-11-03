#!/bin/bash
#SBATCH --job-name=eval_gridfix
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/eval_gridfix_%j.out
#SBATCH --error=slurm_logs/eval_gridfix_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>

# Field II + PICMUS eval of the b1_v5_anchor05_gridfix model.
# Field II uses the resampled targets (same grid as the trainer).

set -euo pipefail
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us
cd "$SLURM_SUBMIT_DIR"
source env.sh

RUN="${RUN:-b1_v5_anchor05_gridfix}"
CKPT="${SLURM_SUBMIT_DIR}/unet_ag/results/${RUN}/checkpoints/best.pt"

echo "=== eval_gridfix ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  ckpt: $CKPT"

python -u -m unet_ag.eval.evaluate_fieldii \
    --ckpt        "$CKPT" \
    --data_root   "${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle" \
    --target_root "${SLURM_SUBMIT_DIR}/results/targets_schiffner_gridfix" \
    --out         "${SLURM_SUBMIT_DIR}/unet_ag/results/${RUN}/eval_fieldii.csv" \
    --model_type  b1_v4 --base_ch 32 --phantom_type both \
    --nz 612 --nx 388 --device cuda

python -u -m unet_ag.eval.evaluate_picmus \
    --ckpt        "$CKPT" \
    --picmus_root "${SLURM_SUBMIT_DIR}/datasets/PICMUS" \
    --out         "${SLURM_SUBMIT_DIR}/unet_ag/results/${RUN}/eval_picmus.csv" \
    --mode both --phantom both --model_type b1_v4 --base_ch 32 \
    --nz 612 --nx 388 \
    --prefilter_fc_mhz 0.0 --prefilter_bw_mhz 0.0 --device cuda

echo "=== Done — $(date) ==="
