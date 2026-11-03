#!/bin/bash
#SBATCH --job-name=drus_ddrm
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48000M
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/drus_ddrm_%j.out
#SBATCH --error=slurm_logs/drus_ddrm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"
source env.sh

# By default, the thesis configuration: prior without fine-tuning, extended
# grid, γ ∈ {20, 25, 50, 75}, seed 0 (the γ of each phantom is reported:
# drus.inference.THESIS_GAMMA). Launch from the repository root.
ART="${DRUS_ARTIFACTS:-methods/04_drus/artifacts}"
SVD_DIR="${SVD_DIR:-$ART/svd}"
PHASE_A_DIR="${PHASE_A_DIR:-$ART/operator}"
CKPT="${CKPT:-$ART/prior/256x256_diffusion_uncond.pt}"
OUT="${OUT:-results/drus_noft}"
GAMMAS="${GAMMAS:-20 25 50 75}"
SEEDS="${SEEDS:-0}"
TIMESTEPS="${TIMESTEPS:-50}"
PICMUS_ROOT="${PICMUS_ROOT:-${THESIS_DATASETS:-data}/PICMUS}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "== drus_ddrm: svd=$SVD_DIR ckpt=$CKPT → $OUT | γ={$GAMMAS} seeds={$SEEDS} =="

# Optional gate G-C1: unconditional samples of the listed ckpts
if [ -n "${GC1_CKPTS:-}" ]; then
    for c in $GC1_CKPTS; do
        base=$(basename "$c" .pt)
        python methods/04_drus/scripts/sample_prior.py --ckpt "$c" \
            --out "$OUT/gc1_${base}.png" --n 4 --steps 250
    done
fi

python methods/04_drus/scripts/sample_ddrm.py \
    --svd_dir "$SVD_DIR" \
    --ckpt "$CKPT" \
    --out "$OUT" \
    --gammas $GAMMAS \
    --seeds $SEEDS \
    --timesteps "$TIMESTEPS"

python methods/04_drus/scripts/evaluate_ddrm.py \
    --pred_dir "$OUT" \
    --phase_a_dir "$PHASE_A_DIR" \
    --picmus_root "$PICMUS_ROOT" \
    --out "$OUT/eval_ddrm.csv" \
    --eval_by

echo "== drus_ddrm DONE =="
