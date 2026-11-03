#!/bin/bash
#SBATCH --job-name=bench_cost
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/bench_cost_%j.out
#SBATCH --error=slurm_logs/bench_cost_%j.err

# Wall-clock time per image of the 6 methods: experimental scenario, das_pw in
# float32 (precision matched to the adjoint), UNet-AG as it runs and DRUS end
# to end, all in the same process and on the same node. Pin the node with
# --nodelist to compare runs: the reported times were measured on g-0-10.
#
# --mem=64000M because DRUS loads Vt.npy (17.2 GB) into RAM: the DDRM spectral
# products run on CPU (spectral_device="cpu" in ddrm_sampler).
# Launch from the repository root with THESIS_DATASETS and DRUS_ARTIFACTS.

set -euo pipefail
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us
cd "$SLURM_SUBMIT_DIR"
source env.sh

OUT="${OUT:-${SLURM_SUBMIT_DIR}/evaluation/metrics/inference_time_fp32.json}"
PHANTOM="${PHANTOM:-exp_resolution}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== bench_cost ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
python -u evaluation/scripts/inference_time.py \
    --device cuda --precision float32 --phantom "$PHANTOM" --out "$OUT"
echo "=== Done — $(date) ==="
