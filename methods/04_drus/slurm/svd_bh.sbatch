#!/bin/bash
#SBATCH --job-name=drus_svd
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=126000M
#SBATCH --time=17:30:00
#SBATCH --output=slurm_logs/drus_svd_%j.out
#SBATCH --error=slurm_logs/drus_svd_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate thesis_us

cd "$SLURM_SUBMIT_DIR"
source env.sh

# Operator (input) and SVD (output): by default, those of $DRUS_ARTIFACTS
# (see methods/04_drus/README.md). Launch from the repository root.
ART="${DRUS_ARTIFACTS:-methods/04_drus/artifacts}"
IN_DIR="${IN_DIR:-$ART/operator}"
OUT_DIR="${OUT_DIR:-$ART/svd}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# libmkl_rt needs to find its sublibs and libiomp5 (pip mkl puts them in env/lib)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "== drus_svd: IN_DIR=$IN_DIR OUT_DIR=$OUT_DIR =="
echo "== node: $(hostname) — $(nproc) cpus =="

# Stage 1: dense BH (GPU if available; peak RAM ~4 GB, writes the 17 GB BH.npy)
if [ -f "$OUT_DIR/BH.npy" ]; then
    echo "== BH.npy already exists, reusing it =="
else
    python methods/04_drus/scripts/run_svd_bh.py --stage bh --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
fi

# Stage 2: SVD (fresh process, no torch; peak RAM ~112 GiB with gesdd)
python methods/04_drus/scripts/run_svd_bh.py --stage svd --in_dir "$IN_DIR" --out_dir "$OUT_DIR"

echo "== drus_svd DONE =="
