# DRUS — the compared diffusion method

Python replica of DRUS (Zhang et al., 2023): 50-step DDRM restoration with a
diffusion prior, on the explicit PICMUS physical model. It is the generative
method the proposed one is compared against: the strongest of the study in
image quality and the most expensive in cost (115 875 GFLOP per image,
552.8 M parameters).

## Thesis conditions

| | |
|---|---|
| Prior | guided-diffusion `256x256_diffusion_uncond.pt` (OpenAI), **without fine-tuning** (the best-performing variant of the replica) |
| Physical model | PICMUS H and apodized beamformer B (Tukey 25 %, f# 1.4), 256 × 256 grid with x ∈ [−18, 18] mm and z ∈ [9, 48] mm (extended with respect to the paper's, z ∈ [10, 46] mm, so that the 7 experimental wires fit) |
| Decomposition | Dense SVD of BH (65 536²): an 8.4 h factorization that takes 51 GB together with the BH matrix it starts from; the factors that are kept (U, S, Vᵀ) take 34 GB |
| Sampling | DDRM, 50 steps, η = 0.85, η_B = 1.0, seed 0 |
| γ per phantom | 75 (sim. resolution), 50 (sim. contrast), 75 (exp. resolution), 25 (exp. contrast): grid search over {20, 25, 50, 75}, as in the original paper |
| Evaluation | \|x\| → 60 dB → [0, 1] → bilinear interpolation to the common 612 × 388 grid |

## Use as a package

It is installed with the rest of the repository (`pip install -e ".[drus]"`)
and exposes the `drus` command and the `drus.inference.DRUS` class, with the
same shape as the proposed method (`unetag`, `unet_ag.inference.UNetAG`).

### 1. Artifacts (once)

```bash
export DRUS_ARTIFACTS=/path/to/drus_artifacts   # or --artifacts on each command
drus prepare operator --picmus /path/to/PICMUS  # minutes, CPU
drus prepare svd                                # ~8.4 h, ~120 GB of RAM, MKL ILP64 (pip install mkl)
drus prepare prior                              # downloads and checks 2.2 GB
drus info                                       # status and thesis parameters
```

Resulting layout:

```
$DRUS_ARTIFACTS/
├── operator/   H.npz, B.npz, grid.npz, meta.json, By_<phantom>.npy
├── svd/        S.npy, U.npy, Vt.npy, Uty_<phantom>.npy, svd_meta.json
└── prior/      256x256_diffusion_uncond.pt
```

On a cluster it is better to split the two SVD stages (`--stage bh` on GPU,
`--stage svd` in a new process with all the RAM); `slurm/svd_bh.sbatch` does
that.

### 2. Reconstruction

```bash
drus reconstruct --picmus contrast_speckle_expe_dataset_rf.hdf5 --out drus.png
drus reconstruct --picmus resolution_distorsion_simu_dataset_rf.hdf5 \
                 --gamma 50 --seed 1 --out drus.npy --save-x x.npy
```

```python
from drus.inference import DRUS, to_grid

model = DRUS.from_artifacts()
res = model.reconstruct_picmus("contrast_speckle_expe_dataset_rf.hdf5")
res.bmode            # (256, 256) in [0, 1]; res.x is the signed output
res.gamma, res.seed  # 25.0 and 0: the thesis ones for this phantom
```

The phantom is inferred from the file name and sets the TGC, the normalization
scale and the thesis γ. For your own RF,
`DRUS.reconstruct(rf, gamma=..., tgc_alpha=..., norm_scale=...)`. It needs a
GPU for the UNet and ~20 GB of RAM for `Vᵀ` (its products run on CPU, as in
the original code). It takes 168.0 s per image end to end on a TITAN RTX,
plus ~105 s of start-up to load the 17.2 GB of `Vᵀ`.

The reconstruction of the four phantoms for the thesis evaluation is
`evaluation/scripts/gen_drus_picmus.py`.

## Contents

| Path | What it is |
|---|---|
| `code/drus/inference.py`, `cli.py`, `preparation.py`, `prior.py` | Public API, `drus` command, construction of the artifacts and prior |
| `code/drus/physics_picmus.py`, `observations.py` | H and B (port of `compute_svdBH.m`), `By` observations (port of `compute_By.m`) |
| `code/drus/svd_bh.py` | Dense SVD via MKL ILP64 (ctypes bindings) |
| `code/drus/ddrm_sampler.py` | DDRM sampler (port of DRUS-v1) |
| `code/drus/tests/` | API contract tests on synthetic artifacts |
| `code/guided_diffusion/` | Prior UNet (from openai/guided-diffusion, without MPI) |
| `scripts/` | Per-phase scripts with which the replica was developed (operator, SVD, prior fine-tuning, sampling and evaluation) |
| `slurm/` | SLURM jobs of the SVD and of the sampling |

Fine-tuning the prior with the 287 supervision images of the proposed method
(`scripts/finetune_prior.py`, `code/drus/ft_dataset.py`) is part of the
replica, but not of the configuration evaluated in the thesis.
