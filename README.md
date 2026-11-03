# 2D B-mode ultrasound image reconstruction from a single plane wave

Repository of the master's thesis of **Héctor David De Lara Berumen**
(CIMAT, 2026): code, trained models, evaluation protocol and material to
reproduce the reported results.

> **Proposed method — UNet-AG** (*B1-v5_anchor05*). A UNet with attention gates
> conditioned on a depth map derived from Schiffner's frequency-dependent
> F-number (Fd-F), which residually refines the Fd-F-apodized adjoint of a
> **single plane wave**. On the PICMUS experimental scenarios and with respect
> to single-plane-wave DAS, it improves lesion detectability (gCNR) by
> **36.6 %** (0.528 → 0.721) and also improves contrast (CNR, −3.974 →
> −3.692 dB) and lateral resolution (FWHM, 3.728 → 2.765 mm), in each of the
> three seeds separately, at **749 GFLOP** per image: 59.7 times less than the
> 75 plane wave Fd-F *beamformer* and 154.6 times less than the compared
> diffusion method (DRUS).

---

## Layout

| Folder | Contents |
|---|---|
| `code/` | The Python packages: `unet_ag` (proposed method), `schiffner_das` (Fd-F port), `drus` (DRUS replica) and `guided_diffusion` (DRUS prior network) |
| `methods/` | One directory per compared method, with its scripts, SLURM jobs and lightweight results |
| `data/` | Generation of the Field II set, how to obtain PICMUS, and the supervision the model was trained with (287 Schiffner targets) |
| `evaluation/` | Evaluation protocol: scripts, metrics (CSV/JSON), arrays on the evaluation grid, MATLAB validation and verifications |
| `discarded_methods/` | Code of the attempts that did not succeed |
| `pyproject.toml` | Installable package with two commands: `unetag` (proposed method) and `drus` (diffusion method) |

### Compared methods

GFLOP per image on the 612 × 388 point grid in the experimental scenarios, on
which the hypothesis is stated; the three scenarios are in
[`evaluation/metrics/cost_per_scenario.csv`](evaluation/metrics/cost_per_scenario.csv).

| Directory | Method | Input | GFLOP |
|---|---|---|---|
| `methods/01_das_1pw` | DAS, full aperture | 1 PW | 596 |
| `methods/02_schiffner_fdf` | Schiffner Fd-F (1 and 75 PW) | 1 / 75 PW | 596 / 44 700 |
| `methods/03_das_75pw` | Coherent compounded DAS | 75 PW | 44 700 |
| `methods/04_drus` | DRUS (diffusion, DDRM) | 1 PW | 115 875 |
| `methods/05_unet_ag_proposed` | **UNet-AG (proposed)** | 1 PW | **749** |

---

## Quick start

```bash
git clone git@github.com:hectorddlb/UNet-AG_Full.git
cd UNet-AG_Full
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"            # installs the packages and the `unetag` and `drus` commands
```

`pip install -e .` makes `unet_ag`, `schiffner_das` and `drus` importable from
any directory. If you prefer not to install anything, `source env.sh` puts
`code/` on the `PYTHONPATH` and `pip install -r requirements.txt` covers the
dependencies.

> **Installing on machines without internet access** (compute nodes, isolated
> environments). By default `pip` builds the package in an isolated
> environment and downloads `setuptools` for it, which fails without network
> access. If `setuptools` is already in the environment, skip that step:
>
> ```bash
> pip install --no-build-isolation --no-deps -e .
> ```

### Using the proposed model

```bash
unetag info                  # checks that it loads (3 529 366 parameters)

# reconstruct a PICMUS phantom
unetag reconstruct --picmus resolution_distorsion_expe_dataset_rf.hdf5 \
                   --out bmode.png --save-adjoint initial.png

# reconstruct your own RF
unetag reconstruct --rf my_data.npy --geometry my_geometry.json --out bmode.png
```

From Python:

```python
from unet_ag.inference import UNetAG, Geometry

model = UNetAG.from_pretrained()                     # seed 42
bmode = model.reconstruct_picmus("...expe_dataset_rf.hdf5")   # (612, 388) in [0,1]

# or with your own RF
geom  = Geometry(n_elements=128, n_samples=1527, pitch_m=300e-6,
                 sampling_freq_hz=20.832e6, speed_of_sound=1540.0)
bmode = model.reconstruct(rf, geom)                  # rf: (n_samples, n_elements)
```

The weights (41 MB each) live in `methods/05_unet_ag_proposed/checkpoints/`
and are found automatically from a clone; outside it, give the folder with
`UNETAG_CHECKPOINTS=/path/to/checkpoints` or with `--ckpt`.

### Using the diffusion method (DRUS)

The DRUS replica is wrapped like the proposed method, with the `drus` command
and the `drus.inference.DRUS` class. Its artifacts do not fit in the
repository and are built once (details in
[`methods/04_drus/README.md`](methods/04_drus/README.md)):

```bash
export DRUS_ARTIFACTS=/path/to/drus_artifacts
drus prepare operator --picmus /path/to/PICMUS  # H, B and observations (minutes, CPU)
drus prepare svd                                # dense SVD of BH (~8.4 h, ~120 GB of RAM, MKL)
drus prepare prior                              # prior published by OpenAI (2.2 GB)
drus info

drus reconstruct --picmus contrast_speckle_expe_dataset_rf.hdf5 --out drus.png
```

```python
from drus.inference import DRUS

model = DRUS.from_artifacts()                        # $DRUS_ARTIFACTS
res = model.reconstruct_picmus("...expe_dataset_rf.hdf5")    # thesis γ, seed 0
res.bmode                                            # (256, 256) in [0,1]
```

It needs a GPU for the 552.8 M parameter UNet and ~20 GB of RAM for the `Vᵀ`
matrix of the spectral decomposition.

---

## Reproducing

### The full evaluation

Requires PICMUS (see [`data/README.md`](data/README.md)) and, for DRUS, its
artifacts. From the repository root:

```bash
export THESIS_DATASETS=/path/to/your/datasets    # with PICMUS/ inside
export DRUS_ARTIFACTS=/path/to/drus_artifacts
python evaluation/scripts/gen_classical_picmus_fixed.py --device cuda   # DAS and Fd-F, 1 and 75 PW
python evaluation/scripts/gen_b1v5_picmus_seeds.py --device cuda        # UNet-AG, 3 seeds
python evaluation/scripts/gen_drus_picmus.py                            # DRUS, thesis γ
python evaluation/scripts/eval_and_figs_final.py                        # metrics and tables
```

Or everything in one SLURM job: `sbatch evaluation/scripts/run_final_results.sbatch`.
The 75 plane wave methods dominate the time (1 to 3.5 h on a TITAN RTX).

### Training the proposed method

Requires the Field II set. The supervision targets are already in the repo.

```bash
python -m unet_ag.train.b1_v5_pulse_aug \
    --data_root  "$THESIS_DATASETS/field_ii_data_multiangle" \
    --target_root data/targets_schiffner_75pw \
    --output_dir  results/my_run \
    --phantom_type points --anchor_weight 0.5 --epochs 20 --seed 42 \
    --device cuda
```

~9 h per seed on a 24 GB NVIDIA TITAN RTX. The equivalent SLURM jobs are in
`methods/05_unet_ag_proposed/slurm/`.

---

## What is and is not in the repository

**Included:** all the code; the 3 checkpoints of the final model (122 MB); the
287 supervision targets (262 MB); the evaluation-grid arrays of the six
methods (29 MB); the images of the MATLAB reference implementation for the
validation of the port (15 MB); all the metrics in CSV/JSON; the results of
the verifications.

**Excluded because of their size**, with reconstruction instructions in the
corresponding folder: the raw PICMUS and Field II datasets (~30 GB); the DRUS
artifacts (operator, a 51 GB SVD together with its source matrix —34 GB for
the factors that are kept— and a 2.2 GB diffusion prior, which are built with
`drus prepare`); the CUBDL dataset.

---

## Citation

The thesis is written in Spanish; cite it with its original title:

```bibtex
@mastersthesis{DeLara2026,
  author  = {De Lara Berumen, Héctor David},
  title   = {Reconstrucción de imágenes de ultrasonido 2D en modo B desde datos
             crudos RF de una sola onda plana mediante una red UNet informada
             por el F-number dependiente de la frecuencia},
  school  = {Centro de Investigación en Matemáticas (CIMAT)},
  year    = {2026},
  address = {Zacatecas, México}
}
```

## Acknowledgement of previous work

This work reimplements, adapts or compares with: the frequency-dependent
F-number of **Schiffner (2021)**; the self-supervised scheme of **Zhang et al.
(2021)**; the DRUS method of **Zhang et al. (2023)**; the attention gates of
**Oktay et al. (2018)**; the **guided-diffusion** diffusion model (OpenAI);
the **Field II** simulator and the **PICMUS** benchmark.
