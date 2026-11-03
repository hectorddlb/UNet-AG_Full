# Code

Four Python packages. `pip install -e .` from the root installs them (with the
`unetag` and `drus` commands); `source env.sh` puts them on the `PYTHONPATH`
without installing.

## `unet_ag/` — the proposed method

| Subpackage | Contents |
|---|---|
| `data/` | RF readers (Field II, PICMUS, CUBDL), partitions and pulse augmentation |
| `physics/` | Measurement operator `OpHFdf`, its wrapper, the F-number map and the classical DAS beamformers |
| `models/` | UNet with conditioned attention gates and the residual wrapper with zero initialization |
| `losses/` | Composite MSE + L1 + SSIM loss |
| `metrics/` | CNR, gCNR and per-wire FWHM |
| `train/` | Entry points of the B1 lineage (the final one is `b1_v5_pulse_aug`) |
| `eval/` | Evaluators on PICMUS, Field II and CUBDL, with their masks |
| `tests/` | Smoke and specification tests |

Entry points:

```bash
unetag --help                                      # inference (unet_ag.inference.UNetAG)
python -m unet_ag.train.b1_v5_pulse_aug --help     # training (final method)
python -m unet_ag.eval.evaluate_picmus  --help     # per-run evaluator on PICMUS
python -m unet_ag.eval.evaluate_fieldii --help     # in-domain evaluation
```

`models/b1_v2_softdas.py` and `models/a2_unet_fourier.py` belong to discarded
variants, but they **are run-time dependencies** of `eval/evaluate_fieldii.py`,
which can load any checkpoint of the lineage. That is why they are here and
not in `discarded_methods/`.

## `schiffner_das/` — port of the frequency-dependent F-number

PyTorch implementation of Schiffner's Fourier-domain beamformer (2021). It
plays two roles: it generates the supervision targets and it is a compared
method in its own right (1 and 75 plane waves).

**Note:** `das_pw()` accepts `tx_sign` and `tx_reference`. The defaults
(`-1.0`, `"edge"`) reproduce the convention of Schiffner's original dataset;
**for PICMUS and Field II you must pass `tx_sign=+1.0, tx_reference="origin"`**
(the correction of the transmit convention).
`evaluation/matlab_validation/` contains the numerical validation against the
MATLAB reference implementation.

## `drus/` — replica of the diffusion method

Wrapped like the proposed method: public API in `drus.inference` (class
`DRUS`, with `from_artifacts()`, `reconstruct()` and `reconstruct_picmus()`)
and the `drus` command (`info`, `prepare`, `reconstruct`).

| Module | Contents |
|---|---|
| `inference.py`, `cli.py` | API and `drus` command; thesis γ per phantom (`THESIS_GAMMA`) |
| `preparation.py` | Location (`$DRUS_ARTIFACTS`) and construction of the artifacts: operator, SVD and prior |
| `prior.py` | Configuration and loading of the diffusion prior (single source of its *flags*) |
| `physics_picmus.py`, `observations.py` | Physical model H, beamformer B and `By` observations |
| `svd_bh.py` | Dense SVD via MKL ILP64 |
| `ddrm_sampler.py` | DDRM sampler |
| `physics_fieldii.py`, `ft_dataset.py` | Field II variant and fine-tuning set of the prior (outside the evaluated configuration) |
| `tests/` | API contract tests |

The per-phase scripts and the SLURM jobs are in `methods/04_drus/`.

## `guided_diffusion/` — DRUS prior network

Copy of [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
(via DRUS-v1), without MPI: the 256 × 256 UNet used by the prior.
