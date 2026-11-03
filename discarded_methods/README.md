# Discarded methods and variants

Code of the attempts that did **not** succeed. It is kept because the
conclusion about where to incorporate the physical knowledge rests on them:
without this code, those claims could not be verified.

**Nothing here is part of the current pipeline.** The final method and the
five compared methods live in `code/` and `methods/`.

## `zhang_edn/` — initial self-supervised proposal

Reimplementation of Zhang et al. (2021): a 12-layer encoder-decoder network,
construction of the measurement matrix `H`, the five loss functions and the
two-stage training. Running it produced the **collapse to x' ≈ 0 in six
configurations**.

Note: the only module of this package that is still alive is
`measurement_operator_fdf.py`, the `OpHFdf` operator — and it is not here but
integrated in the model package, as `code/unet_ag/physics/`.

## `unet_ag_variants/` — variants of the pivot

| File | Corresponds to |
|---|---|
| `train/a1_rf_hfdf_loss.py`, `models/a1_resunet_rf.py`, `losses/fidelity_hfdf.py` | Proposal **A1** (RF→image with a fidelity term) — analogous collapse |
| `train/a2_*.py`, `losses/fourier_weighted.py`, `losses/lateral_deriv.py` | Proposal **A2** (Fourier-weighted loss), 7 variants — `n_lat = 0/20` |
| `train/a2_v7_lateral_deriv.py` | Ablation **A2-v7** (lateral derivative), λ ∈ {0.3, 1.0, 3.0} — all three fail |
| `train/b1_v2_softdas.py` | **B1-v2** (Soft-DAS front end, signed target) — collapse due to the sparse target |
| `train/b1_v4_finetune_picmus.py` | Fine-tuning on PICMUS, 3 modes — metrics identical to 4 decimals |
| `scripts/diagnose_prefilter_picmus.py` | Test-time pre-filter — hurts resolution |
| `scripts/ensemble_anchor05.py` | 3-seed ensemble — ≈ the mean, no gain |
| `physics/classical/{cf,mv}_beamformer.py`, `scripts/precompute_{cf,mv}.py` | CF and MV baselines, never reported |
| `slurm/` | The SLURM jobs of all the above |

**Note:** these files **cannot be run as is** from this directory: their
imports assume they are part of the `unet_ag` package. To run them, copy them
back into `code/unet_ag/` at the relative path given by their subfolder. They
are kept as documentary evidence, not as a pipeline.
