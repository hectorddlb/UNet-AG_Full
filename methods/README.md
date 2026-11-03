# Compared methods

One directory per method. The library code lives in `code/`; here are the
run scripts, the SLURM jobs and the lightweight results.

| Dir | Method | Input | GFLOP | Role in the thesis |
|---|---|---|---|---|
| `01_das_1pw` | DAS, full aperture | 1 PW | 596 | Reference of the hypothesis |
| `02_schiffner_fdf` | Schiffner Fd-F | 1 and 75 PW | 596 / 44 700 | Generates the supervision; parameter-free baseline (1 PW) and compared method (75 PW) |
| `03_das_75pw` | Coherent compounded DAS | 75 PW | 44 700 | High-quality reference |
| `04_drus` | DRUS (DDRM diffusion) | 1 PW | 115 875 | Compared generative method |
| `05_unet_ag_proposed` | **UNet-AG** | 1 PW | **749** | The proposal |

GFLOP per image in the experimental scenarios (on which the hypothesis is
stated); the three scenarios are in `evaluation/metrics/cost_per_scenario.csv`.

All of them are evaluated with the single protocol of `evaluation/`. The final
B-modes of the six (including 1 PW Fd-F) are in `evaluation/npys_eval_grid/`.

The SLURM jobs in each folder keep the paths of the cluster they were run on;
they are reused by passing the paths through environment variables.

## `05_unet_ag_proposed`

- `checkpoints/anchor05_{s42,s1,s2}.pt` — the three trained models (41 MB each).
- `logs/<seed>/` — training and validation curves, and the per-run
  evaluations on PICMUS, Field II and CUBDL.
- `slurm/` — the jobs of the B1 lineage.
- `submit_anchor05_seeds.sh` — launches the three seeds.

**Note:** the CSVs in `logs/` use the **per-run** evaluator, whose "DAS 1 PW"
reference is the Fd-F adjoint, not the full-aperture DAS of the canonical
protocol. They are useful to compare trainings with each other; the
publishable numbers are those in `evaluation/metrics/`.

## `04_drus`

Full Python replica of the method of Zhang et al. (2023), wrapped as a package
(`code/drus/`, `drus` command) like the proposed method. Its `README.md`
explains how to build its artifacts (operator, SVD and prior, which do not fit
in the repository) and how to reconstruct with the thesis conditions.
