# Validation of the Fd-F beamformer reimplementation

The Python port `schiffner_das.das_pw` reproduces the reference implementation
`das_pw.m` of the [mschiffn/f_number](https://github.com/mschiffn/f_number)
repository (commit `ef85e74`, MATLAB R2025b) with differences of the order of
machine precision.

| Case | Rel. L2 error | Max. diff. / peak | \|ΔgCNR\| | \|ΔCNR\| (dB) |
|---|---|---|---|---|
| Fd-F, 1 PW | 9.7 × 10⁻¹⁴ | 1.1 × 10⁻¹³ | 0 | 6.4 × 10⁻¹³ |
| DAS, 1 PW | 1.3 × 10⁻¹³ | 1.3 × 10⁻¹³ | 0 | 4.6 × 10⁻¹³ |
| Fd-F, 75 PW (native convention) | 1.5 × 10⁻¹³ | 1.5 × 10⁻¹³ | 0 | 5.0 × 10⁻¹² |
| DAS, 75 PW (native convention) | 2.2 × 10⁻¹³ | 1.4 × 10⁻¹³ | 0 | 4.8 × 10⁻¹² |

Source: `results/matlab_validation.csv`.

## Conditions

PICMUS experimental contrast scenario; 612 × 388 point grid, z from 5 to
50 mm and x within ±19.2 mm; Tukey window with parameter 0.2; normalization by
the sum of weights; 2.25 to 6.75 MHz band; zero start index; F-number with
χ_lb = 60° and F_ub = 1.5 (or full aperture for DAS); no antialiasing; float64
on both sides.

The reference implementation does not support the PICMUS transmit convention
for non-zero angles, so the two implementations are compared with 75 plane
waves in the reference's **native** convention. That validates the numerical
equivalence of the port, but it does not correspond to the 75 plane wave
comparators of the evaluation, which use the corrected convention. With a
single plane wave, at 0°, both conventions coincide.

The gCNR and CNR differences are computed with the same regions of interest
for both implementations (an inner disc of 0.75 r in each inclusion; a
background ring from 1.25 r to 1.75 r without the other inclusions), on the dB
envelope clipped at 60 dB and mapped to [0, 1].

## Contents

| File | What it is |
|---|---|
| `prepare_input.py` | Packs RF, geometry and grid into `work/exp_contrast_rf.mat` (common input) |
| `run_das_pw_matlab.m` | Runs the reference `das_pw.m` for a list of angles |
| `compare_python_matlab.py` | Runs the port with the same parameters and writes the CSV |
| `results/matlab_<case>.npy` | MATLAB complex images (612 × 388), already summed over angles |
| `results/matlab_validation.csv` | The table above |

## Reproducing

```bash
# 1. common input (needs PICMUS in $THESIS_DATASETS/PICMUS)
python evaluation/matlab_validation/prepare_input.py

# 2. MATLAB reference (optional: the images are already in results/)
git clone https://github.com/mschiffn/f_number evaluation/matlab_validation/work/f_number
git -C evaluation/matlab_validation/work/f_number checkout ef85e74
cd evaluation/matlab_validation
matlab -batch "run_das_pw_matlab('fdf', 38, 'work/matlab_fdf_1pw.mat'); run_das_pw_matlab('das', 38, 'work/matlab_das_1pw.mat')"
matlab -batch "run_das_pw_matlab('fdf', 1:15, 'work/matlab_fdf_75_0.mat')"   # ... 16:30, ..., 61:75
matlab -batch "run_das_pw_matlab('das', 1:15, 'work/matlab_das_75_0.mat')"   # likewise
cd ../..

# 3. comparison (uses the .mat files of work/ if they exist; otherwise results/*.npy)
python evaluation/matlab_validation/compare_python_matlab.py --device cuda
```

Each angle takes ~7.5 min in MATLAB on CPU; the 75 angles were split into five
jobs of 15 angles per method. The reference code is not included in this
repository because of its license (*citationware*); it is downloaded from its
original repository.
