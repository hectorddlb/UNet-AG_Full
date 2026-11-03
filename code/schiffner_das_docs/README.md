# Schiffner Fd-F — Python Port

Python/PyTorch reimplementation of Schiffner's **frequency-dependent F-number**
for coherent plane-wave compounding [1, 2].

This serves as **Baseline 2** in the project (Fd-F beamforming without
deep learning) and provides reusable components for integrating the H_{Fd-F}
operator into the self-supervised DL loss function.

## Quick Start

```bash
pip install torch numpy scipy matplotlib
python examples.py --data data_RF.mat --out results
```

## Project Structure

```
schiffner_das/
├── __init__.py            # Package init with convenient imports
├── f_numbers.py           # F-number models (ConstantFNumber, GratingAngleLB)
├── windows.py             # Window functions (Boxcar, Hann, Triangular, Tukey)
├── normalizations.py      # Normalization policies (Off, On)
├── beamforming.py         # Core DAS beamformer (das_pw)
├── data_io.py             # MATLAB .mat file loading
└── visualization.py       # B-mode plotting utilities

examples.py                # Main demo script (mirrors Schiffner's examples.m)
data_RF.mat                # Example dataset (from Schiffner's repo)
requirements.txt           # Python dependencies
```

## Reuse for the DL Pipeline

The key components designed for reuse in the H_{Fd-F} forward model:

### 1. F-number computation (precomputable)
```python
from schiffner_das import GratingAngleLB
fn = GratingAngleLB(angle_lb_deg=60, F_number_ub=1.5)
# element_pitch_over_lambda = p * f / c  for each frequency bin
F_values = fn.compute_values(element_pitch_over_lambda)  # (N_freq,)
```

### 2. Window functions (element-wise, differentiable)
```python
from schiffner_das import Tukey
window = Tukey(0.2)
apod = window.compute_samples(positions_over_halfwidth_abs)  # any shape
```

### 3. Forward model operator
The `das_pw` function implements the complete Fourier-domain beamformer.
For the DL loss, you'll extract the core operations:
- `torch.fft.rfft` on RF data → Fourier coefficients
- Multiply by Fd-F aperture weights (precomputed per frequency)
- Phase compensation per element/voxel
- `torch.fft.irfft` → reconstructed signal

All operations use native PyTorch and are compatible with `autograd`.

## Parameters (matching Schiffner's paper)

| Parameter | Value | Source |
|-----------|-------|--------|
| N_elements | 128 | L14-5/38 probe |
| element_width | 279.8 µm | |
| element_pitch | 304.8 µm | |
| c_avg | 1538.75 m/s | CIRS Model 040 |
| f_s | 20 MHz | SonixTouch |
| f_lb | 2.25 MHz | |
| f_ub | 6.75 MHz | |
| χ_lb | 60° | Minimum grating lobe distance |
| F_ub | 1.5 | Fixed F-number benchmark |

## Performance Notes

- **CPU-only** (no GPU required)
- Full 512×512 grid: ~30 min per steering angle on a single CPU core
- Reduced 64×128 grid: ~60 s per angle (sufficient for validation)
- Adjust `z_chunk` parameter in `das_pw()` to control memory usage

## References

[1] M. F. Schiffner and G. Schmitz, "Frequency-dependent F-number suppresses
    grating lobes and improves the lateral resolution in coherent plane-wave
    compounding," IEEE TUFFC, 70(9), 2023.

[2] M. F. Schiffner and G. Schmitz, "Frequency-dependent F-number increases
    the contrast and the spatial resolution in fast pulse-echo ultrasound
    imaging," IEEE IUS, 2021.

[3] M. F. Schiffner, MATLAB source: https://github.com/mschiffn/f_number
