"""Computational cost per image of the 6 methods of the thesis.

Analytic FLOPs (hardware independent). Wall-clock time is measured
separately, with inference_time.py.

WHY AN AUTOMATIC COUNTER IS NOT ENOUGH
--------------------------------------
``torch.utils.flop_counter.FlopCounterMode`` only sees conv/matmul/bmm/sdpa.
The three classical methods (das1pw, das75pw, schiffner75pw) and the adjoint
that feeds UNet-AG are **elementwise**: an automatic counter reports them as
0 FLOPs. They have to be counted by hand on the real kernel. The two networks
do go through the automatic counter.

WHAT EXACTLY IS COUNTED
-----------------------
What the evaluation protocol really runs is counted, which is NOT what the
name of each method suggests:

  das1pw, schiffner1pw, das75pw, schiffner75pw -> all four use
      ``schiffner_das.das_pw``, the Fourier beamformer (see
      gen_classical_picmus_fixed.py). Only ``ConstantFNumber(0)`` vs
      ``GratingAngleLB(60,1.5)`` and the number of angles change. That is:
      DAS and Schiffner Fd-F cost the same, and the Fd-F apodization is free
      compared to compounding.
  unet_ag  -> adjoint ``OpHFdf.adjoint`` (1 angle, same Fourier kernel,
      wider band) + forward of the UNet+AG.
  drus     -> 50 DDRM steps, each with 1 forward of the 256x256
      guided-diffusion UNet + 3 products by V/Vt (65536 x 65536).

The **temporal** DAS (``physics/classical/das_1pw.py``) is added as a
reference, since it is what a real scanner does, so that the comparison with
the literature makes sense.

FLOP CONVENTION
---------------
  real add/sub/mul/div ................. 1 FLOP
  sqrt, sin, cos, exp .................. 1 FLOP  (reported separately so the
                                         reader can reweight them)
  complex product ...................... 6 FLOPs (4 mul + 2 add)
  complex sum .......................... 2 FLOPs
  exp(i*phi) ........................... 2 FLOPs (sin + cos)
  matmul (m,k)x(k,n) ................... 2*m*k*n

PER SCENARIO
------------
The number of frequencies processed by the Fourier beamformers depends on the
number of samples of each acquisition, so the cost is computed for the four
PICMUS phantoms (``cost_per_scenario.csv``). There are two bands: that of the
classical methods (2.25-6.75 MHz, the ``das_pw`` one) and that of the adjoint
that feeds UNet-AG (0.5-1.5 fc = 2.604-7.812 MHz, the ``OpHFdf`` one), which is
wider, so the adjoint costs more than 1 PW DAS. The hypothesis is stated on
the experimental scenarios: the ``methods`` list of the JSON, which feeds the
cost comparison, is the one of that scenario.

Usage
-----
    python evaluation/scripts/computational_cost.py      # PICMUS in $THESIS_DATASETS/PICMUS
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "code"))

# Canonical evaluation grid
NZ, NX = 612, 388
N_ANGLES = 75
DRUS_TIMESTEPS = 50
DRUS_SIDE = 256
DRUS_CHANNELS = 3


# ---------------------------------------------------------------------------
# 1. Real problem dimensions (read from PICMUS, not made up)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "sim_resolution": "simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
    "sim_contrast":   "simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
    "exp_resolution": "experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
    "exp_contrast":   "experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
}
SCENARIO_HYPOTHESIS = "exp_resolution"   # = exp_contrast: same dimensions


def picmus_dims(rf_rel: str, band: str = "das") -> dict:
    """Nt, Nel, fs, band and Nf exactly as ``das_pw`` and ``OpHFdf`` derive them.

    ``band="das"``: the dataset band (2.25-6.75 MHz), that of the classical
    beamformers. ``band="adjoint"``: 0.5-1.5 fc with fc = fs/4, the
    ``OpHFdf`` default (measurement_operator_fdf.py), that of the adjoint.
    """
    from schiffner_das import load_picmus_hdf5

    ds = load_picmus_hdf5(DATASETS / "PICMUS" / "database" / rf_rel)
    rf = ds.rf_for_angle(0)
    Nt, Nel = rf.shape
    fs, c = float(ds.f_s), float(ds.c_avg)
    pitch = float(ds.element_pitch)
    if band == "das":
        f_lb, f_ub = float(ds.f_lb), float(ds.f_ub)
    else:
        fc = fs / 4.0
        f_lb, f_ub = 0.5 * fc, 1.5 * fc

    # Zero-padding and frequency axis: exact replica of das_pw (lines 178-196)
    M = (Nel - 1) / 2.0
    ctr_first, ctr_last = -M * pitch, M * pitch
    span = ctr_last - ctr_first
    # z0 of the canonical evaluation grid
    z0 = 5e-3
    Npad = math.ceil((math.sqrt(span ** 2 + z0 ** 2) - z0) * fs / c)
    Ndft = Nt + Npad + 1 - (Nt + Npad) % 2
    ifl = math.ceil(f_lb * Ndft / fs)
    ifu = math.floor(f_ub * Ndft / fs)
    Nf = ifu - ifl + 1

    return dict(Nt=Nt, Nel=Nel, fs=fs, c=c, pitch=pitch,
                f_lb=f_lb, f_ub=f_ub,
                Npad=Npad, Ndft=Ndft, Nf=Nf,
                n_angles=int(len(ds.steering_angles_rad)))


# ---------------------------------------------------------------------------
# 2. Analytic FLOPs
# ---------------------------------------------------------------------------

# Cost per (frequency, pixel, element) tuple of the Fourier kernel.
# Derived line by line from beamforming.py:264-315 / measurement_operator_fdf.py:439-451
FOURIER_KERNEL_COST = {
    "aw * tof":                    1,   # real mul
    "exp(1j * .)":                 2,   # sin + cos
    "dlat_a / hw_elem":            1,   # div (p_hw)
    "Tukey.compute_samples":       5,   # 2 cmp + sub + div + mul + cos + add + div
    "apod * valid":                1,
    "NormalizationOn (sum + div)": 2,
    "apod(real) * eph(complex)":   2,
    "* rf_bp (complex x complex)": 6,
    "sum over elements":           2,   # complex sum
    "sum over frequencies":        2,   # complex sum
}
FOURIER_PER_TUPLE = sum(FOURIER_KERNEL_COST.values())
FOURIER_TRANSC_PER_TUPLE = 3          # sin, cos (exp) + cos (Tukey)

# Cost per (pixel, element) tuple of the temporal DAS (das_1pw.py:44-77)
TIME_DAS_COST = {
    "dx = x - xk":            1,
    "dx*dx + z*z":            2,
    "sqrt":                   1,
    "/ c":                    1,
    "+ t_tx - t_start":       2,
    "* fs":                   1,
    "round":                  1,
    "in_range (2 cmp)":       2,
    "in_aper (abs + cmp)":    2,
    "Hanning (mul+div+cos+add+mul)": 5,
    "clip":                   2,
    "samples * v * w":        2,
    "accumulation":           1,
}
TIME_DAS_PER_TUPLE = sum(TIME_DAS_COST.values())
TIME_DAS_TRANSC_PER_TUPLE = 2         # sqrt + cos


def fourier_kernel_flops(d: dict, nz: int, nx: int, n_angles: int) -> dict:
    """FLOPs of ``n_angles`` passes of the Fourier beamformer."""
    tuples = d["Nf"] * nz * nx * d["Nel"] * n_angles
    focus = tuples * FOURIER_PER_TUPLE
    # FFT per element and angle: 5 N log2 N (usual convention for a complex FFT)
    fft = n_angles * d["Nel"] * 5.0 * d["Ndft"] * math.log2(d["Ndft"])
    # Coherent sum across angles (complex)
    compound = 2.0 * nz * nx * (n_angles - 1) if n_angles > 1 else 0.0
    return {
        "total": focus + fft + compound,
        "focus": focus,
        "fft": fft,
        "compound": compound,
        "tuples": tuples,
        "transcendentals": tuples * FOURIER_TRANSC_PER_TUPLE,
    }


def time_das_flops(d: dict, nz: int, nx: int, n_angles: int) -> dict:
    tuples = nz * nx * d["Nel"] * n_angles
    return {
        "total": tuples * TIME_DAS_PER_TUPLE,
        "tuples": tuples,
        "transcendentals": tuples * TIME_DAS_TRANSC_PER_TUPLE,
    }


def drus_spectral_flops() -> dict:
    """Products by V/Vt of the DDRM sampler (ddrm_sampler.py:53-64, 120-197).

    Each call is a matmul (N,N) x (N,3) = 2*N*N*3 FLOPs, with N = 256^2.
    Per step: Vt(x0_t), Vt(et), V(vt_next) = 3 calls. Plus 1 initial V(init_y).
    """
    n = DRUS_SIDE * DRUS_SIDE
    per_call = 2.0 * n * n * DRUS_CHANNELS
    n_calls = 3 * DRUS_TIMESTEPS + 1
    return {"total": per_call * n_calls, "per_call": per_call, "n_calls": n_calls}


# ---------------------------------------------------------------------------
# 3. Network FLOPs (PyTorch automatic counter)
# ---------------------------------------------------------------------------

def unet_ag_flops(device: str = "cpu") -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    from unet_ag.models.b1_v3_residual import B1V3Residual

    model = B1V3Residual(base_ch=32, embed_ch=16).to(device).eval()
    x0 = torch.zeros(1, 1, NZ, NX, device=device)
    f_map = torch.zeros(1, 1, NZ, NX, device=device)

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x0, f_map)
    n_par = sum(p.numel() for p in model.parameters())
    return {"total": float(counter.get_total_flops()), "params": n_par}


def drus_unet_flops(device: str = "cpu") -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    from guided_diffusion.script_util import create_model_and_diffusion
    from drus.prior import DIFFUSION_FLAGS, MODEL_FLAGS

    flags = dict(MODEL_FLAGS)
    flags["use_fp16"] = False        # FLOPs do not change; avoids convert_to_fp16
    flags["use_checkpoint"] = False  # checkpointing re-runs the forward
    model, _ = create_model_and_diffusion(**flags, **DIFFUSION_FLAGS)
    model.to(device).eval()

    x = torch.zeros(1, DRUS_CHANNELS, DRUS_SIDE, DRUS_SIDE, device=device)
    t = torch.zeros(1, device=device)

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x, t)
    n_par = sum(p.numel() for p in model.parameters())
    del model
    return {"total": float(counter.get_total_flops()), "params": n_par}


# ---------------------------------------------------------------------------
# 4. Assembly
# ---------------------------------------------------------------------------

def build_table(dp: dict, da: dict, ag: dict, du: dict, sp: dict) -> list[dict]:
    """Rows of the cost table for one scenario.

    ``dp``: dimensions in the band of the classical beamformers; ``da``: in
    the band of the UNet-AG adjoint.
    """
    k1 = fourier_kernel_flops(dp, NZ, NX, 1)
    k75 = fourier_kernel_flops(dp, NZ, NX, N_ANGLES)
    adj = fourier_kernel_flops(da, NZ, NX, 1)
    return [
        dict(method="DAS 1 PW", input="1 PW",
             flops=k1["total"], params=0,
             breakdown=f"Fourier x1 angle ({k1['focus']/k1['total']*100:.1f}% focus)"),
        dict(method="Schiffner Fd-F 1 PW", input="1 PW",
             flops=k1["total"], params=0,
             breakdown="identical to DAS 1 PW: only the F-number changes, "
                       "which adds no tuples to the kernel"),
        dict(method="DAS 75 PW", input="75 PW",
             flops=k75["total"], params=0,
             breakdown="Fourier x75 angles"),
        dict(method="Schiffner Fd-F 75 PW", input="75 PW",
             flops=k75["total"], params=0,
             breakdown="identical to DAS 75 PW: Fd-F adds no tuples"),
        dict(method="DRUS (DDRM 50 steps)", input="1 PW",
             flops=DRUS_TIMESTEPS * du["total"] + sp["total"], params=du["params"],
             breakdown=f"50 x UNet256 ({DRUS_TIMESTEPS*du['total']/1e12:.2f} TFLOP) "
                       f"+ V/Vt ({sp['total']/1e12:.2f} TFLOP)"),
        dict(method="UNet-AG (proposed)", input="1 PW",
             flops=adj["total"] + ag["total"], params=ag["params"],
             breakdown=f"adjoint ({adj['total']/1e9:.1f} GFLOP, {da['Nf']} frequencies) "
                       f"+ UNet ({ag['total']/1e9:.1f} GFLOP)"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net_device", default="cpu",
                    help="where to instantiate the networks to count FLOPs (cpu is enough)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "evaluation/metrics/computational_cost.json")
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "evaluation/metrics/cost_per_scenario.csv")
    args = ap.parse_args()

    print("  counting UNet-AG ...", flush=True)
    ag = unet_ag_flops(args.net_device)
    print("  counting the guided-diffusion UNet (552 M params, slow) ...", flush=True)
    du = drus_unet_flops(args.net_device)
    sp = drus_spectral_flops()

    scenarios, csv_rows = {}, []
    for key, rf_rel in SCENARIOS.items():
        dp, da = picmus_dims(rf_rel, "das"), picmus_dims(rf_rel, "adjoint")
        rows = build_table(dp, da, ag, du, sp)
        g = {r["method"]: r["flops"] / 1e9 for r in rows}
        adj = fourier_kernel_flops(da, NZ, NX, 1)["total"] / 1e9
        scenarios[key] = dict(dimensions_das=dp, dimensions_adjoint=da, methods=rows)
        csv_rows.append(dict(
            scenario=key, Nt=dp["Nt"], Nf_das=dp["Nf"], Nf_adjoint=da["Nf"],
            das1pw_gflop=g["DAS 1 PW"], adjoint_gflop=adj, network_gflop=ag["total"] / 1e9,
            proposed_gflop=g["UNet-AG (proposed)"], pw75_gflop=g["DAS 75 PW"],
            drus_gflop=g["DRUS (DDRM 50 steps)"],
            ratio_75pw=g["DAS 75 PW"] / g["UNet-AG (proposed)"],
            ratio_drus=g["DRUS (DDRM 50 steps)"] / g["UNet-AG (proposed)"]))
        print(f"\n=== {key}: Nt={dp['Nt']} Nf(das)={dp['Nf']} Nf(adjoint)={da['Nf']} ===")
        for r in rows:
            print(f"  {r['method']:24s} {r['flops']/1e9:12.3f} GFLOP  {r['breakdown']}")

    hyp = scenarios[SCENARIO_HYPOTHESIS]
    td1 = time_das_flops(hyp["dimensions_das"], NZ, NX, 1)
    td75 = time_das_flops(hyp["dimensions_das"], NZ, NX, N_ANGLES)
    print(f"\n  [reference] temporal DAS 1 PW  = {td1['total']/1e9:.3f} GFLOP")
    print(f"  [reference] temporal DAS 75 PW = {td75['total']/1e9:.3f} GFLOP")

    payload = dict(
        methods_scenario=SCENARIO_HYPOTHESIS,
        dimensions=hyp["dimensions_das"],
        dimensions_adjoint=hyp["dimensions_adjoint"],
        grid=[NZ, NX],
        convention=dict(fourier_kernel_per_tuple=FOURIER_KERNEL_COST,
                        das_temporal_per_tuple=TIME_DAS_COST),
        methods=hyp["methods"],
        scenarios=scenarios,
        network=dict(unet_ag=ag, drus_unet=du, drus_spectral=sp),
        temporal_das_reference=dict(pw1=td1, pw75=td75),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")

    import csv
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
