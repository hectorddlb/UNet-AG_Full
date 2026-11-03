"""Inference time per image of the 6 methods, measured on GPU.

Complements ``computational_cost.py`` (analytic FLOPs). Here wall-clock time
is measured with ``torch.cuda.synchronize()`` around each repetition and a
prior warmup, so that kernel compilation and lazy CUDA context loading are not
timed.

EXTRAPOLATION OF THE 75-ANGLE METHODS
-------------------------------------
``das75pw`` and ``schiffner75pw`` repeat the SAME kernel 75 times, the only
difference being the steering angle; the coherent sum across angles is
negligible (2*Nz*Nx*74 FLOPs vs 23 TFLOP). ``--n_angles`` real angles are
measured (8 by default, spread across the sweep to cover extreme and central
angles) and extrapolated x75, reporting the standard deviation across angles
to show that the extrapolation is legitimate. Measuring all 75 would cost
~35 min of GPU per method.

DRUS
----
Measured in its native configuration with the ``drus`` package: one end-to-end
reconstruction (50 DDRM steps, prior in float16 on GPU, products by the
65536x65536 ``Vt`` on CPU) and, separately, the two components of one step.
Loading ``Vt.npy`` (17.2 GB) is reported separately: it is a start-up cost,
not an inference cost.

UNET-AG
-------
Measured as it runs (``unet_ag.inference``): the Fd-F adjoint with the band
the model processes (0.5-1.5 f_c), the log envelope (CPU) and the network.

SCENARIO AND NODE
-----------------
By default, the experimental scenario (``--phantom exp_resolution``), the same
as the GFLOP of the cost table, and all methods in the same process and on the
same node, with ``das_pw`` in float32 (precision matched to the adjoint).

Usage
-----
    python evaluation/scripts/inference_time.py --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
DRUS_ARTIFACTS = Path(os.environ.get("DRUS_ARTIFACTS", ROOT / "methods/04_drus/artifacts"))
sys.path.insert(0, str(ROOT / "code"))

PHANTOM_RF = {
    "sim_resolution": "simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
    "sim_contrast":   "simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
    "exp_resolution": "experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
    "exp_contrast":   "experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
}

NZ, NX = 612, 388
N_ANGLES_TOTAL = 75
DRUS_TIMESTEPS_TOTAL = 50
DRUS_SIDE, DRUS_CHANNELS = 256, 3


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device: str, n_warmup: int = 1, n_rep: int = 3) -> tuple[float, float]:
    """Return (mean, standard deviation) in seconds."""
    for _ in range(n_warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))


# ---------------------------------------------------------------------------

def load_picmus(phantom: str = "sim_resolution"):
    from schiffner_das import load_picmus_hdf5
    return load_picmus_hdf5(DATASETS / "PICMUS" / "database" / PHANTOM_RF[phantom])


def eval_grid(ds):
    """Canonical 612x388 grid — exact replica of evaluate_fieldii._build_grid."""
    c, fs = float(ds.c_avg), float(ds.f_s)
    ns = int(ds.rf_for_angle(0).shape[0])
    K, pitch = int(ds.n_elements), float(ds.element_pitch)
    z_max = (ns - 1) * c / (2.0 * fs)
    gz = torch.linspace(5e-3, z_max, NZ, dtype=torch.float64)
    gx = torch.linspace(-K * pitch / 2.0, +K * pitch / 2.0, NX, dtype=torch.float64)
    return gz, gx


def bench_fourier_kernel(ds, gz, gx, device: str, n_angles: int, fnum_kind: str,
                         precision: str = "float64"):
    """Time ``das_pw`` for ``n_angles`` angles spread across the sweep.

    ``precision`` matters a lot and must be declared when reporting:
    ``gen_classical_picmus_fixed.py`` does NOT pass the argument, so the
    canonical npys were generated with the ``float64`` default, whereas
    ``OpHFdf`` (the adjoint that feeds UNet-AG) uses ``complex64``. On a TITAN
    RTX FP64 runs at 1/32 of the throughput, so comparing the seconds of the
    two as is measures the precision, not the algorithm. Both are measured.
    """
    from schiffner_das import ConstantFNumber, GratingAngleLB, das_pw

    fnum = ConstantFNumber(0.0) if fnum_kind == "const0" else GratingAngleLB(60.0, 1.5)
    angles = ds.steering_angles_rad
    idx = np.linspace(0, len(angles) - 1, n_angles).round().astype(int)
    index_t0 = -int(round(ds.initial_time_s * ds.f_s))

    def make_run(aidx: int):
        rf = ds.rf_for_angle(aidx)
        ang = float(angles[aidx])

        def run():
            # das_pw prints progress with \r; silenced to keep the log clean
            with contextlib.redirect_stdout(io.StringIO()):
                das_pw(gx, gz, rf, ds.f_s, ang,
                       ds.element_width, ds.element_pitch, ds.c_avg,
                       f_bounds=(ds.f_lb, ds.f_ub), index_t0=index_t0,
                       F_number=fnum, device=device, precision=precision,
                       tx_sign=+1.0, tx_reference="origin")
        return run

    # A single warmup (kernel compilation, CUDA context, FFT cache): the kernel
    # is identical for every angle, so warming up on each one would triple the
    # cost of the benchmark without changing the result.
    make_run(int(idx[0]))()
    sync(device)

    per_angle = []
    for aidx in idx:
        m, _ = timeit(make_run(int(aidx)), device, n_warmup=0, n_rep=1)
        per_angle.append(m)
        print(f"    angle {int(aidx):3d} ({math.degrees(angles[int(aidx)]):+5.1f}°): "
              f"{m:.3f} s", flush=True)

    mu, sd = float(np.mean(per_angle)), float(np.std(per_angle))
    return dict(per_angle_s=mu, per_angle_std_s=sd, n_measured=int(n_angles),
                precision=precision,
                total_1pw_s=mu, total_75pw_s=mu * N_ANGLES_TOTAL)


def bench_unet_ag(phantom: str, device: str):
    """UNet-AG as it runs: Fd-F adjoint, log envelope and network.

    Uses ``unet_ag.inference.UNetAG`` with the seed 42 weights: the adjoint is
    the ``make_op_for_angle`` operator (band 0.5-1.5 f_c, the one the model
    processes), the envelope is ``env_log`` (CPU) and the network receives the
    real input and F_map. The three components are timed separately.
    """
    from unet_ag.inference import UNetAG, load_picmus_rf
    from unet_ag.physics.f_number_map import compute_f_number_map_band_avg
    from unet_ag.physics.h_fdf_wrapper import make_op_for_angle
    from unet_ag.train.b1_attention_gate import _build_grid, _envelope_log_compress

    model = UNetAG.from_pretrained("s42", device=device)
    rf, geom = load_picmus_rf(DATASETS / "PICMUS" / "database" / PHANTOM_RF[phantom])
    gz, gx = _build_grid(geom.as_dict(), NZ, NX)
    op = make_op_for_angle(0.0, geom.as_dict(), gz, gx, device=device, nf_chunk=None)
    y = UNetAG._prepare_rf(rf, geom).to(device).to(op.float_dtype)

    with torch.no_grad():
        m_adj, s_adj = timeit(lambda: op.adjoint(y), device, n_warmup=1, n_rep=5)
        x0c = op.adjoint(y)
    x0_real = (x0c.real if x0c.is_complex() else x0c).float().cpu()
    m_env, s_env = timeit(lambda: _envelope_log_compress(x0_real), "cpu", n_warmup=1, n_rep=5)
    x0 = _envelope_log_compress(x0_real).to(device)[None, None]
    f_map = torch.from_numpy(compute_f_number_map_band_avg(
        gz.numpy(), gx.numpy(), geom.as_dict())).to(device)[None, None]
    with torch.no_grad():
        m_net, s_net = timeit(lambda: model.model(x0, f_map), device, n_warmup=3, n_rep=10)

    return dict(n_frequencies_adjoint=int(op.Nf),
                adjoint_s=m_adj, adjoint_std_s=s_adj,
                envelope_s=m_env, envelope_std_s=s_env,
                network_s=m_net, network_std_s=s_net, total_s=m_adj + m_env + m_net)


def bench_drus(phantom: str, device: str, artifacts: Path):
    """DRUS in its native configuration (``drus`` package): prior in float16 on
    GPU, spectral products with ``Vᵀ`` on CPU, 50 DDRM steps.

    Reported: the loading of ``Vᵀ`` (start-up, not inference), the end-to-end
    time of one reconstruction (mean of two, after a warmup one) and,
    separately, that of the two components of one step: the prior UNet and one
    product by ``Vᵀ``.
    """
    from drus.inference import DRUS

    key = {"sim_resolution": "simu_reso", "sim_contrast": "simu_cont",
           "exp_resolution": "expe_reso", "exp_contrast": "expe_cont"}[phantom]
    model = DRUS.from_artifacts(artifacts, device=device)
    t0 = time.perf_counter()
    ops = model.spectral
    t_load = time.perf_counter() - t0
    net = model.model

    f = DATASETS / "PICMUS" / "database" / PHANTOM_RF[phantom]
    model.reconstruct_picmus(f, phantom=key)           # warmup, not timed
    sync(device)
    e2e = []
    for _ in range(2):
        t0 = time.perf_counter()
        model.reconstruct_picmus(f, phantom=key)
        sync(device)
        e2e.append(time.perf_counter() - t0)

    x = torch.randn(1, DRUS_CHANNELS, DRUS_SIDE, DRUS_SIDE, device=device)
    t = torch.zeros(1, device=device)
    with torch.no_grad():
        m_unet, s_unet = timeit(lambda: net(x, t), device, n_warmup=2, n_rep=5)
    vec = torch.randn(1, DRUS_CHANNELS * ops.n)
    m_vt, s_vt = timeit(lambda: ops.Vt(vec), "cpu", n_warmup=1, n_rep=2)
    return dict(svd_load_s=t_load, end_to_end_s=float(np.mean(e2e)),
                end_to_end_runs_s=e2e, unet_s=m_unet, unet_std_s=s_unet,
                vvt_s=m_vt, vvt_std_s=s_vt,
                spectral_fraction_per_step=3 * m_vt / (3 * m_vt + m_unet))


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_angles", type=int, default=8,
                    help="real angles to measure before extrapolating x75")
    ap.add_argument("--phantom", default="exp_resolution", choices=list(PHANTOM_RF),
                    help="phantom whose RF is timed (default: the experimental "
                         "scenario, the one of the hypothesis)")
    ap.add_argument("--drus_artifacts", type=Path, default=DRUS_ARTIFACTS)
    ap.add_argument("--precision", default="float32",
                    choices=["float64", "float32"],
                    help="das_pw precision; float32 matches the adjoint's (the thesis one)")
    ap.add_argument("--skip_drus", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "evaluation/metrics/inference_time_fp32.json")
    args = ap.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; aborting (CPU timings are not comparable)")
        sys.exit(1)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"=== Inference time — {gpu} | das_pw precision={args.precision} ===\n")

    ds = load_picmus(args.phantom)
    gz, gx = eval_grid(ds)
    import socket
    print(f"  node {socket.gethostname()} | phantom {args.phantom}", flush=True)

    print("  [1/4] DAS (ConstantFNumber(0)) ...", flush=True)
    das = bench_fourier_kernel(ds, gz, gx, dev, args.n_angles, "const0", args.precision)
    print("  [2/4] Schiffner Fd-F (GratingAngleLB(60,1.5)) ...", flush=True)
    sch = bench_fourier_kernel(ds, gz, gx, dev, args.n_angles, "gratinglb", args.precision)
    print("  [3/4] UNet-AG ...", flush=True)
    ag = bench_unet_ag(args.phantom, dev)
    drus = None
    if not args.skip_drus:
        print("  [4/4] DRUS ...", flush=True)
        try:
            drus = bench_drus(args.phantom, dev, args.drus_artifacts)
        except Exception as e:                                  # noqa: BLE001
            print(f"    DRUS failed: {type(e).__name__}: {e}", flush=True)

    rows = [
        ("DAS 1 PW", das["total_1pw_s"]),
        # Fd-F on a single angle: the kernel is already measured per angle, so
        # this number needs no separate measurement.
        ("Schiffner Fd-F 1 PW", sch["total_1pw_s"]),
        ("DAS 75 PW", das["total_75pw_s"]),
        ("Schiffner Fd-F 75 PW", sch["total_75pw_s"]),
        ("DRUS (DDRM 50 steps)", drus["end_to_end_s"] if drus else float("nan")),
        ("UNet-AG (proposed)", ag["total_s"]),
    ]
    print(f"\n=== Time per frame — {gpu} ===")
    hdr = f"{'method':24s} {'seconds':>12s} {'x DAS 1PW':>11s} {'frames/s':>10s}"
    print(hdr); print("-" * len(hdr))
    base = rows[0][1]
    for name, t in rows:
        fps = 1.0 / t if t == t and t > 0 else float("nan")
        print(f"{name:24s} {t:12.3f} {t/base:11.1f} {fps:10.3f}")

    payload = dict(gpu=gpu, node=socket.gethostname(),
                   precision_das_pw=args.precision, phantom=args.phantom,
                   grid=[NZ, NX], das=das, schiffner=sch,
                   unet_ag=ag, drus=drus,
                   table={n: t for n, t in rows})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
