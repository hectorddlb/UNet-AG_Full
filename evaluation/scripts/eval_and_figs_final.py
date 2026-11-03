"""Unified evaluation + figures for the 6 methods on PICMUS.

ONE protocol for every method (the thesis-canonical one):
  image -> envelope -> dB (60 dB DR) -> env_log [0,1] -> bilinear to the
  612x388 eval grid -> _eval_image (FWHM lat/ax, CNR dB, gCNR) with the
  analytic PICMUS masks.

Methods (inputs, produced by the gen_* scripts of this folder):
  das1pw        evaluation/npys/das1pw/<key>_das1pw.npy           (complex, scan grid)
  schiffner1pw  evaluation/npys/schiffner1pw/<key>_*.npy          (complex, scan grid)
  das75pw       evaluation/npys/das75pw/<key>_das75pw.npy         (complex, scan grid)
  schiffner75pw evaluation/npys/schiffner75pw/<key>_*.npy         (complex, scan grid)
  drus          evaluation/npys/drus/x_<ph>_g<γ*>_s*.npy          (signed, 256x256)
  b1v5          evaluation/npys/b1v5_<seed>/<key>_b1v5.npy        (env_log, eval grid)

Outputs:
  evaluation/npys_eval_grid/<method>[_<seed>]/<key>.npy  env_log 612x388
  evaluation/metrics/metrics_picmus_final.csv        long format
  evaluation/metrics/comparison_table.md               aggregated
  evaluation/figures/*.png                               control B-modes

Run from the repository root AFTER gen_classical_picmus_fixed.py,
gen_b1v5_picmus_seeds.py and gen_drus_picmus.py.
"""
from __future__ import annotations

import csv
import re
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("THESIS_DATASETS", ROOT / "data"))  # PICMUS/ and Field II/, see data/README.md
sys.path.insert(0, str(ROOT / "code"))

from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import _build_grid  # noqa: E402
from unet_ag.eval.evaluate_picmus import _eval_image  # noqa: E402

RF_OUT = ROOT / "evaluation"
NPYS = RF_OUT / "npys"
FIGS = RF_OUT / "figures"
METR = RF_OUT / "metrics"
DR_DB = 60.0
DPI = 200

PHANTOMS = [
    ("sim_resolution", "sim", "resolution_distortion", "Simulation — Resolution"),
    ("sim_contrast",   "sim", "contrast_speckle",      "Simulation — Contrast"),
    ("exp_resolution", "exp", "resolution_distortion", "Experimental — Resolution"),
    ("exp_contrast",   "exp", "contrast_speckle",      "Experimental — Contrast"),
]

DRUS_KEYMAP = {
    "sim_resolution": "simu_reso",
    "sim_contrast":   "simu_cont",
    "exp_resolution": "expe_reso",
    "exp_contrast":   "expe_cont",
}
# γ* per phantom for DRUS-noFT on the extended grid (= drus.inference.THESIS_GAMMA)
DRUS_GAMMA = {
    "sim_resolution": 75,
    "sim_contrast":   50,
    "exp_resolution": 75,
    "exp_contrast":   25,
}

METHOD_LABELS = {
    "das1pw":        "DAS 1 PW",
    "schiffner1pw":  "Schiffner Fd-F (1 PW)",
    "das75pw":       "DAS 75 PW",
    "schiffner75pw": "Schiffner Fd-F (75 PW)",
    "drus":          "DRUS (diffusion, 1 PW)",
    "b1v5":          "UNet-AG anchor05 (1 PW)",
}


def env_log01(env: np.ndarray) -> np.ndarray:
    env = np.abs(env).astype(np.float64)
    env = env / (env.max() + 1e-30)
    db = np.clip(20.0 * np.log10(env + 1e-30), -DR_DB, 0.0)
    return ((db + DR_DB) / DR_DB).astype(np.float32)


def interp_to(img01, src_z, src_x, dst_z, dst_x) -> np.ndarray:
    f = RegularGridInterpolator((src_z, src_x), img01.astype(np.float64),
                                bounds_error=False, fill_value=0.0)
    zz, xx = np.meshgrid(dst_z, dst_x, indexing="ij")
    return f(np.stack([zz.ravel(), xx.ravel()], axis=-1)) \
        .reshape(len(dst_z), len(dst_x)).astype(np.float32)


def load_scan_axes(key: str) -> tuple[np.ndarray, np.ndarray]:
    sys.path.insert(0, str(ROOT / "code"))
    from schiffner_das import load_picmus_scan
    sub = {"sim_resolution": "simulation/resolution_distorsion/resolution_distorsion_simu_scan.hdf5",
           "sim_contrast":   "simulation/contrast_speckle/contrast_speckle_simu_scan.hdf5",
           "exp_resolution": "experiments/resolution_distorsion/resolution_distorsion_expe_scan.hdf5",
           "exp_contrast":   "experiments/contrast_speckle/contrast_speckle_expe_scan.hdf5"}[key]
    g = load_picmus_scan(DATASETS / "PICMUS/database" / sub)
    return g.z.numpy().astype(np.float64), g.x.numpy().astype(np.float64)


def collect_images() -> dict:
    """Return {(method, seed): {key: env_log_612x388}} plus grids per phantom."""
    grids = {}
    loaders = {}
    for key, mode, phantom, _ in PHANTOMS:
        loader = PICMUSLoader(DATASETS / "PICMUS", mode=mode,
                              phantom=phantom, normalize=True)
        sample = loader[0]
        gz, gx = _build_grid(sample.geometry, 612, 388)
        grids[key] = (gz.numpy().astype(np.float64),
                      gx.numpy().astype(np.float64))
        loaders[key] = loader

    out: dict = {}

    # classical (complex on scan grid)
    for m in ("das1pw", "schiffner1pw", "das75pw", "schiffner75pw"):
        d = {}
        for key, *_ in PHANTOMS:
            p = NPYS / m / f"{key}_{m}.npy"
            if not p.exists():
                print(f"WARN missing {p}")
                continue
            src_z, src_x = load_scan_axes(key)
            img01 = env_log01(np.load(p))
            gz, gx = grids[key]
            d[key] = interp_to(img01, src_z, src_x, gz, gx)
        if d:
            out[(m, "")] = d

    # DRUS noFT (γ* per phantom, every seed present)
    drus_dir = NPYS / "drus"
    gridnpz = np.load(drus_dir / "grid.npz")
    dz_ax, dx_ax = gridnpz["z_axis"].astype(np.float64), \
        gridnpz["x_axis"].astype(np.float64)
    seeds = set()
    for key, *_ in PHANTOMS:
        ph = DRUS_KEYMAP[key]
        g = DRUS_GAMMA[key]
        for p in sorted(drus_dir.glob(f"x_{ph}_g{g}_s*.npy")):
            seeds.add(re.search(r"_s(\d+)\.npy$", p.name).group(1))
    for s in sorted(seeds):
        d = {}
        for key, *_ in PHANTOMS:
            p = drus_dir / f"x_{DRUS_KEYMAP[key]}_g{DRUS_GAMMA[key]}_s{s}.npy"
            if not p.exists():
                continue
            img01 = env_log01(np.load(p))
            gz, gx = grids[key]
            d[key] = interp_to(img01, dz_ax, dx_ax, gz, gx)
        if d:
            out[("drus", f"s{s}")] = d

    # B1-v5 (env_log already on eval grid)
    for sd in ("s42", "s1", "s2"):
        d = {}
        for key, *_ in PHANTOMS:
            p = NPYS / f"b1v5_{sd}" / f"{key}_b1v5.npy"
            if p.exists():
                d[key] = np.load(p)
        if d:
            out[("b1v5", sd)] = d

    return out, grids, loaders


def save_eval_grid_npys(images: dict) -> None:
    for (m, seed), d in images.items():
        sub = RF_OUT / "npys_eval_grid" / (m if not seed else f"{m}_{seed}")
        sub.mkdir(parents=True, exist_ok=True)
        for key, img in d.items():
            np.save(sub / f"{key}.npy", img)


def run_metrics(images: dict, grids: dict, loaders: dict) -> list[dict]:
    rows = []
    for (m, seed), d in sorted(images.items()):
        for key, mode, phantom, _label in PHANTOMS:
            if key not in d:
                continue
            gz, gx = grids[key]
            met = _eval_image(d[key], loaders[key].phantom_path, phantom,
                              gz, gx)
            row = {"method": m, "seed": seed, "phantom": key}
            row.update(met)
            rows.append(row)
            print(m, seed or "-", key, {k: (round(v, 4) if isinstance(v, float) else v)
                                        for k, v in met.items()})
    return rows


def write_csv(rows: list[dict]) -> None:
    METR.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(METR / "metrics_picmus_final.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def write_table(rows: list[dict]) -> None:
    """Aggregated markdown table: mean±std across seeds per method."""
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[(r["method"], r["phantom"])].append(r)

    def fmt(vals):
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if not vals:
            return "—"
        if len(vals) == 1:
            return f"{vals[0]:.3f}"
        return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"

    lines = [
        "# Final comparison table — PICMUS (canonical env_log 612×388 protocol)",
        "",
        "Lateral FWHM (mm, ↓) on resolution phantoms; CNR (dB, ↑) and gCNR (↑) on contrast phantoms.",
        "Methods with several runs are reported as mean ± std across seeds.",
        "",
        "| Method | sim/res FWHM_lat | exp/res FWHM_lat | sim/con CNR dB | sim/con gCNR | exp/con CNR dB | exp/con gCNR |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in ("das1pw", "schiffner1pw", "das75pw", "schiffner75pw", "drus", "b1v5"):
        cells = [METHOD_LABELS[m]]
        for ph, col in [("sim_resolution", "fwhm_lat_med_mm"),
                        ("exp_resolution", "fwhm_lat_med_mm"),
                        ("sim_contrast", "cnr_db"),
                        ("sim_contrast", "gcnr"),
                        ("exp_contrast", "cnr_db"),
                        ("exp_contrast", "gcnr")]:
            vals = [r.get(col) for r in agg.get((m, ph), [])]
            cells.append(fmt(vals))
        lines.append("| " + " | ".join(cells) + " |")
    (METR / "comparison_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


Z_DISPLAY_MAX_M = 55e-3


def _crop_extent(img01: np.ndarray, grids: dict, key: str):
    """Crop the eval-grid image at Z_DISPLAY_MAX_M and return (img, extent).

    The canonical eval grid extends to (ns-1)c/2fs (123 mm for the exp
    phantoms) but the phantoms only have content down to ~50 mm; displaying
    the full grid squashes the anatomy into the top of the panel.
    """
    gz, gx = grids[key]
    iz = int(np.searchsorted(gz, Z_DISPLAY_MAX_M))
    iz = max(2, min(iz, len(gz)))
    extent = [gx[0] * 1e3, gx[-1] * 1e3, gz[iz - 1] * 1e3, gz[0] * 1e3]
    return img01[:iz], extent


def fig_4panel(images: dict, grids: dict, method: str, seed: str,
               fname: str) -> None:
    d = images.get((method, seed), {})
    fig, axes = plt.subplots(2, 2, figsize=(11, 12), constrained_layout=True)
    order = ["sim_resolution", "sim_contrast", "exp_resolution", "exp_contrast"]
    labels = {k: lab for k, _, _, lab in PHANTOMS}
    last = None
    for ax, key in zip(axes.ravel(), order):
        if key not in d:
            ax.set_axis_off()
            continue
        img, extent = _crop_extent(d[key], grids, key)
        db = img * DR_DB - DR_DB
        last = ax.imshow(db, cmap="gray", vmin=-DR_DB, vmax=0,
                         extent=extent, aspect="equal")
        ax.set_title(labels[key], fontsize=15)
        ax.set_xlabel("Lateral (mm)", fontsize=12)
        ax.set_ylabel("Axial (mm)", fontsize=12)
    if last is not None:
        cb = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.7,
                          pad=0.02)
        cb.set_label("dB", fontsize=13)
    fig.savefig(FIGS / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / fname)


def fig_compare5(images: dict, grids: dict, key: str, fname: str) -> None:
    # Order: the four 1 PW methods first, then the two 75 PW ones, so that the
    # same-input comparison can be read at a glance.
    cols = [("das1pw", ""), ("schiffner1pw", ""), ("drus", "s0"), ("b1v5", "s42"),
            ("das75pw", ""), ("schiffner75pw", "")]
    fig, axes = plt.subplots(1, 6, figsize=(28, 6.5), constrained_layout=True)
    last = None
    for j, ((m, sd), ax) in enumerate(zip(cols, axes)):
        d = images.get((m, sd), {})
        if key not in d:
            ax.set_axis_off()
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            continue
        img, extent = _crop_extent(d[key], grids, key)
        db = img * DR_DB - DR_DB
        last = ax.imshow(db, cmap="gray", vmin=-DR_DB, vmax=0,
                         extent=extent, aspect="equal")
        ax.set_title(METHOD_LABELS[m], fontsize=14)
        ax.set_xlabel("Lateral (mm)", fontsize=11)
        if j == 0:
            ax.set_ylabel("Axial (mm)", fontsize=11)
    if last is not None:
        cb = fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.85,
                          pad=0.01)
        cb.set_label("dB", fontsize=12)
    fig.savefig(FIGS / fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIGS / fname)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    images, grids, loaders = collect_images()
    save_eval_grid_npys(images)
    rows = run_metrics(images, grids, loaders)
    write_csv(rows)
    write_table(rows)

    fig_4panel(images, grids, "das1pw", "", "das_1pw_picmus_4panel.png")
    fig_4panel(images, grids, "schiffner1pw", "",
               "schiffner_fdf_1pw_picmus_4panel.png")
    fig_4panel(images, grids, "das75pw", "", "das_75pw_picmus_4panel.png")
    fig_4panel(images, grids, "schiffner75pw", "",
               "schiffner_fdf_75pw_picmus_4panel.png")
    fig_4panel(images, grids, "drus", "s0", "drus_noft_picmus_4panel.png")
    fig_4panel(images, grids, "b1v5", "s42", "b1v5_anchor05_picmus_4panel.png")
    for key, _, _, _lab in PHANTOMS:
        fig_compare5(images, grids, key, f"compare_6methods_{key}.png")
    print("ALL DONE")


if __name__ == "__main__":
    main()
