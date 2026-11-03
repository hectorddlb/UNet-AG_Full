# Data

The raw sets are not in the repository because of their size (~30 GB). This
folder holds what is needed to rebuild or locate them, plus the derived
supervision, which is included because regenerating it takes days of compute.

## Expected layout at run time

Point `THESIS_DATASETS` to a directory with this layout (or place the data
here):

```
$THESIS_DATASETS/
├── PICMUS/
│   └── database/
│       ├── simulation/{contrast_speckle,resolution_distorsion}/*.hdf5
│       └── experiments/{contrast_speckle,resolution_distorsion}/*.hdf5
└── field_ii_data_multiangle/
    ├── points/points_XXXX.mat      (300 phantoms, 75 angles each)
    └── cysts/cysts_XXXX.mat        (300 phantoms, 75 angles each)
```

## PICMUS

Public benchmark of the *Plane-wave Imaging Challenge in Medical Ultrasound*
(IEEE IUS 2016). Download: <https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/>

The four canonical phantoms are used, each with 75 angles within ±16°:
`simulation/{resolution_distorsion, contrast_speckle}` and
`experiments/{resolution_distorsion, contrast_speckle}`. HDF5 format following
the UFF standard. The project reader is `unet_ag.data.rf_loader.PICMUSLoader`.

**PICMUS is reserved entirely for evaluation**: it never takes part in
training.

## Field II

Simulated training set: 600 phantoms (300 of point reflectors with 20
scatterers each, 300 of anechoic inclusions with a speckle background and 10
cysts), simulated with 75 angles per phantom.

The generation scripts are in `field_ii/`. They require
[Field II](https://www.field-ii.dk/) on MATLAB:

| Script | Purpose |
|---|---|
| `generate_training_data_multiangle.m` | Generates one phantom with its 75 angles |
| `run_generation_multiangle.m` | Launches the generation of the full set |
| `generate_range_cluster.m` / `run_cluster_generation.m` | Range-based variants, to parallelize on a cluster |

Transducer parameters (they replicate PICMUS): `f_c` = 5.208 MHz, `f_s` =
20.832 MHz, 128 elements with a 300 µm pitch and a 270 µm width, 5 mm height
with elevation focus at 20 mm, 2.5-cycle excitation, Gaussian impulse response
with 67 % fractional bandwidth, `c` = 1540 m/s.

## Derived supervision: `targets_schiffner_75pw/`

The **287 Schiffner Fd-F 75 plane wave reconstructions** that make up the
actual training target of the final model, in `points/<id>/compounded.npy`.

They are included in the repository (262 MB) because regenerating them
requires the full Field II set and days of compute. To regenerate them:
`methods/02_schiffner_fdf/scripts/precompute_schiffner_targets_fieldii.py`.

**Note:** **these are the targets that were used to train the evaluated
model**, and they carry transmit-convention and grid defects. They are kept as
is, uncorrected, because they are the actual supervision of the reported
model. The corrected version is regenerated with
`evaluation/scripts/precompute_schiffner_targets_fieldii_fixed.py` followed by
`align_targets_to_train_grid.py`; the corresponding control experiment is in
`evaluation/supervision_control/`.

## CUBDL

Used only for the cross-domain evaluation. Not included; available at
<https://cubdl.jhu.edu/>. The subset used is the plane-wave acquisitions with
a compatible geometry (~L7-4/L10-5).
