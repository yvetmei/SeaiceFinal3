# SeaIce-PI-LSTM

Physics-Informed LSTM for Arctic Sea Ice Drift Prediction with Continual Learning and Uncertainty Estimation.

## Features

- **PI-LSTM**: Physics-informed loss incorporating momentum conservation, concentration conservation, and thickness conservation equations
- **Multi-source Data Fusion**: Buoy trajectories + velocity TIF (EPSG:3413) + monthly thickness/concentration NetCDF + FY3C MWRI HDF5
- **Continual Learning**: Memory Replay + optional Elastic Weight Consolidation (EWC), year-by-year fine-tuning on new data distributions
- **Weak Label Supervision**: OSI SAF low-resolution sea ice drift products (OSI-405-d) as real-physics calibration signals, with per-sample uncertainty weighting and per-dimension masking
- **Uncertainty Estimation**: MC Dropout and Deep Ensemble with 95% confidence intervals, propagated through recursive multi-step prediction
- **OOD Detection**: Mahalanobis distance-based out-of-distribution warning for extreme conditions
- **Physical Constraints**: Prediction clipping to physical bounds (A in [0,1], h >= 0), static depth field protection during recursive rollout

## Architecture

```
Input [u, v, A, h] (12-step lookback)
    |
LSTM (64 units, return_sequences) -> Dropout(0.3)
    |
LSTM (32 units) -> Dropout(0.3)
    |
Dense (feat_dim) -> Physics-Informed Loss
    |
Output: next-step [u, v, A, h] prediction
```

## Data Sources

| Data | Format | Source |
|------|--------|--------|
| Buoy trajectories | XLSX | IABP / MOSAiC |
| Ice velocity u/v | GeoTIFF (EPSG:3413, 25km) | Remote sensing products |
| Ice thickness & concentration | NetCDF (monthly, 25km) | SIT products |
| Weak labels (2021-2026) | NetCDF | [OSI SAF LR Ice Drift (OSI-405-d)](https://osi-saf.eumetsat.int/) |

## Usage

### Basic training with dataForIce directory structure
```bash
python SeaiceFinal3.py --dataforice --base_dir E:/dataForIce \
    --nc_dir E:/dataForIce/tif_data/thicknessM \
    --epochs 200 --predict_steps 7
```

### Full pipeline: weak label calibration + uncertainty
```bash
python SeaiceFinal3.py --dataforice --base_dir E:/dataForIce \
    --nc_dir E:/dataForIce/tif_data/thicknessM \
    --weak_label --drift_nc_dir E:/dataForIce/osisaf_drift \
    --sim_years 2021 2022 2023 2024 2025 2026 \
    --epochs 200 --epochs_per_year 15 --batch_size 32 \
    --uncertainty mc_dropout --mc_samples 30
```

### Key arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--dataforice` | - | Use dataForIce directory loader |
| `--weak_label` | - | Enable OSI SAF weak label fine-tuning |
| `--continual` | - | Enable free-drift simulation + continual learning |
| `--uncertainty` | none | `mc_dropout` or `ensemble` |
| `--ewc_lambda` | 0 | EWC regularization strength (>0 enables EWC) |
| `--epochs` | 50 | Training epochs (early stopping enabled) |
| `--look_back` | 12 | LSTM input sequence length |

## Directory Structure

```
E:/dataForIce/
├── excel_data/           # Buoy XLSX files (multiple formats supported)
│   ├── Buoy01.xlsx       # Format: Time, Day of year, Lat, Long
│   ├── buoy1.xlsx        # Format: Unnamed:0(datetime), time0(doy), lat0, lon0
│   └── Buoy13.xlsx       # Format: time0(doy), lat0, lon0, ice_u, ice_v, ...
├── tif_data/
│   └── v/                # u and v velocity TIF files in same directory
│       ├── 20160101v_uv.tif   # u-component (eastward)
│       └── 20160101v_vv.tif   # v-component (northward)
├── tif_data/thicknessM/  # Monthly SIT NetCDF files
│   └── SIT_25km_monthly_201601.nc
└── osisaf_drift/         # OSI SAF weak label files (flat directory)
    └── ice_drift_nh_polstere-625_multi-oi_*.nc
```

## Citation

If you use this code, please cite the OSI SAF data:

> OSI SAF Global Low Resolution Sea Ice Drift, OSI-405-d,
> doi:10.15770/EUM_SAF_OSI_NRT_2007

## License

MIT
