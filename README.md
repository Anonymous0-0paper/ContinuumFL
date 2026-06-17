# ContinuumFL: Federated Learning with Spatial-Aware Aggregation in non-IID Edge Zones

![ContinuumFL Logo](https://img.shields.io/badge/ContinuumFL-v1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

The implementation of the **ContinuumFL** framework for spatial-aware federated learning in heterogeneous edge environments. This framework introduces novel spatial-aware aggregation techniques that exploit geographical relationships and computational heterogeneity in edge zones.

## Overview

ContinuumFL addresses the fundamental challenges of deploying federated learning across heterogeneous edge computing environments where devices exhibit spatial correlations in data distributions and diverse computational capabilities. Unlike traditional FL approaches that treat edge devices uniformly, ContinuumFL incorporates spatial awareness into the aggregation process.

### Key Features

- **Spatial-Aware Zone Discovery**: Dynamic clustering based on spatial proximity, data similarity, and network characteristics
- **Hierarchical Aggregation**: Two-tier aggregation with intra-zone and inter-zone spatial-aware weighting
- **Communication Optimization**: Gradient compression, delta encoding, and opportunistic caching
- **Fairness-Aware Weighting**: Adaptive weight calculation with fairness constraints
- **Comprehensive Evaluation**: Built-in comparison with baseline FL methods
- **Rich Visualizations**: Detailed analysis and plotting capabilities

## Architecture

```
ContinuumFL/
├── config.py                 # Configuration management
├── main.py                   # Main execution script
├── requirements.txt          # Dependencies
├── scripts/
│   ├── continuumfl_common.sh         # Shared helpers and presets
│   ├── run_continuumfl.sh            # Run ContinuumFL (local)
│   ├── run_continuumfl_slurm.sh      # Run ContinuumFL (SLURM/VSC-5)
│   ├── run_continuumfl_slurm_quick.sh# Quick SLURM job
│   ├── run_baselines_ucihar.sh       # Baselines on UCI-HAR (local/SLURM)
│   └── run_baselines_sweep.sh        # Sweep baselines across fault configs
├── src/
│   ├── core/                 # Core framework components
│   │   ├── device.py         # Edge device implementation
│   │   ├── zone.py           # Spatial zone management
│   │   └── zone_discovery.py # Dynamic zone discovery
│   ├── aggregation/          # Hierarchical aggregation
│   │   └── hierarchical_aggregator.py
│   ├── data/                 # Data loading and distribution
│   │   └── federated_dataset.py
│   ├── models/               # Model definitions
│   │   └── model_factory.py
│   ├── communication/        # Communication optimization
│   │   └── compression.py
│   ├── baselines/            # Baseline FL methods
│   │   └── baseline_fl.py
│   ├── visualization/        # Plotting and analysis
│   │   └── visualizer.py
│   └── continuum_fl_coordinator.py  # Main coordinator
├── data/                     # Dataset storage (auto-downloaded on first run)
├── logs/                     # Training logs
├── checkpoints/              # Model checkpoints
└── results/                  # Experiment results
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/continuumfl.git
cd continuumfl
```

2. Install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Datasets are downloaded automatically on first run — no manual setup required.

---

## Running ContinuumFL (Our Model)

### Option 1 — Convenience script with presets

```bash
bash scripts/run_continuumfl.sh [PRESET] [extra main.py args...]
```

Available presets (defined in `scripts/continuumfl_common.sh`):

| Preset | Dataset | Devices | Zones | Rounds | Notes |
|--------|---------|---------|-------|--------|-------|
| `quick` | femnist | 10 | 2 | 10 | Fast smoke-test |
| `standard` | femnist | 100 | 20 | 100 | Default |
| `large` | femnist | 500 | 50 | 150 | Large-scale |
| `baseline` | femnist | 100 | 20 | 100 | ContinuumFL + baselines |
| `comm` | femnist | 100 | 20 | 100 | Compression enabled |
| `femnist` | femnist | 100 | 20 | 100 | FEMNIST + baselines |
| `shakespeare` | shakespeare | 35 | 7 | 100 | Text prediction |

```bash
# Quick smoke-test
bash scripts/run_continuumfl.sh quick

# Standard run on FEMNIST
bash scripts/run_continuumfl.sh standard

# Override any parameter after the preset
bash scripts/run_continuumfl.sh standard --num_rounds 50 --dataset ucihar

# Dry-run (print command, don't execute)
DRY_RUN=true bash scripts/run_continuumfl.sh standard
```

### Option 2 — Direct `main.py` invocation

```bash
python main.py \
  --dataset ucihar \
  --num_devices 50 \
  --num_zones 5 \
  --num_rounds 200 \
  --local_epochs 5 \
  --learning_rate 0.001 \
  --batch_size 64 \
  --device cuda \
  --save_results \
  --create_visualizations
```

### Option 3 — SLURM (VSC-5)

```bash
sbatch scripts/run_continuumfl_slurm.sh
# or for a quick job:
sbatch scripts/run_continuumfl_slurm_quick.sh
```

---

## Running Baselines Only

Use `--baselines_only` to skip ContinuumFL training and run only the comparison methods.

### Via `main.py` directly

```bash
python main.py \
  --dataset ucihar \
  --num_devices 50 \
  --num_zones 5 \
  --num_rounds 200 \
  --local_epochs 5 \
  --learning_rate 0.001 \
  --batch_size 64 \
  --baselines_only \
  --run_baselines \
  --baseline_methods FedAvg FedProx HierFL ClusterFL IFCA APCfl GeoFL SnapCFL \
  --ifca_k 5 \
  --device cuda \
  --save_results
```

### Via the UCI-HAR baseline script

The script runs all baseline methods across 8 fault scenarios automatically:

```bash
# Local run
bash scripts/run_baselines_ucihar.sh

# SLURM submission
sbatch scripts/run_baselines_ucihar.sh

# Dry-run (print commands, skip execution)
DRY_RUN=true bash scripts/run_baselines_ucihar.sh
```

Edit the variables at the top of the script to change dataset, methods, or fault scenarios.

### Via the sweep script

```bash
bash scripts/run_baselines_sweep.sh
```

### Available baseline methods

| Method | Description |
|--------|-------------|
| `FedAvg` | Federated Averaging |
| `FedProx` | FedAvg + proximal term |
| `HierFL` | Hierarchical FL |
| `ClusterFL` | Clustered FL |
| `IFCA` | Iterative Federated Clustering Algorithm |
| `APCfl` | Adaptive Personalized Clustered FL |
| `GeoFL` | Geography-aware FL |
| `SnapCFL` | Snapshot Clustered FL |

---

## Running ContinuumFL + Baselines Together

To run ContinuumFL and compare it against baselines in a single experiment, use `--run_baselines` (without `--baselines_only`):

```bash
python main.py \
  --dataset ucihar \
  --num_devices 50 \
  --num_zones 5 \
  --num_rounds 200 \
  --run_baselines \
  --baseline_methods FedAvg FedProx HierFL ClusterFL \
  --create_visualizations \
  --save_results \
  --device cuda
```

Or use the `baseline` preset:

```bash
bash scripts/run_continuumfl.sh baseline
```

---

## Fault Tolerance Experiments

Add device and zone failure simulation:

```bash
python main.py \
  --dataset ucihar \
  --num_devices 50 \
  --num_zones 5 \
  --num_rounds 200 \
  --enable_failure \
  --device_failure_probability 0.10 \
  --zone_failure_probability 0.05 \
  --device cuda \
  --save_results
```

The `run_baselines_ucihar.sh` script already iterates over these fault configurations:

| Scenario | Device Fail | Zone Fail |
|----------|-------------|-----------|
| fault_free | 0% | 0% |
| dev_low | 5% | 0% |
| dev_moderate | 10% | 0% |
| dev_high | 20% | 0% |
| dev_low__zone_low | 5% | 2% |
| dev_moderate__zone_moderate | 10% | 5% |
| dev_high__zone_high | 20% | 10% |
| severe | 30% | 15% |

---

## Key Algorithms

### 1. Dynamic Zone Discovery

ContinuumFL implements a multi-dimensional similarity metric for device clustering:

```
Sim(d_i, d_j) = ω₁·S_spatial + ω₂·S_data + ω₃·S_network
```

Where:
- `S_spatial`: Spatial proximity based on geographical distance
- `S_data`: Data similarity using gradient similarity
- `S_network`: Network similarity considering bandwidth and latency

### 2. Hierarchical Aggregation

**Intra-zone Aggregation:**
```
w_k^(t+1) = Σ α_i^k w_i^(t)
```

**Inter-zone Aggregation:**
```
w^(t+1) = Σ β_k^(t) w_k^(t+1) + λ Σ Σ ρ(z_k, z_j)(w_k - w_j)
```

### 3. Communication Optimization

- **Top-k Sparsification**: Transmit only top-k% gradient components (controlled by `--compression_rate`)
- **Delta Encoding**: Send only model differences
- **Opportunistic Caching**: Cache stable model layers

---

## Supported Datasets

| Dataset | Task | Auto-downloaded |
|---------|------|----------------|
| `ucihar` | Activity recognition (sensor) | Yes |
| `femnist` | Handwritten character recognition | Yes (HuggingFace) |
| `cifar100` | Image classification (100 classes) | Yes (torchvision) |
| `shakespeare` | Next-character prediction | Yes (HuggingFace) |
| `speechcommands` | Keyword spotting | Yes |

---

## Configuration Reference

### Core parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `shakespeare` | Dataset: `ucihar`, `femnist`, `cifar100`, `shakespeare`, `speechcommands` |
| `--num_devices` | 100 | Number of edge devices |
| `--num_zones` | 20 | Number of spatial zones |
| `--num_rounds` | 200 | Training rounds |
| `--local_epochs` | 5 | Local epochs per round |
| `--learning_rate` | 0.001 | Learning rate |
| `--batch_size` | 16 | Batch size |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--random_seed` | 42 | Reproducibility seed |
| `--max_samples` | 50000 | Dataset size limit (`-1` = no limit) |

### Spatial / aggregation parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--spatial_weight` | 0.4 | Weight for spatial similarity |
| `--data_weight` | 0.4 | Weight for data similarity |
| `--network_weight` | 0.2 | Weight for network similarity |
| `--spatial_regularization` | 0.05 | Spatial regularization λ |
| `--correlation_threshold` | 0.05 | Zone discovery threshold |
| `--intra_zone_alpha` | 100 | Dirichlet α for intra-zone heterogeneity |
| `--inter_zone_alpha` | 5 | Dirichlet α for inter-zone heterogeneity |
| `--compression_rate` | 0.6 | Fraction of gradients to transmit (top-k) |
| `--enable_compression` | off | Enable gradient compression |

### Baseline flags

| Argument | Description |
|----------|-------------|
| `--run_baselines` | Run baselines alongside ContinuumFL |
| `--baselines_only` | Skip ContinuumFL, run only baselines |
| `--baseline_methods A B C` | Methods to run (space-separated list) |
| `--ifca_k N` | Number of clusters for IFCA (should equal `--num_zones`) |

### Output flags

| Argument | Description |
|----------|-------------|
| `--save_results` | Save metrics to `results/` |
| `--create_visualizations` | Generate plots after training |
| `--log_dir PATH` | Override log directory |
| `--results_dir PATH` | Override results directory |
| `--checkpoint_dir PATH` | Override checkpoint directory |
| `--config_file PATH` | Load parameters from a JSON file |

---

## Visualizations

Generate plots after a run:

```bash
python main.py --create_visualizations
```

Or pass it during training to auto-generate at the end:

```bash
python main.py --dataset femnist --num_rounds 100 --create_visualizations --save_results
```

Plots include training curves, per-zone performance, spatial device layout, communication costs, and convergence comparison with baselines.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Reduce `--batch_size` or `--num_devices` |
| Dataset download fails | Check internet and disk space; `requests` package required for UCI-HAR / Speech Commands |
| Visualization errors | Set `MPLBACKEND=Agg` for headless environments |
| Import errors | Run `pip install -r requirements.txt` |
| `module: command not found` | Only needed on HPC (VSC-5); not required locally |

---

## Development

### Adding New Datasets

1. Extend `FederatedDataset` in [src/data/federated_dataset.py](src/data/federated_dataset.py)
2. Add model definition in [src/models/model_factory.py](src/models/model_factory.py)
3. Update argument choices in [main.py](main.py)

### Adding New Baselines

1. Implement method in [src/baselines/baseline_fl.py](src/baselines/baseline_fl.py)
2. Add its name to `--baseline_methods` choices in [main.py](main.py)
3. Update comparison visualization if needed

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- University of Innsbruck Institute of Computer Science
- PyTorch and scikit-learn communities
- LEAF benchmark framework for federated learning datasets

---

**ContinuumFL** — Bringing spatial awareness to federated learning in edge environments.
