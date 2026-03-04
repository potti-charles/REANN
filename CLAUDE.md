# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

REANN (Recursively Embedded Atom Neural Network) is a PyTorch-based deep learning package for training interatomic potentials, dipole moments, transition dipole moments, and polarizabilities for molecular, reactive, and periodic systems. It supports distributed training via PyTorch DDP on both GPU and CPU.

## Dependencies

- PyTorch 2.0.0 / LibTorch 2.0.0
- opt_einsum 3.2.0
- CMake 3.1.0 (for LAMMPS interface builds)

## Build & Run Commands

### Training
```bash
# Single process
python -m reann

# Distributed training (multi-GPU)
torchrun --nproc_per_node=<N> -m reann
```

Training expects `para/input_nn` and `para/input_density` config files plus `train/` and `val/` data directories in the working directory.

### Building LAMMPS Interface
```bash
cd reann/lammps-REANN-interface/cmake
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=<path-to-libtorch>
make
```

### Building ASE Fortran Neighbor List
Compile with f2py using the `run` script in `reann/ASE/fortran/`.

## Architecture

### Entry Point
`reann/__main__.py` → `reann/run/train.py` — the training script executes top-level module code (not wrapped in functions/classes), importing globals from `src.read`.

### Core Modules (`reann/src/`)

- **read.py** — Loads all configuration from `para/input_nn`, `para/input_density`, and data from `train/`/`val/` directories. Exports globals (hyperparameters, data tensors) used by `train.py`.
- **MODEL.py** — `NNMod` (element-specific neural network with `ResBlock` residual blocks) and `ResBlock` classes.
- **density.py** — `GetDensity` computes embedded atomic density using Gaussian radial basis functions and angular components via opt_einsum.
- **Property_*.py** — Property calculators selected by `start_table` parameter:
  - `0` → `Property_energy` (energy only)
  - `1` → `Property_force` (energy + forces via autograd)
  - `2` → `Property_DM` (dipole moment)
  - `3` → `Property_TDM` (transition dipole moment)
  - `4` → `Property_POL` (polarizability, uses 3 NNMod instances)
- **optimize.py** — `Optimize()` function: main training loop with AdamW, ReduceLROnPlateau scheduler, EMA, and checkpointing.
- **dataloader.py** — `DataLoader` (batching/shuffling) and `CudaDataLoader` (async GPU prefetching).
- **Loss.py** — Weighted MSE loss computation.
- **EMA.py** — Exponential moving average for model parameters.
- **activate.py** — Custom activation functions: `Relu_like`, `Tanh_like`.

### External Interfaces
- **reann/pes/**, **reann/dm/**, **reann/tdm/**, **reann/pol/** — Inference modules with `script_PES.py` for exporting TorchScript models.
- **reann/lammps-interface/**, **reann/lammps-REANN-interface/** — C++ pair_style implementations linking LibTorch for LAMMPS MD.
- **reann/ASE/** — ASE calculator interface with Fortran neighbor list.

## Key Design Patterns

- **Global state via `src.read`**: `train.py` does `from src.read import *`, making all config values and loaded data available as module-level globals. This is intentional — the entire training pipeline is orchestrated as top-level script code.
- **Property dispatch**: The `start_table` integer selects which `Property_*.py` module to import, determining what physical quantity is trained.
- **Model export**: After training, models are saved as TorchScript (`.pt`) for use in LAMMPS or ASE inference.

## Configuration

Two input files in `para/` directory:
- **input_nn**: NN hyperparameters (batch sizes, learning rates, network layers `nl`, residual blocks `nblock`, dropout, activation, `oc_loop`/`oc_nl`/`oc_nblock` for orbital coefficients)
- **input_density**: Density parameters (`cutoff`, `nipsin` for max angular momentum, `nwave` for radial Gaussians — must be power of 2)

## Data Format

Training/validation data stored in `train/configuration` and `val/configuration` files. Each configuration block contains: lattice vectors, PBC flags, atomic coordinates with optional forces, and target properties prefixed with `abprop:`.
