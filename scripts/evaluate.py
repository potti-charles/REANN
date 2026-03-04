#!/usr/bin/env python3
"""
Evaluate NNP vs DFT accuracy and select structures for active learning.

Usage:
    # Evaluate DFT trajectories against NNP
    python scripts/evaluate.py --model PES.pt --dft_dir ./new_dft --atomtype C N Cu

    # Custom thresholds (meV/atom)
    python scripts/evaluate.py --model PES.pt --dft_dir ./new_dft --atomtype C N Cu \
        --threshold_low 7 --threshold_high 50

Reads OUTCAR/vasprun.xml from dft_dir, computes NNP predictions for each frame,
and outputs:
  - Per-trajectory RMSE/MAE statistics
  - Structures with RMSE in [threshold_low, threshold_high] for dataset augmentation
  - Summary report
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

from ase.io import read, write


def get_calculator(model_path, atomtype, device="cpu"):
    """Load REANN model as a calculator for single-point evaluation."""
    pes = torch.jit.load(model_path, map_location=device)
    pes.to(torch.device(device)).to(torch.float64)
    pes.eval()
    cutoff = float(pes.cutoff)
    pes = torch.jit.optimize_for_inference(pes)
    return pes, cutoff


def compute_nnp_energy_forces(pes, atoms, atomtype, cutoff, device="cpu"):
    """Compute NNP energy and forces for a single ASE Atoms object."""
    from ase.neighborlist import neighbor_list

    cell = np.array(atoms.cell, dtype=np.float64)
    positions = atoms.get_positions().astype(np.float64)

    # Neighbor list
    idx_i, idx_j, shifts_arr = neighbor_list("ijS", atoms, cutoff)
    shift_vectors = (shifts_arr @ cell).astype(np.float64)

    dev = torch.device(device)
    tcell = torch.from_numpy(cell).to(torch.float64).to(dev)
    cart = torch.from_numpy(positions).to(torch.float64).to(dev)
    cart.requires_grad = True
    neighlist = torch.tensor(np.stack([idx_i, idx_j]), device=dev, dtype=torch.long)
    shifts = torch.from_numpy(shift_vectors).to(torch.float64).to(dev)

    symbols = list(atoms.symbols)
    species = torch.tensor([atomtype.index(s) for s in symbols], device=dev, dtype=torch.long)
    disp_cell = torch.zeros_like(tcell)

    energy = pes(tcell, disp_cell, cart, neighlist, shifts, species)
    forces_grad = torch.autograd.grad(energy, cart)[0]
    forces = -forces_grad.detach().cpu().numpy()

    return float(energy.detach().cpu().numpy()), forces


def evaluate_trajectory(pes, frames, atomtype, cutoff, device="cpu"):
    """Evaluate a trajectory: compute per-frame energy/force errors."""
    results = []
    for atoms in frames:
        try:
            e_dft = atoms.get_potential_energy()
            f_dft = atoms.get_forces()
        except Exception:
            continue

        try:
            e_nnp, f_nnp = compute_nnp_energy_forces(pes, atoms, atomtype, cutoff, device)
        except Exception as e:
            print(f"  Warning: NNP evaluation failed: {e}")
            continue

        n_atoms = len(atoms)
        e_err = (e_nnp - e_dft) / n_atoms  # eV/atom
        f_err_norms = np.linalg.norm(f_nnp - f_dft, axis=1)  # per-atom force error norm

        results.append({
            "atoms": atoms,
            "n_atoms": n_atoms,
            "e_dft": e_dft,
            "e_nnp": e_nnp,
            "e_err_per_atom": e_err,
            "f_rmse": np.sqrt(np.mean(f_err_norms**2)),
            "f_mae": np.mean(f_err_norms),
        })

    return results


def compute_trajectory_stats(results):
    """Compute aggregate RMSE/MAE for a trajectory."""
    if not results:
        return None

    e_errors = np.array([r["e_err_per_atom"] for r in results])
    f_rmses = np.array([r["f_rmse"] for r in results])
    f_maes = np.array([r["f_mae"] for r in results])

    return {
        "n_frames": len(results),
        "e_rmse_meV": np.sqrt(np.mean(e_errors**2)) * 1000,  # meV/atom
        "e_mae_meV": np.mean(np.abs(e_errors)) * 1000,       # meV/atom
        "f_rmse_meV": np.mean(f_rmses) * 1000,               # meV/Angstrom
        "f_mae_meV": np.mean(f_maes) * 1000,                 # meV/Angstrom
    }


def find_dft_files(dft_dir):
    """Find OUTCAR/vasprun.xml files."""
    dft_path = Path(dft_dir)
    files = sorted(dft_path.rglob("OUTCAR"))
    if not files:
        files = sorted(dft_path.rglob("vasprun.xml"))
    return [str(f) for f in files]


def main():
    parser = argparse.ArgumentParser(description="Evaluate NNP vs DFT accuracy")
    parser.add_argument("--model", type=str, required=True, help="Path to PES.pt")
    parser.add_argument("--dft_dir", type=str, required=True,
                        help="Directory containing OUTCAR/vasprun.xml files")
    parser.add_argument("--atomtype", nargs="+", required=True,
                        help="Atom types matching input_density")
    parser.add_argument("--threshold_low", type=float, default=7.0,
                        help="Lower RMSE threshold in meV/atom (default: 7)")
    parser.add_argument("--threshold_high", type=float, default=50.0,
                        help="Upper RMSE threshold in meV/atom (default: 50)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="./al_selected",
                        help="Directory to save selected structures")
    parser.add_argument("--report", type=str, default="evaluation_report.txt",
                        help="Output report file")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    pes, cutoff = get_calculator(args.model, args.atomtype, args.device)
    print(f"  Cutoff: {cutoff} Angstrom")

    # Find DFT files
    dft_files = find_dft_files(args.dft_dir)
    if not dft_files:
        print(f"No OUTCAR/vasprun.xml found in {args.dft_dir}")
        return
    print(f"Found {len(dft_files)} DFT output files")

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = []
    traj_stats = []
    selected_structures = []

    for fi, dft_file in enumerate(dft_files):
        print(f"\n[{fi+1}/{len(dft_files)}] {dft_file}")
        try:
            frames = read(dft_file, index=":")
            if not isinstance(frames, list):
                frames = [frames]
        except Exception as e:
            print(f"  Could not read: {e}")
            continue

        print(f"  {len(frames)} frames")

        results = evaluate_trajectory(pes, frames, args.atomtype, cutoff, args.device)
        if not results:
            print("  No valid frames")
            continue

        stats = compute_trajectory_stats(results)
        stats["file"] = dft_file
        traj_stats.append(stats)

        print(f"  Energy RMSE: {stats['e_rmse_meV']:.2f} meV/atom, "
              f"MAE: {stats['e_mae_meV']:.2f} meV/atom")
        print(f"  Force  RMSE: {stats['f_rmse_meV']:.2f} meV/A, "
              f"MAE: {stats['f_mae_meV']:.2f} meV/A")

        # Select structures for active learning
        for r in results:
            e_rmse_meV = abs(r["e_err_per_atom"]) * 1000
            if args.threshold_low <= e_rmse_meV <= args.threshold_high:
                selected_structures.append(r["atoms"])

        all_results.extend(results)

    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)

    if all_results:
        overall = compute_trajectory_stats(all_results)
        print(f"Total frames evaluated: {overall['n_frames']}")
        print(f"Energy RMSE: {overall['e_rmse_meV']:.2f} meV/atom")
        print(f"Energy MAE:  {overall['e_mae_meV']:.2f} meV/atom")
        print(f"Force  RMSE: {overall['f_rmse_meV']:.2f} meV/A")
        print(f"Force  MAE:  {overall['f_mae_meV']:.2f} meV/A")

    # Convergence check
    converged_trajs = sum(1 for s in traj_stats if s["e_rmse_meV"] < args.threshold_low)
    total_trajs = len(traj_stats)
    print(f"\nConverged trajectories (RMSE < {args.threshold_low} meV/atom): "
          f"{converged_trajs}/{total_trajs}")

    if converged_trajs == total_trajs and total_trajs > 0:
        print("*** NNP CONVERGED — all trajectories below threshold ***")
    else:
        print(f"Selected {len(selected_structures)} structures for dataset augmentation "
              f"({args.threshold_low} < RMSE < {args.threshold_high} meV/atom)")

    # Save selected structures
    if selected_structures:
        sel_file = os.path.join(args.output_dir, "selected_structures.extxyz")
        write(sel_file, selected_structures)
        print(f"Saved selected structures to {sel_file}")

    # Write report
    report_path = os.path.join(args.output_dir, args.report)
    with open(report_path, "w") as f:
        f.write("REANN Active Learning Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        if all_results:
            f.write(f"Total frames: {overall['n_frames']}\n")
            f.write(f"Energy RMSE: {overall['e_rmse_meV']:.2f} meV/atom\n")
            f.write(f"Energy MAE:  {overall['e_mae_meV']:.2f} meV/atom\n")
            f.write(f"Force  RMSE: {overall['f_rmse_meV']:.2f} meV/A\n")
            f.write(f"Force  MAE:  {overall['f_mae_meV']:.2f} meV/A\n\n")

        f.write(f"Convergence threshold: {args.threshold_low} meV/atom\n")
        f.write(f"Converged: {converged_trajs}/{total_trajs}\n")
        f.write(f"Selected structures: {len(selected_structures)}\n\n")

        f.write("Per-trajectory results:\n")
        f.write("-" * 60 + "\n")
        for s in traj_stats:
            status = "OK" if s["e_rmse_meV"] < args.threshold_low else "REFINE"
            f.write(f"[{status}] {s['file']}\n")
            f.write(f"  {s['n_frames']} frames, E_RMSE={s['e_rmse_meV']:.2f}, "
                    f"F_RMSE={s['f_rmse_meV']:.2f} meV/A\n")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
