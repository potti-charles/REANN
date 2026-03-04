#!/usr/bin/env python3
"""
Convert VASP output (OUTCAR/vasprun.xml) to REANN configuration format.

Usage:
    python scripts/vasp2reann.py --input_dir ./dft_data --output_dir ./work --ratio 0.8

Input: Directory containing OUTCAR or vasprun.xml files (can be nested).
Output: train/configuration and val/configuration in REANN format.

REANN format uses fractional coordinates for positions.
Forces are kept in Cartesian (eV/Angstrom) as read from VASP.
"""

import argparse
import os
import random
import numpy as np
from pathlib import Path

try:
    from ase.io import read
    from ase.data import atomic_masses, chemical_symbols
except ImportError:
    raise ImportError("ASE is required: pip install ase")


# Standard atomic masses lookup
ATOMIC_MASS = {s: m for s, m in zip(chemical_symbols, atomic_masses)}


def atoms_to_reann_block(atoms, point_idx, include_forces=True):
    """Convert an ASE Atoms object to a REANN configuration block string."""
    cell = atoms.get_cell()
    pbc = atoms.get_pbc().astype(int)
    positions_frac = atoms.get_scaled_positions()
    symbols = atoms.get_chemical_symbols()
    energy = atoms.get_potential_energy()

    lines = []
    lines.append(f"point=   {point_idx} ")

    # Cell vectors
    for i in range(3):
        lines.append(f"{cell[i][0]}   {cell[i][1]}  {cell[i][2]} ")

    # PBC
    lines.append(f"pbc {pbc[0]}  {pbc[1]}  {pbc[2]} ")

    # Atoms: element mass frac_x frac_y frac_z [fx fy fz]
    if include_forces:
        try:
            forces = atoms.get_forces()  # Cartesian, eV/Angstrom
        except Exception:
            include_forces = False

    for i, sym in enumerate(symbols):
        mass = ATOMIC_MASS.get(sym, 1.0)
        fx, fy, fz = positions_frac[i]
        line = f"{sym}  {mass}  {fx}  {fy}  {fz}"
        if include_forces:
            line += f" {forces[i][0]}  {forces[i][1]}  {forces[i][2]}"
        lines.append(line)

    # Property (energy)
    lines.append(f"abprop: {energy}   ")

    return "\n".join(lines) + "\n"


def find_vasp_files(input_dir):
    """Find all OUTCAR and vasprun.xml files recursively."""
    input_path = Path(input_dir)
    files = []

    # Look for OUTCAR files
    for f in sorted(input_path.rglob("OUTCAR")):
        files.append(str(f))

    # If no OUTCAR found, look for vasprun.xml
    if not files:
        for f in sorted(input_path.rglob("vasprun.xml")):
            files.append(str(f))

    return files


def read_vasp_structures(filepath):
    """Read all structures from a VASP output file."""
    try:
        # Try reading all frames (ionic steps)
        atoms_list = read(filepath, index=":")
        if not isinstance(atoms_list, list):
            atoms_list = [atoms_list]

        # Filter: only keep frames that have energy and forces
        valid = []
        for atoms in atoms_list:
            try:
                atoms.get_potential_energy()
                atoms.get_forces()
                valid.append(atoms)
            except Exception:
                continue
        return valid
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Convert VASP output to REANN format")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing OUTCAR/vasprun.xml files")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Working directory for REANN (default: ./)")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Fraction of data for training (default: 0.8)")
    parser.add_argument("--no_forces", action="store_true",
                        help="Do not include forces in output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames per OUTCAR file (None=all)")
    args = parser.parse_args()

    random.seed(args.seed)
    include_forces = not args.no_forces

    # Find VASP files
    vasp_files = find_vasp_files(args.input_dir)
    if not vasp_files:
        print(f"No OUTCAR or vasprun.xml files found in {args.input_dir}")
        return

    print(f"Found {len(vasp_files)} VASP output files")

    # Read all structures
    all_atoms = []
    for vf in vasp_files:
        print(f"  Reading {vf} ...")
        frames = read_vasp_structures(vf)
        if args.max_frames and len(frames) > args.max_frames:
            frames = frames[:args.max_frames]
        all_atoms.extend(frames)
        print(f"    -> {len(frames)} frames")

    if not all_atoms:
        print("No valid structures found!")
        return

    print(f"\nTotal structures: {len(all_atoms)}")

    # Shuffle and split
    indices = list(range(len(all_atoms)))
    random.shuffle(indices)
    n_train = int(len(indices) * args.ratio)
    train_indices = sorted(indices[:n_train])
    val_indices = sorted(indices[n_train:])

    print(f"Training: {len(train_indices)}, Validation: {len(val_indices)}")

    # Create output directories
    train_dir = os.path.join(args.output_dir, "train")
    val_dir = os.path.join(args.output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Write training data
    train_file = os.path.join(train_dir, "configuration")
    with open(train_file, "w") as f:
        for point_idx, idx in enumerate(train_indices, 1):
            block = atoms_to_reann_block(all_atoms[idx], point_idx, include_forces)
            f.write(block)
    print(f"Written {len(train_indices)} configs to {train_file}")

    # Write validation data
    val_file = os.path.join(val_dir, "configuration")
    with open(val_file, "w") as f:
        for point_idx, idx in enumerate(val_indices, 1):
            block = atoms_to_reann_block(all_atoms[idx], point_idx, include_forces)
            f.write(block)
    print(f"Written {len(val_indices)} configs to {val_file}")

    # Print atom types found
    all_symbols = set()
    for atoms in all_atoms:
        all_symbols.update(atoms.get_chemical_symbols())
    atomtype_str = str(sorted(all_symbols))
    print(f"\nAtom types found: {atomtype_str}")
    print(f"Use in input_density: atomtype={atomtype_str}")


if __name__ == "__main__":
    main()
