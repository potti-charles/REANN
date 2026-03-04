#!/usr/bin/env python3
"""
Structure optimization using trained REANN model via ASE.

Usage:
    # Single structure optimization
    python scripts/ase_optimize.py --model PES.pt --input POSCAR --atomtype C N Cu --fmax 0.05

    # Batch optimization of multiple structures
    python scripts/ase_optimize.py --model PES.pt --input_dir ./structures --atomtype C N Cu --fmax 0.05

    # MC-like random perturbation sampling
    python scripts/ase_optimize.py --model PES.pt --input POSCAR --atomtype C N Cu --mode sample \
        --n_samples 100 --perturb 0.2 --temperature 1000

The Fortran getneigh module must be compiled first (see reann/ASE/fortran/run).
If unavailable, a pure Python neighbor list from ASE is used as fallback.
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS, FIRE
from ase.units import kB


def get_calculator(model_path, atomtype, device="cpu", dtype=torch.float32, maxneigh=50000):
    """Create a REANN ASE calculator."""
    reann_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ase_calc_dir = os.path.join(reann_root, "reann", "ASE", "calculators")
    fortran_dir = os.path.join(reann_root, "reann", "ASE", "fortran")

    # Add paths for REANN calculator and getneigh
    for p in [ase_calc_dir, fortran_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Try to import Fortran getneigh; fall back to ASE neighborlist
    try:
        import getneigh
        has_fortran = True
    except ImportError:
        has_fortran = False
        print("Warning: Fortran getneigh not found. Using ASE built-in neighbor list.")
        print("For better performance, compile it: cd reann/ASE/fortran && bash run")

    from reann import REANN

    if has_fortran:
        calc = REANN(atomtype, maxneigh, getneigh,
                     properties=["energy", "forces"],
                     nn=model_path, device=device, dtype=dtype)
    else:
        # Fallback: wrap with a simple Python neighbor list adapter
        calc = REANN_Fallback(atomtype, model_path, device, dtype)

    return calc


class REANN_Fallback:
    """Fallback calculator using ASE's built-in neighbor list when Fortran getneigh is unavailable."""

    def __init__(self, atomtype, model_path, device="cpu", dtype=torch.float32):
        from ase.calculators.calculator import Calculator, all_changes
        self._atomtype = atomtype
        self._device = torch.device(device)
        self._dtype = dtype

        pes = torch.jit.load(model_path)
        pes.to(self._device).to(self._dtype)
        pes.eval()
        self._cutoff = float(pes.cutoff)
        self._pes = torch.jit.optimize_for_inference(pes)
        self.results = {}
        self.atoms = None

    def calculate(self, atoms):
        from ase.neighborlist import neighbor_list
        self.atoms = atoms
        cell = np.array(atoms.cell)
        positions = atoms.get_positions()

        # Build neighbor list
        idx_i, idx_j, shifts_arr = neighbor_list("ijS", atoms, self._cutoff)
        shift_vectors = shifts_arr @ cell

        tcell = torch.from_numpy(cell).to(self._dtype).to(self._device)
        cart = torch.from_numpy(positions).to(self._dtype).to(self._device)
        cart.requires_grad = True
        neighlist = torch.tensor(np.stack([idx_i, idx_j]), device=self._device, dtype=torch.long)
        shifts = torch.from_numpy(shift_vectors).to(self._dtype).to(self._device)
        symbols = list(atoms.symbols)
        species = torch.tensor([self._atomtype.index(s) for s in symbols],
                               device=self._device, dtype=torch.long)
        disp_cell = torch.zeros_like(tcell)

        energy = self._pes(tcell, disp_cell, cart, neighlist, shifts, species)
        forces = torch.autograd.grad(energy, cart)[0]
        forces = torch.neg(forces).detach().cpu().numpy()

        self.results["energy"] = float(energy.detach().cpu().numpy())
        self.results["forces"] = forces


def optimize_structure(atoms, calc, fmax=0.05, optimizer="BFGS", steps=500, traj_file=None):
    """Run structure optimization."""
    atoms.calc = calc
    if hasattr(calc, "reset"):
        calc.reset()

    if optimizer == "FIRE":
        opt = FIRE(atoms, trajectory=traj_file)
    else:
        opt = BFGS(atoms, trajectory=traj_file)

    opt.run(fmax=fmax, steps=steps)
    return atoms


def perturb_structure(atoms, amplitude=0.2):
    """Apply random perturbation to atomic positions."""
    new_atoms = atoms.copy()
    positions = new_atoms.get_positions()
    positions += np.random.randn(*positions.shape) * amplitude
    new_atoms.set_positions(positions)
    return new_atoms


def mc_sampling(atoms, calc, n_samples, temperature, perturb_amp, fmax, output_dir):
    """Simple Monte Carlo sampling with NNP optimization."""
    os.makedirs(output_dir, exist_ok=True)

    # Optimize reference structure
    ref_atoms = optimize_structure(atoms.copy(), calc, fmax=fmax)
    ref_energy = ref_atoms.get_potential_energy()
    accepted = [ref_atoms.copy()]
    print(f"Reference energy: {ref_energy:.4f} eV")

    beta = 1.0 / (kB * temperature)
    n_accepted = 0

    for i in range(n_samples):
        trial = perturb_structure(ref_atoms, perturb_amp)
        trial = optimize_structure(trial, calc, fmax=fmax)
        trial_energy = trial.get_potential_energy()

        dE = trial_energy - ref_energy
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            accepted.append(trial.copy())
            ref_atoms = trial
            ref_energy = trial_energy
            n_accepted += 1
            tag = "accepted"
        else:
            tag = "rejected"

        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/{n_samples}: E={trial_energy:.4f} eV, dE={dE:.4f}, {tag}, "
                  f"acceptance={n_accepted/(i+1):.2%}")

    # Save all accepted structures
    output_file = os.path.join(output_dir, "mc_samples.extxyz")
    write(output_file, accepted)
    print(f"\nSaved {len(accepted)} accepted structures to {output_file}")
    return accepted


def main():
    parser = argparse.ArgumentParser(description="Structure optimization with REANN")
    parser.add_argument("--model", type=str, required=True, help="Path to PES.pt")
    parser.add_argument("--input", type=str, default=None,
                        help="Input structure file (POSCAR, .xyz, .extxyz, etc.)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory of structure files for batch optimization")
    parser.add_argument("--atomtype", nargs="+", required=True,
                        help="Atom types matching input_density, e.g. C N Cu")
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="Force convergence criterion in eV/A (default: 0.05)")
    parser.add_argument("--optimizer", choices=["BFGS", "FIRE"], default="BFGS")
    parser.add_argument("--steps", type=int, default=500, help="Max optimization steps")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="optimized.extxyz",
                        help="Output file for optimized structures")
    parser.add_argument("--save_traj", action="store_true",
                        help="Save optimization trajectory")
    # MC sampling options
    parser.add_argument("--mode", choices=["optimize", "sample"], default="optimize",
                        help="optimize=structure relaxation, sample=MC sampling")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of MC sampling steps (mode=sample)")
    parser.add_argument("--perturb", type=float, default=0.2,
                        help="Perturbation amplitude in Angstrom (mode=sample)")
    parser.add_argument("--temperature", type=float, default=1000,
                        help="MC temperature in K (mode=sample)")
    parser.add_argument("--maxneigh", type=int, default=50000,
                        help="Max neighbor pairs (default: 50000)")
    args = parser.parse_args()

    calc = get_calculator(args.model, args.atomtype, args.device, maxneigh=args.maxneigh)

    if args.mode == "sample":
        if not args.input:
            print("Error: --input is required for sample mode")
            return
        atoms = read(args.input)
        mc_sampling(atoms, calc, args.n_samples, args.temperature, args.perturb,
                    args.fmax, os.path.dirname(args.output) or ".")
        return

    # Optimization mode
    structures = []
    if args.input:
        structures = read(args.input, index=":")
        if not isinstance(structures, list):
            structures = [structures]
    elif args.input_dir:
        input_path = Path(args.input_dir)
        for ext in ["*.vasp", "POSCAR*", "*.xyz", "*.extxyz", "*.cif"]:
            for f in sorted(input_path.glob(ext)):
                try:
                    s = read(str(f), index=":")
                    if not isinstance(s, list):
                        s = [s]
                    structures.extend(s)
                except Exception as e:
                    print(f"Warning: could not read {f}: {e}")
    else:
        print("Error: provide --input or --input_dir")
        return

    print(f"Optimizing {len(structures)} structures...")
    optimized = []
    for i, atoms in enumerate(structures):
        traj_file = f"opt_{i}.traj" if args.save_traj else None
        print(f"\n--- Structure {i+1}/{len(structures)} ({len(atoms)} atoms) ---")
        try:
            opt_atoms = optimize_structure(atoms.copy(), calc, args.fmax,
                                           args.optimizer, args.steps, traj_file)
            e = opt_atoms.get_potential_energy()
            fmax_val = np.max(np.linalg.norm(opt_atoms.get_forces(), axis=1))
            print(f"  E = {e:.4f} eV, max|F| = {fmax_val:.4f} eV/A")
            optimized.append(opt_atoms)
        except Exception as e:
            print(f"  Failed: {e}")

    if optimized:
        write(args.output, optimized)
        print(f"\nSaved {len(optimized)} optimized structures to {args.output}")


if __name__ == "__main__":
    main()
