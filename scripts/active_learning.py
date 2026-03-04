#!/usr/bin/env python3
"""
Active learning loop for REANN NNP training.

Workflow (following the paper methodology):
  1. Convert initial DFT data → REANN format
  2. Train REANN model
  3. Export model to TorchScript (PES.pt)
  4. Use NNP (via ASE) to explore configuration space (MC sampling / optimization)
  5. Re-optimize NNP-sampled structures with DFT (VASP)
  6. Evaluate NNP vs DFT on new trajectories
  7. Select structures with 7 < RMSE < 50 meV/atom → augment dataset
  8. Repeat from step 2 until converged (all trajectories RMSE < 7 meV/atom)

Usage:
    python scripts/active_learning.py --config al_config.yaml

Or step-by-step (manual mode):
    python scripts/active_learning.py --step train --workdir ./work
    python scripts/active_learning.py --step export --workdir ./work
    python scripts/active_learning.py --step sample --workdir ./work --model PES.pt --input POSCAR
    python scripts/active_learning.py --step evaluate --workdir ./work --model PES.pt --dft_dir ./new_dft
    python scripts/active_learning.py --step augment --workdir ./work --new_data ./al_selected/selected_structures.extxyz
"""

import argparse
import os
import sys
import subprocess
import shutil
import yaml
from pathlib import Path
from datetime import datetime


SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd, cwd=None, desc=""):
    """Run a shell command and print output."""
    print(f"\n{'='*60}")
    print(f"[{desc}] {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        return False
    return True


def step_convert(config):
    """Step 1: Convert VASP data to REANN format."""
    cmd = (f"python {SCRIPTS_DIR}/vasp2reann.py "
           f"--input_dir {config['dft_data_dir']} "
           f"--output_dir {config['workdir']} "
           f"--ratio {config.get('train_ratio', 0.8)}")
    return run_command(cmd, desc="VASP → REANN conversion")


def step_setup(config):
    """Step 1b: Generate REANN config files."""
    atomtype = " ".join(config["atomtype"])
    nl = " ".join(str(x) for x in config.get("nl", [128, 128]))
    cmd = (f"python {SCRIPTS_DIR}/setup_reann.py "
           f"--workdir {config['workdir']} "
           f"--atomtype {atomtype} "
           f"--cutoff {config.get('cutoff', 4.5)} "
           f"--nl {nl} "
           f"--nblock {config.get('nblock', 2)} "
           f"--nwave {config.get('nwave', 8)} "
           f"--oc_loop {config.get('oc_loop', 1)} "
           f"--epoch {config.get('epoch', 20000)} "
           f"--batchsize_train {config.get('batchsize_train', 64)}")
    return run_command(cmd, desc="Generate REANN config")


def step_train(config):
    """Step 2: Train REANN model."""
    nproc = config.get("nproc", 1)
    workdir = config["workdir"]

    if nproc > 1:
        cmd = f"torchrun --nproc_per_node={nproc} -m reann"
    else:
        cmd = "python -m reann"

    return run_command(cmd, cwd=workdir, desc="Training REANN")


def step_export(config):
    """Step 3: Export to TorchScript."""
    cmd = (f"python {SCRIPTS_DIR}/export_model.py "
           f"--workdir {config['workdir']}")
    return run_command(cmd, desc="Export TorchScript")


def step_sample(config):
    """Step 4: NNP sampling via ASE."""
    atomtype = " ".join(config["atomtype"])
    model = os.path.join(config["workdir"], "PES.pt")
    input_struct = config.get("sample_input", "")

    if not input_struct:
        print("Warning: no sample_input specified, skipping sampling step")
        return True

    sample_dir = os.path.join(config["workdir"], "sampled")
    cmd = (f"python {SCRIPTS_DIR}/ase_optimize.py "
           f"--model {model} "
           f"--input {input_struct} "
           f"--atomtype {atomtype} "
           f"--mode sample "
           f"--n_samples {config.get('n_samples', 100)} "
           f"--perturb {config.get('perturb_amp', 0.2)} "
           f"--temperature {config.get('mc_temperature', 1000)} "
           f"--fmax {config.get('fmax', 0.05)} "
           f"--output {sample_dir}/samples.extxyz")
    return run_command(cmd, desc="NNP MC sampling")


def step_evaluate(config, cycle):
    """Step 6: Evaluate NNP vs DFT."""
    atomtype = " ".join(config["atomtype"])
    model = os.path.join(config["workdir"], "PES.pt")
    dft_dir = config.get("new_dft_dir", "")

    if not dft_dir:
        print("Warning: no new_dft_dir specified, skipping evaluation")
        return True, False

    eval_dir = os.path.join(config["workdir"], f"eval_cycle_{cycle}")
    cmd = (f"python {SCRIPTS_DIR}/evaluate.py "
           f"--model {model} "
           f"--dft_dir {dft_dir} "
           f"--atomtype {atomtype} "
           f"--threshold_low {config.get('threshold_low', 7)} "
           f"--threshold_high {config.get('threshold_high', 50)} "
           f"--output_dir {eval_dir}")
    success = run_command(cmd, desc=f"Evaluate cycle {cycle}")

    # Check if converged by reading report
    converged = False
    report_file = os.path.join(eval_dir, "evaluation_report.txt")
    if os.path.exists(report_file):
        with open(report_file) as f:
            content = f.read()
            if "NNP CONVERGED" in content:
                converged = True

    return success, converged


def step_augment(config, cycle):
    """Step 7: Augment dataset with selected structures."""
    eval_dir = os.path.join(config["workdir"], f"eval_cycle_{cycle}")
    selected_file = os.path.join(eval_dir, "selected_structures.extxyz")

    if not os.path.exists(selected_file):
        print(f"No selected structures found at {selected_file}")
        return True

    # Convert selected structures to REANN format and append to training data
    from ase.io import read as ase_read

    # Import conversion function
    sys.path.insert(0, SCRIPTS_DIR)
    from vasp2reann import atoms_to_reann_block

    structures = ase_read(selected_file, index=":")
    if not isinstance(structures, list):
        structures = [structures]

    train_config = os.path.join(config["workdir"], "train", "configuration")

    # Count existing points
    existing_points = 0
    if os.path.exists(train_config):
        with open(train_config) as f:
            for line in f:
                if line.startswith("point="):
                    existing_points += 1

    # Append new structures
    with open(train_config, "a") as f:
        for i, atoms in enumerate(structures):
            block = atoms_to_reann_block(atoms, existing_points + i + 1, include_forces=True)
            f.write(block)

    print(f"Appended {len(structures)} structures to {train_config}")
    print(f"Total training structures: {existing_points + len(structures)}")
    return True


def run_full_loop(config):
    """Run the complete active learning loop."""
    max_cycles = config.get("max_cycles", 8)
    log_file = os.path.join(config["workdir"], "active_learning.log")

    with open(log_file, "a") as log:
        log.write(f"\n{'='*60}\n")
        log.write(f"Active Learning started: {datetime.now()}\n")
        log.write(f"Config: {config}\n")

    # Step 1: Initial data conversion (only if no existing train/configuration)
    train_config = os.path.join(config["workdir"], "train", "configuration")
    if not os.path.exists(train_config):
        if not step_convert(config):
            return
        if not step_setup(config):
            return

    for cycle in range(1, max_cycles + 1):
        print(f"\n{'#'*60}")
        print(f"# ACTIVE LEARNING CYCLE {cycle}/{max_cycles}")
        print(f"{'#'*60}")

        with open(log_file, "a") as log:
            log.write(f"\n--- Cycle {cycle} started: {datetime.now()} ---\n")

        # Train
        if not step_train(config):
            print(f"Training failed at cycle {cycle}")
            break

        # Export
        if not step_export(config):
            print(f"Export failed at cycle {cycle}")
            break

        # Sample new configurations with NNP
        step_sample(config)

        # At this point, user needs to run DFT on sampled structures
        # Check if new DFT results are available
        new_dft_dir = config.get("new_dft_dir", "")
        if not new_dft_dir or not os.path.exists(new_dft_dir):
            print(f"\n*** Cycle {cycle} training complete. ***")
            print(f"Next steps:")
            print(f"  1. Run DFT (VASP) on sampled structures in {config['workdir']}/sampled/")
            print(f"  2. Place DFT results in: {new_dft_dir or '<new_dft_dir>'}")
            print(f"  3. Re-run with --step evaluate or continue the loop")
            break

        # Evaluate
        success, converged = step_evaluate(config, cycle)

        with open(log_file, "a") as log:
            log.write(f"Cycle {cycle} evaluation: success={success}, converged={converged}\n")

        if converged:
            print(f"\n*** NNP CONVERGED at cycle {cycle}! ***")
            break

        # Augment dataset
        step_augment(config, cycle)

    print(f"\nActive learning log saved to {log_file}")


def main():
    parser = argparse.ArgumentParser(description="REANN Active Learning Loop")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file for full loop")
    parser.add_argument("--step", type=str, default=None,
                        choices=["convert", "setup", "train", "export", "sample",
                                 "evaluate", "augment", "full"],
                        help="Run a single step manually")
    parser.add_argument("--workdir", type=str, default="./")
    parser.add_argument("--dft_data_dir", type=str, default=None)
    parser.add_argument("--new_dft_dir", type=str, default=None)
    parser.add_argument("--atomtype", nargs="+", default=None)
    parser.add_argument("--sample_input", type=str, default=None)
    parser.add_argument("--cycle", type=int, default=1)
    args = parser.parse_args()

    # Load config from YAML or build from args
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    else:
        config = {
            "workdir": args.workdir,
            "atomtype": args.atomtype or [],
            "dft_data_dir": args.dft_data_dir or "",
            "new_dft_dir": args.new_dft_dir or "",
            "sample_input": args.sample_input or "",
        }

    # Ensure workdir exists
    os.makedirs(config.get("workdir", "./"), exist_ok=True)

    step = args.step or config.get("step", "full")

    if step == "convert":
        step_convert(config)
    elif step == "setup":
        step_setup(config)
    elif step == "train":
        step_train(config)
    elif step == "export":
        step_export(config)
    elif step == "sample":
        step_sample(config)
    elif step == "evaluate":
        step_evaluate(config, args.cycle)
    elif step == "augment":
        step_augment(config, args.cycle)
    elif step == "full":
        run_full_loop(config)


if __name__ == "__main__":
    main()
