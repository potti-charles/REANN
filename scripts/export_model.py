#!/usr/bin/env python3
"""
Export trained REANN model (REANN.pth) to TorchScript (PES.pt).

Usage:
    python scripts/export_model.py --workdir ./work

Requires para/input_nn and para/input_density in workdir.
Reads REANN.pth and outputs PES.pt.
"""

import argparse
import os
import sys
import torch
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description="Export REANN model to TorchScript")
    parser.add_argument("--workdir", type=str, default="./",
                        help="Working directory containing REANN.pth and para/ (default: ./)")
    parser.add_argument("--output", type=str, default="PES.pt",
                        help="Output TorchScript file name (default: PES.pt)")
    parser.add_argument("--model", type=str, default="REANN.pth",
                        help="Input model checkpoint (default: REANN.pth)")
    args = parser.parse_args()

    # Change to workdir so PES.py can read para/ files
    original_dir = os.getcwd()
    os.chdir(args.workdir)

    # Add REANN package to path
    reann_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reann_pkg = os.path.join(reann_root, "reann")
    if reann_pkg not in sys.path:
        sys.path.insert(0, reann_pkg)

    from pes import PES as PES_module

    # Initialize model (reads para/input_nn and para/input_density)
    print("Initializing PES model from para/ config files...")
    init_pes = PES_module.PES()

    # Load trained weights
    model_path = args.model
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found in {args.workdir}")
        os.chdir(original_dir)
        return

    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")

    # Handle DDP module. prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict["reannparam"].items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    init_pes.load_state_dict(new_state_dict)

    # Convert to TorchScript
    print("Converting to TorchScript...")
    scripted_pes = torch.jit.script(init_pes)
    for params in scripted_pes.parameters():
        params.requires_grad = False
    scripted_pes.to(torch.double)

    output_path = args.output
    scripted_pes.save(output_path)
    print(f"Saved TorchScript model to {os.path.join(args.workdir, output_path)}")

    os.chdir(original_dir)


if __name__ == "__main__":
    main()
