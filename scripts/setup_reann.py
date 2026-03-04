#!/usr/bin/env python3
"""
Generate REANN configuration files (input_nn and input_density).

Usage:
    python scripts/setup_reann.py --workdir ./work --atomtype C N Cu --cutoff 4.5

Creates para/input_nn and para/input_density in the specified working directory.
Parameters follow the paper conventions (2 hidden layers x 128 neurons, nblock=2, etc.)
"""

import argparse
import os


INPUT_NN_TEMPLATE = """\
# required parameters
  start_table = {start_table}                # 0=energy, 1=energy+force, 2=DM, 3=TDM, 4=polarizability
  table_coor = 0                 # 0=cartesian, 1=fractional coordinates
  nl = {nl}          # neural network architecture
  nblock = {nblock}
  dropout_p = {dropout_p}
  table_init = 0                 # 1 to restart from REANN.pth
# NN epoch and optimizer parameters
  Epoch = {epoch}
  patience_epoch = {patience_epoch}
  decay_factor = 0.5
  start_lr = {start_lr}
  end_lr = {end_lr}
  re_ceff = 0
# data parameters
  ratio = 0.9
  batchsize_train = {batchsize_train}
  batchsize_val = {batchsize_val}
  e_ceff = {e_ceff}
  init_f = {init_f}
  final_f = {final_f}
# misc
  queue_size = 10
  print_epoch = {print_epoch}
  table_norm = True
  DDP_backend = 'nccl'
  activate = '{activate}'
  dtype = "float32"
# orbital coefficient parameters
  oc_nl = {oc_nl}
  oc_nblock = {oc_nblock}
  oc_dropout_p = {oc_dropout_p}
  oc_activate = '{activate}'
  oc_table_norm = True
  oc_loop = {oc_loop}
# data folder (relative to working directory)
  folder = "{folder}"
"""

INPUT_DENSITY_TEMPLATE = """\
# density parameters
  neigh_atoms = {neigh_atoms}
  cutoff = {cutoff}
  nipsin = {nipsin}
  atomtype = {atomtype}
  nwave = {nwave}
"""


def main():
    parser = argparse.ArgumentParser(description="Generate REANN config files")
    parser.add_argument("--workdir", type=str, default="./",
                        help="Working directory (default: ./)")
    parser.add_argument("--atomtype", nargs="+", required=True,
                        help="Atom types, e.g. C N Cu")
    parser.add_argument("--cutoff", type=float, default=4.5,
                        help="Cutoff distance in Angstrom (default: 4.5)")
    parser.add_argument("--nipsin", type=int, default=2,
                        help="Max angular momentum (default: 2)")
    parser.add_argument("--nwave", type=int, default=8,
                        help="Number of radial Gaussians, must be power of 2 (default: 8)")
    parser.add_argument("--nl", type=int, nargs="+", default=[128, 128],
                        help="Hidden layer sizes (default: 128 128)")
    parser.add_argument("--nblock", type=int, default=2,
                        help="Number of residual blocks (default: 2)")
    parser.add_argument("--activate", type=str, default="Relu_like",
                        choices=["Relu_like", "Tanh_like"],
                        help="Activation function (default: Relu_like)")
    parser.add_argument("--start_table", type=int, default=1,
                        help="0=energy, 1=energy+force (default: 1)")
    parser.add_argument("--epoch", type=int, default=20000,
                        help="Max training epochs (default: 20000)")
    parser.add_argument("--batchsize_train", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--batchsize_val", type=int, default=256,
                        help="Validation batch size (default: 256)")
    parser.add_argument("--neigh_atoms", type=int, default=150,
                        help="Max neighbors per atom (default: 150)")
    parser.add_argument("--oc_loop", type=int, default=1,
                        help="Orbital coefficient iterations (default: 1)")
    parser.add_argument("--folder", type=str, default="./",
                        help="Data folder relative to workdir (default: ./)")
    parser.add_argument("--start_lr", type=float, default=0.001)
    parser.add_argument("--end_lr", type=float, default=1e-5)
    parser.add_argument("--patience_epoch", type=int, default=100)
    args = parser.parse_args()

    para_dir = os.path.join(args.workdir, "para")
    os.makedirs(para_dir, exist_ok=True)

    # Format lists as Python list strings
    nl_str = str(args.nl)
    dropout_str = str([0.0] * len(args.nl))
    atomtype_str = str(args.atomtype)
    oc_nl_str = str(args.nl)
    oc_dropout_str = str([0.0] * len(args.nl))

    # Write input_nn
    nn_content = INPUT_NN_TEMPLATE.format(
        start_table=args.start_table,
        nl=nl_str,
        nblock=args.nblock,
        dropout_p=dropout_str,
        epoch=args.epoch,
        patience_epoch=args.patience_epoch,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        batchsize_train=args.batchsize_train,
        batchsize_val=args.batchsize_val,
        e_ceff=0.1,
        init_f=50,
        final_f=0.5,
        print_epoch=5,
        activate=args.activate,
        oc_nl=oc_nl_str,
        oc_nblock=1,
        oc_dropout_p=oc_dropout_str,
        oc_loop=args.oc_loop,
        folder=args.folder,
    )
    nn_path = os.path.join(para_dir, "input_nn")
    with open(nn_path, "w") as f:
        f.write(nn_content)
    print(f"Written {nn_path}")

    # Write input_density
    density_content = INPUT_DENSITY_TEMPLATE.format(
        neigh_atoms=args.neigh_atoms,
        cutoff=args.cutoff,
        nipsin=args.nipsin,
        atomtype=atomtype_str,
        nwave=args.nwave,
    )
    density_path = os.path.join(para_dir, "input_density")
    with open(density_path, "w") as f:
        f.write(density_content)
    print(f"Written {density_path}")

    print(f"\nConfig files created in {para_dir}")
    print(f"To train: cd {args.workdir} && python -m reann")


if __name__ == "__main__":
    main()
