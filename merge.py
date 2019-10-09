# merge.py
"""Merge existing analysis and state files for a single simulation directory.

usage: merge.py -h
       merge.py -d DIRECTORY
       merge.py -Ra RAYLEIGH [-m MU] [-Pr PRANDTL] [-N N] [--movie]

optional arguments:
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        folder containing simulation files to merge
  -Ra RAYLEIGH, --Rayleigh RAYLEIGH
                        Rayleigh number (conduction vs convection)
  -m MU, --mu MU        relaxation parameter for projection coupling
  -Pr PRANDTL, --Prandtl PRANDTL
                        Prandtl number (viscous vs thermal diffusion)
  -N N, --N N           number of Fourier/Chebyshev projection modes
  --movie               indicate that this is a movie-making experiment
"""

import os
import argparse

from boussinesq2d import BoussinesqDataAssimilation2D
from data_assimilation import folder_name
from base_simulator import RANK


# Parse command line arguments ------------------------------------------------

parser = argparse.ArgumentParser(description="Merge existing analysis and "
                            "state files for a single simulation directory")
parser.usage = """merge.py -h
       merge.py -d DIRECTORY
       merge.py -Ra RAYLEIGH [-m MU] [-Pr PRANDTL] [-N N] [--movie]
"""

# mutually exclusive group: --directory, -Ra
group = parser.add_mutually_exclusive_group()
group.add_argument("-d", "--directory",
                    help="folder containing simulation files to merge")
group.add_argument("-Ra", "--Rayleigh", type=int, default=10000,
                    help="Rayleigh number (conduction vs convection)")
# Other arguments used if --directory is not specified
parser.add_argument("-m", "--mu", type=int,
                    help="relaxation parameter for projection coupling")
parser.add_argument("-Pr", "--Prandtl", type=int,
                    help="Prandtl number (viscous vs thermal diffusion)")
parser.add_argument("-N", "--N", type=int,
                    help="number of Fourier/Chebyshev projection modes")
parser.add_argument("--movie", action="store_true", default=False,
                    help="indicate that this is a movie-making experiment")
parser.add_argument("--no-force", action="store_true", default=False,
                    help="do not force the merge.")

args = parser.parse_args()

# Get the name of the directory to merge --------------------------------------
if args.directory:
    label = args.directory
else:
    label = folder_name(Rayleigh=args.Rayleigh, mu=args.mu, N=args.N,
                        Prandtl=args.Prandtl, movie=args.movie)

if not os.path.isdir(label):
    raise NotADirectoryError(label)

# Run the actual merge.
b2d = BoussinesqDataAssimilation2D(label)
b2d.merge_results(force=not args.no_force)

# Process the merged data (only on the root process).
if RANK == 0:
    if os.path.isfile(os.path.join(label, "analysis", "analysis.h5")):
        b2d.plot_convergence()
        b2d.plot_nusselt()
    if os.path.isfile(os.path.join(label, "states", "states.h5")):
        b2d.animate_temperature()
        b2d.animate_clusters()
