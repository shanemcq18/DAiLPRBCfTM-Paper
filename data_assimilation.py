# data_assimilation.py
"""Class-based Dedalus script on Data Assimilation for Mantle Convection.

usage: data_assimilation.py -h
       data_assimilation.py --test
       data_assimilation.py --main [-Ra RAYLEIGH] [-Pr PRANDTL] [-N N]
                            [-m MU] [-t TIME] [-init INITIAL] [-s SAVE]
                            [--tight] [--no-anlaysis]
       data_assimilation.py --main --movie [-Ra RAYLEIGH] [-Pr PRANDTL] [-N N]
                            [-m MU] [-t TIME] [-init INITIAL] [--tight]

Run a Rayleigh-Benard convection simulation in 2D using the Boussinesq
equations; assimilate the data from that system into another system that
starts from a Fourier projection of the initial data.

optional arguments:
  -h, --help            show this help message and exit
  --main                run a full simulation with specified parameters
  --test                run a short test with default parameters
  -Ra RAYLEIGH, --Rayleigh RAYLEIGH
                        Rayleigh number (conduction vs convection)
  -Pr PRANDTL, --Prandtl PRANDTL
                        Prandtl number (viscous vs thermal diffusion)
  -N N, --N N           number of Fourier/Chebyshev projection modes
  -m MU, --mu MU        relaxation parameter for projection coupling
  -t TIME, --time TIME  simulation time
  -init INITIAL, --initial INITIAL
                        data file to use as initial conditions
  -s SAVE, --save SAVE  interval at which to save the simulation state
  --tight               simulate with smaller time steps than usual
  --no-analysis         simulate without convergence measurements
  --movie               indicate that this is a movie-making experiment

Authors: Shane McQuarrie, Jared Whitehead
"""

import argparse
import numpy as np

from boussinesq2d import (BoussinesqDataAssimilation2D,
                          BoussinesqDataAssimilation2Dmovie)


# Global variables ------------------------------------------------------------

DEFAULT_N = 32
DEFAULT_MU = 20000
TEST_DIR = "__TEST__"

# Folder management tools -----------------------------------------------------

def folder_name(Rayleigh, mu=None, N=None, Prandtl=None, movie=False):
    """Get the name of the folder for the specified simulation parameters."""
    # Always label with Rayleigh number.
    label = "RA_{:0>10}".format(Rayleigh)

    # Label by mu, N, Prandlt, and movie next.
    if N and N != DEFAULT_N:
        label += "_N_{:0>2}".format(N)
    if mu and mu != DEFAULT_MU:
        label += "_{:0>6}".format(mu)
    if Prandtl:
        label += "_PR_{:0>3}".format(Prandtl)
    if movie:
        label += "_movie"

    return label

# Command line argument parser ------------------------------------------------

parser = argparse.ArgumentParser(description="Run a Rayleigh-Benard "
                "convection simulation in 2D using the Boussinesq equations; "
                "assimilate the data from that system into another system "
                "that starts from a Fourier projection of the initial data.")
parser.usage = """data_assimilation.py -h
       data_assimilation.py --test
       data_assimilation.py --main [-Ra RAYLEIGH] [-Pr PRANDTL] [-N N]
                            [-m MU] [-t TIME] [-init INITIAL] [-s SAVE]
                            [--tight] [--no-anlaysis]
       data_assimilation.py --main --movie [-Ra RAYLEIGH] [-Pr PRANDTL] [-N N]
                            [-m MU] [-t TIME] [-init INITIAL] [--tight]
"""

# Mutually exclusive group: --main and --test
group1 = parser.add_mutually_exclusive_group()
group1.add_argument("--main", action="store_true",
                    help="run a full simulation with specified parameters")
group1.add_argument("--test", action="store_true",
                    help="run a short test with default parameters")

parser.add_argument("-Ra", "--Rayleigh", type=int, default=10000,
                    help="Rayleigh number (conduction vs convection)")
parser.add_argument("-Pr", "--Prandtl", type=int,
                    help="Prandtl number (viscous vs thermal diffusion)")
parser.add_argument("-N", "--N", type=int, default=DEFAULT_N,
                    help="number of Fourier/Chebyshev projection modes")
parser.add_argument("-m", "--mu", type=int, default=DEFAULT_MU,
                    help="relaxation parameter for projection coupling")
parser.add_argument("-t", "--time", type=float, default=1.0,
                    help="simulation time")
parser.add_argument("-init", "--initial",
                    help="data file to use as initial conditions")

# Mutually exclusive group: --movie and --save
group2 = parser.add_mutually_exclusive_group()
group2.add_argument("-s", "--save", type=float, default=.05,
                    help="interval at which to save the simulation state")
parser.add_argument("--tight", action="store_true",
                    help="simulate with smaller time steps than usual")
parser.add_argument("--no-analysis", action="store_true", default=False,
                    help="simulate without convergence measurements")
group2.add_argument("--movie", action="store_true", default=False,
                    help="indicate that this is a movie-making experiment")


if __name__ == "__main__":
    args = parser.parse_args()

    # Run a full simulation ---------------------------------------------------
    if args.main:
        label = folder_name(args.Rayleigh, args.mu,
                            args.N, args.Prandtl, args.movie)
        if args.movie:
            b2d = BoussinesqDataAssimilation2Dmovie(label)
            b2d.setup(Rayleigh=args.Rayleigh, mu=args.mu, N=args.N,
                      Prandtl=args.Prandtl)
            b2d.simulate(initial_conditions=args.initial,
                         sim_time=args.time, tight=args.tight)
        else:
            b2d = BoussinesqDataAssimilation2D(label)
            b2d.setup(Rayleigh=args.Rayleigh, mu=args.mu, N=args.N,
                      Prandtl=args.Prandtl)
            b2d.simulate(initial_conditions=args.initial,
                         sim_time=args.time, tight=args.tight,
                         save=args.save, analysis=not args.no_analysis)

    # Run a short test with default parameters --------------------------------
    elif args.test:
        b2d = BoussinesqDataAssimilation2D(TEST_DIR)
        b2d.setup(N=8)
        try:
            b2d.simulate(initial_conditions="default.h5", sim_time=.1501,
                         save=.0001)
        except KeyboardInterrupt:
            pass
        b2d.process_results()
