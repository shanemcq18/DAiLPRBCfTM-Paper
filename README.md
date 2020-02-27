Source code and data for the paper [_Data Assimilation in Large-Prandtl Rayleigh-Benard Convection from Thermal Measurements_](https://epubs.siam.org/doi/abs/10.1137/19M1248327) by [A. Farhat](https://scholar.google.com/citations?user=LlBckhUAAAAJ&hl=en&oi=ao), [N. E. Glatt-Holtz](https://scholar.google.com/citations?user=1GRq340AAAAJ&hl=en&oi=ao), [V. R. Martinez](https://scholar.google.com/citations?user=zml74fIAAAAJ&hl=en&oi=sra), [S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ&hl=en&oi=sra), and [J. P. Whitehead](https://scholar.google.com/citations?hl=en&user=lLR_YEYAAAAJ).

## Contents

**Disclaimer:** most scripts in this repository require the [`dedalus`](http://dedalus-project.org/) package; see the [dedalus install guide](https://dedalus-project.readthedocs.io/en/latest/installation.html).
A quick way to test that your `dedalus` installation is working properly for this repository is to run `python3 data_assimilation.py --test`.

#### Code

- [`base_simulator.py`](base_simulator.py): defines a `BaseSimulator` class base for managing data storage locations, simulation information, and other housekeeping items.

- [`boussinesq2d.py`](boussinesq2d.py): defines classes that extend `BaseSimulator` and set up the simulation with `dedalus`.
The actual simulation code is in `BoussinesqDataAssimilation2D.setup()` and `BoussinesqDataAssimilation2D.simulate()`; the class also includes visualization routines for post-processing.

- [`data_assimilation.py`](data_assimilation.py): script for parsing arguments and starting a simulation with `boussinesq2d.py`.

- [`merge.py`](merge.py): script for combining simulation data and calling visualization routines.

- [`plot_tools.py`](plot_tools.py): visualization routines for comparing results from multiple simulations.

- [`plot_mu_ra_relation.py`](plot_mu_ra_relation.py): a simple curve-fitting routine for one of the figures.

- [`plot_all.sh`](plot_all.sh) and [`plot_all_p3.sh`](plot_all_p3.sh): commands for figure creation with `merge.py`, `plot_tools.py`, and `plot_mu_ra_relation.py`.

#### Data Files

- [`default.h5`](default.h5): an initial data set for testing.

- [`infinite_prandtl`/](infinite_prandtl): numerical simulation results for experiments in which _Pr_ is infinite.

- [`finite_prandtl`/](finite_prandtl): numerical simulation results for experiments in which _Pr_ is finite, but large.

- [`hybrid_prandtl`/](hybrid_prandtl): numerical simulation results for experiments in which the data comes from a model with finite _Pr_, but the assimilating model assumes _Pr_ to be infinite.

- [`old_data/`](old_data): numerical simulation results from a previous version of the paper.

#### Figures

- [`figures/`](figures) folder containing the actual figures used in the paper.

- [`.mplstyle`](.mplstyle): `matplotlib` configuration file.
