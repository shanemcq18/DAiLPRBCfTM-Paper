Source code and data for the paper _Data Assimilation in Large-Prandtl Rayleigh-Benard Convection from Thermal Measurements_ by [A. Farhat](https://scholar.google.com/citations?user=LlBckhUAAAAJ&hl=en&oi=ao), [N. E. Glatt-Holtz](https://scholar.google.com/citations?user=1GRq340AAAAJ&hl=en&oi=ao), [V. R. Martinez](https://scholar.google.com/citations?user=zml74fIAAAAJ&hl=en&oi=sra), [S. A. McQuarrie](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ&hl=en&oi=sra), and [J. P. Whitehead](https://scholar.google.com/citations?hl=en&user=lLR_YEYAAAAJ).
The paper has been submitted to the [SIAM Journal on Applied Dynamical Systems (SIADS)](https://www.siam.org/Publications/Journals/SIAM-Journal-on-Applied-Dynamical-Systems-SIADS).
Read an early draft of the paper [on ArXiV](https://arxiv.org/abs/1903.01508).

## Contents

#### Code

The following files were used to run numerical simulations with [`dedalus`](http://dedalus-project.org/), analyze the data, and create the figures that appear in the paper.
As a disclaimer, these scripts all require `dedalus` to be installed beforehand.

- [`base_simulator.py`](base_simulator.py): defines a `BaseSimulator` class base for managing data storage locations, simulation information, and other housekeeping items.

- [`boussinesq2d.py`](boussinesq2d.py): defines classes that extend `BaseSimulator` and set up the simulation with `dedalus`.
The actual simulation code is in `BoussinesqDataAssimilation2D.setup()` and `BoussinesqDataAssimilation2D.simulate()`; the class also includes visualization routines for post-processing.

- [`data_assimilation.py`](data_assimilation.py): script for parsing arguments and starting a simulation with `boussinesq2d.py`.

- [`merge.py`](merge.py): script for combining simulation data and calling visualization routines.

- [`plot_tools.py`](plot_tools.py): visualization routines for comparing results from multiple simulations.

- [`plot_mu_ra_relation.py`](plot_mu_ra_relation.py): a simple curve-fitting routine for one of the figures.

- [`plot_all.sh`](plot_all.sh) and [`plot_all_p3.sh`](plot_all_p3.sh): commands for figure creation with `merge.py`, `plot_tools.py`, and `plot_mu_ra_relation.py`.

#### Data Files

The following folders contain numerical simulation results, organized by the value of the Prandtl number _Pr_.

- [`infinite_prandtl`/](infinite_prandtl): experiments in which _Pr_ is infinite.

- [`finite_prandtl`/](finite_prandtl): experiments in which _Pr_ is finite, but large.

- [`hybrid_prandtl`/](hybrid_prandtl): experiments in which the data comes from a model with finite _Pr_, but the assimilating model assumes _Pr_ to be infinite.

- [`old_data/`](old_data): numerical results from a previous version of the paper.

#### Figures

The [`figures/`](figures) folder contains the actual figures used in the paper.


