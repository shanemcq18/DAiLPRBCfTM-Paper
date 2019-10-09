#!/bin/bash

# The need for higher mu
python plot_tools.py --outfile mu_1_bad.pdf --xmax 0.05 --ymin 1e-4 sum_norm Ra infinite_prandtl/RA_0000464158_000001 infinite_prandtl/RA_0000623550_000001 infinite_prandtl/RA_0000837677_000001 infinite_prandtl/RA_0001125335_000001

# Varying mu (infinite Pr)
python plot_tools.py --outfile RA_0038881551_mu.pdf --xmax 0.01 sum_norm mu infinite_prandtl/RA_0038881551_000001 infinite_prandtl/RA_0038881551_008000 infinite_prandtl/RA_0038881551_010700 infinite_prandtl/RA_0038881551_025000 --title "Ra\approx 3.89\times 10^7"
python plot_tools.py --outfile RA_0052233450_mu.pdf --xmax 0.01 sum_norm mu infinite_prandtl/RA_0052233450_000001 infinite_prandtl/RA_0052233450_011000 infinite_prandtl/RA_0052233450_012700 infinite_prandtl/RA_0052233450_025000 --title "Ra\approx 5.22\times 10^7"
python plot_tools.py --outfile RA_0070170382_mu.pdf --xmax 0.01 sum_norm mu infinite_prandtl/RA_0070170382_000001 infinite_prandtl/RA_0070170382_013000 infinite_prandtl/RA_0070170382_015300 infinite_prandtl/RA_0070170382_025000 --title "Ra\approx 7.02\times 10^7"

# Varying N (infinite Pr)
python plot_tools.py --outfile RA_0038881551_N.pdf --xmax 0.01 sum_norm N infinite_prandtl/RA_0038881551_010700 infinite_prandtl/RA_0038881551_N_20_010700 infinite_prandtl/RA_0038881551_N_10_010700 infinite_prandtl/RA_0038881551_N_06_010700 --title "Ra\approx 3.89\times 10^7"
python plot_tools.py --outfile RA_0052233450_N.pdf --xmax 0.01 sum_norm N infinite_prandtl/RA_0052233450_012700 infinite_prandtl/RA_0052233450_N_22_012700 infinite_prandtl/RA_0052233450_N_14_012700 infinite_prandtl/RA_0052233450_N_06_012700 --title "Ra\approx 5.22\times 10^7"
python plot_tools.py --outfile RA_0070170382_N.pdf --xmax 0.01 sum_norm N infinite_prandtl/RA_0070170382_015300 infinite_prandtl/RA_0070170382_N_26_015300 infinite_prandtl/RA_0070170382_N_10_015300 infinite_prandtl/RA_0070170382_N_06_015300 --title "Ra\approx 7.02\times 10^7"

# Varying Pr (finite Pr)
python plot_tools.py --xmax 0.015 --outfile RA_0052233450_018000_PR.pdf sum_norm Pr infinite_prandtl/RA_0052233450_018000 finite_prandtl/RA_0052233450_018000_PR_100 finite_prandtl/RA_0052233450_018000_PR_042 finite_prandtl/RA_0052233450_018000_PR_007 --title "Ra\approx 5.22\times 10^7"
python plot_tools.py --xmax 0.015 --outfile RA_0038881551_014000_PR.pdf sum_norm Pr infinite_prandtl/RA_0038881551_014000 finite_prandtl/RA_0038881551_014000_PR_100 finite_prandtl/RA_0038881551_014000_PR_042 finite_prandtl/RA_0038881551_014000_PR_007 --title "Ra\approx 3.89\times 10^7"
python plot_tools.py --xmax 0.015 --outfile RA_0070170382_020001_PR.pdf sum_norm Pr infinite_prandtl/RA_0070170382_020001 finite_prandtl/RA_0070170382_020001_PR_100 finite_prandtl/RA_0070170382_020001_PR_042 finite_prandtl/RA_0070170382_020001_PR_007 --title "Ra\approx 7.02\times 10^7"

# Hybrid Pr convergence rates
python plot_tools.py --outfile hybrid_pr_vary.pdf --xmax 0.05 --title "Ra\approx 5.22\times 10^7" h2_l2_sum Pr hybrid_prandtl/RA_0052233450_018000_PR_010 hybrid_prandtl/RA_0052233450_018000_PR_100 hybrid_prandtl/RA_0052233450_018000_PR_1000 hybrid_prandtl/RA_0052233450_018000_PR_10000 hybrid_prandtl/RA_0052233450_018000_PR_100000 hybrid_prandtl/RA_0052233450_018000_PR_1000000 hybrid_prandtl/RA_0052233450_018000_PR_10000000
python plot_tools.py --outfile hybrid_pr_const.pdf h2_l2_avg Pr hybrid_prandtl/RA_0052233450_018000_PR_010 hybrid_prandtl/RA_0052233450_018000_PR_100 hybrid_prandtl/RA_0052233450_018000_PR_1000 hybrid_prandtl/RA_0052233450_018000_PR_10000 hybrid_prandtl/RA_0052233450_018000_PR_100000 hybrid_prandtl/RA_0052233450_018000_PR_1000000 hybrid_prandtl/RA_0052233450_018000_PR_10000000

# Full convergence plots
python merge.py --directory infinite_prandtl/RA_0052233450_012700 --no-force
mv infinite_prandtl/RA_0052233450_012700/convergence.pdf RA_0052233450_012700_convergence.pdf
python merge.py --directory finite_prandtl/RA_0052233450_018000_PR_100 --no-force
mv finite_prandtl/RA_0052233450_018000_PR_100/convergence.pdf RA_0052233450_018000_PR_100_convergence.pdf
python merge.py --directory hybrid_prandtl/RA_0052233450_018000_PR_100 --no-force
mv hybrid_prandtl/RA_0052233450_018000_PR_100/convergence.pdf RA_0052233450_018000_PR_100_hybrid_convergence.pdf
