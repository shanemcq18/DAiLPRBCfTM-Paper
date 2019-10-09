# plot_tools.py
"""Functions for extracting data to plot, etc."""

import os
import h5py
import json
import argparse
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt

from base_simulator import BaseSimulator


plt.style.use(".mplstyle")

# Helper Functions ============================================================

def get_data(dirname, task):
    """Get the specified taks from <dirname>/analysis/analysis.h5

    Returns:
        ((n,) ndarray): sim_time
        ((n,) ndarray): task
    """
    if task == "sum_norm":
        return get_sum_norm(dirname)
    elif task == "h2_l2_sum":
        return get_h2_l2_sum(dirname)
    elif task == "h2_l2_avg":
        return get_h2_l2_avg(dirname)
    # Make sure the directory and corresponding analysis file exist.
    if not os.path.isdir(dirname):
        raise NotADirectoryError(dirname)
    target = os.path.join(dirname, "analysis", "analysis.h5")
    if not os.path.isfile(target):
        raise FileNotFoundError(target)

    # Get the data.
    try:
        with h5py.File(target, 'r') as datafile:
            return np.array(datafile["scales/sim_time"]), datafile[task][:,0,0]
    except KeyError:
        print("\n Missing data in {}: {}".format(dirname, task), end='\n\n')
        raise

def get_label(dirname, label):
    """Strip the label of interest out of a directory name. For example, if
    dirname == "RA_123_N_08_456",
        label=="RA" -> return 123
        label=="N"  -> return 8
        label=="mu" -> return 456
    For label=="RA", also put it in scientific notation.
    """
    try:
        out = BaseSimulator._load_params(dirname)[label]
    except KeyError:
        if label == "Pr":
            return "$Pr = \infty$"
        else:
            raise
    out = "{:.2e}".format(out) if label in ["Ra","Pr"] else out #JPW: change back to 0.2e for most plots
    return r"${} = {}$".format(r"\mu" if label == "mu" else label, out)

def get_sum_norm(dirname):
    """Extract ||u_ - u||_H2 + ||T_ - T||_H2."""
    simtime1, T_h2_err = get_data(dirname, "tasks/T_h2_err")
    extra_time, T_h2 = get_data(dirname, "tasks/T_h2")
    simtime2, u_h2_err = get_data(dirname, "tasks/u_h2_err")
    extra_time, u_h2 = get_data(dirname, "tasks/u_h2")
    assert np.allclose(simtime1, simtime2)
    assert T_h2_err.shape == u_h2_err.shape
    return simtime1, (T_h2_err + u_h2_err)/(T_h2[0]+u_h2[0])

def get_h2_l2_sum(dirname):
    """Extract (1/Ra)*||u_ - u||_H2 + ||T_ - T||_l2^2."""
    print(dirname)
    simtime1, T_l2_err = get_data(dirname, "tasks/T_err")
    extra_time, T_l2 = get_data(dirname, "tasks/T_L2")
    simtime2, u_h2_err = get_data(dirname, "tasks/u_h2_err")
    extra_time, u_h2 = get_data(dirname, "tasks/u_h2")

    target = os.path.join(dirname, "params.json")
    with open(os.path.join(dirname, "params.json"), 'r') as infile:
        Ra = json.load(infile)["Ra"]

    assert np.allclose(simtime1, simtime2)
    assert T_l2_err.shape == u_h2_err.shape
#    return simtime1, T_l2_err**2 + u_h2_err/Ra
    return simtime1, (T_l2_err + u_h2_err/(Ra))/(T_l2[0]+u_h2[0]/Ra)

def get_h2_l2_avg(dirname):
    """Extract the mean of (1/Ra)*||u_ - u||_H2 + ||T_ - T||_l2^2."""
    simtime1, T_l2_err = get_data(dirname, "tasks/T_err")
    extra_time, T_l2 = get_data(dirname, "tasks/T_L2")
    simtime2, u_h2_err = get_data(dirname, "tasks/u_h2_err")
    extra_time, u_h2 = get_data(dirname, "tasks/u_h2")

    target = os.path.join(dirname, "params.json")
    with open(os.path.join(dirname, "params.json"), 'r') as infile:
        params = json.load(infile)

    assert np.allclose(simtime1, simtime2)
    assert T_l2_err.shape == u_h2_err.shape
    error = (T_l2_err + u_h2_err/(params["Ra"]))/(T_l2[0] + u_h2[0]/params["Ra"])
    return params["Pr"], error[(len(error)//8):].mean()


# Main Routine ================================================================

def make_plot(dirs, task, variable, outfile, xmax=.01, ymin=1e-11, title=None):
    """Plot the task from the given directories.

    Parameters:
        dirs (list(str)): directories to pull data from.
        task (str): which item to search for in each analysis.h5 file.
        variable (str): the kind of labels to use in the plot legend.
        outfile (str): the name of the resulting figure file.
        xmax (float): upper bound for the x-axis of the plot.
        ymin (float): lower bound for the y-axis of the plot.
        title (str): title for the plot.
    """
    print("Plotting {} with label {} from".format(task, variable))
    print('\t' + "\n\t".join(dirs))

    fig,ax = plt.subplots(1, 1)
    data, missing = [], []

    # Get the data from each directory and record errors.
    for d in dirs:
        try:
            data.append(get_data(d, task))
        except FileNotFoundError:
            missing.append(d)
            dirs.remove(d)
    if not dirs:
        raise FileNotFoundError(missing)
    elif missing:
        print("MISSING FILES:\n\t", "\n\t".join(missing))

    if task == "h2_l2_avg":
        pr, avg = np.transpose(data)
#        def func(t, a, b, c):
#            return a*t**2 + b*t + c
        def func(t, a, b):
            return a*t + b
        popt = opt.curve_fit(func, np.log10(pr), np.log10(avg))[0]
        print(popt)
        x = np.linspace(np.log10(pr.min()), np.log10(pr.max()), 200)
        ax.loglog(10**x, 10**func(x, *popt))
        ax.loglog(pr, avg, 'k*')
        ax.set_xlabel("$Pr$")
        ax.set_title(r"$Ra\approx 5.22 \times 10^7$")
        ax.set_ylim(1e-6, 1e1)
        # ax.set_ylabel(r"$\frac{1}{Ra}||\tilde{\mathbf{u}} - \mathbf{u}||_{H^2(\Omega)} + ||\tilde{T} - T||_{L^2(\Omega)}^2$")
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        print("Figure saved as", outfile)
        return

    # Plot the data over [0, xmax].
    for [t,y],d in zip(data, dirs):
        ax.semilogy(t, y, '-', lw=.5, label=get_label(d, variable))
        dum_t = t
    ax.semilogy(dum_t,np.exp(-dum_t),'-',lw=.5, label="$e^{-\mu t}$") #JPW: specific to Figure 3
    ax.set_xlim(0, xmax)
    ax.set_ylim(ymin, 1e1)
    ax.set_xlabel(r"Simulation Time $t$")
    ax.legend(loc="lower left", ncol=2, fontsize=8)
    if title:
        ax.set_title(r"${}$".format(title))
    # if task == "sum_norm":
    #     ax.set_ylabel(r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||_{H^2(\Omega)} + ||(\tilde{T} - T)(t)||_{H^2(\Omega)}$")
    if task == "h2_l2_sum":
        # ax.set_ylabel(r"$\frac{1}{Ra}||(\tilde{\mathbf{u}} - \mathbf{u})(t)||_{H^2(\Omega)} + ||(\tilde{T} - T)(t)||_{L^2(\Omega)}^2$", fontsize=8)
        ax.legend(loc="upper right", ncol=4, fontsize=6)
    if variable == "Ra":
        ax.set_ylabel("Error")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print("Figure saved as", outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("TASK",
                        help="which task to plot (e.g. tasks/T_err, sum_norm)")
    parser.add_argument("LABEL",
                        help="the variable to use for a label (Ra, Pr, mu, or N)")
    parser.add_argument("DIR", nargs='+',
                        help="directories to pull data from")
    parser.add_argument("--outfile", default="TEST.png",
                        help="name of the file to save")
    parser.add_argument("--xmax", type=float, default=.01,
                        help="upper bound of x-axis on plot")
    parser.add_argument("--ymin", type=float, default=1e-11,
                        help="lower bound of y-axis on plot")
    parser.add_argument("--title", default=None,
                        help="the figure title (if any)")
    args = parser.parse_args()

    make_plot(dirs=args.DIR, task=args.TASK, variable=args.LABEL,
              outfile=args.outfile, xmax=args.xmax, ymin=args.ymin, title=args.title)
