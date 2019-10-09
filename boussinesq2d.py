# boussinesq2d.py
"""Dedalus script for data assimilation in the Boussinesq equations.

Authors: Shane McQuarrie, Jared Whitehead
"""

import os
import re
import h5py
import time
import numpy as np
from scipy.integrate import simps
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.animation import writers as mplwriters
try:
    from tqdm import tqdm
except ImportError:
    print("Recommended: install tqdm (pip install tqdm)")
    tqdm = lambda x: x

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.core.operators import GeneralFunction

from base_simulator import BaseSimulator, RANK, SIZE

# Simulation Classes ==========================================================

class BoussinesqDataAssimilation2D(BaseSimulator):
    """Manager for dedalus simulations of the 2D Boussinesq system.

    Let Psi = [0,L]x[0,1] with coordinates (x,z). Defining
    u = [v, w] = [-psi_z, psi_x] and zeta = laplace(psi),
    the Boussinesq equations can be written as follows.

    Pr [Ra T_x + laplace(zeta)] - zeta_t = u.grad(zeta)
                        laplace(T) - T_t = u.grad(T)
    subject to
        u(z=0) = 0 = u(z=1)
        T(z=0) = 1, T(z=1) = 0
        u, T periodic in (x,y) (use a Fourier basis)

    Variables:
        u:R2xR -> R2: the fluid velocity vector field.
        T:R2xR -> R: the fluid temperature.
        p:R2xR -> R: the pressure.
        Ra: the Rayleigh number.
        Pr: the Prandtl number

    If the Prandtl number is infinite, the first equations can be simplified.

    - zeta_t = u.grad(zeta)
    """
    @staticmethod
    def P_N(F, N, scale=False):
        """Calculate the Fourier mode projection of F with N terms."""
        # Set the c_n to zero wherever n > N (in both axes).
        X,Y = np.indices(F['c'].shape)
        F['c'][(X >= N) | (Y >= N)] = 0

        if scale:
            F.set_scales(1)
        return F['g']

    def setup(self, L=4, xsize=256, zsize=128, Prandtl=None, Rayleigh=10000,
                                            mu=1, N=32, BCs="no-slip"):
        """Set up the systems of equations as a dedalus Initial Value Problem,
        without providing initial conditions yet.

        Parameters:
            L (float): the length of the x domain. In x and z, the domain is
                therefore [0,L]x[0,1].
            xsize (int): the number of points to discretize in the x direction.
            zsize (int): the number of points to discretize in the z direction.
            Prandtl (None or float): the ratio of momentum diffusivity to
                thermal diffusivity of the fluid. If None (default), then
                the system is set up as if Prandtl = infinity.
            Rayleigh (float): measures the amount of heat transfer due to
                convection, as opposed to conduction.
            mu (float): constant on the Fourier projection in the
                Data Assimilation system.
            N (int): the number of modes to keep in the Fourier projection.
            BCs (str): if 'no-slip', use the no-slip BCs u(z=0,1) = 0.
                If 'free-slip', use the free-slip BCs u_z(z=0,1) = 0.
        """
        # Validate BCs parameter.
        if BCs not in {"no-slip", "free-slip"}:
            raise ValueError("'BCs' must be 'no-slip' or 'free-slip'")

        # Validate N parameter.
        minsize = min(xsize, zsize)
        if not 0 <= N <= minsize:
            raise ValueError("0 <= N <= {} is required".format(minsize))

        # Bases and Domain ----------------------------------------------------
        x_basis = de.Fourier('x', xsize, interval=(0, L), dealias=3/2)
        z_basis = de.Chebyshev('z', zsize, interval=(0, 1), dealias=3/2)
        domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

        # Initialize the problem as an IVP and add variables ------------------
        problem = de.IVP(domain, variables=['T', 'T_',      # Temperature
                                            'Tz', 'Tz_',
                                            'psi', 'psi_',  # Stream function
                                            'psiz', 'psiz_',
                                            'zeta', 'zeta_',# Laplace of stream
                                            'zetaz', 'zetaz_'])

        driving = GeneralFunction(domain, 'g',
                                  BoussinesqDataAssimilation2D.P_N, args=[])

        # System parameters (these are saved to a JSON file).
        problem.parameters['L'] = L                 # Domain parameters
        problem.parameters['xsize'] = xsize
        problem.parameters['zsize'] = zsize

        problem.parameters['Ra'] = Rayleigh         # Fluid parameters
        if Prandtl:
            problem.parameters['Pr'] = Prandtl

        problem.parameters["N"] = N                 # Assimilation parameters
        problem.parameters["mu"] = mu
        problem.parameters["driving"] = driving

        # Stream function substitutions: u = [v, w] = [-psi_z, psi_w]
        problem.substitutions['v']  = "-dz(psi)"
        problem.substitutions['v_'] = "-dz(psi_)"
        problem.substitutions['w']  =  "dx(psi)"
        problem.substitutions['w_'] =  "dx(psi_)"

        # Relate higher-order z derivatives (b/c Chebyshev).
        problem.add_equation("psiz  - dz(psi)  = 0")
        problem.add_equation("psiz_ - dz(psi_) = 0")
        problem.add_equation("zetaz  - dz(zeta)  = 0")
        problem.add_equation("zetaz_ - dz(zeta_) = 0")
        problem.add_equation("Tz  - dz(T)  = 0")
        problem.add_equation("Tz_ - dz(T_) = 0")

        # zeta = laplace(psi)
        problem.add_equation("zeta  - dx(dx(psi))  - dz(psiz)  = 0")
        problem.add_equation("zeta_ - dx(dx(psi_)) - dz(psiz_) = 0")

        # 2D Boussinesq Equations ---------------------------------------------
        if Prandtl is None: # Ra T_x + laplace(zeta) = 0
            problem.add_equation("Ra*dx(T)  + dx(dx(zeta))  + dz(zetaz)  = 0")
            problem.add_equation("Ra*dx(T_) + dx(dx(zeta_)) + dz(zetaz_) = 0")
        else:              # Pr(Ra T_x + laplace(zeta)) - zeta_t = u.grad(zeta)
            problem.add_equation("Pr*(Ra*dx(T)  + dx(dx(zeta))  + dz(zetaz))"
                                 " - dt(zeta)  = v*dx(zeta)   + w*zetaz")
            problem.add_equation("Pr*(Ra*dx(T_) + dx(dx(zeta_)) + dz(zetaz_))"
                                 " - dt(zeta_) = v_*dx(zeta_) + w_*zetaz_")

        # T_t - laplace(T) = -u . grad(T) (+ driving Fourier projection)
        problem.add_equation("dt(T)  - dx(dx(T))  - dz(Tz)  "
                             "= -v*dx(T)   - w*Tz")
        problem.add_equation("dt(T_) - dx(dx(T_)) - dz(Tz_) "
                             "= -v_*dx(T_) - w_*Tz_ - mu*driving")

        # Boundary Conditions -------------------------------------------------
        # Temperature heating from 'left' (bottom), cooling from 'right' (top).
        problem.add_bc("left(T)  = 1")          # T(z=0) = 1
        problem.add_bc("left(T_) = 1")
        problem.add_bc("right(T)  = 0")         # T(z=1) = 0
        problem.add_bc("right(T_) = 0")

        # Velocity field boundary conditions: no-slip or free-slip.
        # w(z=1) = w(z=0) = 0 (part of no-slip and free-slip)
        problem.add_bc("left(psi)  = 0")
        problem.add_bc("left(psi_) = 0")
        problem.add_bc("right(psi)  = 0")
        problem.add_bc("right(psi_) = 0")

        if BCs == "no-slip":
            # u(z=0) = 0 --> v(z=0) = 0 = w(z=0)
            problem.add_bc("left(psiz)  = 0")
            problem.add_bc("left(psiz_) = 0")

            # u(z=1) = 0 --> v(z=1) = 0 = w(z=1)
            problem.add_bc("right(psiz)  = 0")
            problem.add_bc("right(psiz_) = 0")

        elif BCs == "free-slip":
            # u'(z=0) = 0 --> v'(z=0) = 0 = w(z=0)
            problem.add_bc("left(dz(psiz))  = 0")
            problem.add_bc("left(dz(psiz_)) = 0")

            # u'(z=1) = 0 --> v'(z=1) = 0 = w(z=1)
            problem.add_bc("right(dz(psiz))  = 0")
            problem.add_bc("right(dz(psiz_)) = 0")

        self.problem = problem
        self.logger.info("Problem constructed")

        # Save system parameters in JSON format.
        if RANK == 0:
            self._save_params()

    def simulate(self, initial_conditions=None, scheme=de.timesteppers.RK443,
                       sim_time=2, wall_time=np.inf, tight=False, save=.05,
                       analysis=True):
        """Load initial conditions, run the simulation, and merge results.

        Parameters:
            initial_conditions (None, str): determines from what source to
                draw the initial conditions. Valid options are as follows:
                - None: use trivial conditions (T_ = 1 - z, T = 1 - z + eps).
                - 'resume': use the most recent state file in the
                    records directory (load both model and DA system).
                - An .h5 filename: load state variables for the model and
                    reset the data assimilation state variables to zero.
            scheme (de.timesteppers): The kind of solver to use. Options are
                RK443 (de.timesteppers.RK443), RK111, RK222, RKSMR, etc.
            sim_time (float): The maximum amount of simulation time allowed
                (in seconds) before ending the simulation.
            wall_time (float): The maximum amound of computing time allowed
                (in seconds) before ending the simulation.
            tight (bool): If True, set a low cadence and min_dt for refined
                simulation. If False, set a higher cadence and min_dt for a
                more coarse (but faster) simulation.
            save (float): The number of simulation seconds that pass between
                saving the state files. Higher save result in smaller data
                files, but lower numbers result in better animations.
                Set to 0 to disable saving state files.
            analysis (bool): Whether or not to track convergence measurements.
                Disable for faster simulations (less message passing via MPI)
                when convergence estimates are not needed (i.e. movie only).
        """
        if not self.problem:
            raise TypeError("problem not initialized (run setup())")

        self.logger.debug("\n")
        self.logger.debug("NEW SIMULATION")
        solver = self.problem.build_solver(scheme)
        self.logger.info("Solver built")

        N = int(self.problem.parameters['N'])

        # Initial conditions --------------------------------------------------
        if initial_conditions is None:              # "Trivial" conditions.
            eps = 1e-4
            k = 3.117
            dt = 1e-4

            x,z = self.problem.domain.grids(scales=1)
            T, T_ = solver.state['T'], solver.state['T_']
            # Start T from rest plus a small perturbation.
            T['g']  = 1 - z + eps*np.sin(k*x)*np.sin(2*np.pi*z)
            T.differentiate('z', out=solver.state['Tz'])
            # Start T_ from rest.
            T_['g'] = 1 - z
            T_.differentiate('z', out=solver.state['Tz_'])
            self.logger.info("Using trivial initial conditions")

        elif isinstance(initial_conditions, str):   # Load data from a file.
            # Resume: load the state of the last (merged) state file.
            resume = initial_conditions == "resume"
            if resume:
                initial_conditions = self._get_merged_file("states")
            if not initial_conditions.endswith(".h5"):
                raise ValueError("'{}' is not an h5 file".format(
                                                        initial_conditions))
            # Load the data from the specified h5 file into the system.
            self.logger.info("Loading initial conditions from {}".format(
                                                        initial_conditions))

            with h5py.File(initial_conditions, 'r') as infile:
                dt = infile["scales/timestep"][-1] * .01    # initial dt
                errs = []
                tasks = ["T", "Tz", "psi", "psiz", "zeta", "zetaz"]
                if resume:      # Only load assimilating variables to resume.
                    tasks += ["T_", "Tz_", "psi_", "psiz_", "zeta_", "zetaz_"]
                    solver.sim_time = infile["scales/sim_time"][-1]
                    niters = infile["scales/iteration"][-1]
                    solver.initial_iteration = niters
                    solver.iteration = niters
                for name in tasks:
                    # Get task data from the h5 file (recording failures).
                    try:
                        data = infile["tasks/"+name][-1,:,:]
                    except KeyError as e:
                        errs.append("tasks/"+name)
                        continue
                    # Determine the chunk belonging to this process.
                    chunk = data.shape[1] // SIZE
                    subset = data[:,RANK*chunk:(RANK+1)*chunk]
                    # Change the corresponding state variable.
                    scale = solver.state[name]['g'].shape[0] / \
                                        self.problem.parameters["xsize"]
                    solver.state[name].set_scales(1)
                    solver.state[name]['g'] = subset
                    solver.state[name].set_scales(scale)
                if errs:
                    raise KeyError("Missing keys in '{}': '{}'".format(
                                    initial_conditions, "', '".join(errs)))
            # Initial conditions for assimilating system: T_0 = P_4(T0).
            if not resume:
                G = self.problem.domain.new_field()
                G['c'] = solver.state['T']['c'].copy()
                solver.state['T_']['g'] = BoussinesqDataAssimilation2D.P_N(
                                                                    G, 4, True)
                solver.state['T_'].differentiate('z', out=solver.state['Tz_'])

        # Driving / projection function arguments -----------------------------

        dT = solver.state['T_'] - solver.state['T']
        self.problem.parameters["driving"].args = [dT, N]
        self.problem.parameters["driving"].original_args = [dT, N]

        # Stopping Parameters -------------------------------------------------

        solver.stop_sim_time = sim_time         # Length of simulation.
        solver.stop_wall_time = wall_time       # Real time allowed to compute.
        solver.stop_iteration = np.inf          # Maximum iterations allowed.

        # State snapshots -----------------------------------------------------
        if save:
            # Save the temperature measurements in states/ files. Use sim_dt.
            snaps = solver.evaluator.add_file_handler(
                                    os.path.join(self.records_dir, "states"),
                                    sim_dt=save, max_writes=5000,mode="append")
                                    # Set save=0.005 or lower for more writes.
            snaps.add_task("T")
            snaps.add_task("T_")
            snaps.add_task("driving", name="P_N")

        # Convergence analysis ------------------------------------------------
        if analysis:
            # Save specific tasks in analysis/ files every few iterations.
            annals = solver.evaluator.add_file_handler(
                                    os.path.join(self.records_dir, "analysis"),
                                    iter=20, max_writes=73600, mode="append")

            # Nusselt Number measurements - - - - - - - - - - - - - - - - - - -
            # 1 + int(wT)/L
            annals.add_task("1 + integ(w *T , 'x','z')/L", name="Nu_1")
            annals.add_task("1 + integ(w_*T_, 'x','z')/L", name="Nu_1_da")
            # int(grad(T)^2)/L
            annals.add_task("integ(dx(T )**2 + Tz **2, 'x','z')/L",
                                                            name="Nu_2")
            annals.add_task("integ(dx(T_)**2 + Tz_**2, 'x','z')/L",
                                                            name="Nu_2_da")
            # 1 + int(grad(u)^2)/(Ra L)
            annals.add_task("1 + "
                "integ(dx(v )**2 + dz(v )**2 + dx(w )**2 + dz(w )**2,'x','z')"
                                                    "/(Ra*L)", name="Nu_3")
            annals.add_task("1 + "
                "integ(dx(v_)**2 + dz(v_)**2 + dx(w_)**2 + dz(w_)**2,'x','z')"
                                                    "/(Ra*L)", name="Nu_3_da")

            # Convergence estimates - - - - - - - - - - - - - - - - - - - - - -
            # ||T - T_||_L2
            annals.add_task("sqrt( integ((T - T_)**2, 'x','z')"
                                    "/integ(T**2, 'x', 'z') )", name="T_err")
            # ||grad(T) - grad(T_)||_L2
            # Could use dz(T-T_) or dz(T)-dz(T_) or Tz-Tz_
            annals.add_task("sqrt( integ(dx(T-T_)**2 + dz(T-T_)**2, 'x','z')"
                                    "/integ(dx(T)**2 + dz(T)**2, 'x','z') )",
                                                            name="gradT_err")
            # ||u - u_||_L2
            annals.add_task("sqrt( integ((v-v_)**2 + (w-w_)**2, 'x','z')"
                                    "/integ(v**2 + w**2, 'x','z') )",
                                                            name="u_err")
            # ||grad(u - u_)||_L2
            annals.add_task("sqrt( integ(dx(v-v_)**2 + dz(v-v_)**2"
                                       " + dx(w-w_)**2 + dz(w-w_)**2, 'x','z')"
                                   "/integ(dx(v)**2 + dz(v)**2"
                                       " + dx(w)**2 + dz(w)**2, 'x','z') )",
                                                            name="gradu_err")
            # ||T - T_||_H2
            annals.add_task("sqrt( integ(dx(dx(T-T_))**2 + dx(dz(T-T_))**2 "
                                                "+ dz(dz(T-T_))**2, 'x','z')"
                                "/integ(dx(dx(T))**2 + dx(dz(T))**2 "
                                                "+ dz(dz(T))**2, 'x','z') )",
                                                            name="T_h2_err")
            # ||u - u_||_H2
            annals.add_task("sqrt("
                            "integ( dx(dx(v-v_))**2 + dz(dz(v-v_))**2"
                              " + dx(dz(v-v_))**2 + dx(dz(w-w_))**2"
                              " + dx(dx(w-w_))**2 + dz(dz(w-w_))**2, 'x','z')"
                           "/integ( dx(dx(v))**2 + dz(dz(v))**2"
                              " + dx(dz(v))**2 + dx(dz(w))**2"
                              " + dx(dx(w))**2 + dz(dz(w))**2, 'x','z') )",
                                                            name="u_h2_err")

        # Control Flow --------------------------------------------------------
        if tight:                   # Tighter control flow (slower but safer).
            cfl = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=1,
                                            max_change=1.5, min_change=0.01,
                                            max_dt=0.01,    min_dt=1e-10)
        else:                       # Looser control flow (faster but risky).
            cfl = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                                            max_change=1.5, min_change=0.5,
                                            max_dt=0.01,    min_dt=1e-6)
        cfl.add_velocities(('v',  'w' ))
        cfl.add_velocities(('v_', 'w_'))

        # Flow properties (print during run; not recorded in the records files)
        flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
        flow.add_property("sqrt(v **2 + w **2) / Ra", name='Re' )
        flow.add_property("sqrt(v_**2 + w_**2) / Ra", name='Re_')

        # MAIN COMPUTATION LOOP -----------------------------------------------
        try:
            self.logger.info("Starting main loop")
            start_time = time.time()
            while solver.ok:
                # Recompute time step and iterate.
                dt = cfl.compute_dt()
                dt = solver.step(dt) #, trim=True)

                # Print info to the screen every 10 iterations.
                if solver.iteration % 10 == 0:
                    info = "Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}".format(
                                        solver.iteration, solver.sim_time, dt)
                    Re  = flow.max("Re")
                    Re_ = flow.max("Re_")
                    info += ", Max Re = {:f}".format(Re)
                    info += ", Max Re_ = {:f}".format(Re_)
                    self.logger.info(info)
                    # Make sure the simulation hasn't blown up.
                    if np.isnan(Re) or np.isnan(Re_):
                        raise ValueError("Reynolds number went to infinity!!"
                                         "\nRe = {}, Re_ = {}".format(Re, Re_))
        except BaseException as e:
            self.logger.error("Exception raised, triggering end of main loop.")
            raise
        finally:
            total_time = time.time() - start_time
            cpu_hr = total_time / 60 / 60 * SIZE
            self.logger.info("Iterations: {:d}".format(solver.iteration))
            self.logger.info("Sim end time: {:.3e}".format(solver.sim_time))
            self.logger.info("Run time: {:.3e} sec".format(total_time))
            self.logger.info("Run time: {:.3e} cpu-hr".format(cpu_hr))
            self.logger.debug("END OF SIMULATION\n")

    def _get_merged_file(self, label):
        """Return the name of the oldest merged (full or partial) h5 file with
        the specified label.
        """
        if label not in {"states", "analysis"}:
            raise ValueError("label must be 'states' or 'analysis'")
        out = self.get_files(label)
        if out[0].endswith("{}.h5".format(label)):
            return out[0]
        return out[-1]

    @staticmethod
    def _get_fully_merged_state_file(records_dir):
        """Return the name of the fully merged h5 state file, without doing
        any file merges if the file does not exist.

        Parameters:
            records_dir (str): The base folder containing the simulation files.

        Raises:
            NotADirectoryError: if the states/ subdirectory does not exist.
            FileNotFoundError: if the states/states.h5 file does not exist.
        """
        subdir = os.path.join(records_dir, "states")
        if not os.path.isdir(subdir):
            raise NotADirectoryError(subdir)

        target = os.path.join(records_dir, "states", "states.h5")
        if not os.path.isfile(target):
            raise FileNotFoundError(target)

        return target

    def merge_results(self, force=False):
        """Merge the different process state and analysis files together."""
        for label in ["analysis", "states"]:
            # Check that the folder exists and is nonempty.
            folder = os.path.join(self.records_dir, label)
            if os.path.isdir(folder) and os.listdir(folder):
                # Call the parent merge function.
                BaseSimulator.merge_results(self, label, True, force=force)
            else:
                # Inform the user that merge files were not found.
                self.logger.info("No {} files to merge".format(label))

    def plot_convergence(self, savefig=True):
        """Plot the six measures of convergence over time."""
        # self.merge_results()
        datafile = self._get_merged_file("analysis")
        self.logger.info("Plotting convergence estimates from '{}'...".format(
                                                                    datafile))
        # Gather data from the source file.
        with h5py.File(datafile, 'r') as data:
            times = list(data["scales/sim_time"])
            T_err = data["tasks/T_err"][:,0,0]
            gradT_err = data["tasks/gradT_err"][:,0,0]
            u_err = data["tasks/u_err"][:,0,0]
            gradu_err = data["tasks/gradu_err"][:,0,0]
            T_h2_err = data["tasks/T_h2_err"][:,0,0]
            u_h2_err = data["tasks/u_h2_err"][:,0,0]

        with plt.style.context(".mplstyle"):
            # Make subplots and a big plot for an overlay.
            fig = plt.figure(figsize=(12,6))
            ax1 = plt.subplot2grid((3,4), (0,0))
            ax2 = plt.subplot2grid((3,4), (0,1))
            ax3 = plt.subplot2grid((3,4), (1,0))
            ax4 = plt.subplot2grid((3,4), (1,1))
            ax5 = plt.subplot2grid((3,4), (2,0))
            ax6 = plt.subplot2grid((3,4), (2,1))
            axbig = plt.subplot2grid((3,4), (0,2), rowspan=3, colspan=2)

            # Plot the data.
            ax1.semilogy(times, T_err, 'C0', lw=.5)
            ax2.semilogy(times, u_err, 'C1', lw=.5)
            ax3.semilogy(times, gradT_err, 'C2', lw=.5)
            ax4.semilogy(times, gradu_err, 'C3', lw=.5)
            ax5.semilogy(times, T_h2_err, 'C4', lw=.5)
            ax6.semilogy(times, u_h2_err, 'C5', lw=.5)
            axbig.semilogy(times, T_err, 'C0', lw=.5,
                           label=r"$||(\tilde{T} - T)(t)||_{L^2(\Omega)}$")
            axbig.semilogy(times, u_err, 'C1', lw=.5,
                           label=r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||"
                                 r"_{L^2(\Omega)}$")
            axbig.semilogy(times, gradT_err, 'C2', lw=.5,
                           label=r"$||(\nabla\tilde{T} - \nabla T)(t)||"
                                 r"_{L^2(\Omega)}$")
            axbig.semilogy(times, gradu_err, 'C3', lw=.5,
                           label=r"$||(\nabla\tilde{\mathbf{u}} - \nabla"
                                 r"\mathbf{u})(t)||_{L^2(\Omega)}$")
            axbig.semilogy(times, T_h2_err, 'C4', lw=.5,
                           label=r"$||(\tilde{T} - T)(t)||_{H^2(\Omega)}$")
            axbig.semilogy(times, u_h2_err, 'C5', lw=.5,
                           label=r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||"
                                 r"_{H^2(\Omega)}$")
            axbig.legend(loc="upper right")

            # Set minimal axis and tick labels.
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xticklabels([])
            for ax in [ax2, ax4, ax6]:
                ax.set_yticklabels([])
            ax5.set_xlabel("Simulation Time", color="white")
            ax6.set_xlabel("Simulation Time", color="white")
            axbig.set_xlabel("Simulation Time", color="white")
            fig.text(0.5, 0.01, r"Simulation Time $t$", ha="center",
                     fontsize=16)
            ax1.set_title(r"$||(\tilde{T} - T)(t)||_{L^2(\Omega)}$")
            ax2.set_title(r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||"
                          r"_{L^2(\Omega)}$")
            ax3.set_title(r"$||(\nabla\tilde{T} - \nabla T)(t)||"
                          r"_{L^2(\Omega)}$")
            ax4.set_title(r"$||(\nabla\tilde{\mathbf{u}} - \nabla"
                          r"\mathbf{u})(t)||_{L^2(\Omega)}$")
            ax5.set_title(r"$||(\tilde{T} - T)(t)||_{H^2(\Omega)}$")
            ax6.set_title(r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||"
                          r"_{H^2(\Omega)}$")
            axbig.set_title("Overlay")

            # Make the axes uniform and use tight spacing.
            xlims = axbig.get_xlim()
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6, axbig]:
                ax.set_xlim(xlims)
            #    ax.set_ylim(1e-11, 1e1)
            plt.tight_layout()

            # Save or show the figure.
            if savefig:
                outfile = os.path.join(self.records_dir, "convergence.pdf")
                plt.savefig(outfile, dpi=300, bbox_inches="tight")
                self.logger.info("\tFigure saved as '{}'".format(outfile))
            else:
                plt.show()
            plt.close()

    def plot_nusselt(self, savefig=True):
        """Plot the three measures of the Nusselt number over time for the
        base and DA systems.
        """
        # self.merge_results()
        datafile = self._get_merged_file("analysis")
        self.logger.info("Plotting Nusselt number from '{}'...".format(
                                                                    datafile))
        # Gather data from the source file.
        times = []
        nusselt = [[] for _ in range(6)]
        with h5py.File(datafile, 'r') as data:
            times = list(data["scales/sim_time"])
            for i in range(1,4):
                label = "tasks/Nu_{}".format(i)
                nusselt[i-1] = data[label][:,0,0]
                nusselt[i+2] = data[label+"_da"][:,0,0]
        t, nusselt = np.array(times), np.array(nusselt)

        # Calculate time averages (integrate using Simpson's rule).
        nuss_avg = np.array([[simps(nu[:n], t[:n]) for n in range(1,len(t)+1)]
                                                            for nu in nusselt])
        nuss_avg[:,1:] /= t[1:]

        with plt.style.context(".mplstyle"):
            # Plot results in 4 subplots (raw nusselt vs time avg, nonDA vs DA)
            fig = plt.figure(figsize=(12,6))
            ax1 = plt.subplot2grid((2,4), (0,0))
            ax2 = plt.subplot2grid((2,4), (0,1), sharey=ax1)
            ax3 = plt.subplot2grid((2,4), (1,0))
            ax4 = plt.subplot2grid((2,4), (1,1), sharey=ax3)
            axbig = plt.subplot2grid((2,4), (0,2), rowspan=2, colspan=2)
            for i in [0,1,2]:
                ax1.plot(t[1:], nusselt[i,1:])
                ax3.plot(t[1:], nuss_avg[i,1:])
                ax2.plot(t[1:], nusselt[i+3,1:])
                ax4.plot(t[1:], nuss_avg[i+3,1:])
            axbig.plot(t[1:], nuss_avg[:3,1:].mean(axis=0),
                       label='Data ("Truth")')
            axbig.plot(t[1:], nuss_avg[3:,1:].mean(axis=0),
                       label="Assimilating System")
            ax1.set_title("Raw Nusselt", fontsize=8)
            ax3.set_title("Time Average", fontsize=8)
            ax2.set_title("DA Raw Nusselt", fontsize=8)
            ax4.set_title("DA Time Average", fontsize=8)
            axbig.set_title("Overlay of Mean Time Averages", fontsize=8)
            axbig.legend(loc="lower right")
            plt.tight_layout()

            if savefig:
                outfile = os.path.join(self.records_dir, "nusselt.pdf")
                plt.savefig(outfile, dpi=300, bbox_inches="tight")
                self.logger.info("\tFigure saved as '{}'".format(outfile))
            else:
                plt.show()
            plt.close()

    def animate_temperature(self, max_frames=np.inf, fps=100):
        """Animate the temperature results of the simulation (model and DA
        system) and save it to an mp4 file called 'temperature.mp4'.
        """
        # self.merge_results()
        state_file = self._get_merged_file("states")
        self.logger.info("Creating temperature animation from '{}'...".format(
                                                                state_file))

        # Set up the figure / movie writer.
        fig = plt.figure(figsize=(12,6))
        ax1 = plt.subplot2grid((2,2), (0,0))
        ax2 = plt.subplot2grid((2,2), (0,1))
        ax4 = plt.subplot2grid((2,2), (1,0), colspan=2)
        # fig, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2, 2)
        ax1.axis("off"); ax2.axis("off") #; ax3.axis("off")
        ax1.set_title('Data ("Truth")')
        ax2.set_title("Assimilating System")
        # ax3.set_title("Projected Temperature Difference", fontsize=8)
        writer = mplwriters["ffmpeg"](fps=fps) # frames per second, sets speed.

        # Rename the old animation if it exists (it will be deleted later).
        outfile = os.path.join(self.records_dir, "temperature.mp4")
        oldfile = os.path.join(self.records_dir, "old_temperature.mp4")
        if os.path.isfile(outfile):
            self.logger.info("\tRenaming old animation '{}' -> '{}'".format(
                                                            outfile, oldfile))
            os.rename(outfile, oldfile)

        # Write the movie at 200 DPI (resolution).
        with writer.saving(fig,outfile,200), h5py.File(state_file,'r') as data:
            print("Extracting data...", end="", flush=True)
            T = data["tasks/T"]
            T_ = data["tasks/T_"]
            # dT = data["tasks/P_N"]
            times = list(data["scales/sim_time"])
            assert len(times) == len(T) == len(T_), "mismatched dimensions"
            print("done")

            # Plot ||T_ - T||_L^infinity.
            print("Calculating / plotting ||T_ - T||_L^infty(Omega)...",
                  end='', flush=True)
            L_inf = np.max(np.abs(T_[:] - T[:]), axis=(1,2))
            ax4.semilogy(times, L_inf, lw=1)
            ax4_line = plt.axvline(x=times[0], color='r', lw=.5)
            _, ylims = ax4_line.get_data()
            ax4.set_xlim(times[0], times[-1])
            ax4.set_ylim(1e-11, 1e1)
            ax4.set_title(r"$||\tilde{T} - T||_{L^\infty(\Omega)} =$" \
                          + "{:.2e}".format(L_inf[0]))
            ax4.spines["right"].set_visible(False)
            ax4.spines["top"].set_visible(False)
            ax4.set_xlabel(r"Simulation Time $t$")
            print("done")

            # Set up color maps for each temperature layer.
            im1 = ax1.imshow( T[0].T, animated=True, cmap="inferno",
                             vmin=0, vmax=1)
            im2 = ax2.imshow(T_[0].T, animated=True, cmap="inferno",
                             vmin=0, vmax=1)
            # im3 = ax3.imshow(dT[0].T, animated=True, cmap="RdBu_r",
            #                  vmin=-.05, vmax=.05)
                             # norm=SymLogNorm(linthresh=1e-10, vmin=-1, vmax=1))
            # im3 = ax3.imshow(np.log(np.abs(T[0] - T_[0]) + 1e-16).T,
            #                  animated=True, cmap="viridis") # log difference
            # fig.colorbar(im3, ax=ax3, fraction=0.023)
            ax1.invert_yaxis() # Flip the images right-side up.
            ax2.invert_yaxis()
            # ax3.invert_yaxis()

            # Save a frame for each layer of task data.
            for j in tqdm(range(min(T.shape[0], max_frames))):
                im1.set_array( T[j].T)     # Truth
                im2.set_array(T_[j].T)     # Approximation
                # im3.set_array(dT[j].T)     # Difference
                # im3.set_array(np.log(np.abs(T[j] - T_[j]) + 1e-16).T)

                # Moving line for ||T - T_||_L^infty error plot.
                t = times[j]
                ax4_line.set_data([[t,t], ylims])
                ax4.set_title(r"$||(\tilde{T}-T)(t)||_{L^\infty(\Omega)} =$" \
                              + "{:.2e}".format(L_inf[j]))
                writer.grab_frame()
        self.logger.info("\tAnimation saved as '{}'".format(outfile))
        plt.close()

        # Delete the old animation.
        if os.path.isfile(oldfile):
            self.logger.info("\tDeleting old animation '{}'".format(oldfile))
            os.remove(oldfile)

    def _cluster(self, index=0, tasks=["T"], nsamples=1000):
        """Stack the specified tasks from a single given datafile at one index.
        Cluster the data with a KMeans classifier into two groups (k=2).
        Return a mask of shape (xsize, zsize) specifying the groups.

        Parameters:
            TODO
        """
        np.random.seed(11181991)
        datafile = self._get_merged_file("states")

        # Pull data 'pixel' data from the file.
        with h5py.File(datafile, 'r') as data:
            pixels  = data["tasks/T"][index]
            pixels_ = data["tasks/T_"][index]
            # TODO: stack the pixels with other data, depending on kwargs
            # TODO: differentiate between 1-D cases and n-D cases

        M,N = pixels.shape[:2]
        num_pixels = M*N
        pixels = np.ravel(pixels) # alter for n-d.
        nsamples = min(nsamples, num_pixels)

        # Train the kmeans cluster algorithm on the model data.
        kmeans = KMeans(n_clusters=2)
        sample_indices = np.random.choice(np.arange(num_pixels),
                                          size=nsamples, replace=False)
        kmeans.fit(pixels[sample_indices].reshape((-1,1)))

        # Get labels for the model and data assimilation systems.
        md_labels = kmeans.predict(pixels.reshape((-1, 1)))
        DA_labels = kmeans.predict(np.ravel(pixels_).reshape((-1, 1)))
        md_image = np.reshape(md_labels == md_labels[0], (M,N))
        DA_image = np.reshape(DA_labels == md_labels[0], (M,N))

        return md_image, DA_image

    def animate_clusters(self, tasks=["T"], max_frames=np.inf):
        """TODO"""
        # self.merge_results()
        state_file = self._get_merged_file("states")
        self.logger.info(
                "Creating clustered temperature animation from {}...".format(
                                                                state_file))
        # Set up the figure / movie writer.
        fig, [ax1, ax2] = plt.subplots(2, 1)
        ax1.axis("off"); ax2.axis("off")
        ax1.set_title('Data ("Truth")', fontsize=8)
        ax2.set_title("Assimilating System", fontsize=8)
        writer = mplwriters["ffmpeg"](fps=25) # frames per second, sets speed.

        # Remove the old animation if it exists.
        outfile = os.path.join(self.records_dir, "clusters.mp4")
        oldfile = os.path.join(self.records_dir, "old_clusters.mp4")
        if os.path.isfile(outfile):
            self.logger.info("\tRenaming old animation '{}' -> '{}'".format(
                                                            outfile, oldfile))
            os.rename(outfile, oldfile)

        # Write the movie at 200 DPI (resolution).
        with writer.saving(fig, outfile, 200):
            with h5py.File(state_file, 'r') as data:
                num_indices = min(data["tasks/T"].shape[0], max_frames)
            # Save a frame for each layer of data.
            mask, mask_ = self._cluster(0, tasks)
            im1 = ax1.imshow( mask.T, animated=True, cmap="gray")
            im2 = ax2.imshow(mask_.T, animated=True, cmap="gray")
            writer.grab_frame()
            for i in tqdm(range(num_indices)):
                if i == 0:          # First iteration is already done,
                    continue        # but trick tqdm into showing it.
                mask, mask_ = self._cluster(i, tasks)
                im1.set_array( mask.T)
                im2.set_array(mask_.T)
                writer.grab_frame()
        self.logger.info("\tAnimation saved as '{}'".format(outfile))
        plt.close()

        # Delete the old animation.
        if os.path.isfile(oldfile):
            self.logger.info("\tDeleting old animation '{}'".format(oldfile))
            os.remove(oldfile)

    def process_results(self):
        """Call all post-processing methods."""
        self.merge_results()
        if RANK == 0:
            self.plot_convergence()
            self.plot_nusselt()
            self.animate_temperature()
            self.animate_clusters()


class BoussinesqDataAssimilation2Dmovie(BoussinesqDataAssimilation2D):
    """Same as BoussinesqDataAssimilation2D, but simulate() never saves
    analysis files and always saves the temperature fields at every iteration.
    """
    def simulate(self, initial_conditions=None, scheme=de.timesteppers.RK443,
                    sim_time=2, wall_time=np.inf, tight=False):
        """Load initial conditions, run the simulation, and merge results.

        Parameters:
            initial_conditions (None, str): determines from what source to
                draw the initial conditions. Valid options are as follows:
                - None: use trivial conditions (T_ = 1 - z, T = 1 - z + eps).
                - 'resume': use the most recent state file in the
                    records directory (load both model and DA system).
                - An .h5 filename: load state variables for the model and
                    reset the data assimilation state variables to zero.
            scheme (de.timesteppers): The kind of solver to use. Options are
                RK443 (de.timesteppers.RK443), RK111, RK222, RKSMR, etc.
            sim_time (float): The maximum amount of simulation time allowed
                (in seconds) before ending the simulation.
            wall_time (float): The maximum amound of computing time allowed
                (in seconds) before ending the simulation.
            tight (bool): If True, set a low cadence and min_dt for refined
                simulation. If False, set a higher cadence and min_dt for a
                more coarse (but faster) simulation.
        """
        if not self.problem:
            raise TypeError("problem not initialized (run setup())")

        self.logger.debug("\n")
        self.logger.debug("NEW SIMULATION")
        solver = self.problem.build_solver(scheme)
        self.logger.info("Solver built")

        N = int(self.problem.parameters['N'])

        # Initial conditions --------------------------------------------------
        if initial_conditions is None:              # "Trivial" conditions.
            dt = 1e-4
            eps = 1e-4
            k = 3.117

            x,z = self.problem.domain.grids(scales=1)
            T, T_ = solver.state['T'], solver.state['T_']
            # Start T from rest plus a small perturbation.
            T['g']  = 1 - z + eps*np.sin(k*x)*np.sin(2*np.pi*z)
            T.differentiate('z', out=solver.state['Tz'])
            # Start T_ from rest.
            T_['g'] = 1 - z
            T_.differentiate('z', out=solver.state['Tz_'])
            self.logger.info("Using trivial initial conditions")

        elif isinstance(initial_conditions, str):   # Load data from a file.
            # Resume: load the state of the last (merged) state file.
            resume = initial_conditions == "resume"
            if resume:
                initial_conditions = self._get_merged_file("states")
            if not initial_conditions.endswith(".h5"):
                raise ValueError("'{}' is not an h5 file".format(
                                                        initial_conditions))
            # Load the data from the specified h5 file into the system.
            self.logger.info("Loading initial conditions from {}".format(
                                                        initial_conditions))

            with h5py.File(initial_conditions, 'r') as infile:
                dt = infile["scales/timestep"][-1] * .01    # initial dt
                errs = []
                tasks = ["T", "Tz", "psi", "psiz", "zeta", "zetaz"]
                if resume:      # Only load assimilating variables to resume.
                    tasks += ["T_", "Tz_", "psi_", "psiz_", "zeta_", "zetaz_"]
                    solver.sim_time = infile["scales/sim_time"][-1]
                    niters = infile["scales/iteration"][-1]
                    solver.initial_iteration = niters
                    solver.iteration = niters
                for name in tasks:
                    # Get task data from the h5 file (recording failures).
                    try:
                        data = infile["tasks/"+name][-1,:,:]
                    except KeyError as e:
                        errs.append("tasks/"+name)
                        continue
                    # Determine the chunk belonging to this process.
                    chunk = data.shape[1] // SIZE
                    subset = data[:,RANK*chunk:(RANK+1)*chunk]
                    # Change the corresponding state variable.
                    scale = solver.state[name]['g'].shape[0] / \
                                        self.problem.parameters["xsize"]
                    solver.state[name].set_scales(1)
                    solver.state[name]['g'] = subset
                    solver.state[name].set_scales(scale)
                if errs:
                    raise KeyError("Missing keys in '{}': '{}'".format(
                                    initial_conditions, "', '".join(errs)))
            # Initial conditions for assimilating system: T_0 = P_4(T0).
            if not resume:
                G = self.problem.domain.new_field()
                G['c'] = solver.state['T']['c'].copy()
                solver.state['T_']['g'] = BoussinesqDataAssimilation2D.P_N(
                                                                    G, 4, True)
                solver.state['T_'].differentiate('z', out=solver.state['Tz_'])

        # Driving / projection function arguments -----------------------------

        dT = solver.state['T_'] - solver.state['T']
        self.problem.parameters["driving"].args = [dT, N]
        self.problem.parameters["driving"].original_args = [dT, N]

        # Stopping Parameters -------------------------------------------------

        solver.stop_sim_time = sim_time         # Length of simulation.
        solver.stop_wall_time = wall_time       # Real time allowed to compute.
        solver.stop_iteration = np.inf          # Maximum iterations allowed.

        # State snapshots -----------------------------------------------------

        # Save the entire state in states/ files. USE iter, NOT sim_dt.
        # NOTE: This is where BoussinesqDataAssimilation2Dmovie differs.
        snaps = solver.evaluator.add_file_handler(
                                os.path.join(self.records_dir, "states"),
                                iter=1, max_writes=5000, mode="append")
        snaps.add_task("T")
        snaps.add_task("T_")
        snaps.add_task("driving", name="P_N")

        # Control Flow --------------------------------------------------------
        if tight:                   # Tighter control flow (slower but safer).
            cfl = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=1,
                                            max_change=1.5, min_change=0.01,
                                            max_dt=0.01,    min_dt=1e-10)
        else:                       # Looser control flow (faster but risky).
            cfl = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=1,
                                            max_change=1.5, min_change=0.5,
                                            max_dt=0.01,    min_dt=1e-6)
        cfl.add_velocities(('v',  'w' ))
        cfl.add_velocities(('v_', 'w_'))

        # Flow properties (print during run; not recorded in the records files)
        flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
        flow.add_property("sqrt(v **2 + w **2) / Ra", name='Re' )
        flow.add_property("sqrt(v_**2 + w_**2) / Ra", name='Re_')

        # MAIN COMPUTATION LOOP -----------------------------------------------
        try:
            self.logger.info("Starting main loop")
            start_time = time.time()
            while solver.ok:
                # Recompute time step and iterate.
                dt = cfl.compute_dt()
                dt = solver.step(dt)

                # Print info to the screen every 10 iterations.
                if solver.iteration % 10 == 0:
                    info = "Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}".format(
                                        solver.iteration, solver.sim_time, dt)
                    Re  = flow.max("Re")
                    Re_ = flow.max("Re_")
                    info += ", Max Re = {:f}".format(Re)
                    info += ", Max Re_ = {:f}".format(Re_)
                    self.logger.info(info)
                    # Make sure the simulation hasn't blown up.
                    if np.isnan(Re) or np.isnan(Re_):
                        raise ValueError("Reynolds number went to infinity!!"
                                         "\nRe = {}, Re_ = {}".format(Re, Re_))
        except BaseException as e:
            self.logger.error("Exception raised, triggering end of main loop.")
            raise
        finally:
            total_time = time.time() - start_time
            cpu_hr = total_time / 60 / 60 * SIZE
            self.logger.info("Iterations: {:d}".format(solver.iteration))
            self.logger.info("Sim end time: {:.3e}".format(solver.sim_time))
            self.logger.info("Run time: {:.3e} sec".format(total_time))
            self.logger.info("Run time: {:.3e} cpu-hr".format(cpu_hr))
            self.logger.debug("END OF SIMULATION\n")
