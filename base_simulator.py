# base_simulator.py
"""Base class / variables for Dedalus simulations with good file management.

Author: Shane McQuarrie
"""

import os
import re
import json
import time
import logging
from glob import glob
from mpi4py import MPI

from dedalus.tools import post

# Global variables ============================================================

RANK = MPI.COMM_WORLD.rank        # Which process this is running on
SIZE = MPI.COMM_WORLD.size        # How many processes are running

# Log file formatting
LOG_FORMAT = "(%(asctime)s, {:0>2}/{:0>2}) %(levelname)s: %(message)s".format(
                                                                RANK+1, SIZE)
LOG_FILE = "process{:0>2}.log".format(RANK)
logging.basicConfig(format=LOG_FORMAT)

# Regex for finding data files
FILE_INDEX = re.compile(r"_s(\d+?)(?:_p\d)?\.h5$")

# Parameter storage filename
PARAMS = "params.json"

# Simulation Class ============================================================

class BaseSimulator:
    def __init__(self, records_dir=None, log=True):
        """Store variables for record keeping.

        Parameters:
            records_dir (str): the directory in which all simulation data
                is stored or retrieved. If None (default), the directory is
                B2D__date_mm_dd_yyyy__time_hh_mm/, which depends on the day
                and time of creation.
        """
        self._name = type(self).__name__
        if not records_dir:             # Default: directory name by timestamp.
            records_dir = self._name + time.strftime("__%m_%d_%Y__%H_%M")
        records_dir = records_dir.strip(os.sep)
        if os.path.isdir(records_dir):  # Notify user if directory exists.
            print("Connecting to existing directory {}".format(records_dir))
        else:                           # Default: make new directory.
            try:
                os.mkdir(records_dir)
                print("Created new directory {}".format(records_dir))
            except FileExistsError:     # Other process might create it first.
                pass

        # Log information to a logfile in the records directory.
        logger = logging.getLogger(records_dir + str(RANK))
        logger.setLevel(logging.DEBUG)
        if log and len(logger.handlers) == 0:
            logfile = logging.FileHandler(os.path.join(records_dir, LOG_FILE))
            logfile.setFormatter(logging.Formatter(LOG_FORMAT))
            logfile.setLevel(logging.DEBUG)
            logger.addHandler(logfile)

        # Store variables.
        self.records_dir = records_dir
        self.params_file = os.path.join(records_dir, PARAMS)
        self.logger = logger
        self.problem = None

        # If the parameter file already exists, load and print it.
        if os.path.isfile(self.params_file) and RANK == 0:
            self.logger.info("Previous simulation parameters:")
            for key, value in self._load_params(self.records_dir).items():
                self.logger.info("\t'{}': {}".format(key, value))

    def setup(*args, **kwargs):
        """Define the dedalus problem (de.IVP, de.EVP, de.LBVP, or de.NLBVP),
        including variables, substitutions, equations, and boundary conditions.

        IMPORTANT: Save the dedalus problem object as 'self.problem'.
        """
        raise NotImplementedError("setup() must be implemented in subclasses")

    def simulate(*args, **kwargs):
        """Create a solver (self.problem.build_solver(scheme)), set initial
        conditions on the problem (if appropriate), set up analyses, and run
        the actual simulation.
        """
        raise NotImplementedError("simulate() must be implemented in"
                                                                " subclasses")

    def _save_params(self):
        """Save a dictionary of problem parameters to a readable JSON file.
        If such a file already exists, compare the current problem parameters
        to the old parameters, log any differences, and overwrite the file.
        """
        if not self.problem:
            raise NotImplementedError("Problem not initialized (call setup())")

        def JSONish(x):
            """Return True if x is JSON serializable, False otherwise."""
            try:
                json.dumps(x)
                return True
            except TypeError:
                return False

        params = {k:v for k,v in self.problem.parameters.items() if JSONish(v)}
        self.logger.info("Writing parameters to '{}'".format(self.params_file))
        for key, value in params.items():
            self.logger.info("\t'{}': {}".format(key, value))

        # If there is already a params file, compare old and new parameters.
        if os.path.isfile(self.params_file):
            old_params = self._load_params(self.records_dir)
            if old_params != params:
                self.logger.info("Saved parameters updated")
            old_keys, new_keys = set(old_params.keys()), set(params.keys())
            # Report missing or new parameter keys.
            for key in old_keys - new_keys:
                self.logger.info("\tOld param '{}' removed (was {})".format(
                                                        key, old_keys[key]))
            for key in new_keys - old_keys:
                self.logger.info("\tNew param '{}'".format(key))
            # Report changed parameters.
            for key in old_keys & new_keys:
                old, new = old_params[key], params[key]
                if old != new:
                    self.logger.info("\tParam '{}' changed: {} -> {}".format(
                                                                key, old, new))

        with open(self.params_file, 'w') as outfile:
            json.dump(params, outfile)

    @staticmethod
    def _load_params(records_dir):
        """Load a dictionary of problem parameters from a given directory."""
        # Make sure the directory exists.
        if not os.path.isdir(records_dir):
            raise NotADirectoryError(records_dir)

        # Load the parameters.
        with open(os.path.join(records_dir, PARAMS), 'r') as infile:
            return json.load(infile)

    @staticmethod
    def _file_index(f):
        """Get the index of a dedalus h5 file. For example,
        states_s1.h5 -> 1, analysis_s10.h5 -> 10, etc.
        """
        out = FILE_INDEX.findall(f)
        try:
            return int(out[0])
        except IndexError:
            return -1

    def get_files(self, label):
        """Return a sorted list of the merged h5 data files.

        Parameters:
            label (str): The name of the subdirectory containing merged .h5
                files. For example, if label="states", there should be a folder
                self.records_dir/states/ containing at least one .h5 file.

        Raises:
            NotADirectoryError: if the specified subdirectory does not exist.
            FileNotFoundError: if the specified subdirectory exists but there
                are no matching files.
        """
        # Check that the relevant folder exists.
        subdir = os.path.join(self.records_dir, label)
        if not os.path.isdir(subdir):
            raise NotADirectoryError(subdir)

        # Get the list of files.
        out = sorted(glob(os.path.join(subdir, "*.h5")), key=self._file_index)
        if len(out) == 0:
            raise FileNotFoundError("no {} files found".format(label))

        return out

    def merge_results(self, label, full_merge=False, force=False):
        """Merge the different process result files together.

        Parameters:
            label (str): The name of the subdirectory containing folders where
                each process computes results. For example, if label="states",
                then self.records_dir/states/ should exist and contain at least
                one subfolder named states_s1/ (or similar), which in turn
                contains .h5 files from each process (states_s1_p0.h5, etc.).
            full_merge (bool): If true, merge the process files AND merge
                the resulting files into one large file. For example,
                states_s1_p0.h5 and states_s1_p1.h5 are merged into
                states_s1.h5 like usual, and then states_s1.h5 and states_s2.h5
                are merged into one large states.h5.
            force (bool): If true, merge the files even if there is an existing
                merged file.
        """
        # Check that the relevant folder exists.
        subdir = os.path.join(self.records_dir, label)
        if not os.path.isdir(subdir):
            raise NotADirectoryError(subdir)

        # Check for existing folders and missing files before merging.
        work_todo = False
        if full_merge:
            work_todo = not os.path.isfile(os.path.join(subdir, label+".h5"))
        else:
            for d in os.listdir(subdir):
                target = os.path.join(subdir, d)
                if os.path.isdir(target) and not os.path.isfile(target+".h5"):
                    work_todo = True
                    break
        if work_todo or force:
            self.logger.info("Merging {} files...".format(label))
            post.merge_process_files(subdir,cleanup=False,comm=MPI.COMM_WORLD)
            if full_merge:
                # Wait for other processes to finish.
                MPI.COMM_WORLD.Barrier()
                # Do the last merge.
                set_paths = glob(os.path.join(subdir, label+"_s*.h5"))
                post.merge_sets(os.path.join(subdir, label+".h5"),
                                set_paths, cleanup=True, comm=MPI.COMM_WORLD)
            self.logger.info("\t{} files now {}merged".format(label,
                                            "fully " if full_merge else ""))

    def __str__(self):
        """String representation: the raw equations, boundary conditions, and
        parameters of the dedalus problem object.
        """
        if not self.problem:
            return "Problem not initialized (call setup())"
        out = self._name + " System\n\nEquations:\n\t"
        out += "\n\t".join([q["raw_equation"] for q in self.problem.equations])
        out += "\nBoundary Conditions:\n\t"
        out += "\n\t".join([q["raw_equation"] for q in self.problem.bcs])
        out += "\nParameters:\n\t"
        out += "\n\t".join("{}: {}".format(key,value)
                            for key,value in self.problem.parameters.items())
        return out


def all_experiment_params():
    """Get a dictionary mapping foldername to parameter dictionaries, as given
    by BaseSimulator._load_params().
    """
    return {f: BaseSimulator._load_params(f) for f in os.listdir()
               if os.path.isdir(f) and os.path.isfile(os.path.join(f, PARAMS))}
