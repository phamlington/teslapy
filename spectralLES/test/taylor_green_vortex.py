"""
Description:
------------
Simulation and analysis of incompressible Taylor-Green vortex
using draft versions of TESLaCU Python modules and routines from the
spectralDNS3D_short Taylor-Green Vortex simulation code written by Mikael
Mortensen (<https://github.com/spectralDNS/spectralDNS/spectralDNS3D_short.py>)

Notes:
------

Definitions:
------------

Authors:
--------
Colin Towery, colin.towery@colorado.edu

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
"""

from mpi4py import MPI
import numpy as np
from spectralLES import spectralLES
import sys
# from memory_profiler import profile
# import os
import time
comm = MPI.COMM_WORLD


def timeofday():
    return time.strftime("%H:%M:%S")


def taylor_green_vortex():
    if comm.rank == 0:
        print("Python MPI spectral DNS simulation of problem "
              "`Taylor-Green vortex' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    # -------------------------------------------------------------------------

    L = 2*np.pi
    N = 64
    nu = 0.000625

    if N % comm.size > 0:
        if comm.rank == 0:
            print('Job started with improper number of MPI tasks for the '
                  'size of the data specified!', flush=True)
        MPI.Finalize()
        sys.exit(1)

    solver = spectralLES(comm, N, L, nu, epsilon=0, Gtype='spectral')

    # -------------------------------------------------------------------------

    t = 0.0
    dt= 0.01
    tlimit= 1.0
    tstep = 0

    start = time.time()

    solver.initialize_TaylorGreen_vortex()
    solver.computeAD = solver.computeAD_vorticity_form

    while t < tlimit-1.e-8:
        k = comm.reduce(0.5*np.sum(np.square(solver.U)*(1./N)**3))
        if comm.rank == 0:
            print('cycle = %2.0d, KE = %12.10f' % (tstep, k), flush=True)

        t += dt
        tstep += 1
        solver.RK4_integrate(dt)

    k = comm.reduce(0.5*np.sum(np.square(solver.U)*(1./N)**3))
    k_true = 0.12451526736699045  # from spectralDNS3D_short.py run until T=1.0
    if comm.rank == 0:
        print("Time = %12.8f, KE = %16.13f" % (time.time()-start, k),
              flush=True)

        # assert that the two codes must be within single-precision round-off
        # error of each other
        assert round(abs(k - k_true), 7) < 1.e-7

        # if code passes assertion then output the relative error in codes for
        # proper bragging rights
        print("relative error in avg. KE compared to spectralDNS3D_short.py: "
              "{}".format(abs(k - k_true)/k_true))

    return


###############################################################################
if __name__ == "__main__":
    taylor_green_vortex()
