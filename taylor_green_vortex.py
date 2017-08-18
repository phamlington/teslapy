"""
Description:
------------
Simulation and analysis of incompressible Taylor-Green vortex
using draft versions of TESLaCU Python modules and routines from the
spectralDNS3D_short Taylor-Green Vortex simulation code written by Mikael
Mortensen (<https://github.com/spectralDNS/spectralDNS/spectralDNS3D_short.py>)

Command Line Options:
---------------------
-i <input directory>    default: 'data/'
-o <output directory>   default: 'analysis/'
-p <problem ID>         defualt: 'no_problem_id'
-N <Nx>                 default: 64
-L <L>                  default: 2*pi
-t <tlimit>             default: 0.1

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
# import mpiAnalyzer
# import mpiReader
# import mpiWriter
from spectralLES import *
from teslacu.fft_mpi4py_numpy import *  # FFT transforms
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
            print ('Job started with improper number of MPI tasks for the '
                   'size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Generate a spectralLES solver, data writer, and data analyzer with the
    # appropriate class factories, MPI communicators, and paramters

    # writer = mpiReader.mpiBinaryWriter(
    #             mpi_comm=comm, idir=idir, ndims=3,
    #             decomp=[True, False, False], nx=[N]*3, nh=None,
    #             periodic=[True]*3, byteswap=False)

    # analyzer = mpiAnalyzer.factory(
    #             comm=comm, idir=idir, odir=odir, probID=pid, L=L, nx=N,
    #             geo='hit', method='spectral')

    solver = spectralLES(comm, L, N, nu)

    # -------------------------------------------------------------------------
    # Initialize the simulation

    t = 0.0
    tstep = 0
    dt = 0.01
    tlimit = 8*np.pi
    t0 = time.time()

    solver.Initialize_Taylor_Green_vortex()
    solver.computeAD = solver.computeAD_vorticity_formulation

    while t < tlimit-1.e-8:
        if tstep % 10 == 0:
            k = comm.reduce(0.5*np.sum(np.square(solver.U))*(1./N)**3)
            if comm.rank == 0:
                print "{}\t{}".format(k, dt)

        t += dt
        tstep += 1
        solver.RK4_integrate(dt)

        # Update the dynamic dt based on CFL constraint
        dt = solver.new_dt_const_nu(0.45)

    solver.U[0] = irfft3(comm, solver.U_hat[0])
    solver.U[1] = irfft3(comm, solver.U_hat[1])
    solver.U[2] = irfft3(comm, solver.U_hat[2])

    k = 0.5*comm.reduce(np.sum(np.square(solver.U))*(1./N)**3)
    k_true = 0.124515267367  # from spectralDNS3D_short.py run until T=1.0
    if comm.rank == 0:
        print("Time = {}".format(time.time()-t0))

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
