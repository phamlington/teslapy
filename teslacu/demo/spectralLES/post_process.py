"""
Analysis of compressible, ideal gas, HIT using draft versions of TESLaCU
python modules.

Command Line Options:
---------------------
-i <input directory>    default: 'data/'
-o <output directory>   default: 'analysis/'
-p <problem ID>         defualt: 'no_problem_id'
-N <Nx>                 default: 512
-g <gamma>              default: 1.4
-L <L>                  default: 1.0
-R <R>                  default: 8.3144598e7/21
-r <irs:ire:rint>       default: 1:20:1
-t <its:ite:tint>       default: 1:20:1
--Texp <texp>           default: 0.7
--Tcoef <tcoef>         default: 3.1e-6
--Tmp0 <tmp0>           default: 293.0

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
# from math import sqrt
import sys
from teslacu import mpiAnalyzer, mpiReader, get_inputs, timeofday

comm = MPI.COMM_WORLD


def spectralLES_post_process(args):
    if comm.rank == 0:
        print("Python MPI job `spectralLES_post_process' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, T_0) = args

    # -------------------------------------------------------------------------
    # Divide COMM_WORLD amongst the data snapshots

    if N % comm.size > 0:
        if comm.rank == 0:
            print('Job started with improper number of MPI tasks for the '
                  'size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Generate a data reader and analyzer with the appropriate MPI comms

    reader = mpiReader(comm=comm, idir=idir, ndims=3,
                       decomp=[True, False, False], nx=[N]*3, nh=None,
                       periodic=[True]*3, byteswap=False)

    analyzer = mpiAnalyzer(comm=comm, idir=idir, odir=odir, probID=pid,
                           L=L, nx=N, geo='hit', method='fourier')

    update = '{0}\t{1}\t{2}\t{3:4d}\t%s'.format
    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    s = list(analyzer.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)
    analyzer.tol = 1.0e-16

    for it in range(its, ite+1, tint):
        tstep = str(it).zfill(3)

        u[0] = reader.read_variable('Velocity1_%s.rst' % tstep,
                                    ftype=np.float64)
        u[1] = reader.read_variable('Velocity2_%s.rst' % tstep,
                                    ftype=np.float64)
        u[2] = reader.read_variable('Velocity3_%s.rst' % tstep,
                                    ftype=np.float64)

        if comm.rank % 64 == 0:
            print(update(timeofday(), tstep, 0, comm.rank)
                  % 'variables loaded into memory')

        analyzer.spectral_density(u, 'u', 'velocity PSD', Ek_fmt('u_i'))

        # if comm.rank == 0:

    if comm.rank == 0:
        print("Python MPI job `spectralLES_post_process'"
              " finished at "+timeofday())


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    spectralLES_post_process(get_inputs())
