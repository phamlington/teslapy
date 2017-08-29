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
import mpiAnalyzer
import mpiReader
# import mpiWriter
from single_comm_functions import *
import sys
comm = MPI.COMM_WORLD


###############################################################################
def derived_variable_preprocess(args):
    if comm.rank == 0:
        print("Python MPI job `derived_variable_preprocess' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, T_0) = args

    # nr = (ire-irs)/rint + 1
    # nt = (ite-its)/tint + 1
    # Ne = nr*N**3

    prefix = ['Density', 'Velocity1', 'Velocity2', 'Velocity3',
              'Total_Energy', 'Scalar0',
              'Pressure', 'Reaction', 'Diffusion', 'Momentum', 'Dilatation']

    # -------------------------------------------------------------------------
    # Divide COMM_WORLD amongst the data snapshots

    if N % comm.size > 0:
        if comm.rank == 0:
            print ('Job started with improper number of MPI tasks for the '
                   'size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Generate a data reader and analyzer with the appropriate MPI comms

    reader = mpiReader.mpiBinaryReader(
                mpi_comm=comm, idir=idir, ndims=3,
                decomp=[True, False, False], nx=[N]*3, nh=None,
                periodic=[True]*3, byteswap=False)

    analyzer = mpiAnalyzer.factory(
                comm=comm, idir=idir, odir=odir, probID=pid, L=L, nx=N,
                geo='hit', method='akima')

    # writer = mpiWriter.mpiBinaryWriter(
    #             mpi_comm=comm, odir=odir, ndims=3,
    #             decomp=[True, False, False], nx=[N]*3, nh=None,
    #             periodic=[True]*3, byteswap=False)

    update = '{0}\t{1}\t{2}\t{3:4d}\t%s'.format
    fmt = '{0}/{0}_{1}_{2}.bin'.format

    s = list(analyzer.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)
    analyzer.tol = 1.0e-16

    # B = 1.0e13
    # T_a = 13500

    for it in xrange(its, ite+1, tint):
        tstep = str(it).zfill(4)

        for ir in xrange(irs, ire+1, rint):

            rho = reader.get_variable(fmt(prefix[0], tstep, ir))
            u[0] = reader.get_variable(fmt(prefix[1], tstep, ir))
            u[1] = reader.get_variable(fmt(prefix[2], tstep, ir))
            u[2] = reader.get_variable(fmt(prefix[3], tstep, ir))
            re = reader.get_variable(fmt(prefix[4], tstep, ir))
            # rY = reader.get_variable(fmt(prefix[5], tstep, ir))

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       % 'variables loaded into memory')

            # -------------------------------------------------------------------------
            ru = rho*np.sum(np.square(u), axis=0)
            P = (gamma-1)*(re-0.5*ru)  # pressure
            # T = P/(rho*R)                        # thermodynamic temperature
            # Y = rY/rho
            # rwdot = -rho*rY*B*np.exp(-T_a/T)
            # Smm = analyzer.div(u)
            # mu = tcoef*np.power(T, texp)
            # rD = analyzer.div(mu*analyzer.scl_grad(Y))

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       % 'variables computed')

            print
            print (update(timeofday(), tstep, ir, comm.rank) %
                   ('Pts of P < 0 atm: %d' % np.sum(P <= 0.0)))
            print (update(timeofday(), tstep, ir, comm.rank) %
                   ('Pts of 0 < P < 1 atm: %d' %
                    np.sum((P <= 1.01325e6) & (P > 0.0))))
            print

            print
            print (update(timeofday(), tstep, ir, comm.rank) %
                   ('Pts of rho < 0 K: %d' % np.sum(rho <= 0.0)))
            print (update(timeofday(), tstep, ir, comm.rank) %
                   ('Pts of 0 < rho < 1.0e-3 K: %d' %
                    np.sum((rho <= 1.0e-3) & (rho > 0.0))))
            print

            # writer.set_variable(fmt(prefix[6], tstep, ir), P)
            # writer.set_variable(fmt(prefix[7], tstep, ir), rwdot)
            # writer.set_variable(fmt(prefix[8], tstep, ir), rD)
            # writer.set_variable(fmt(prefix[9], tstep, ir), ru)
            # writer.set_variable(fmt(prefix[10], tstep, ir), Smm)

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'derived variables written to file')

    if comm.rank == 0:
        print ("Python MPI job `derived_variable_preprocess'"
               " finished at "+timeofday())


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    derived_variable_preprocess(get_inputs())
