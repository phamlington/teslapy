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
from single_comm_functions import *
import sys
comm = MPI.COMM_WORLD


###############################################################################
def derived_variable_slices(args):
    if comm.rank == 0:
        print("Python MPI job `statistics_serial' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, tmp0) = args

    prefix = ['Density', 'Velocity1', 'Velocity2', 'Velocity3',
              'Total_Energy', 'Scalar0', 'Velocity_Perturbations1',
              'Velocity_Perturbations2', 'Velocity_Perturbations3']

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
    #             mpi_comm=comm, odir=ddir, ndims=3,
    #             decomp=[True, False, False], nx=[N]*3, nh=None,
    #             periodic=[True]*3, byteswap=False)

    update = '{0}\t{1}\t{2}\t{3:4d}\t'.format
    fmt = '{0}/{0}_{1}_{2}.bin'.format

    s = list(analyzer.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)

    # psum = mpiAnalyzer.psum
    # MMIN = MPI.MIN
    MMAX = MPI.MAX
    # MSUM = MPI.SUM

    # B = 6.85e12
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
                       +'variables loaded into memory')

            rek = 0.5*rho*np.sum(np.square(u), axis=0)
            rei = re-rek
            P = rei*(gamma-1)
            T = P/(rho*R)
            # Y = rY/rho
            dil2 = np.square(analyzer.div(u))
            Enst = 0.5*np.sum(np.square(analyzer.curl(u)), axis=0)
            dTdT = np.sum(np.square(analyzer.scl_grad(T)), axis=0)

            smax0 = np.max(dil2)
            emax0 = comm.allreduce(smax0, op=MMAX)

            smax1 = np.max(Enst)
            emax1 = comm.allreduce(smax1, op=MMAX)

            smax2 = np.max(dTdT)
            emax2 = comm.allreduce(smax2, op=MMAX)

            if comm.rank == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'emax0, emax1, emax2: {}\t{}\t{}'
                       .format(emax0, emax1, emax2))

            if ((abs(smax0-emax0)/emax0 < 1.0e-2)
                or (abs(smax1-emax1)/emax1 < 1.0e-2)
                or (abs(smax2-emax2)/emax2 < 1.0e-2)):

                with open(analyzer.odir+analyzer.probID
                          +'-slices_{}_{}_{}.bin'
                          .format(tstep, ir, comm.rank), 'w') as fh:
                    rho[0].tofile(fh)
                    T[0].tofile(fh)
                    dil2[0].tofile(fh)
                    Enst[0].tofile(fh)
                    dTdT[0].tofile(fh)

            comm.Barrier()

            # minY = np.min(Y)
            # maxY = np.max(Y)

    if comm.rank == 0:
        print ("Python MPI job `derived_variable_slices'"
               " finished at "+timeofday())

if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    derived_variable_slices(get_inputs())


# # ------------------------------------------------------------

# print (update(timeofday(), tstep, ir, comm.rank)
#        +'found an autoignition kernel!')

# # ------------------------------------------------------------

# # writer.variable_subdomain(fmt(prefix[5], tstep, ir), T)
# # writer.variable_subdomain(fmt(prefix[6], tstep, ir), P)
# # writer.variable_subdomain(fmt(prefix[7], tstep, ir), Skk)

# # Tp = reader.variable_subdomain(fmt(prefix[5], tstep, ir))
# # L2 = np.sqrt(analyzer.psum(np.square(T-Tp)))

# # print(update(timeofday(), tstep, ir, comm.rank)
# #       +'L2 Error for T: {}'.format(L2))
