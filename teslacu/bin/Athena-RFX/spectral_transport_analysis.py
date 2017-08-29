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
import mpi_single_comm_Analyzer
import mpiReader
from single_comm_functions import *
import fft_mpi4py_numpy as tcfft        # FFT transforms and math functions
import sys
# from memory_profiler import profile
# import os
# import time
comm = MPI.COMM_WORLD


# rho = upinterp(rho1, analyzer1)
# s = list(rho.shape)
# s.insert(0, 3)
# u = np.empty(s, dtype=np.float64)
# u[0] = upinterp(u1[0], analyzer1)
# u[1] = upinterp(u1[1], analyzer1)
# u[2] = upinterp(u1[2], analyzer1)
# re = upinterp(re1, analyzer1)
###############################################################################
# @profile
def spec_trans_serial(args):
    if comm.rank == 0:
        print("Python MPI job `spec_trans_serial' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
     texp, tcoef, tmp0) = args

    # nr = (ire-irs)/rint + 1
    # nt = (ite-its)/tint + 1
    prefix = ['Density', 'Velocity1', 'Velocity2', 'Velocity3',
              'Total_Energy', 'Scalar0', 'Velocity_Perturbations1',
              'Velocity_Perturbations2', 'Velocity_Perturbations3']

    # -------------------------------------------------------------------------
    # Divide COMM_WORLD amongst the data snapshots
    # ENS_COMM is the communicator for purely ENSemble statistics (all spatial
    # data over a single timestep)
    # SP_COMM  is the communicator for a single snapshot of SPatial data
    # ROOT_COMM is the (intra)communicator between root tasks of each SP_COMM
    # in an ENS_COMM

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

    analyzer = mpi_single_comm_Analyzer.factory(
                idir=idir, odir=odir, probID=pid, L=L, nx=N, geo='hit',
                method='akima')

    update = '{0}\t{1}\t{2}\t{3:4d}\t'.format
    fmt = '{}_{}_{}.bin'.format

    s = list(analyzer.nnx)
    s.insert(0, 3)
    u = np.empty(s, dtype=np.float64)
    s.insert(0, 3)
    # Rij = np.empty(s, dtype=np.float64)

    # Ghat = filter_kernel(analyzer, 6.0*L/N)

    for it in xrange(its, ite+1, tint):
        tstep = str(it).zfill(4)

        for ir in xrange(irs, ire+1, rint):
            analyzer.prefix = '%s-%s_%s_' % (pid, tstep, str(ir))

            rho = reader.variable_subdomain(fmt(prefix[0], tstep, ir))
            u[0] = reader.variable_subdomain(fmt(prefix[1], tstep, ir))
            u[1] = reader.variable_subdomain(fmt(prefix[2], tstep, ir))
            u[2] = reader.variable_subdomain(fmt(prefix[3], tstep, ir))
            re = reader.variable_subdomain(fmt(prefix[4], tstep, ir))

            # rho = tcfft.irfft3(comm, Ghat*tcfft.rfft3(comm, rho))
            # ruh = Ghat*analyzer.vec_fft(rho*u)   # 'rho-u hat'
            # re = tcfft.irfft3(comm, Ghat*tcfft.rfft3(comm, re))
            # u = analyzer.vec_ifft(ruh)/rho

            if comm.rank % 64 == 0:
                print(update(timeofday(), tstep, ir, comm.rank)
                      +'variables loaded into memory')

            # -----------------------------------------------------------------
            # Compute 'basic' physical-space variables

            P = (re-rho*0.5*np.sum(np.square(u), axis=0))*(gamma-1)
            T = P/(rho*R)
            mu = tcoef*np.power(T, texp)

            re = None
            T = None

            A = analyzer.grad(u)
            S = A

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'physical-space variables computed')

            # -----------------------------------------------------------------
            T3 = np.zeros_like(u)
            ruh = analyzer.vec_fft(rho*u)   # 'rho-u hat'
            uh = analyzer.vec_fft(u)        # 'u hat'
            T3h = np.zeros_like(uh)
            Tk = np.empty((9, analyzer.nk[-1]))

            Tk[0] = 0.25*analyzer.shell_average(np.sum(np.real(
                                                 ruh*np.conj(uh)), axis=0))

            # -----------------------------------------------------------------

            for j in range(3):
                for i in range(3):
                    T3[i] += analyzer.deriv(rho*u[i]*u[j], dim=j)
                    # T3h[i] += 1j*analyzer.K[j]*tcfft.rfft3(comm,
                    # rho*u[i]*u[j])

            T3h = analyzer.vec_fft(T3)
            Tk[1] = 0.5*analyzer.shell_average(np.sum(np.real(
                                                T3h*np.conj(uh)), axis=0))

            # -----------------------------------------------------------------

            T3 = u[0]*A[0]
            T3+= u[1]*A[1]
            T3+= u[2]*A[2]
            T3h = analyzer.vec_fft(T3)
            Tk[2] = 0.5*analyzer.shell_average(np.sum(np.real(
                                               T3h*np.conj(ruh)), axis=0))

            # -----------------------------------------------------------------

            T3 = analyzer.scl_grad(P)
            T3h = analyzer.vec_fft(T3)
            # Ph = tcfft.rfft3(comm, P)
            # T3h[0] = 1j*analyzer.K[0]*Ph
            # T3h[1] = 1j*analyzer.K[1]*Ph
            # T3h[2] = 1j*analyzer.K[2]*Ph

            Tk[3] = 0.5*analyzer.shell_average(np.sum(np.real(
                                                T3h*np.conj(uh)), axis=0))

            # T3 = analyzer.vec_ifft(T3h)
            T3h = analyzer.vec_fft(T3/rho)
            Tk[4] = 0.5*analyzer.shell_average(np.sum(np.real(
                                                T3h*np.conj(ruh)), axis=0))

            # -----------------------------------------------------------------

            S = A + np.rollaxis(A, 1)  # ACTUALLY 2*S!!!!!!
            Skk = 0.5*np.einsum('kk...', S)

            T3 = analyzer.scl_grad((5./3.)*mu*Skk)
            T3h = analyzer.vec_fft(T3)
            # tmp = Ph
            # tmp = (5./3.)*tcfft.rfft3(comm, mu*Skk)
            # T3h[0] = 1j*analyzer.K[0]*tmp
            # T3h[1] = 1j*analyzer.K[1]*tmp
            # T3h[2] = 1j*analyzer.K[2]*tmp

            Tk[7] = 0.5*analyzer.shell_average(np.sum(np.real(
                                                T3h*np.conj(uh)), axis=0))

            # T3 = analyzer.vec_ifft(T3h)
            T3h = analyzer.vec_fft(T3/rho)
            Tk[8] = 0.5*analyzer.shell_average(np.sum(np.real(
                                                T3h*np.conj(ruh)), axis=0))

            T3[:] *= -0.4
            # T3h = analyzer.vec_fft(T3)
            for j in range(3):
                for i in range(3):
                    T3[i] += analyzer.deriv(mu*S[j, i], dim=j)
                    # T3h[i] += 1j*analyzer.K[j]*tcfft.rfft3(comm, mu*S[j, i])
            T3h = analyzer.vec_fft(T3)

            Tk[5] = 0.5*analyzer.shell_average(np.sum(np.real(
                                                T3h*np.conj(uh)), axis=0))

            # T3 = analyzer.vec_ifft(T3h)
            T3h = analyzer.vec_fft(T3/rho)
            Tk[6] = 0.5*analyzer.shell_average(np.sum(np.real(
                                                T3h*np.conj(ruh)), axis=0))

            # -----------------------------------------------------------------

            if comm.rank == 0:
                fh = open('%s%sSKE_transport_tilde.bin'
                          % (odir, analyzer.prefix), 'w')
                Tk.tofile(fh)
                fh.close()

                comm.Barrier()
            else:
                comm.Barrier()

            if comm.rank % 64 == 0:
                print (update(timeofday(), tstep, ir, comm.rank)
                       +'ir loop finished')

    if comm.rank == 0:
        print("\nPython MPI job `spec_trans_serial' finished at {}"
              .format(timeofday()))

    return

###############################################################################
if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{: .8e}'.format})
    spec_trans_serial(get_inputs())
