"""
"""

from mpi4py import MPI
import numpy as np
from math import *
import time
import sys
import getopt
import fft_mpi4py_numpy as tcfft        # FFT transforms and math functions
from scipy.interpolate import Akima1DInterpolator as interp
# import hashlib

world_comm = MPI.COMM_WORLD
help_string = ("Athena analyzer command line options:\n"
               "-i <input directory>,\t defualt: 'data/'\n"
               "-o <output directory>,\t defualt: 'analysis/'\n"
               "-p <problem ID>,\t\t defualt: 'no_problem_id'\n"
               "-N <Nx>,\t\t default: 512\n"
               "-g <gamma>,\t\t default: 1.4\n"
               "-L <L>,\t\t\t default: 1.0\n"
               "-r <irs:ire:rint>,\t\t default: 1:20:1\n"
               "-t <its:ite:tint>,\t\t default: 1:20:1\n"
               "-R <R>,\t\t\t default: 8.3144598e7/21\n"
               "--Texp <texp>,\t\t default: 0.7\n"
               "--Tcoef <tcoef>,\t default: 3.1e-6\n"
               "--Tmp0 <tmp0>,\t\t default: 293.0\n")


def timeofday():
    return time.strftime("%H:%M:%S")


def get_inputs():
    """
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
    """
    idir = 'data/'                  # input folder
    odir = 'analysis/'              # input folder
    N = 512                         # linear grid size
    pid = 'no_problem_id'           # problem ID
    L = 1.0                         # domain size
    irs = 1                         # ensemble run index start
    ire = 20                        # ensemble run index end
    rint = 1                        # ensemble run index interval
    its = 1                         # vtk time index start
    ite = 20                        # vtk time index end
    tint = 1                        # vtk time index interval
    gamma = 1.4                     # heat capacity ratio
    R = 8.3144598e7/21              # gas constant
    texp = 0.7                      # transport properties exponent "n"
    tcoef = 3.1e-6                  # transport properties Th1^n coefficient
    tmp0 = 293.0                    # reference temperature

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:p:N:n:M:g:L:r:t:R:",
                                   ["Texp=", "Tcoef=", "Tmp0="])
    except getopt.GetoptError as e:
        if world_comm.rank == 0:
            print e
            print help_string
        MPI.Finalize()
        sys.exit(999)
    except Exception as e:
        if world_comm.rank == 0:
            print ('Unknown exception while getting input arguments!')
            print e
        MPI.Finalize()
        try:
            sys.exit(e.errno)
        except:
            sys.exit(999)

    for opt, arg in opts:
        try:
            if opt=='-h':
                if world_comm.rank == 0:
                    print help_string
                MPI.Finalize()
                sys.exit(1)
            elif opt=='-i':
                idir = arg
                if world_comm.rank == 0:
                    print 'input directory:\t'+idir
            elif opt=='-o':
                odir = arg
                if world_comm.rank == 0:
                    print 'output directory:\t'+odir
            elif opt=='-p':
                pid = arg
                if world_comm.rank == 0:
                    print 'problem ID:\t\t'+pid
            elif opt=='-N':
                N = int(arg)
                if world_comm.rank == 0:
                    print 'N:\t\t\t{}'.format(N)
            elif opt=='-g':
                gamma = float(arg)
                if world_comm.rank == 0:
                    print 'gamma:\t\t\t{}'.format(gamma)
            elif opt=='-L':
                L = float(arg)
                if world_comm.rank == 0:
                    print 'L:\t\t\t{}'.format(L)
            elif opt=='-r':
                try:
                    [irs, ire, rint] = [int(i) for i in arg.split(':')]
                except ValueError as e:
                    if world_comm.rank == 0:
                        print ('Input Error: option -r <irs:ire:rint> requires'
                               ' three integer values separated by colons.')
                        print e
                        print help_string
                    MPI.Finalize()
                    sys.exit(e.errno)
                if world_comm.rank == 0:
                    print 'ensemble runs:\t\t{}'.format((irs, ire, rint))
            elif opt=='-t':
                try:
                    [its, ite, tint] = [int(i) for i in arg.split(':')]
                except ValueError as e:
                    if world_comm.rank == 0:
                        print ('Input Error: option -t <its:ite:tint> requires'
                               ' three integer values separated by colons.')
                        print e
                        print help_string
                    MPI.Finalize()
                    sys.exit(e.errno)
                if world_comm.rank == 0:
                    print 'time steps:\t\t{}'.format((its, ite, tint))
            elif opt=='-R':
                R = float(arg)
                if world_comm.rank == 0:
                    print 'R:\t\t\t{}'.format(R)
            elif opt=='--Texp':
                texp = float(arg)
                if world_comm.rank == 0:
                    print 'texp:\t\t\t{}'.format(texp)
            elif opt=='--Tcoef':
                tcoef = float(arg)
                if world_comm.rank == 0:
                    print 'tcoef:\t\t\t{}'.format(tcoef)
            elif opt=='--Tmp0':
                tmp0 = float(arg)
                if world_comm.rank == 0:
                    print 'tcoef:\t\t\t{}'.format(tmp0)
            else:
                if world_comm.rank == 0:
                    print help_string
                MPI.Finalize()
                sys.exit(1)
        except Exception as e:
            if world_comm.rank == 0:
                print ('Unknown exception while reading argument {} '
                       'from option {}!'.format(opt, arg))
                print e
            MPI.Finalize()
            sys.exit(e.errno)

    args = (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, R,
            texp, tcoef, tmp0)

    return args


def scalar_analysis(mA, phi, minmax, w, wbar, fname, title, symb):
    """
    Compute all the 'stand-alone' statistics and scalings related to
    a scalar field such as density, pressure, or tracer mass
    fraction.

    Arguments
    ---------
    mA    : mpiAnalyzer object
    phi   : scalar data field
    w     : scalar weights field
    wbar  : mean of weights field
    fname : file name string
    title : written name of phi
    symb  : latex math string symbolizing phi
    """

    # update = '{:4d}\t{}'.format

    if w is None:
        ylabel = "\mathrm{pdf}"
        xlabel = "%s\t\left\langle{}\\right\\rangle" % symb
    else:
        ylabel = "\widetilde{\mathrm{pdf}}"
        xlabel = "%s\t\left\{{{}\\right\}}" % symb

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    # if mA.comm.rank % 8 == 0:
    #     print (update(mA.comm.rank, '\tphi: %s'
    #            % hashlib.md5(phi).hexdigest()))

    mA.mpi_histogram1(phi.copy(), fname, xlabel, ylabel, minmax, 100, w, wbar)

    # if mA.comm.rank % 8 == 0:
    #     print (update(mA.comm.rank, '\tphi post-histogram: %s'
    #            % hashlib.md5(phi).hexdigest()))

    mA.write_mpi_moments(phi, title, symb, w, wbar, m1=0)

    # if mA.comm.rank % 8 == 0:
    #     print (update(mA.comm.rank, '\tphi post-moments: %s'
    #            % hashlib.md5(phi).hexdigest()))

    if fname in ['rho', 'P', 'T', 'Smm', 'Y']:
        mA.spectral_density(phi, fname, title+' PSD', Ek_fmt(symb))

    # if mA.comm.rank % 8 == 0:
    #     print (update(mA.comm.rank, '\tphi post-PSD: %s'
    #            % hashlib.md5(phi).hexdigest()))

    # insert structure functions, scalar increments, two-point
    # correlations here.

    return


def vector_analysis(mA, v, minmax, w, wbar, fname, title, symb):
    """
    Compute all the 'stand-alone' statistics and scalings related to
    a vector field such as velocity, momentum, or vorticity.

    Arguments
    ---------
    v     : vector field (1st order tensor)
    fname : file name string
    symb  : latex math string
    """

    if w is None:
        xlabel = "%s\t\left\langle{}\\right\\rangle" % symb
        ylabel = "\mathrm{pdf}"
        wbar = None
        wvec = None
    else:
        xlabel = "%s\t\left\{{{}\\right\}}" % symb
        ylabel = "\widetilde{\mathrm{pdf}}"
        if w.size == v.size/v.shape[0]:
            s = [1]*w.ndim
            s.insert(0, v.shape[0])
            wvec = np.tile(w, s)
        elif w.size == v.size:
            wvec = w
        else:
            raise ValueError("w should either be the same size as v or"+
                             "the same size as one component of v")

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    # vector components analyzed
    mA.mpi_histogram1(v.copy(), fname, xlabel, ylabel, minmax,
                      100, wvec, wbar, norm=3.0)
    mA.spectral_density(v, fname, title+' PSD', Ek_fmt(symb))
    mA.write_mpi_moments(v, title, symb, wvec, wbar, m1=0, norm=3.0)

    # insert structure functions, scalar increments, two-point
    # correlations here.

    return


def gradient_analysis(mA, A, minmax, w, wbar, fname, title, symb):
    """
    Compute all the 'stand-alone' statistics of the velocity-
    gradient tensor field.

    Arguments
    ---------
    A     : velocity gradient tensor field (2nd order)
    fname : file name string
    symb  : latex math string
    """

    for j in xrange(0, 3):
        for i in xrange(0, 3):
            tij = ' {}{}'.format(i+1, j+1)
            sij = '_{{{}{}}}'.format(i+1, j+1)
            mA.write_mpi_moments(
                            A[j, i], title+tij, symb+sij, w, wbar, m1=0)

    if w is None:
        xlabel = "%s\t\left\langle{}\\right\\rangle" % symb
        ylabel = "\mathrm{pdf}"
        wbar = None
        W = None
    else:
        xlabel = "%s\t\left\{{{}\\right\}}" % symb
        ylabel = "\widetilde{\mathrm{pdf}}"
        if w.size == A.size/(A.shape[0]*A.shape[1]):
            s = [1]*w.ndim
            s.insert(0, A.shape[1])
            s.insert(0, A.shape[0])
            W = np.tile(w, s)
        elif w.size == A.size:
            W = w
        else:
            raise ValueError("w should either be the same size as v or"
                             "the same size as one component of v")

    symb += '_{ij}'

    mA.mpi_histogram1(A.copy(), fname, xlabel, ylabel, minmax, 100, W, wbar,
                      norm=9.0)

    # Aii = np.einsum('ii...', A)
    # I = np.identity(3)/3.0
    # s = np.ones(len(Aii.shape)+2, dtype=np.int)
    # s[0:2] = 3
    # Atl = A-I.reshape(s)*Aii
    # m1 = (m1.sum()-m1[0, 0]-m1[1, 1]-m1[2, 2])/9.0

    # symb += "^\mathrm{tl}"
    # fname += '_tl'
    # mA.mpi_histogram1(Atl, fname, xlabel, ylabel, 100, W, wbar,
    #                   m1, 9.0)

    # add tensor invariants here.

    return None  # add success flag or error


def helmholtz(mA, u):
    """
    Computes the k-parallel component of a Fourier-space vector
    using the projection operator, which is the physical-space
    Helmholtz decomposition of a vector field into its dilatational
    component.

    TCFFT.HELMHOLTZ MAY BE BROKEN AFTER SOME CHANGES TO THE WAVENUMBER
    VARIABLES IN THE ANALYZER CLASS. JUST NEED TO CHECK FIRST.
    """
    nnz, ny, nx = u.shape[1:]
    nk = nx/2+1
    nny = ny/mA.comm.size
    nz = nnz*mA.comm.size

    if u.dtype.itemsize == 8:
        fft_complex = np.complex128
    elif u.dtype.itemsize == 4:
        fft_complex = np.complex64
    else:
        raise

    fu = np.empty([3, nz, nny, nk], dtype=fft_complex)
    fu[0] = tcfft.rfft3(mA.comm, u[0])
    fu[1] = tcfft.rfft3(mA.comm, u[1])
    fu[2] = tcfft.rfft3(mA.comm, u[2])

    fud, fus = tcfft.helmholtz(fu, mA.K)

    ud = np.empty_like(u)
    us = np.empty_like(u)

    ud[0] = tcfft.irfft3(mA.comm, fud[0])
    ud[1] = tcfft.irfft3(mA.comm, fud[1])
    ud[2] = tcfft.irfft3(mA.comm, fud[2])

    us[0] = tcfft.irfft3(mA.comm, fus[0])
    us[1] = tcfft.irfft3(mA.comm, fus[1])
    us[2] = tcfft.irfft3(mA.comm, fus[2])

    return ud, us


def upinterp(phi, analyzer):
        phi = analyzer.z2y_slab_exchange(phi)

        s = list(phi.shape)
        if analyzer.comm.rank==0:
            print 'phi.shape:', s
        x = np.arange(-2, s[0]+2, dtype=np.float64)
        xi = np.arange(0, s[0], 0.5, dtype=np.float64)

        s[0] += 4
        tmp = np.empty(s, dtype=phi.dtype)

        tmp[2:-2] = phi
        tmp[:2] = phi[-2:]
        tmp[-2:] = phi[:2]

        spline = interp(x, tmp, axis=0)
        phi0 = spline(xi)

        phi0 = analyzer.y2z_slab_exchange(phi0)
        phi0 = np.swapaxes(phi0, 1, 0)

        s = list(phi0.shape)
        if analyzer.comm.rank==0:
            print 'phi0.shape:', s
        x = np.arange(-2, s[0]+2, dtype=np.float64)
        xi = np.arange(0, s[0], 0.5, dtype=np.float64)

        s[0] += 4
        tmp = np.empty(s, dtype=phi.dtype)

        tmp[2:-2] = phi0
        tmp[:2] = phi0[-2:]
        tmp[-2:] = phi0[:2]

        spline = interp(x, tmp, axis=0)
        phi1 = spline(xi)

        phi1 = np.swapaxes(phi1, 2, 0)

        s = list(phi1.shape)
        if analyzer.comm.rank==0:
            print 'phi1.shape:', s
        x = np.arange(-2, s[0]+2, dtype=np.float64)
        xi = np.arange(0, s[0], 0.5, dtype=np.float64)

        s[0] += 4
        tmp = np.empty(s, dtype=phi.dtype)

        tmp[2:-2] = phi1
        tmp[:2] = phi1[-2:]
        tmp[-2:] = phi1[:2]

        spline = interp(x, tmp, axis=0)
        phi2 = spline(xi)
        if analyzer.comm.rank==0:
            print 'phi2.shape:', phi2.shape

        return np.transpose(phi2, (1, 2, 0))
