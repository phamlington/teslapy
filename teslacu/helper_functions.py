"""
"""

from mpi4py import MPI
import numpy as np
# from math import sqrt
import time
import sys
import getopt
# import hashlib

# print(update(mA.comm.rank, '\tphi post-PSD: %s'
#       % hashlib.md5(phi).hexdigest()))

__all__ = []

comm = MPI.COMM_WORLD
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
    -g <gamma>              default: 1.3
    -L <L>                  default: 1.0
    -M <M>                  default: 21
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
    gamma = 1.3                     # heat capacity ratio
    M = 24.62                       # molecular weight
    texp = 0.7                      # transport properties exponent "n"
    tcoef = 3.1e-6                  # transport properties Th1^n coefficient
    tmp0 = 293.0                    # reference temperature

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:p:N:n:M:g:L:r:t:R:",
                                   ["Texp=", "Tcoef=", "Tmp0="])
    except getopt.GetoptError as e:
        if comm.rank == 0:
            print(e)
            print(help_string)
        MPI.Finalize()
        sys.exit(999)
    except Exception as e:
        if comm.rank == 0:
            print('Unknown exception while getting input arguments!')
            print(e)
        MPI.Finalize()

    for opt, arg in opts:
        try:
            if opt=='-h':
                if comm.rank == 0:
                    print(help_string)
                MPI.Finalize()
                sys.exit(1)
            elif opt=='-i':
                idir = arg
                if comm.rank == 0:
                    print('input directory:\t'+idir)
            elif opt=='-o':
                odir = arg
                if comm.rank == 0:
                    print('output directory:\t'+odir)
            elif opt=='-p':
                pid = arg
                if comm.rank == 0:
                    print('problem ID:\t\t'+pid)
            elif opt=='-N':
                try:
                    N = [int(i) for i in arg.split(':')]
                except ValueError as e:
                    if comm.rank == 0:
                        print('Input Error: option -N <Nx[:Ny:Nz]> requires'
                              ' 1-3 integer values separated by colons.')
                        print(e)
                        print(help_string)
                    MPI.Finalize()
                    sys.exit(e.errno)
                if comm.rank == 0:
                    print('N:\t\t\t{}'.format(N))
            elif opt=='-g':
                gamma = float(arg)
                if comm.rank == 0:
                    print('gamma:\t\t\t{}'.format(gamma))
            elif opt=='-L':
                try:
                    L = [float(i) for i in arg.split(':')]
                except ValueError as e:
                    if comm.rank == 0:
                        print('Input Error: option -L <NL[:Ly:Lz]> requires'
                              ' 1-3 integer values separated by colons.')
                        print(e)
                        print(help_string)
                    MPI.Finalize()
                    sys.exit(e.errno)
                if comm.rank == 0:
                    print('L:\t\t\t{}'.format(L))
            elif opt=='-r':
                try:
                    [irs, ire, rint] = [int(i) for i in arg.split(':')]
                except ValueError as e:
                    if comm.rank == 0:
                        print('Input Error: option -r <irs:ire:rint> requires'
                              ' three integer values separated by colons.')
                        print(e)
                        print(help_string)
                    MPI.Finalize()
                    sys.exit(e.errno)
                if comm.rank == 0:
                    print('ensemble runs:\t\t{}'.format((irs, ire, rint)))
            elif opt=='-t':
                try:
                    [its, ite, tint] = [int(i) for i in arg.split(':')]
                except ValueError as e:
                    if comm.rank == 0:
                        print('Input Error: option -t <its:ite:tint> requires'
                              ' three integer values separated by colons.')
                        print(e)
                        print(help_string)
                    MPI.Finalize()
                    sys.exit(e.errno)
                if comm.rank == 0:
                    print('time steps:\t\t{}'.format((its, ite, tint)))
            elif opt=='-M':
                M = float(arg)
                if comm.rank == 0:
                    print('M:\t\t\t{}'.format(M))
            elif opt=='--Texp':
                texp = float(arg)
                if comm.rank == 0:
                    print('texp:\t\t\t{}'.format(texp))
            elif opt=='--Tcoef':
                tcoef = float(arg)
                if comm.rank == 0:
                    print('tcoef:\t\t\t{}'.format(tcoef))
            elif opt=='--Tmp0':
                tmp0 = float(arg)
                if comm.rank == 0:
                    print('tcoef:\t\t\t{}'.format(tmp0))
            else:
                if comm.rank == 0:
                    print(help_string)
                MPI.Finalize()
                sys.exit(1)
        except Exception as e:
            if comm.rank == 0:
                print('Unknown exception while reading argument {} '
                      'from option {}!'.format(opt, arg))
                print(e)
            MPI.Finalize()
            sys.exit(e.errno)

    args = (idir, odir, pid, N, L, irs, ire, rint, its, ite, tint, gamma, M,
            texp, tcoef, tmp0)

    return args


def scalar_analysis(mA, phi, range, m1, w, wbar, fname, title, symb, norm=1.):
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

    mA.mpi_histogram1(phi, fname, '%s\t%s' % (xlabel, ylabel),
                      range, 100, w, wbar, m1, norm)

    mA.write_mpi_moments(phi, title, w, wbar, m1, norm)

    if fname in ['rho', 'P', 'T', 'Smm', 'Y']:
        mA.spectral_density(phi, fname, '%s PSD\t%s' % (title, Ek_fmt(symb)))

    # insert structure functions, scalar increments, two-point
    # correlations here.

    return


def vector_analysis(mA, v, minmax, m1, w, wbar, fname, title, symb):
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
    mA.write_mpi_moments(v, title, symb, wvec, wbar, m1=m1, norm=3.0)

    # insert structure functions, scalar increments, two-point
    # correlations here.

    return


def gradient_analysis(mA, A, minmax, m1, w, wbar, fname, title, symb):
    """
    Compute all the 'stand-alone' statistics of the velocity-
    gradient tensor field.

    Arguments
    ---------
    A     : velocity gradient tensor field (2nd order)
    fname : file name string
    symb  : latex math string
    """

    for j in range(0, 3):
        for i in range(0, 3):
            tij = ' {}{}'.format(i+1, j+1)
            sij = '_{{{}{}}}'.format(i+1, j+1)
            mA.write_mpi_moments(
                            A[j, i], title+tij, symb+sij, w, wbar, m1=m1)

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
