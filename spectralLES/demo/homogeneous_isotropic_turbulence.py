"""
Description:
------------
Simulation and analysis of incompressible homogeneous isotropic
turbulence using the TESLaCU Python scientific analysis and spectralLES
pure-Python pseudo-spectral large eddy simulation solver
for model development and testing.

Notes:
------
To execute the program in serial with an input file on the command line, run
`mpiexec -n 1 python homogeneous_isotropic_turbulence.py \
              -f HIT_demo_inputs.txt`.

For help with program options, run
`mpiexec -n 1 python homogeneous_isotropic_turbulence.py -h`.

Authors:
--------
Colin Towery, colin.towery@colorado.edu

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
https://github.com/teslacu/teslapy.git
https://github.com/teslacu/spectralLES.git
"""

from mpi4py import MPI
import numpy as np
import sys
import time
from math import sqrt, pi
import argparse
from spectralLES import spectralLES
from teslacu import mpiAnalyzer, mpiWriter
from teslacu.fft import irfft3  # FFT transforms

comm = MPI.COMM_WORLD


# -----------------------------------------------------------------------------
def homogeneous_isotropic_turbulence(pp=None, sp=None):
    """
    Arguments:
    ----------
    pp: (optional) program parameters, parsed by argument parser
        provided by this file
    sp: (optional) solver parameters, parsed by spectralLES.parser
    """

    if comm.rank == 0:
        print("\n----------------------------------------------------------")
        print("MPI-parallel Python spectralLES simulation of problem \n"
              "`Homogeneous Isotropic Turbulence' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))
        print("----------------------------------------------------------")

    # if function called without passing in parsed arguments, then parse
    # the arguments from the command line

    if pp is None:
        pp = hit_parser.parse_known_args()[0]

    if sp is None:
        sp = spectralLES.parser.parse_known_args()[0]

    if comm.rank == 0:
        print('\nProblem Parameters:\n-------------------')
        for k, v in vars(pp).items():
            print(k, v)
        print('\nSpectralLES Parameters:\n-----------------------')
        for k, v in vars(sp).items():
            print(k, v)
        print("\n----------------------------------------------------------\n")

    assert len(set(pp.N)) == 1, ('Error, this beta-release HIT program '
                                 'requires equal mesh dimensions')
    N = pp.N[0]
    assert len(set(pp.L)) == 1, ('Error, this beta-release HIT program '
                                 'requires equal domain dimensions')
    L = pp.L[0]

    if N % comm.size > 0:
        if comm.rank == 0:
            print('Error: job started with improper number of MPI tasks for '
                  'the size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Configure the solver, writer, and analyzer

    # -- construct solver instance from sp's attribute dictionary
    solver = spectralLES(comm, **vars(sp))

    # -- configure solver instance to solve the NSE with the vorticity
    #    formulation of the advective term, linear forcing, and
    #    Smagorinsky SGS model.
    solver.computeAD = solver.computeAD_vorticity_form
    Sources = [solver.computeSource_linear_forcing,
               solver.computeSource_Smagorinksy_SGS]

    Ck = 1.6
    Cs = sqrt((pi**-2)*((3*Ck)**-1.5))  # == 0.098...
    # Cs = 0.2
    kwargs = {'Cs': Cs, 'dvScale': None}

    # -- form HIT initial conditions from either user-defined values or
    #    physics-based relationships using epsilon and L
    Urms = 1.2*(pp.epsilon*L)**(1./3.)             # empirical coefficient
    Einit= getattr(pp, 'Einit', None) or Urms**2   # == 2*KE_equilibrium
    kexp = getattr(pp, 'kexp', None) or -1./3.     # -> E(k) ~ k^(-2./3.)
    kpeak= getattr(pp, 'kpeak', None) or N//4      # ~ kmax/2

    # !  currently using a fixed random seed of comm.rank for testing
    solver.initialize_HIT_random_spectrum(Einit, kexp, kpeak, rseed=comm.rank)

    U_hat = solver.U_hat
    U = solver.U
    omega = solver.omega
    K = solver.K

    # -- configure the writer and analyzer from both pp and sp attributes
    writer = mpiWriter(comm, odir=pp.odir, N=N)
    analyzer = mpiAnalyzer(comm, odir=pp.adir, pid=pp.pid, L=L, N=N,
                           config='hit', method='spectral')

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format
    emin = np.inf
    emax = np.NINF
    analyzer.mpi_moments_file = '%s%s.moments' % (analyzer.odir, pp.pid)

    # -------------------------------------------------------------------------
    # Setup the various time and IO counters

    tauK = sqrt(pp.nu/pp.epsilon)           # Kolmogorov time-scale
    taul = 0.2*L*sqrt(3)/Urms               # 0.2 is empirical coefficient
    c = pp.cfl*sqrt(2*Einit)/Urms
    dt = solver.new_dt_constant_nu(c)       # use as estimate
    print("Integral time scale = {}".format(taul))

    if pp.tlimit == np.Inf:   # put a very large but finite limit on the run
        pp.tlimit = 262*taul  # such as (256+6)*tau, for spinup and 128 samples

    dt_rst = getattr(pp, 'dt_rst', None) or 4*taul
    dt_bin = getattr(pp, 'dt_bin', None) or taul
    dt_stat= getattr(pp, 'dt_stat', None) or max(0.2*taul, 2*tauK, 20*dt)
    dt_spec= getattr(pp, 'dt_spec', None) or max(0.1*taul, tauK, 10*dt)
    dt_drv = getattr(pp, 'dt_drv', None) or max(tauK, 10*dt)

    t_sim = t_rst = t_bin = t_stat = t_spec = t_drv = 0.0
    tstep = irst = ibin = istat = ispec = 0

    # -- ensure that analysis and simulation outputs are properly synchronized
    #    This assumes that dt_spec < dt_stat < dt_bin < dt_rst, and that
    #    division remainders smaller than 0.1 are potentially
    #    consequential round-off errors in what should be integer multiples
    #    due to the user supplying insufficient significant digits
    if ((dt_stat % dt_spec) < 0.1*dt_spec):
        dt_stat -= dt_stat % dt_spec

    if ((dt_bin % dt_spec) < 0.1*dt_spec):
        dt_bin -= dt_bin % dt_spec

    if ((dt_bin % dt_stat) < 0.1*dt_stat):
        dt_bin -= dt_bin % dt_stat

    if ((dt_rst % dt_bin) < 0.1*dt_bin):
        dt_rst -= dt_rst % dt_bin

    # -------------------------------------------------------------------------
    # Run the simulation

    while t_sim < pp.tlimit+1.e-8:

        # -- Update the dynamic dt based on CFL constraint
        dt = solver.new_dt_constant_nu(pp.cfl)
        t_test = t_sim + 0.5*dt
        compute_vorticity = True  # reset the vorticity computation flag

        # -- output log messages every step if needed/wanted
        KE = 0.5*comm.allreduce(np.sum(np.square(U)))*(1./N)**3
        if comm.rank == 0:
            print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e"
                  % (tstep, t_sim, dt, KE))

        # - output snapshots and data analysis products
        if t_test >= t_spec:
            analyzer.spectral_density(U_hat, '%3.3d_u' % ispec,
                                      'velocity PSD\t%s' % Ek_fmt('u_i'))

            omega[2] = irfft3(comm, 1j*(K[0]*U_hat[1] - K[1]*U_hat[0]))
            omega[1] = irfft3(comm, 1j*(K[2]*U_hat[0] - K[0]*U_hat[2]))
            omega[0] = irfft3(comm, 1j*(K[1]*U_hat[2] - K[2]*U_hat[1]))

            analyzer.spectral_density(omega, '%3.3d_omga' % ispec,
                                      'vorticity PSD\t%s' % Ek_fmt('\omega_i'))

            t_spec += dt_spec
            ispec += 1
            compute_vorticity = False

        # if t_test >= t_stat:

        #     if compute_vorticity:
        #         omega[2] = irfft3(comm, 1j*(K[0]*U_hat[1] - K[1]*U_hat[0]))
        #         omega[1] = irfft3(comm, 1j*(K[2]*U_hat[0] - K[0]*U_hat[2]))
        #         omega[0] = irfft3(comm, 1j*(K[1]*U_hat[2] - K[2]*U_hat[1]))

        #     enst = 0.5*np.sum(np.square(omega), axis=0)

        #     emin = min(emin, comm.allreduce(np.min(enst), op=MPI.MIN))
        #     emax = max(emax, comm.allreduce(np.max(enst), op=MPI.MAX))

        #     scalar_analysis(analyzer, enst, (emin, emax), None, None,
        #                     '%3.3d_enst' % istat, 'enstrophy', '\Omega')
        #     t_stat += dt_stat
        #     istat += 1

        # -- output singe-precision binary files and restart checkpoints
        # if t_test >= t_bin:
        #     writer.write_scalar('Enstrophy_%3.3d.bin' % ibin, enst, np.float32)
        #     t_bin += dt_bin
        #     ibin += 1

        if t_test >= t_rst:
            writer.write_scalar('Velocity1_%3.3d.rst' % irst, U[0], np.float64)
            writer.write_scalar('Velocity2_%3.3d.rst' % irst, U[1], np.float64)
            writer.write_scalar('Velocity3_%3.3d.rst' % irst, U[2], np.float64)
            t_rst += dt_rst
            irst += 1

        # -- Update the forcing pattern
        if t_test >= t_drv:
            # call solver.computeSource_linear_forcing to compute dvScale only
            kwargs['dvScale'] = Sources[0](computeRHS=False)
            t_drv += dt_drv
            if comm.rank == 0:
                print("------ updated dvScale for linear forcing ------")
                # print(kwargs['dvScale'])

        # -- integrate the solution forward in time
        solver.RK4_integrate(dt, *Sources, **kwargs)

        t_sim += dt
        tstep += 1

        sys.stdout.flush()  # forces Python 3 to flush print statements

    # -------------------------------------------------------------------------
    # Finalize the simulation

    omega[2] = irfft3(comm, 1j*(K[0]*U_hat[1] - K[1]*U_hat[0]))
    omega[1] = irfft3(comm, 1j*(K[2]*U_hat[0] - K[0]*U_hat[2]))
    omega[0] = irfft3(comm, 1j*(K[1]*U_hat[2] - K[2]*U_hat[1]))
    enst = 0.5*np.sum(np.square(omega), axis=0)

    analyzer.spectral_density(U_hat, '%3.3d_u' % ispec, 'velocity PSD\t%s'
                              % Ek_fmt('u_i'))
    analyzer.spectral_density(omega, '%3.3d_omga' % ispec,
                              'vorticity PSD\t%s' % Ek_fmt('\omega_i'))

    # emin = min(emin, comm.allreduce(np.min(enst), op=MPI.MIN))
    # emax = max(emax, comm.allreduce(np.max(enst), op=MPI.MAX))
    # scalar_analysis(analyzer, enst, (emin, emax), None, None,
    #                 '%3.3d_enst' % istat, 'enstrophy', '\Omega')

    writer.write_scalar('Enstrophy_%3.3d.bin' % ibin, enst, np.float32)

    writer.write_scalar('Velocity1_%3.3d.rst' % irst, U[0], np.float64)
    writer.write_scalar('Velocity2_%3.3d.rst' % irst, U[1], np.float64)
    writer.write_scalar('Velocity3_%3.3d.rst' % irst, U[2], np.float64)

    return


# -----------------------------------------------------------------------------
def timeofday():
    return time.strftime("%H:%M:%S")


def scalar_analysis(mA, phi, minmax, w, wbar, fname, title, symb):
    """
    Compute all the 'stand-alone' statistics and scalings related to
    a scalar field such as density, pressure, or tracer mass
    fraction.

    Arguments
    ---------
    mA      : mpiAnalyzer object
    phi     : scalar data field
    minmax  : (min, max) of histogram binning
    w       : scalar weights field
    wbar    : mean of weights field
    fname   : file name string
    title   : written name of phi
    symb    : latex math string symbolizing phi
    """

    # update = '{:4d}\t{}'.format

    if w is None:
        ylabel = "\mathrm{pdf}"
        xlabel = "%s\t\left\langle{}\\right\\rangle" % symb
    else:
        ylabel = "\widetilde{\mathrm{pdf}}"
        xlabel = "%s\t\left\{{{}\\right\}}" % symb

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    mA.mpi_histogram1(phi.copy(), fname, '%s\t%s' % (xlabel, ylabel),
                      minmax, 100, w, wbar)
    mA.write_mpi_moments(phi, title, symb, w, wbar)

    if fname in ['rho', 'P', 'T', 'Smm', 'Y']:
        mA.spectral_density(phi, fname, '%s PSD\t%s' % (title, Ek_fmt(symb)))

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
    mA.mpi_histogram1(v.copy(), fname, '%s\t%s' % (xlabel, ylabel), minmax,
                      100, wvec, wbar, norm=3.0)
    mA.spectral_density(v, fname, '%s PSD\t%s' % (title, Ek_fmt(symb)))
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

    for j in range(3):
        for i in range(3):
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

    mA.mpi_histogram1(A.copy(), fname, '%s\t%s' % (xlabel, ylabel),
                      minmax, 100, W, wbar, norm=9.0)

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


# -----------------------------------------------------------------------------
hit_parser = argparse.ArgumentParser(prog='Homogeneous Isotropic Turbulence',
                                     parents=[spectralLES.parser])

hit_parser.description = ("A large eddy simulation model testing and analysis "
                          "script for homogeneous isotropic turbulence")
hit_parser.epilog = ('This program uses spectralLES, %s'
                     % spectralLES.parser.description)

config_group = hit_parser._action_groups[2]

config_group.add_argument('-p', '--pid', type=str, default='test',
                          help='problem prefix for analysis outputs')
config_group.add_argument('--dt_drv', type=float,
                          help='refresh-rate of forcing pattern')

time_group = hit_parser.add_argument_group('time integration arguments')

time_group.add_argument('--cfl', type=float, default=0.5, help='CFL number')
time_group.add_argument('-t', '--tlimit', type=float, default=np.inf,
                        help='solution time limit')
time_group.add_argument('-w', '--twall', type=float,
                        help='run wall-time limit (ignored for now!!!)')

init_group = hit_parser.add_argument_group('initial condition arguments')

init_group.add_argument('-i', '--init', '--initial-condition',
                        metavar='IC', default='GamieOstriker',
                        choices=['GamieOstriker', 'TaylorGreen'],
                        help='use specified initial condition')
init_group.add_argument('--kexp', type=float,
                        help=('Gamie-Ostriker power-law scaling of '
                              'initial velocity condition'))
init_group.add_argument('--kpeak', type=float,
                        help=('Gamie-Ostriker exponential-decay scaling of '
                              'initial velocity condition'))
init_group.add_argument('--Einit', type=float,
                        help='specify KE of initial velocity field')

rst_group = hit_parser.add_argument_group('simulation restart arguments')

rst_group.add_argument('-l', '--last', '--restart-from-last', dest='restart',
                       action='store_const', const=-1,
                       help='restart from last *.rst checkpoint in IDIR')
rst_group.add_argument('-r', '--rst', '--restart-from-num', type=int,
                       dest='restart', metavar='NUM',
                       help=('restart from specified checkpoint in IDIR, '
                             'negative numbers index backwards from last'))
rst_group.add_argument('--idir', type=str, default='./data/',
                       help='input directory for restarts')

io_group = hit_parser.add_argument_group('simulation output arguments')

io_group.add_argument('--odir', type=str, default='./data/',
                      help='output directory for simulation fields')
io_group.add_argument('--dt_rst', type=float,
                      help='time between restart checkpoints')
io_group.add_argument('--dt_bin', type=float,
                      help='time between single-precision outputs')

anlzr_group = hit_parser.add_argument_group('analysis output arguments')

anlzr_group.add_argument('--adir', type=str, default='./analysis/',
                         help='output directory for analysis products')
anlzr_group.add_argument('--dt_stat', type=float,
                         help='time between statistical analysis outputs')
anlzr_group.add_argument('--dt_spec', type=float,
                         help='time between isotropic power spectral density'
                              ' outputs')


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    homogeneous_isotropic_turbulence()
