"""
Description:
------------
244-coefficient truncated Volterra series ALES model static test program

Notes:
------
run `mpiexec -n 1 python ales244_static_test.py -h` for help

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
from math import *
import argparse
from spectralLES import spectralLES
from teslacu import mpiAnalyzer, mpiWriter
from teslacu.fft import rfft3, irfft3  # FFT transforms
from teslacu.stats import psum          # statistical functions

comm = MPI.COMM_WORLD


def timeofday():
    return time.strftime("%H:%M:%S")


###############################################################################
# Define the problem ("main" function)
###############################################################################
def ales244_static_les_test(pp=None, sp=None):
    """
    Arguments:
    ----------
    pp: (optional) program parameters, parsed by argument parser
        provided by this file
    sp: (optional) solver parameters, parsed by spectralLES.parser
    """

    if comm.rank == 0:
        print("MPI-parallel Python spectralLES simulation of problem "
              "`Homogeneous Isotropic Turbulence' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    # if function called without passing in parsed arguments, then parse
    # the arguments from the command line

    if pp is None:
        pp = hit_parser.parse_known_args()[0]

    if sp is None:
        sp = spectralLES.parser.parse_known_args()[0]

    if comm.rank == 0:
        print(pp)
        print(sp)

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
    solver = ales244_solver(comm, **vars(sp))

    U_hat = solver.U_hat
    U = solver.U
    omega = solver.omega
    K = solver.K

    # -- configure solver instance to solve the NSE with the vorticity
    #    formulation of the advective term, linear forcing, and
    #    the ales244 SGS model
    solver.computeAD = solver.computeAD_vorticity_form
    Sources = [solver.computeSource_linear_forcing,
               solver.computeSource_ales244_SGS]

    H_244 = np.loadtxt('h_ij.dat', usecols=(1, 2, 3, 4, 5, 6), unpack=True)

    kwargs = {'H_244': H_244, 'dvScale': None}

    # -- form HIT initial conditions from either user-defined values or
    #    physics-based relationships using epsilon and L
    Urms = 1.2*(pp.epsilon*L)**(1./3.)             # empirical coefficient
    Einit= getattr(pp, 'Einit', None) or Urms**2   # == 2*KE_equilibrium
    kexp = getattr(pp, 'kexp', None) or -1./3.     # -> E(k) ~ k^(-2./3.)
    kpeak= getattr(pp, 'kpeak', None) or N//4      # ~ kmax/2

    # -- currently using a fixed random seed of comm.rank for testing
    solver.initialize_HIT_random_spectrum(Einit, kexp, kpeak, rseed=comm.rank)

    # -- configure the writer and analyzer from both pp and sp attributes
    writer = mpiWriter(comm, odir=pp.odir, N=N)
    analyzer = mpiAnalyzer(comm, odir=pp.adir, pid=pp.pid, L=L, N=N,
                           config='hit', method='spectral')

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    # -------------------------------------------------------------------------
    # Setup the various time and IO counters

    tauK = sqrt(pp.nu/pp.epsilon)           # Kolmogorov time-scale
    taul = 0.2*L*sqrt(3)/Urms               # 0.2 is empirical coefficient
    c = pp.cfl*sqrt(2*Einit)/Urms
    dt = solver.new_dt_constant_nu(c)          # use as estimate

    if pp.tlimit == np.Inf:   # put a very large but finite limit on the run
        pp.tlimit = 262*taul  # such as (256+6)*tau, for spinup and 128 samples

    dt_rst = getattr(pp, 'dt_rst', None) or 2*taul
    dt_spec= getattr(pp, 'dt_spec', None) or max(0.1*taul, tauK, 10*dt)
    dt_drv = getattr(pp, 'dt_drv', None) or max(tauK, 10*dt)

    t_sim = t_rst = t_spec = t_drv = 0.0
    tstep = irst = ispec = 0

    # -------------------------------------------------------------------------
    # Run the simulation

    while t_sim < pp.tlimit+1.e-8:

        # -- Update the dynamic dt based on CFL constraint
        dt = solver.new_dt_constant_nu(pp.cfl)
        t_test = t_sim + 0.5*dt

        # -- output log messages every step if needed/wanted
        KE = 0.5*comm.allreduce(psum(np.square(U)))/solver.Nx
        if comm.rank == 0:
            print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e"
                  % (tstep, t_sim, dt, KE))

        # - output snapshots and data analysis products
        if t_test >= t_spec:
            analyzer.spectral_density(U_hat, '%3.3d_u' % ispec,
                                      'velocity PSD\t%s' % Ek_fmt('u_i'))

            irfft3(comm, 1j*(K[0]*U_hat[1] - K[1]*U_hat[0]), omega[2])
            irfft3(comm, 1j*(K[2]*U_hat[0] - K[0]*U_hat[2]), omega[1])
            irfft3(comm, 1j*(K[1]*U_hat[2] - K[2]*U_hat[1]), omega[0])

            analyzer.spectral_density(omega, '%3.3d_omga' % ispec,
                                      'vorticity PSD\t%s' % Ek_fmt('\omega_i'))

            t_spec += dt_spec
            ispec += 1

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
                print("------ updated linear forcing pattern ------")

        # -- integrate the solution forward in time
        solver.RK4_integrate(dt, *Sources, **kwargs)

        t_sim += dt
        tstep += 1

        sys.stdout.flush()  # forces Python 3 to flush print statements

    # -------------------------------------------------------------------------
    # Finalize the simulation

    irfft3(comm, 1j*(K[0]*U_hat[1] - K[1]*U_hat[0]), omega[2])
    irfft3(comm, 1j*(K[2]*U_hat[0] - K[0]*U_hat[2]), omega[1])
    irfft3(comm, 1j*(K[1]*U_hat[2] - K[2]*U_hat[1]), omega[0])

    analyzer.spectral_density(U_hat, '%3.3d_u' % ispec, 'velocity PSD\t%s'
                              % Ek_fmt('u_i'))
    analyzer.spectral_density(omega, '%3.3d_omga' % ispec,
                              'vorticity PSD\t%s' % Ek_fmt('\omega_i'))

    writer.write_scalar('Velocity1_%3.3d.rst' % irst, U[0], np.float64)
    writer.write_scalar('Velocity2_%3.3d.rst' % irst, U[1], np.float64)
    writer.write_scalar('Velocity3_%3.3d.rst' % irst, U[2], np.float64)

    return


###############################################################################
# Extend the spectralLES class
###############################################################################
class ales244_solver(spectralLES):
    """
    Just adding extra memory and the ales244 SGS model. By using the
    spectralLES class as a super-class and defining a subclass for each
    SGS model we want to test, spectralLES doesn't get cluttered with
    an excess of models over time.
    """

    # Class Constructor -------------------------------------------------------
    def __init__(self, comm, N, L, nu, epsilon, Gtype, **kwargs):

        # First: call spectralLES.__init__()
        super().__init__(comm, N, L, nu, epsilon, Gtype, **kwargs)

        # Second: add extra memory for ales244 model
        nnz, ny, nx = self.nnx
        nz, nny, nk = self.nnk

        self.tau_hat = np.empty((6, nz, nny, nk), dtype=complex)
        self.UU_hat = np.empty_like(self.tau_hat)

    # Instance Methods --------------------------------------------------------
    def computeSource_ales244_SGS(self, H_244, **ignored):
        """
        H_244 - ALES coefficients h_ij for 244-term Volterra series
                truncation. H_244.shape = (6, 244)
        """
        tau_hat = self.tau_hat
        UU_hat = self.UU_hat

        m = 0
        for j in range(3):
            for i in range(j, 3):
                rfft3(self.comm, self.U[i]*self.U[j], UU_hat[m])
                m+=1

        m = 0
        # loop over 6 stress tensor components
        for j in range(3):
            for i in range(j, 3):
                tau_hat[m] = H_244[m, 0]  # constant coefficient
                n = 1

                # loop over 27 stencil points
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        for c in range(-1, 2):
                            # compute stencil shift operator
                            # dx = 2*pi/N for standard incompressible HIT
                            # but really shift theorem needs 2*pi/N, not dx
                            pos = np.array([a, b, c])*self.dx
                            pos.resize((3, 1, 1, 1))
                            shift = np.exp(1j*np.sum(self.K*pos, axis=0))

                            # 3 ui Volterra series components
                            for d in range(3):
                                tau_hat[m] += H_244[m, n]*shift*self.U_hat[d]
                                n+=1

                            # 6 uiuj collocated Volterra series components
                            p = 0
                            for e in range(3):
                                for f in range(e, 3):
                                    tau_hat[m] += H_244[m, n]*shift*UU_hat[p]
                                    n+=1
                                    p+=1
                # end of nested loops a-f

                m+=1
        # end of ij loops

        self.W_hat[:] = 0.0
        m = 0
        for j in range(3):
            for i in range(j, 3):
                self.W_hat[i] += 1j*(1+(i!=j))*self.K[j]*tau_hat[m]

        self.dU += self.W_hat

        return


###############################################################################
# Add a parser for this problem
###############################################################################
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


###############################################################################
if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    ales244_static_les_test()
