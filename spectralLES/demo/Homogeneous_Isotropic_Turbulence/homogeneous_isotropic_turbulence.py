"""
Description:
------------
Simulation and analysis of incompressible homogeneous isotropic turbulence
using the TESLaCU Python scientific analysis and spectralLES
pure-Python pseudo-spectral large eddy simulation solver
for model development and testing.

Notes:
------
run `mpiexec -n 1 python homogeneous_isotropic_turbulence -h` for help message

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
from math import *
import argparse
from spectralLES import spectralLES, LoadInputFile
from teslacu import mpiAnalyzer, mpiWriter
from teslacu.fft import irfft3  # FFT transforms
from .analysis_functions import *    # not _quite_ done with this
# from .output_functions import *    # not done working on this yet

comm = MPI.COMM_WORLD

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(prog='Homogeneous Isotropic Turbulence')
parser.description = ("A large eddy simulation model testing and analysis "
                      "script for homogeneous isotropic turbulence")
parser.epilog = ('This program uses spectralLES, %s'
                 % spectralLES.parser.description)

parser.add_argument('inputfile', type=open, action=LoadInputFile,
                    help='path to file-based input arguments')

anlzr_group.add_argument('-p', '--pid', type=str, default='test',
                         help='problem ID (file prefix)')
time_group.add_argument('--dt_drv', type=float,
                        help='refresh rate of forcing source term')

time_group = parser.add_argument_group('simulation runtime parameters')

time_group.add_argument('--cfl', type=float, default=0.5, help='CFL number')
time_group.add_argument('--tlimit', type=float, default=np.inf,
                        help='solution time limit')
time_group.add_argument('--twall', type=float,
                        help='run wall-time limit (ignored for now!!!)')
time_group.add_argument('-l', '--last', '--restart-from-last', dest='restart',
                        action='store_const', const=-1,
                        help='restart from last *.rst output in IDIR')
time_group.add_argument('-r', '--rst', '--restart-from-num', type=int,
                        dest='restart',
                        help=('restart from specified output in IDIR, '
                              'negative numbers index backwards from last'))
io_group.add_argument('--idir', type=str, default='./data/',
                      help='input directory for restarts')

init_group = parser.add_argument_group('simulation initialization parameters')

init_group.add_argument('-i', '--init', '--initial-condition',
                        default='GamieOstriker',
                        choice=['GamieOstriker', 'TaylorGreen'],
                        help='use specified initial condition')
init_group.add_argument('--kexp', type=float,
                        help=('Gamie-Ostriker power-law scaling of '
                              'initial velocity condition'))
init_group.add_argument('--kpeak', type=float,
                        help=('Gamie-Ostriker exponential-decay scaling of '
                              'initial velocity condition'))
init_group.add_argument('--Einit', type=float,
                        help='specify KE of initial velocity field')

io_group = parser.add_argument_group('simulation data input/output parameters')

io_group.add_argument('--odir', type=str, default='./data/',
                      help='output directory for simulation fields')
io_group.add_argument('--dt_rst', type=float,
                      help='time between restart checkpoints')
io_group.add_argument('--dt_bin', type=float,
                      help='time between single-precision outputs')

anlzr_group = parser.add_argument_group('in-situ analysis output parameters')

anlzr_group.add_argument('--adir', type=str, default='./analysis/',
                         help='output directory for analysis products')
anlzr_group.add_argument('--dt_stat', type=float,
                         help='time between statistical analysis outputs')
anlzr_group.add_argument('--dt_psd', type=float,
                         help='time between isotropic spectra outputs')


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
        print("MPI-parallel Python spectralLES simulation of problem "
              "`Homogeneous Isotropic Turbulence' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    # if function called without passing in parsed arguments, then parse
    # the arguments from the command line
    # this assumes that if pp is None, then so is sp
    if pp is None:
        pp, remainder = parser.parse_known_args()
        sp = spectralLES.parser.parse_args(remainder)

    if comm.rank == 0:
        print(pp)
        print(sp)

    assert len(set(sp.N)) == 1, ('Error, this beta-release HIT program '
                                 'requires equal mesh dimensions')
    N = sp.N[0]
    assert len(set(sp.L)) == 1, ('Error, this beta-release HIT program '
                                 'requires equal domain dimensions')
    L = sp.L[0]

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
    Urms = 1.2*(sp.epsilon*L)**(1./3.)             # empirical coefficient
    Einit = getattr(pp, 'Einit', None) or Urms**2  # == 2*KE_equilibrium
    kexp = getattr(pp, 'kexp', None) or -1./3.     # -> E(k) ~ k^(-2./3.)
    kpeak = getattr(pp, 'kpeak', None) or N//4     # ~ kmax/2
    # !  currently using a fixed random seed of comm.rank for testing
    solver.initialize_HIT_random_spectrum(Einit, kexp, kpeak, rseed=comm.rank)

    # -- configure the writer and analyzer from both pp and sp attributes
    writer = mpiWriter(comm, odir=pp.odir, N=sp.N)
    analyzer = mpiAnalyzer(comm, odir=pp.adir, pid=pp.pid, L=sp.L, N=sp.N,
                           config='hit', method='spectral')

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format
    emin = np.inf
    emax = np.NINF
    analyzer.mpi_moments_file = '%s%s.moments' % (analyzer.odir, pid)

    # -------------------------------------------------------------------------
    # Setup the various time and IO counters

    tauK = sqrt(sp.nu/sp.epsilon)           # Kolmogorov time-scale
    taul = 0.2*L*sqrt(3)/Urms               # 0.2 is empirical coefficient
    c = pp.cfl*sqrt(2*Einit)/Urms
    dt = solver.new_dt_const_nu(c)          # use as estimate

    if pp.tlimit == np.Inf:   # put a very large but finite limit on the run
        pp.tlimit = 262*taul  # such as (256+6)*tau, for spinup and 128 samples

    dt_rst = getattr(pp, 'dt_rst', None) or 4*taul
    dt_bin = getattr(pp, 'dt_bin', None) or taul
    dt_hst = getattr(pp, 'dt_hst', None) or max(0.2*taul, 2*tauK, 20*dt)
    dt_psd = getattr(pp, 'dt_psd', None) or max(0.1*taul, tauK, 10*dt)
    dt_drv = getattr(pp, 'dt_drv', None) or max(tauK, 10*dt)

    t_sim = t_rst = t_bin = t_hst = t_psd = t_drv = 0.0
    tstep = irst = ibin = ihst = ipsd = 0

    # -- ensure that analysis and simulation outputs are properly synchronized
    #
    #    This assumed that dt_psd < dt_hst < dt_bin < dt_rst, and that
    #    fractional remainders between 0.01 and 0.0001 are potentially
    #    consequential round-off errors in what should be integer multiples
    #    due to the user supplying insufficient significant digits
    if 0.01*dt_psd > (dt_hst % dt_psd) > 1.e-4*dt_psd:
        dt_hst -= dt_hst % dt_psd
    if 0.01*dt_hst > (dt_bin % dt_hst) > 1.e-4*dt_hst:
        dt_bin -= dt_bin % dt_hst
    if 0.01*dt_bin > (dt_rst % dt_bin) > 1.e-4*dt_bin:
        dt_rst -= dt_rst % dt_bin

    # -------------------------------------------------------------------------
    # Run the simulation

    while t_sim < pp.tlimit+1.e-8:

        # -- Update the dynamic dt based on CFL constraint
        dt = solver.new_dt_const_nu(pp.cfl)
        t_test = t_sim + 0.5*dt

        # -- output log messages every step if needed/wanted
        KE = 0.5*comm.allreduce(np.sum(np.square(solver.U)))*(1./N)**3
        if comm.rank == 0:
            print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e"
                  % (tstep, t_sim, dt, KE))

        # - output snapshots and data analysis products
        if t_test >= t_psd:
            analyzer.spectral_density(solver.U_hat, '%3.3d_u' % ipsd,
                                      'velocity PSD', Ek_fmt('u_i'))
            t_psd += dt_psd

            if t_test >= t_hst:
                solver.omega[2] = irfft3(comm,
                                         1j*(solver.K[0]*solver.U_hat[1]
                                             -solver.K[1]*solver.U_hat[0]))
                solver.omega[1] = irfft3(comm,
                                         1j*(solver.K[2]*solver.U_hat[0]
                                             -solver.K[0]*solver.U_hat[2]))
                solver.omega[0] = irfft3(comm,
                                         1j*(solver.K[1]*solver.U_hat[2]
                                             -solver.K[2]*solver.U_hat[1]))
                enst = 0.5*np.sum(np.square(solver.omega), axis=0)

                emin = min(emin, comm.allreduce(np.min(enst), op=MPI.MIN))
                emax = max(emax, comm.allreduce(np.max(enst), op=MPI.MAX))

                scalar_analysis(analyzer, enst, (emin, emax), None, None,
                                '%3.3d_enst' % ihst, 'enstrophy', '\Omega')

                analyzer.spectral_density(solver.omega, '%3.3d_omga' % ipsd,
                                          'vorticity PSD', Ek_fmt('\omega_i'))
                t_hst += dt_hst
                ihst += 1

            ipsd += 1

        # -- output singe-precision binary files and restart checkpoints
        if t_test >= t_bin:
            writer.write_scalar('Enstrophy_%3.3d.bin' % ibin, enst,
                                dtype=np.float32)
            t_bin += dt_bin
            ibin += 1

        if t_test >= t_rst:
            writer.write_scalar('Velocity1_%3.3d.rst' % irst, solver.U[0],
                                dtype=np.float64)
            writer.write_scalar('Velocity2_%3.3d.rst' % irst, solver.U[1],
                                dtype=np.float64)
            writer.write_scalar('Velocity3_%3.3d.rst' % irst, solver.U[2],
                                dtype=np.float64)
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

    solver.omega[2] = irfft3(comm, 1j*(solver.K[0]*solver.U_hat[1]
                                       -solver.K[1]*solver.U_hat[0]))
    solver.omega[1] = irfft3(comm, 1j*(solver.K[2]*solver.U_hat[0]
                                       -solver.K[0]*solver.U_hat[2]))
    solver.omega[0] = irfft3(comm, 1j*(solver.K[1]*solver.U_hat[2]
                                       -solver.K[2]*solver.U_hat[1]))
    enst = 0.5*np.sum(np.square(solver.omega), axis=0)

    analyzer.spectral_density(solver.U_hat, '%3.3d_u' % ipsd, 'velocity PSD',
                              Ek_fmt('u_i'))
    analyzer.spectral_density(solver.omega, '%3.3d_omga' % ipsd,
                              'vorticity PSD', Ek_fmt('\omega_i'))

    emin = min(emin, comm.allreduce(np.min(enst), op=MPI.MIN))
    emax = max(emax, comm.allreduce(np.max(enst), op=MPI.MAX))
    scalar_analysis(analyzer, enst, (emin, emax), None, None,
                    '%3.3d_enst' % ihst, 'enstrophy', '\Omega')

    writer.write_scalar('Enstrophy_%3.3d.bin' % ibin, enst, dtype=np.float32)

    writer.write_scalar('Velocity1_%3.3d.rst' % irst, solver.U[0],
                        dtype=np.float64)
    writer.write_scalar('Velocity2_%3.3d.rst' % irst, solver.U[1],
                        dtype=np.float64)
    writer.write_scalar('Velocity3_%3.3d.rst' % irst, solver.U[2],
                        dtype=np.float64)

    return
# -----------------------------------------------------------------------------


def timeofday():
    return time.strftime("%H:%M:%S")


if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    homogeneous_isotropic_turbulence()
