"""
Description:
------------
Pope75 static coefficient forward testing

Notes:
------
run `mpiexec -n 1 python Pope75_static_test.py -h` for help

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
from math import sqrt
import argparse
from spectralLES import spectralLES
from teslacu import mpiWriter
from teslacu.fft import rfft3, irfft3, shell_average
from teslacu.stats import psum

comm = MPI.COMM_WORLD


def timeofday():
    return time.strftime("%H:%M:%S")


###############################################################################
# Extend the spectralLES class
###############################################################################
class staticGeneralizedEddyViscosityLES(spectralLES):
    """
    Empty Docstring!
    """

    # -------------------------------------------------------------------------
    # Class Constructor
    # -------------------------------------------------------------------------
    def __init__(self, Smagorinsky=False, **kwargs):
        """
        Empty Docstring!
        """

        super().__init__(**kwargs)

        self.Aij = np.empty(self.nnk, dtype=complex)
        self.Aji = np.empty(self.nnk, dtype=complex)
        self.tau_ij_hat = np.empty(self.nnk, dtype=complex)

        if Smagorinsky:
            self.S = np.empty((6, *self.nnx))     # sparse symmetric tensor

        else:
            self.S = np.empty((3, 3, *self.nnx))  # dense tensor
            self.R = np.empty((3, 3, *self.nnx))  # dense tensor
            self.tau_ij = np.empty(self.nnx)

    # -------------------------------------------------------------------------
    # Instance Methods
    # -------------------------------------------------------------------------
    def computeSource_Smagorinsky_SGS(self, C1=-6.39e-2, **ignored):
        """
        SPARSE SYMMETRIC TENSOR INDEXING:
        m == 0 -> ij == 00
        m == 1 -> ij == 01
        m == 2 -> ij == 02
        m == 3 -> ij == 11
        m == 4 -> ij == 12
        m == 5 -> ij == 22
        """

        # --------------------------------------------------------------
        # Explicitly filter the solution field
        self.W_hat[:] = self.les_filter*self.U_hat

        # --------------------------------------------------------------
        # Compute S_ij and |S|^2
        S_sqr = self.W[0]
        S_sqr[:] = 0.0
        m = 0
        for i in range(3):
            for j in range(i, 3):
                self.Aij[:] = 0.5j*self.K[j]*self.W_hat[i]

                if i == j:
                    self.S[m] = irfft3(self.comm, 2*self.Aij)
                    S_sqr += self.S[m]**2

                else:
                    self.Aji[:] = 0.5j*self.K[i]*self.W_hat[j]
                    self.S[m] = irfft3(self.comm, self.Aij + self.Aji)
                    S_sqr += 2*self.S[m]**2

                m+=1

        # --------------------------------------------------------------
        # Compute C_1 Delta^2 |S| S_ij
        coef = self.W[1]
        coef[:] = C1*self.D_les**2*np.sqrt(2.0*S_sqr)

        # --------------------------------------------------------------
        # Compute FFT{div(tau)} and add to RHS update
        m = 0
        for i in range(3):
            for j in range(i, 3):
                rfft3(self.comm, coef*self.S[m], self.tau_ij_hat)

                self.dU[i] -= 1j*self.K[j]*self.tau_ij_hat
                if i != j:
                    self.dU[j] -= 1j*self.K[i]*self.tau_ij_hat

                m+=1

        return

    def computeSource_4termGEV_SGS(self, C=None, **ignored):
        """
        Empty Docstring!
        """

        # --------------------------------------------------------------
        # Explicitly filter the solution field
        self.W_hat[:] = self.les_filter*self.U_hat

        # --------------------------------------------------------------
        # Compute S_ij, R_ij, S_kl S_kl, R_kl R_kl, and |S|
        S = self.S
        R = self.R

        S_mod = self.W[0]
        S_sqr = self.W[1]
        R_sqr = self.W[2]

        S_sqr[:] = 0.0
        R_sqr[:] = 0.0
        for i in range(3):
            for j in range(3):
                self.Aij[:] = 0.5j*self.K[j]*self.W_hat[i]

                if i == j:
                    S[i, j] = irfft3(self.comm, 2*self.Aij)
                    R[i, j] = 0.0

                else:
                    self.Aji[:] = 0.5j*self.K[i]*self.W_hat[j]

                    S[i, j] = irfft3(self.comm, self.Aij + self.Aji)
                    R[i, j] = irfft3(self.comm, self.Aij - self.Aji)

                S_sqr += S[i, j]**2
                R_sqr += R[i, j]**2

        S_mod[:] = np.sqrt(2.0*S_sqr)

        # --------------------------------------------------------------
        # Compute tau_ij = Delta**2 C_m G_ij^m and update RHS
        for i in range(3):
            for j in range(3):

                # G_ij^1 = |S| S_ij
                self.tau_ij[:] = C[0]*S_mod*S[i, j]

                # G_ij^2 = -(S_ik R_jk + R_ik S_jk)
                self.tau_ij -= C[1]*np.sum(S[i]*R[j] + S[j]*R[i], axis=0)

                # G_ij^3 = S_ik S_jk - 1/3 delta_ij S_kl S_kl
                self.tau_ij += C[2]*np.sum(S[i]*S[j], axis=0)
                if i == j:
                    self.tau_ij -= C[2]*(1/3)*S_sqr

                # G_ij^4 = - R_ik R_jk - 1/3 delta_ij R_kl R_kl
                self.tau_ij -= C[3]*np.sum(R[i]*R[j], axis=0)
                if i == j:
                    self.tau_ij -= C[3]*(1/3)*R_sqr

                rfft3(self.comm, self.tau_ij, self.tau_ij_hat)
                self.dU[i] -= 1j*self.K[j]*self.tau_ij_hat

        return


###############################################################################
# Define the problem
###############################################################################
def ABC_static_test(pp=None, sp=None):
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

    # ------------------------------------------------------------------
    # Get the problem and solver parameters and assert compliance
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
            print('Error: job started with improper number of MPI tasks'
                  ' for the size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # ------------------------------------------------------------------
    # Configure the LES solver
    solver = staticGeneralizedEddyViscosityLES(
                Smagorinsky=True, comm=comm, **vars(sp))

    solver.computeAD = solver.computeAD_vorticity_form
    Sources = [solver.computeSource_linear_forcing,
               solver.computeSource_Smagorinsky_SGS,
               # solver.computeSource_4termGEV_SGS,
               ]

    # C1 = np.array([-6.39e-02])
    C3 = np.array([-3.75e-02, 6.2487e-02, 6.9867e-03, 0.0])
    C4 = np.array([-3.15e-02, -5.25e-02, 2.7e-02, 2.7e-02])
    kwargs = dict(C1=-6.39e-02, C=C3*solver.D_les**2, dvScale=None)

    U_hat = solver.U_hat
    U = solver.U
    Kmod = np.floor(np.sqrt(solver.Ksq)).astype(int)

    # ------------------------------------------------------------------
    # form HIT initial conditions from either user-defined values or
    # physics-based relationships
    Urms = 1.083*(pp.epsilon*L)**(1./3.)             # empirical coefficient
    Einit= getattr(pp, 'Einit', None) or Urms**2   # == 2*KE_equilibrium
    kexp = getattr(pp, 'kexp', None) or -1./3.     # -> E(k) ~ k^(-2./3.)
    kpeak= getattr(pp, 'kpeak', None) or N//4      # ~ kmax/2

    # currently using a fixed random seed for testing
    solver.initialize_HIT_random_spectrum(Einit, kexp, kpeak, rseed=comm.rank)

    # ------------------------------------------------------------------
    # Configure a spatial field writer
    writer = mpiWriter(comm, odir=pp.odir, N=N)
    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format

    # -------------------------------------------------------------------------
    # Setup the various time and IO counters
    tauK = sqrt(pp.nu/pp.epsilon)           # Kolmogorov time-scale
    taul = 0.11*sqrt(3)*L/Urms              # 0.11 is empirical coefficient

    if pp.tlimit == np.Inf:
        pp.tlimit = 200*taul

    dt_rst = getattr(pp, 'dt_rst', None) or taul
    dt_spec= getattr(pp, 'dt_spec', None) or 0.2*taul
    dt_drv = getattr(pp, 'dt_drv', None) or 0.25*tauK

    t_sim = t_rst = t_spec = t_drv = 0.0
    tstep = irst = ispec = 0
    tseries = []

    if comm.rank == 0:
        print('\ntau_ell = %.6e\ntau_K = %.6e\n' % (taul, tauK))

    # -------------------------------------------------------------------------
    # Run the simulation
    if comm.rank == 0:
        t1 = time.time()

    while t_sim < pp.tlimit+1.e-8:

        # -- Update the dynamic dt based on CFL constraint
        dt = solver.new_dt_constant_nu(pp.cfl)
        t_test = t_sim + 0.5*dt

        # -- output/store a log every step if needed/wanted
        KE = 0.5*comm.allreduce(psum(np.square(U)))/solver.Nx
        tseries.append([tstep, t_sim, KE])

        # -- output KE and enstrophy spectra
        if t_test >= t_spec:

            # -- output message log to screen on spectrum output only
            if comm.rank == 0:
                print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e"
                      % (tstep, t_sim, dt, KE))

            # -- output kinetic energy spectrum to file
            spect3d = np.sum(np.real(U_hat*np.conj(U_hat)), axis=0)
            spect3d[..., 0] *= 0.5
            spect1d = shell_average(comm, spect3d, Kmod)

            if comm.rank == 0:
                fname = '%s/%s-%3.3d_KE.spectra' % (pp.adir, pp.pid, ispec)
                fh = open(fname, 'w')
                metadata = Ek_fmt('u_i')
                fh.write('%s\n' % metadata)
                spect1d.tofile(fh, sep='\n', format='% .8e')
                fh.close()

            t_spec += dt_spec
            ispec += 1

        # -- output physical-space solution fields for restarting and analysis
        if t_test >= t_rst:
            writer.write_scalar('%s-Velocity1_%3.3d.rst' %
                                (pp.pid, irst), U[0], np.float64)
            writer.write_scalar('%s-Velocity2_%3.3d.rst' %
                                (pp.pid, irst), U[1], np.float64)
            writer.write_scalar('%s-Velocity3_%3.3d.rst' %
                                (pp.pid, irst), U[2], np.float64)
            t_rst += dt_rst
            irst += 1

        # -- Update the forcing mean scaling
        if t_test >= t_drv:
            # call solver.computeSource_linear_forcing to compute dvScale only
            kwargs['dvScale'] = Sources[0](computeRHS=False)
            t_drv += dt_drv

        # -- integrate the solution forward in time
        solver.RK4_integrate(dt, *Sources, **kwargs)

        t_sim += dt
        tstep += 1

        sys.stdout.flush()  # forces Python 3 to flush print statements

    # -------------------------------------------------------------------------
    # Finalize the simulation
    if comm.rank == 0:
        t2 = time.time()
        print('Program took %12.7f s' % ((t2-t1)))

    KE = 0.5*comm.allreduce(psum(np.square(U)))/solver.Nx
    tseries.append([tstep, t_sim, KE])

    if comm.rank == 0:
        fname = '%s/%s-%3.3d_KE_tseries.txt' % (pp.adir, pp.pid, ispec)
        header = 'Kinetic Energy Timeseries,\n# columns: tstep, time, KE'
        np.savetxt(fname, tseries, fmt='%10.5e', header=header)

        print("cycle = %7d  time = %15.8e  dt = %15.8e  KE = %15.8e"
              % (tstep, t_sim, dt, KE))
        print("\n----------------------------------------------------------")
        print("MPI-parallel Python spectralLES simulation finished at {}."
              .format(timeofday()))
        print("----------------------------------------------------------")

    # -- output kinetic energy spectrum to file
    spect3d = np.sum(np.real(U_hat*np.conj(U_hat)), axis=0)
    spect3d[..., 0] *= 0.5
    spect1d = shell_average(comm, spect3d, Kmod)

    if comm.rank == 0:
        fh = open('%s/%s-%3.3d_KE.spectra' %
                  (pp.adir, pp.pid, ispec), 'w')
        metadata = Ek_fmt('u_i')
        fh.write('%s\n' % metadata)
        spect1d.tofile(fh, sep='\n', format='% .8e')
        fh.close()

    # -- output physical-space solution fields for restarting and analysis
    writer.write_scalar('%s-Velocity1_%3.3d.rst' %
                        (pp.pid, irst), U[0], np.float64)
    writer.write_scalar('%s-Velocity2_%3.3d.rst' %
                        (pp.pid, irst), U[1], np.float64)
    writer.write_scalar('%s-Velocity3_%3.3d.rst' %
                        (pp.pid, irst), U[2], np.float64)

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

time_group.add_argument('--cfl', type=float, default=0.45, help='CFL number')
time_group.add_argument('-t', '--tlimit', type=float, default=1.0,
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

anlzr_group.add_argument('--adir', type=str, default='./analysis',
                         help='output directory for analysis products')
anlzr_group.add_argument('--dt_stat', type=float,
                         help='time between statistical analysis outputs')
anlzr_group.add_argument('--dt_spec', type=float,
                         help='time between isotropic power spectral density'
                              ' outputs')


###############################################################################
if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    ABC_static_test()
