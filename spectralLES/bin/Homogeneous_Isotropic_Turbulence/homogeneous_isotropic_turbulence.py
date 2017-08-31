"""
Description:
------------
Simulation and analysis of incompressible homogeneous isotropic turbulence
using draft versions of TESLaCU Python modules and routines from the
spectralDNS3D_short Taylor-Green Vortex simulation code written by Mikael
Mortensen (<https://github.com/spectralDNS/spectralDNS/spectralDNS3D_short.py>)

Parameter File and Command Line Options:
----------------------------------------
*See Notes for more details
-i <input directory>
-o <output directory>
-p <problem ID>
-L <Domain box size>
-N <grid cells per dimension>
-c <CFL coefficient>
--tlimit <simulation time limit>
--dt_rst
--dt_bin
--dt_hst
--dt_spec
--nu <kinematic viscosity>
--eps <energy injection rate>
--Uinit <Urms of random IC>
--k_exp <power law exponent of random IC>
--k_peak <exponential decay scaling>

Notes:
------
*Coming soon

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
import sys
import getopt
# import os
import time
from math import *

from spectralLES import spectralLES
from teslacu import mpiAnalyzer, mpiWriter
from teslacu.fft import irfft3  # FFT transforms
from HIT_analysis_functions import scalar_analysis

comm = MPI.COMM_WORLD


def homogeneous_isotropic_turbulence(args):
    if comm.rank == 0:
        print("Python MPI spectral DNS simulation of problem "
              "`Homogeneous Isotropic Turbulence' started with "
              "{} tasks at {}.".format(comm.size, timeofday()))

    # -------------------------------------------------------------------------

    (idir, odir, pid, L, N, cfl, tlimit, dt_rst, dt_bin, dt_hst, dt_spec, nu,
     eps_inj, Urms, k_exp, k_peak) = args
    eps_inj *= L**3  # convert power density to absolute power

    if N % comm.size > 0:
        if comm.rank == 0:
            print('Job started with improper number of MPI tasks for the '
                  'size of the data specified!')
        MPI.Finalize()
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Configure a spectralLES solver, data writer, and data analyzer with the
    # appropriate class factories, MPI communicators, and paramters.
    # In the future a data plotter object could also be generated here!

    writer = mpiWriter(comm=comm, odir=("%s/data/" % odir), nx=[N]*3)

    analyzer = mpiAnalyzer(comm=comm, idir=idir, odir=("%s/analysis/" % odir),
                           probID=pid, L=L, nx=N, geo='hit', method='akima')

    Ek_fmt = "\widehat{{{0}}}^*\widehat{{{0}}}".format
    emin = np.inf
    emax = np.NINF
    analyzer.mpi_moments_file = '%s%s.moments' % (analyzer.odir, pid)

    # CONFIGURE SOLVER - use default Gtype, les_scale, and test_scale
    solver = spectralLES(comm, L, N, nu, eps_inj=eps_inj)

    # - create filter kernel for spectrally-truncated forcing
    # -- start with a low-pass filter
    solver.forcing_filter = solver.filter_kernel(4.0, 'spectral')
    # -- multipy by high-pass filter to get a band-pass filter
    solver.forcing_filter*= 1.0-solver.filter_kernel(2.0, 'spectral')

    # - finish configuring solver with RHS terms
    solver.computeAD = solver.computeAD_vorticity_formulation
    Sources = [solver.computeSource_HIT_linear_forcing,
               solver.computeSource_Smagorinksy_SGS]
    kwargs = {}

    # -------------------------------------------------------------------------
    # Initialize the simulation

    t_sim = t_rst = t_bin = t_hst = t_spec = 0.0
    # t_drv = 0
    tstep = irst = iout = 0
    # idrv = 0

    if dt_hst % dt_spec > 1.e-6*dt_spec:
        # ensures that all analysis outputs are in sync (see below)
        dt_hst -= dt_hst % dt_spec
    if dt_bin % dt_hst > 1.e-6*dt_hst:
        # ensures that Enstrophy is computed for binary output (see below)
        dt_bin -= dt_bin % dt_hst

    # currently using a fixed random seed of comm.rank for testing
    np.random.seed(comm.rank)  # sets random seed for all RNG functions
    # int_seeds = np.random.randint(1, 2147483648, size=(10000,), dtype=int)
    solver.Initialize_HIT_random_spectrum(Urms, k_exp, k_peak)
    # solver.Initialize_Taylor_Green_vortex()

    # Update the dynamic dt based on CFL constraint
    dt = solver.new_dt_const_nu(cfl)

    # -------------------------------------------------------------------------
    # Run the simulation

    while t_sim < tlimit+1.e-8:

        # output stdout/log messages every step if needed/wanted
        KE = 0.5*comm.allreduce(np.sum(np.square(solver.U)))*(1./N)**3
        if comm.rank == 0:
            print("time = %15.8e\tdt = %15.8e\tKE = %15.8e"
                  % (t_sim, dt, KE))

        # Output snapshots and data analysis products
        t_test = t_sim + 0.5*dt
        if t_test >= t_spec:
            analyzer.spectral_density(solver.U_hat, '%3.3d_u' % iout,
                                      'velocity PSD', Ek_fmt('u_i'))
            t_spec += dt_spec

            if t_test >= t_hst:
                solver.omga[2] = irfft3(comm,
                                        1j*(solver.K[0]*solver.U_hat[1]
                                            -solver.K[1]*solver.U_hat[0]))
                solver.omga[1] = irfft3(comm,
                                        1j*(solver.K[2]*solver.U_hat[0]
                                            -solver.K[0]*solver.U_hat[2]))
                solver.omga[0] = irfft3(comm,
                                        1j*(solver.K[1]*solver.U_hat[2]
                                            -solver.K[2]*solver.U_hat[1]))
                enst = 0.5*np.sum(np.square(solver.omga), axis=0)

                emin = min(emin, comm.allreduce(np.min(enst), op=MPI.MIN))
                emax = max(emax, comm.allreduce(np.max(enst), op=MPI.MAX))

                scalar_analysis(analyzer, enst, (emin, emax), None, None,
                                '%3.3d_enst' % iout, 'enstrophy', '\Omega')

                analyzer.spectral_density(solver.omga, '%3.3d_omga' % iout,
                                          'vorticity PSD', Ek_fmt('\omega_i'))
                t_hst += dt_hst

                if t_test >= t_bin:
                    writer.write_scalar('Enstrophy_%3.3d.bin' % iout, enst,
                                        dtype=np.float32)
                    t_bin += dt_bin

            iout += 1

        # Output restart files
        if t_test >= t_rst:
            writer.write_scalar('Velocity1_%3.3d.rst' % irst, solver.U[0],
                                dtype=np.float64)
            writer.write_scalar('Velocity2_%3.3d.rst' % irst, solver.U[1],
                                dtype=np.float64)
            writer.write_scalar('Velocity3_%3.3d.rst' % irst, solver.U[2],
                                dtype=np.float64)
            t_rst += dt_rst
            irst += 1

        # Update the random forcing pattern by generating a new random seed
        # to be passed to the source functions
        # if t_test >= t_drv:
        #     # get the next random integer in the range [1, 2**31)
        #     rseed = int(int_seeds[idrv])  # convert to basic Python int type
        #     # set the driving refresh rate dynamically
        #     # driving should allow for 3 integral-velocity cell-crossing times
        #     # according to Alexei
        #     t_drv+= 3.0*solver.dx.max()/sqrt(2*KE/3.)
        #     idrv += 1
        #     if comm.rank == 0:
        #         print("---- updated random seed ----")

        # Integrate the solution forward in time
        solver.RK4_integrate(dt, *Sources, **kwargs)

        t_sim += dt
        tstep += 1

        # Update the dynamic dt based on CFL constraint
        dt = solver.new_dt_const_nu(cfl)

        sys.stdout.flush()  # forces Python 3 to flush print statements

    # -------------------------------------------------------------------------
    # Finalize the simulation

    # Output restarting and data analysis files
    analyzer.spectral_density(solver.U_hat, '%3.3d_u' % iout, 'velocity PSD',
                              Ek_fmt('u_i'))

    solver.omga[2] = irfft3(comm, 1j*(solver.K[0]*solver.U_hat[1]
                                      -solver.K[1]*solver.U_hat[0]))
    solver.omga[1] = irfft3(comm, 1j*(solver.K[2]*solver.U_hat[0]
                                      -solver.K[0]*solver.U_hat[2]))
    solver.omga[0] = irfft3(comm, 1j*(solver.K[1]*solver.U_hat[2]
                                      -solver.K[2]*solver.U_hat[1]))
    enst = 0.5*np.sum(np.square(solver.omga), axis=0)

    writer.write_scalar('Enstrophy_%3.3d.bin' % iout, enst, dtype=np.float32)

    emin = min(emin, comm.allreduce(np.min(enst), op=MPI.MIN))
    emax = max(emax, comm.allreduce(np.max(enst), op=MPI.MAX))
    scalar_analysis(analyzer, enst, (emin, emax), None, None,
                    '%3.3d_enst' % iout, 'enstrophy', '\Omega')
    analyzer.spectral_density(solver.omga, '%3.3d_omga' % iout,
                              'vorticity PSD', Ek_fmt('\omega_i'))

    writer.write_scalar('Velocity1_%3.3d.rst' % irst, solver.U[0],
                        dtype=np.float64)
    writer.write_scalar('Velocity2_%3.3d.rst' % irst, solver.U[1],
                        dtype=np.float64)
    writer.write_scalar('Velocity3_%3.3d.rst' % irst, solver.U[2],
                        dtype=np.float64)

    return


###############################################################################
def timeofday():
    return time.strftime("%H:%M:%S")


def get_inputs():
    """
    Command Line Options:
    ---------------------
    -i <input directory>
    -o <output directory>
    -p <problem ID>
    -L <Domain box size>
    -N <grid cells per dimension>
    -c <CFL coefficient>
    --tlimit <simulation time limit>
    --dt_rst
    --dt_bin
    --dt_hst
    --dt_spec
    --nu <kinematic viscosity>
    --eps <energy injection rate>
    --Uinit <Urms of random IC>
    --k_exp <power law exponent of random IC>
    --k_peak <exponential decay scaling>
    """

    # import 'defaults' from parameters file
    from HIT_parameters import idir, odir, pid, L, N, cfl, tlimit, dt_rst, \
        dt_bin, dt_hst, dt_spec, nu, eps_inj, Urms, k_exp, k_peak

    help_string = ("spectralLES HIT solver command line options:\n"
                   "-i <input directory>\n"
                   "-o <output directory>\n"
                   "-p <problem ID>\n"
                   "-L <Domain box size>\n"
                   "-N <grid cells per dimension>\n"
                   "-c <CFL coefficient>\n"
                   "--tlimit <simulation time limit>\n"
                   "--dt_rst <output rate of restart fields>\n"
                   "--dt_bin <output rate of single-precision fields>\n"
                   "--dt_hst <output rate of analysis histograms>\n"
                   "--dt_spec <output rate of 1D velocity spectra>\n"
                   "--nu <kinematic viscosity>\n"
                   "--eps <energy injection rate>\n"
                   "--Uinit <Urms of random IC>\n"
                   "--k_exp <power law exponent of random IC>\n"
                   "--k_peak <exponential decay scaling>\n")

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:p:L:N:c:",
                                   ["tlimit=", "dt_rst=", "dt_bin=", "dt_hst=",
                                    "dt_spec=", "nu=", "eps=", "Uinit=",
                                    "k_exp=", "k_peak="])
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
        try:
            sys.exit(e.errno)
        except:
            sys.exit(999)

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
            elif opt=='-L':
                L = float(arg)
                if comm.rank == 0:
                    print('L:\t\t\t{}'.format(L))
            elif opt=='-N':
                N = int(arg)
                if comm.rank == 0:
                    print('N:\t\t\t{}'.format(N))
            elif opt=='-c':
                cfl = float(arg)
                if comm.rank == 0:
                    print('CFL:\t\t\t{}'.format(cfl))
            elif opt=='--tlimit':
                tlimit = float(arg)
                if comm.rank == 0:
                    print('tlimit:\t\t\t{}'.format(tlimit))
            elif opt=='--dt_rst':
                dt_rst = float(arg)
                if comm.rank == 0:
                    print('dt_rst:\t\t\t{}'.format(dt_rst))
            elif opt=='--dt_bin':
                dt_bin = float(arg)
                if comm.rank == 0:
                    print('dt_bin:\t\t\t{}'.format(dt_bin))
            elif opt=='--dt_hst':
                dt_hst = float(arg)
                if comm.rank == 0:
                    print('dt_hst:\t\t\t{}'.format(dt_hst))
            elif opt=='--dt_spec':
                dt_spec = float(arg)
                if comm.rank == 0:
                    print('dt_spec:\t\t\t{}'.format(dt_spec))
            elif opt=='--nu':
                nu = float(arg)
                if comm.rank == 0:
                    print('nu:\t\t\t{}'.format(nu))
            elif opt=='--eps':
                eps_inj = float(arg)
                if comm.rank == 0:
                    print('eps_inj:\t\t\t{}'.format(eps_inj))
            elif opt=='--Uinit':
                Urms = float(arg)
                if comm.rank == 0:
                    print('Uinit:\t\t\t{}'.format(Urms))
            elif opt=='--k_exp':
                k_exp = float(arg)
                if comm.rank == 0:
                    print('k_exp:\t\t\t{}'.format(k_exp))
            elif opt=='--k_peak':
                k_peak = float(arg)
                if comm.rank == 0:
                    print('k_peak:\t\t\t{}'.format(k_peak))
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

    return (idir, odir, pid, L, N, cfl, tlimit, dt_rst, dt_bin, dt_hst,
            dt_spec, nu, eps_inj, Urms, k_exp, k_peak)

if __name__ == "__main__":
    # np.set_printoptions(formatter={'float': '{: .8e}'.format})
    homogeneous_isotropic_turbulence(get_inputs())
