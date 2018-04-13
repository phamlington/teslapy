"""
spectralLES: a pure-Python pseudo-spectral large eddy simulation solver
for model development and testing originally based upon the
spectralDNS3D_short Taylor-Green Vortex simulation code written by
Mikael Mortensen.
(<https://github.com/spectralDNS/spectralDNS/spectralDNS3D_short.py>)

Description:
============

Notes:
======

Indexing convention:
--------------------
Since TESLa has mostly worked in MATLAB and Fortran, it is common for us
to think in terms of column-major index order, i.e., [x1, x2, x3], where
x1 is contiguous in memory and x3 is always the inhomogenous dimension
in the Athena-RFX flame geometry. However, Python and C/C++ are row-
major index order, i.e., where x3 is contiguous in memory and x1 is the
inhomogenous dimension.

Coding Style Guide:
-------------------
This module generally adheres to the Python style guide published in
PEP 8, with the following exceptions:
- Warning W503 (line break occurred before a binary operator) is
  ignored, since this warning is backwards and PEP 8 actually recommends
  breaking before operators rather than after
- Error E225 (missing whitespace around operator) is ignored because
  sometimes breaking this rule produces more readable code.

For more information see <http://pep8.readthedocs.org/en/latest/intro.html>

Additionally, I have not yet, but plan to eventually get all of these
docstrings whipped into shape and in compliance with the Numpy/Scipy
style guide for documentation:
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>

Finally, this module should always strive to achieve enlightenment by
following the the Zen of Python (PEP 20, just `import this` in a Python
shell) and using idiomatic Python (i.e. 'Pythonic') concepts and design
patterns.

Definitions:
============
K     - the Fourier-space spatial-frequency vector, or "wavevector"
Ksq   - the wavevector magnitude squared, k^2 = k_ik_i

Authors:
========
Colin Towery (colin.towery@colorado.edu)
Olga Doronina (olga.doronina@colorado.edu)

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
https://github.com/teslacu/spectralLES.git
"""
from mpi4py import MPI
import numpy as np
from math import *
import argparse

from teslacu.fft import rfft3, irfft3   # FFT transforms
from teslacu.stats import psum          # statistical functions


class LoadInputFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        """
        this overloads the template argparse.Action.__call__ method
        and must keep the same argument names
        """

        # argument values is an open file handle, with statement is just added
        # prettiness to make it clear that values is a file handle
        with values as fh:
            raw_lines = fh.read().splitlines()
            fh.close()

        stripped_lines = [rl.strip().split('#')[0] for rl in raw_lines]
        # the following strip blank lines, since '' evaluates to False
        input_lines = [sl for sl in stripped_lines if sl]
        args = []

        for line in input_lines:
            if line.startswith('<'):
                # These file lines are considered special just in case
                # we ever want to do something with these headers, like
                # spit the whole simulation config back out to stdout/log
                # like Athena does.
                #
                # line.strip()[1:-1]
                pass
            else:
                key, val = [kv.strip() for kv in line.strip().split('=')]
                args.append('--%s=%s' % (key, val))

        parser.parse_known_args(args, namespace)

        return


###############################################################################
class spectralLES(object):
    """
    Class Variables:
        parser:

    Class Constructor:

        Regular Arguments:
            comm:
            N:
            L:
            nu:
            epsilon:
            Gtype:

        Optional Keyword Arguments:
            k_les:
            k_test:
            kfLow:
            kfHigh:
    """

    # Class Variables ---------------------------------------------------------
    parser = argparse.ArgumentParser(prog='spectralLES', add_help=False)

    parser.description = ('a pure-Python pseudo-spectral large eddy simulation'
                          ' solver for model development and testing')

    parser.add_argument('-f', type=open, action=LoadInputFile, metavar='file',
                        help='path to file-based input arguments')

    _config_group = parser.add_argument_group(
                        'problem configuration arguments')

    _config_group.add_argument('-N', '--N', type=int, nargs='*', default=[64],
                               metavar=('Nx', 'Ny'),
                               help='mesh dimensions')
    _config_group.add_argument('-L', '--L', type=float, nargs='*',
                               default=[2.0*pi], metavar=('Lx', 'Ly'),
                               help='domain dimensions')
    _config_group.add_argument('--nu', type=float, default=0.0011,
                               help='viscosity')
    _config_group.add_argument('--epsilon', type=float, default=1.2,
                               help='target energy dissipation rate')
    _config_group.add_argument('--kfLow', type=int,
                               help='low-wavenumber cutoff of HIT forcing')
    _config_group.add_argument('--kfHigh', type=int,
                               help='high-wavenumber cutoff of HIT forcing')

    _solver_group = parser.add_argument_group(
                        'solver configuration arguments')

    _solver_group.add_argument('--filter_type', type=str, dest='Gtype',
                               default='comp_exp',
                               choices=['spectral', 'comp_exp', 'tophat'],
                               help='shape of the test and LES filters')
    _solver_group.add_argument('--k_les', type=int,
                               help='cutoff wavenumber of LES filter')
    _solver_group.add_argument('--k_test', type=int,
                               help='cutoff wavenumber of test filter')

    # Class Constructor -------------------------------------------------------
    def __init__(self, comm, N, L, nu, epsilon, Gtype, **kwargs):

        # process input arguments ---------------------------------------------
        self.comm = comm

        if np.iterable(N):
            if len(N) == 1:
                self.nx = np.array(list(N)*3, dtype=int)
            elif len(N) == 3:
                self.nx = np.array(N, dtype=int)  # "analysis nx"
            else:
                raise IndexError("The length of nx must be either 1 or 3")
        else:
            self.nx = np.array([int(N)]*3, dtype=int)

        if np.iterable(L):
            if len(L) == 1:
                self.L = np.array(list(L)*3)
            elif len(L) == 3:
                self.L = np.array(L)  # "analysis nx"
            else:
                raise IndexError("The length of L must be either 1 or 3")
        else:
            self.L = np.array([float(L)]*3)

        self.nu = nu
        self.epsilon = epsilon

        self.k_les = kwargs.pop('k_les', None) or int(sqrt(2)*self.nx.min()/3)
        self.k_test = kwargs.pop('k_test', None) or self.k_les//2

        # so long as k_les is integer, need to dimensionalize
        self.D_les = self.L.min()/self.k_les
        self.D_test= self.L.min()/self.k_test

        # add all remaining arguments to the namespace, so that the user
        # may store them in the solver instance for use later
        self.kfHigh = kwargs.pop('kfHigh', None)
        self.kfLow = kwargs.pop('kfLow', None)
        for name in kwargs:
            setattr(self, name, kwargs[name])

        # compute remaining instance variables from inputs --------------------
        # -- MPI Global domain variables (1D Decomposition)
        self.dx = self.L/self.nx
        self.Nx = self.nx.prod()
        self.Nxinv = 1.0/self.Nx

        self.nk = self.nx.copy()
        self.nk[2] = self.nx[2]//2+1
        self.dk = 1.0/self.L

        # -- MPI Local physical-space subdomain variables (1D Decomposition)
        self.nnx = self.nx.copy()
        self.ixs = np.zeros(3, dtype=int)
        self.ixe = self.nx.copy()

        self.nnx[0] = self.nx[0]//self.comm.size
        self.ixs[0] = self.nnx[0]*self.comm.rank
        self.ixe[0] = self.ixs[0]+self.nnx[0]

        self.X = np.mgrid[self.ixs[0]:self.ixe[0],
                          self.ixs[1]:self.ixe[1],
                          self.ixs[2]:self.ixe[2]].astype(np.float64)
        self.X *= self.dx.reshape((3, 1, 1, 1))

        # -- MPI Local spectral-space subdomain variables (1D Decomposition)
        self.nnk = self.nk.copy()
        self.nnk[1] = self.nx[1]//self.comm.size

        self.iks = np.zeros(3, dtype=int)
        self.iks[1] = self.nnk[1]*self.comm.rank
        self.ike = self.iks+self.nnk

        # !WARNING!
        # these following wavevectors and wavenumbers are in units of
        # dk_i, and each dk_i can be different, such that equal integer
        # values of ki do not correspond to equal dimensionalized
        # spatial frequencies. I kept K as integer values to match the
        # results of spectralDNS32_short.py for the time being.
        # This will give incorrect derivatives for domains with outer
        # dimensions L_i /= 2pi, which would make dk_i = 2pi/L_i /= 1!
        # However, Leray-Hopf/Helmholtz/Hodge projection is unnaffected
        # by the magnitudes of dk_i
        k0 = np.fft.fftfreq(self.nx[0])*self.nx[0]
        k1 = np.fft.fftfreq(self.nx[1])[self.iks[1]:self.ike[1]]*self.nx[1]
        k2 = np.fft.rfftfreq(self.nx[2])*self.nx[2]
        self.K = np.array(np.meshgrid(k0, k1, k2, indexing='ij'), dtype=int)

        self.Ksq = np.sum(np.square(self.K), axis=0)
        self.K_Ksq = self.K * np.where(self.Ksq==0, 1.0,
                                       self.Ksq.astype(np.float64))**-1

        # -- MPI Local subdomain filter kernel arrays
        self.les_filter = self.filter_kernel(self.k_les, Gtype)
        self.test_filter = self.filter_kernel(self.k_test, Gtype)

        if self.kfHigh or self.kfLow:
            self.hit_filter = np.ones_like(self.les_filter)
            if self.kfHigh:
                self.hit_filter *= self.filter_kernel(self.kfHigh)
            if self.kfLow:
                self.hit_filter *= 1.0 - self.filter_kernel(self.kfLow)
        else:
            self.hit_filter = 1.0

        # -- MPI Local subdomain data arrays (1D Decomposition)
        nnz, ny, nx = self.nnx
        nz, nny, nk = self.nnk

        # real vector field memory
        self.U = np.empty((3, nnz, ny, nx))     # solution vector
        self.W = np.empty_like(self.U)          # work vector
        self.omega = self.W                     # vorticity

        # complex vector field memory
        self.U_hat = np.empty((3, nz, nny, nk), dtype=complex)
        self.U_hat0= np.empty_like(self.U_hat)
        self.U_hat1= np.empty_like(self.U_hat)
        self.W_hat = np.empty_like(self.U_hat)  # work vector
        self.dU = np.empty_like(self.U_hat)     # RHS accumulator

        # real and complex tensor field memory
        self.A = np.empty((3, 3, nnz, ny, nx))
        # self.A_hat = np.empty((3, 3, nz, nny, nk), dtype=complex)

    # Object-Handling Methods -------------------------------------------------

    def __enter__(self):
        # with-statement initialization
        return self

    def __exit__(self, type, value, tb):
        # with-statement finalization
        pass

    # Instance Methods --------------------------------------------------------

    def filter_kernel(self, kf, Gtype='spectral', k_kf=None,
                      dtype=np.complex128):
        """
        kf    - input cutoff wavenumber for ensured isotropic filtering
        Gtype - (Default='spectral') filter kernel type
        k_kf  - (Default=None) spectral-space wavenumber field pre-
                normalized by filter cutoff wavenumber. Pass this into
                FILTER_KERNEL for anisotropic filtering since this
                function generates isotropic filter kernels by default.
                If not None, kf is ignored.
        """

        if k_kf is None:
            A = self.L/self.L.min()  # domain size aspect ratios
            A.resize((3, 1, 1, 1))   # ensure proper array broadcasting
            kmag = np.sqrt(np.sum(np.square(self.K/A), axis=0))
            k_kf = kmag/kf

        Ghat = np.empty(k_kf.shape, dtype=dtype)

        if Gtype == 'spectral':
            Ghat[:] = (np.abs(k_kf) < 1.0).astype(dtype)

        elif Gtype == 'comp_exp':
            # A 'COMPact EXPonential' filter which:
            #   1) has compact support in a ball of spectral (physical)
            #       radius kf (1/kf)
            #   2) is strictly positive, and
            #   3) is smooth (infinitely differentiable)
            #      in _both_ physical and spectral space!
            with np.errstate(divide='ignore'):
                Ghat[:] = np.exp(-k_kf**2/(0.25-k_kf**2), where=k_kf < 0.5,
                                 out=np.zeros_like(k_kf)).astype(dtype)

            G = irfft3(self.comm, Ghat)
            G[:] = np.square(G)
            rfft3(self.comm, G, Ghat)
            Ghat *= 1.0/self.comm.allreduce(Ghat[0, 0, 0], op=MPI.MAX)
            Ghat -= 1j*np.imag(Ghat)

            # elif Gtype == 'inv_comp_exp':
            #     # Same as 'comp_exp' but the physical-space and spectral-
            #     # space kernels are swapped so that the physical-space kernel
            #     # has only a central lobe of support.
            #     H = np.exp(-r_rf**2/(1.0-r_rf**2))
            #     G = np.where(r_rf < 1.0, H, 0.0)
            #     rfft3(self.comm, G, Ghat)
            #     Ghat[:] = Ghat**2
            #     G[:] = irfft3(self.comm, Ghat)
            #     G /= self.comm.allreduce(psum(G), op=MPI.SUM)
            #     rfft3(self.comm, G, Ghat)

        elif Gtype == 'tophat':
            Ghat[:] = np.sin(pi*k_kf)/(pi*k_kf**2)

        else:
            raise ValueError('did not understand filter type')

        return Ghat

    def initialize_Taylor_Green_vortex(self):
        """
        Generates the Taylor-Green vortex velocity initial condition
        """
        self.U[0] = np.sin(self.X[0])*np.cos(self.X[1])*np.cos(self.X[2])
        self.U[1] =-np.cos(self.X[0])*np.sin(self.X[1])*np.cos(self.X[2])
        self.U[2] = 0.0
        rfft3(self.comm, self.U[0], self.U_hat[0])
        rfft3(self.comm, self.U[1], self.U_hat[1])
        rfft3(self.comm, self.U[2], self.U_hat[2])

        return

    def compute_random_HIT_spectrum(self, kexp, kpeak, rseed=None):
        """
        Generates a random, incompressible, velocity field with an
        un-scaled Gamie-Ostriker isotropic turbulence spectrum
        """
        if type(rseed) is int and rseed > 0:
            np.random.seed(rseed)

        # Give each wavevector component a random phase and random magnitude
        # where magnitude is normally-distributed with variance 1 and mean 0
        # This will give RMS magnitude of 1.0
        q1 = np.random.rand(*self.W_hat.shape)   # standard uniform samples
        q2 = np.random.randn(*self.W_hat.shape)  # standard normal samples
        self.W_hat[:] = q2*(np.cos(2*pi*q1)+1j*np.sin(2*pi*q1))

        # Rescale to give desired spectrum

        # - First ensure that the wavenumber magnitudes are isotropic
        A = self.L/self.L.min()  # domain size aspect ratios
        A.resize((3, 1, 1, 1))   # ensure proper array broadcasting
        kmag = np.sqrt(np.sum(np.square(self.K/A), axis=0))

        # - Second, scale to Gamie-Ostriker spectrum with kexp and kpeak
        #   and do not excite modes smaller than dk along the shortest
        #   dimension L (kmag < 1.0).
        with np.errstate(divide='ignore'):
            self.W_hat *= np.power(kmag, kexp-1.0, where=kmag >= 1.0,
                                   out=np.zeros_like(kmag))
        self.W_hat *= self.les_filter*np.exp(-kmag/kpeak)

        return

    def initialize_HIT_random_spectrum(self, Einit=None, kexp=-5./6.,
                                       kpeak=None, rseed=None):
        """
        Generates a random, incompressible, velocity initial condition
        with a scaled Gamie-Ostriker isotropic turbulence spectrum
        """
        if Einit is None:
            Einit = 0.72*(self.epsilon*self.L.max())**(2./3.)
            # the constant of 0.72 is empirically-based
        if kpeak is None:
            a = self.L/self.L.min()         # domain size aspect ratios
            kpeak = np.max((self.nx//8)/a)  # this gives kmax/4

        self.compute_random_HIT_spectrum(kexp, kpeak, rseed)

        # Solenoidally-project, U_hat*(1-ki*kj/k^2)
        self.W_hat -= np.sum(self.W_hat*self.K_Ksq, axis=0)*self.K

        # - Third, scale to Einit
        irfft3(self.comm, self.W_hat[0], self.U[0])
        irfft3(self.comm, self.W_hat[1], self.U[1])
        irfft3(self.comm, self.W_hat[2], self.U[2])

        Urms = sqrt(2.0*Einit)
        self.U *= Urms*sqrt(self.Nx/self.comm.allreduce(psum(self.U**2)))

        # transform to finish initial conditions
        rfft3(self.comm, self.U[0], self.U_hat[0])
        rfft3(self.comm, self.U[1], self.U_hat[1])
        rfft3(self.comm, self.U[2], self.U_hat[2])

        return

    def computeSource_HIT_random_forcing(self, rseed=None, **ignored):
        """
        Source function to be added to spectralLES solver instance

        Takes one keyword argument:
        rseed: (positive integer, optional), changes the random seed of
            the pseudo-RNG inside the np.random module
        """
        self.compute_random_HIT_spectrum(-5./3., self.nk[-1], rseed)
        self.W_hat *= self.hit_filter

        irfft3(self.comm, self.W_hat[0], self.W[0])
        irfft3(self.comm, self.W_hat[1], self.W[1])
        irfft3(self.comm, self.W_hat[2], self.W[2])
        dvScale = self.epsilon/self.comm.allreduce(psum(self.W*self.U))

        self.W_hat *= dvScale
        self.dU += self.W_hat

        return dvScale

    def computeSource_linear_forcing(self, dvScale=None, computeRHS=True,
                                     **ignored):
        """
        Source function to be added to spectralLES solver instance
        inclusion of keyword dvScale necessary to actually compute the
        source term

        Takes one keyword argument:
        dvScale: (optional) user-provided linear scaling
        computeRHS: (default=True) add source term to RHS accumulator
        """
        # Update the HIT forcing function
        self.W_hat[:] = self.U_hat*self.hit_filter

        if dvScale is None:
            irfft3(self.comm, self.W_hat[0], self.W[0])
            irfft3(self.comm, self.W_hat[1], self.W[1])
            irfft3(self.comm, self.W_hat[2], self.W[2])
            dvScale = self.epsilon*self.Nx/self.comm.allreduce(
                                                        psum(self.U*self.W))

        if computeRHS:
            self.dU += dvScale*self.W_hat

        return dvScale

    def computeSource_Smagorinksy_SGS(self, Cs=1.2, **ignored):
        """
        Smagorinsky Model (takes Cs as input)

        Takes one keyword argument:
        Cs: (float, optional), Smagorinsky constant
        """
        for i in range(3):
            for j in range(3):
                self.A[j, i] = 0.5*irfft3(self.comm,
                                          1j*(self.K[j]*self.U_hat[i]
                                              +self.K[i]*self.U_hat[j]))

        # compute SGS flux tensor, nuT = 2|S|(Cs*D)**2
        nuT = self.W[0]
        nuT = np.sqrt(2.0*np.sum(np.square(self.A), axis=(0, 1)))
        nuT*= 2.0*(Cs*self.D_les)**2

        self.W_hat[:] = 0.0
        for i in range(3):
            for j in range(3):
                self.W_hat[i]+= 1j*self.K[j]*rfft3(self.comm, self.A[j, i]*nuT)

        self.dU += self.W_hat

        return

    def computeAD_vorticity_form(self, **ignored):
        """
        Computes right-hand-side (RHS) advection and diffusion term of
        the incompressible Navier-Stokes equations using a vorticity
        formulation for the advection term.

        This function overwrites the previous contents of self.dU.
        """
        K = self.K
        U_hat = self.U_hat
        U = self.U
        omega = self.omega
        comm = self.comm

        # take curl of velocity to get vorticity and inverse transform
        irfft3(comm, 1j*(K[1]*U_hat[2] - K[2]*U_hat[1]), omega[0])
        irfft3(comm, 1j*(K[2]*U_hat[0] - K[0]*U_hat[2]), omega[1])
        irfft3(comm, 1j*(K[0]*U_hat[1] - K[1]*U_hat[0]), omega[2])

        # compute convective transport as the physical-space cross-product of
        # vorticity and velocity and forward transform
        rfft3(comm, U[1]*omega[2] - U[2]*omega[1], self.dU[0])
        rfft3(comm, U[2]*omega[0] - U[0]*omega[2], self.dU[1])
        rfft3(comm, U[0]*omega[1] - U[1]*omega[0], self.dU[2])

        # Compute the diffusive transport term and add to the convective term
        self.dU -= self.nu*self.Ksq*self.U_hat

        return

    def new_dt_constant_nu(self, cfl):
        u1m = u2m = u3m = 0.0
        u1m = self.comm.allreduce(np.max(self.U[0]), op=MPI.MAX)
        u2m = self.comm.allreduce(np.max(self.U[1]), op=MPI.MAX)
        u3m = self.comm.allreduce(np.max(self.U[2]), op=MPI.MAX)

        dtMinHydro = cfl*min(self.dx[0]/u1m, self.dx[1]/u2m, self.dx[2]/u3m)
        dtMinDiff = min(self.dx)**2/(2.0*self.nu)
        dtMin = min(dtMinHydro, dtMinDiff)
        if dtMinDiff < dtMinHydro:
            if self.comm.rank == 0:
                print("timestep limited by diffusion! {} {}"
                      .format(dtMinHydro, dtMinDiff))

        return dtMin

    def RK4_integrate(self, dt, *Sources, **kwargs):
        """
        4th order Runge-Kutta time integrator for spectralLES

        Arguments:
        ----------
        dt: current timestep
        *Sources: (Optional) User-supplied source terms. This is a
            special Python syntax, basically any argument you feed
            RK4_integrate() after dt will be stored in the list
            Source_terms. If no arguments are given, Sources = [],
            in which case the loop is skipped.
        **kwargs: (Optional) the keyword arguments to be passed to all
            Sources.
        """

        a = [1./6., 1./3., 1./3., 1./6.]
        b = [0.5, 0.5, 1.]

        self.U_hat1[:] = self.U_hat0[:] = self.U_hat

        for rk in range(4):

            irfft3(self.comm, self.U_hat[0], self.U[0])
            irfft3(self.comm, self.U_hat[1], self.U[1])
            irfft3(self.comm, self.U_hat[2], self.U[2])

            self.computeAD(**kwargs)
            for computeSource in Sources:
                computeSource(**kwargs)

            # Filter the nonlinear contributions to the RHS
            self.dU *= self.les_filter

            # Apply the Leray-Hopf projection operator (1 - Helmholtz
            # operator) to filtered nonlinear contributions in order to
            # enforce the divergence-free continuity condition.
            # This operation is equivalent to computing the pressure
            # field using a physical-space pressure-Poisson solver and
            # then adding the pressure-gradient transport term to the RHS.
            self.dU -= np.sum(self.dU*self.K_Ksq, axis=0)*self.K

            if rk < 3:
                self.U_hat[:] = self.U_hat0 + b[rk]*dt*self.dU
            self.U_hat1[:] += a[rk]*dt*self.dU

        irfft3(self.comm, self.U_hat[0], self.U[0])
        irfft3(self.comm, self.U_hat[1], self.U[1])
        irfft3(self.comm, self.U_hat[2], self.U[2])

        return

###############################################################################
