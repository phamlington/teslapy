"""
`SpectralLES` is a pure-Python pseudospectral large eddy simulation solver
for educational instruction and model development based upon the
spectralDNS3D_short CFD simulation code written by Mikael Mortensen.
(<github.com/spectralDNS/spectralDNS/blob/master/spectralDNS3D_short.py>)


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
- Warnings W503/W504 (line break occurred before/after a binary operator)
- Errors E225/E226 (missing whitespace around operator)
These codes are not enforced by PEP 8 as a rigorous standard and can be
safely ignored.

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
------------
K     - the Fourier-space spatial-frequency vector, or "wavevector"
Ksq   - the wavevector magnitude squared, k^2 = k_ik_i

Authors:
--------
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
from math import sqrt, pi
import argparse

from teslacu.fft import rfft3, irfft3, shell_average   # FFT transforms
from teslacu.stats import psum                         # statistical functions


###############################################################################
# Derived class for argument parsing from a special-format text file
###############################################################################
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
# Base spectralLES solver class (essentially a pure-Python DNS solver)
###############################################################################
class spectralLES(object):
    """
    Empty Docstring!
    """

    # -------------------------------------------------------------------------
    # Class Attributes (an argparser and its argument groups)
    # -------------------------------------------------------------------------
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
    _config_group.add_argument('--nu', type=float, default=0.000185,
                               help='viscosity')
    _config_group.add_argument('--epsilon', type=float, default=0.103,
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

    # -------------------------------------------------------------------------
    # Class Constructor
    # -------------------------------------------------------------------------
    def __init__(self, comm, N, L, nu, epsilon, Gtype, **kwargs):

        # --------------------------------------------------------------
        # Process all of the input arguments
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

        self.k_les = kwargs.pop('k_les', None)

        # add all remaining arguments to the namespace, so that the user
        # may store them in the solver instance for use later
        self.kfHigh = kwargs.pop('kfHigh', None)
        self.kfLow = kwargs.pop('kfLow', None)
        for name in kwargs:
            setattr(self, name, kwargs[name])

        # --------------------------------------------------------------
        # Compute global domain variables
        self.dx = self.L/self.nx
        self.Nx = self.nx.prod()
        self.Nxinv = 1.0/self.Nx

        self.nk = self.nx.copy()
        self.nk[2] = self.nx[2]//2+1
        self.dk = 1.0/self.L

        # --------------------------------------------------------------
        # Compute local subdomain physical-space variables
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

        # --------------------------------------------------------------
        # Compute local subdomain spectral-space variables
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
        with np.errstate(divide='ignore'):
            self.K_Ksq = np.where(self.Ksq > 0, self.K/self.Ksq, 0.0)

        self.Kmod = np.floor(np.sqrt(self.Ksq)).astype(int)

        # --------------------------------------------------------------
        # Compute local subdomain filter kernels
        self.k_dealias = int(sqrt(2)*self.nx.min()/3)
        self.dealias = self.filter_kernel(self.k_dealias)

        self.forcing_filter = np.ones(self.nnk, dtype=complex)
        if self.kfHigh:
            self.forcing_filter *= self.filter_kernel(self.kfHigh)
        if self.kfLow:
            self.forcing_filter *= 1.0 - self.filter_kernel(self.kfLow)

        if self.k_les:
            self.les_filter = self.filter_kernel(self.k_les, Gtype)
            self.D_les = self.L.min()/self.k_les
        else:
            self.les_filter = self.dealias
            self.D_les = self.L.min()/self.k_dealias

        # --------------------------------------------------------------
        # Allocate local subdomain real vector field memory
        self.U = np.empty((3, *self.nnx))       # solution vector
        self.W = np.empty((3, *self.nnx))       # work vector
        self.omega = np.empty((3, *self.nnx))   # vorticity

        # --------------------------------------------------------------
        # Allocate local subdomain real vector field memory
        self.U_hat = np.empty((3, *self.nnk), dtype=complex)  # solution vector
        self.U_hat0= np.empty((3, *self.nnk), dtype=complex)  # solution vector
        self.U_hat1= np.empty((3, *self.nnk), dtype=complex)  # solution vector
        self.W_hat = np.empty((3, *self.nnk), dtype=complex)  # work vector
        self.dU = np.empty((3, *self.nnk), dtype=complex)     # RHS accumulator

    def __enter__(self):
        # with-statement initialization
        return self

    def __exit__(self, type, value, tb):
        # with-statement finalization
        pass

    # -------------------------------------------------------------------------
    # Instance Methods
    # -------------------------------------------------------------------------
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

        Notes
        -----
        This function applies the Leray-Hopf projection operator (aka
        the residual of the Helmholtz operator) to the entire RHS.
        By performing this operation on the entire RHS, we
        simultaneously enforce the divergence-free continuity condition
        _and_ automatically make all SGS source terms deviatoric!

        This operation is equivalent to computing the total _mechanical_
        pressure (aka the thermodynamic pressure plus the non-deviatoric
        component of all source terms) using a physical-space Poisson
        solver and then adding a mechanical-pressure-gradient transport
        term to the RHS.
        """

        a = [1/6, 1/3, 1/3, 1/6]
        b = [0.5, 0.5, 1.0, 0.0]

        self.U_hat1[:] = self.U_hat0[:] = self.U_hat[:]

        for rk in range(4):
            # ----------------------------------------------------------
            # ensure all computeAD and computeSource methods have an
            # updated physical-space solution field
            irfft3(self.comm, self.U_hat[0], self.U[0])
            irfft3(self.comm, self.U_hat[1], self.U[1])
            irfft3(self.comm, self.U_hat[2], self.U[2])

            # ----------------------------------------------------------
            # compute all RHS terms
            self.computeAD(**kwargs)
            for computeSource in Sources:
                computeSource(**kwargs)

            # ----------------------------------------------------------
            # dealias and project the entire RHS
            self.dU *= self.dealias
            self.dU -= np.sum(self.dU*self.K_Ksq, axis=0)*self.K

            # ----------------------------------------------------------
            # accumulate the intermediate RK stages
            self.U_hat[:] = self.U_hat0 + b[rk]*dt*self.dU
            self.U_hat1[:] += a[rk]*dt*self.dU

        # --------------------------------------------------------------
        # update the spectral-space solution field with the final RK stage
        self.U_hat[:] = self.U_hat1[:]

        # --------------------------------------------------------------
        # ensure the user has an updated physical-space solution field
        irfft3(self.comm, self.U_hat[0], self.U[0])
        irfft3(self.comm, self.U_hat[1], self.U[1])
        irfft3(self.comm, self.U_hat[2], self.U[2])

        return

    def initialize_TaylorGreen_vortex(self):
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

        # --------------------------------------------------------------
        # Generate a random, incompressible, velocity field with a
        # Gamie-Ostriker isotropic turbulence spectrum
        self.compute_random_HIT_spectrum(kexp, kpeak, rseed)

        # --------------------------------------------------------------
        # Solenoidally-project, U_hat*(1-ki*kj/k^2)
        self.W_hat -= np.sum(self.W_hat*self.K_Ksq, axis=0)*self.K

        # --------------------------------------------------------------
        # scale to Einit
        irfft3(self.comm, self.W_hat[0], self.U[0])
        irfft3(self.comm, self.W_hat[1], self.U[1])
        irfft3(self.comm, self.W_hat[2], self.U[2])

        Urms = sqrt(2.0*Einit)
        self.U *= Urms*sqrt(self.Nx/self.comm.allreduce(psum(self.U**2)))

        # --------------------------------------------------------------
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
        mpi_reduce = self.comm.allreduce

        # --------------------------------------------------------------
        # generate a random, band-pass-filtered velocity field
        self.compute_random_HIT_spectrum(-5./3., self.nk[-1], rseed)
        self.W_hat *= self.forcing_filter

        # --------------------------------------------------------------
        # scale to constant energy injection rate
        irfft3(self.comm, self.W_hat[0], self.W[0])
        irfft3(self.comm, self.W_hat[1], self.W[1])
        irfft3(self.comm, self.W_hat[2], self.W[2])
        dvScale = self.epsilon*self.Nx/mpi_reduce(psum(self.W*self.U))

        self.W_hat *= dvScale

        # --------------------------------------------------------------
        # add forcing term to the RHS accumulator
        self.dU += self.W_hat

        return dvScale

    def computeSource_linear_forcing(self, dvScale=1.0, **ignored):
        """
        Source function to be added to spectralLES solver instance
        inclusion of keyword dvScale necessary to properly scale the
        source term

        Takes one keyword argument:
        dvScale: user-provided linear scaling
        """
        # Band-pass filter the solution velocity field,
        # scale to constant energy injection rate,
        # and add to the RHS accumulator
        self.W_hat[:] = dvScale*self.U_hat*self.forcing_filter
        self.dU += self.W_hat

        return self.W_hat

    # def computeSource_const_spectrum_forcing(self, Ek, **ignored):
    #     """
    #     Empty docstring!
    #     """
    #     self.W_hat[:] = self.U_hat*self.forcing_filter
    #     self.W_hat *= np.conj(self.W_hat)

    #     spect3d = np.sum(np.real(self.W_hat), axis=0)
    #     spect3d[..., 0] *= 0.5
    #     spect1d = shell_average(self.comm, spect3d, self.Kmod)

    #     self.W_hat[:] = (Ek/spect1d)*self.U_hat*self.forcing_filter
    #     self.dU += self.W_hat

    #     return e_inj

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

        # --------------------------------------------------------------
        # take curl of velocity and inverse transform to get vorticity
        irfft3(comm, 1j*(K[1]*U_hat[2] - K[2]*U_hat[1]), omega[0])
        irfft3(comm, 1j*(K[2]*U_hat[0] - K[0]*U_hat[2]), omega[1])
        irfft3(comm, 1j*(K[0]*U_hat[1] - K[1]*U_hat[0]), omega[2])

        # --------------------------------------------------------------
        # compute the convective transport as the physical-space
        # cross-product of vorticity and velocity
        rfft3(comm, U[1]*omega[2] - U[2]*omega[1], self.dU[0])
        rfft3(comm, U[2]*omega[0] - U[0]*omega[2], self.dU[1])
        rfft3(comm, U[0]*omega[1] - U[1]*omega[0], self.dU[2])

        # --------------------------------------------------------------
        # add the diffusive transport term
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

    def compute_random_HIT_spectrum(self, kexp, kpeak, rseed=None):
        """
        Generates a random, incompressible, velocity field with a
        Gamie-Ostriker isotropic turbulence spectrum
        """
        if type(rseed) is int and rseed > 0:
            np.random.seed(rseed)

        # --------------------------------------------------------------
        # Give each wavevector component a random uniform phase and a
        # random normal magnitude.
        q1 = np.random.rand(*self.W_hat.shape)   # standard uniform samples
        q2 = np.random.randn(*self.W_hat.shape)  # standard normal samples
        self.W_hat[:] = q2*(np.cos(2*pi*q1)+1j*np.sin(2*pi*q1))

        # --------------------------------------------------------------
        # ensure that the wavenumber magnitudes are isotropic
        A = self.L/self.L.min()  # Aspect ratios
        A.resize((3, 1, 1, 1))
        kmag = np.sqrt(np.sum(np.square(self.K/A), axis=0))

        # --------------------------------------------------------------
        # rescale each wavenumber magnitude to Gamie-Ostriker spectrum
        # and do not excite modes |k|/dk < 1.0
        with np.errstate(divide='ignore'):
            self.W_hat *= np.power(kmag, kexp-1.0, where=kmag >= 1.0,
                                   out=np.zeros_like(kmag))
        self.W_hat *= self.dealias*np.exp(-kmag/kpeak)

        return

    def compute_dvScale_constant_injection(self):
        """
        empty docstring!
        """
        mpi_reduce = self.comm.allreduce

        # Band-pass filter the solution velocity field
        self.W_hat[:] = self.U_hat*self.forcing_filter

        # scale to constant energy injection rate
        irfft3(self.comm, self.W_hat[0], self.W[0])
        irfft3(self.comm, self.W_hat[1], self.W[1])
        irfft3(self.comm, self.W_hat[2], self.W[2])
        dvScale = self.epsilon*self.Nx/mpi_reduce(psum(self.U*self.W))

        return dvScale

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

        elif Gtype == 'tophat':
            Ghat[:] = np.sin(pi*k_kf)/(pi*k_kf**2)

        elif Gtype == 'comp_exp':
            # A Compact Exponential filter that:
            #   1) has compact support in _both_ physical and spectral space
            #   2) is strictly positive in _both_ spaces
            #   3) is smooth (infinitely differentiable) in _both_ spaces
            #   4) has simply-connected support in spectral space with
            #      an outer radius kf, and
            #   5) has disconnected (lobed) support in physical space
            #      with an outer radius of 2*pi/kf
            with np.errstate(divide='ignore'):
                Ghat[:] = np.exp(-k_kf**2/(0.25-k_kf**2),
                                 where=k_kf < 0.5,
                                 out=np.zeros_like(k_kf)
                                 ).astype(dtype)

            G = irfft3(self.comm, Ghat)
            G[:] = np.square(G)
            rfft3(self.comm, G, Ghat)
            Ghat *= 1.0/self.comm.allreduce(Ghat[0, 0, 0], op=MPI.MAX)
            Ghat -= 1j*np.imag(Ghat)

        elif Gtype == 'inv_comp_exp':
            # Same as 'comp_exp' but the physical-space and
            # spectral-space kernels are swapped so that the
            # physical-space support is a simply-connected ball
            raise ValueError('inv_comp_exp not yet implemented!')

        else:
            raise ValueError('did not understand filter type')

        return Ghat

    def jit_linear_forcing(dU, U_hat, W_hat, bandpass, dvScale):
        """
        non-object-oriented function for JIT compiling to optimized bit-code
        """
        W_hat[:] = dvScale*U_hat*bandpass
        dU += W_hat

        return W_hat


###############################################################################
# Example of how to extend the spectralLES class
###############################################################################
class staticSmagorinskyLES(spectralLES):
    """
    Empty Docstring!
    """

    # -------------------------------------------------------------------------
    # Class Constructor
    # -------------------------------------------------------------------------
    def __init__(self, **kwargs):
        """
        Just adds some extra working memory on top of the base class.
        See the documentation for the `spectralLES` class.
        """

        super().__init__(**kwargs)
        self.S = np.empty((3, 3, *self.nnx))

    # -------------------------------------------------------------------------
    # Instance Methods
    # -------------------------------------------------------------------------
    def computeSource_Smagorinsky_SGS(self, Cs=1.2, **ignored):
        """
        Smagorinsky Model (takes Cs as input)

        Takes one keyword argument:
        Cs: (float, optional), Smagorinsky constant
        """

        # --------------------------------------------------------------
        # Explicitly filter the solution field
        self.W_hat[:] = self.les_filter*self.U_hat

        for i in range(3):
            for j in range(3):
                self.S[i, j] = irfft3(self.comm,
                                      1j*self.K[j]*self.W_hat[i] +
                                      1j*self.K[i]*self.W_hat[j])

        # --------------------------------------------------------------
        # compute the leading coefficient, nu_T = 2|S|(Cs*D)**2
        nuT = self.W[0]
        nuT[:] = np.sqrt(np.sum(np.square(self.S), axis=(0, 1)))
        nuT *= (Cs*self.D_les)**2

        # --------------------------------------------------------------
        # Compute FFT{div(tau)} and add to RHS update
        self.W_hat[:] = 0.0
        for i in range(3):
            for j in range(3):
                self.W_hat[i]+= 1j*self.K[j]*rfft3(self.comm, nuT*self.S[i, j])

        self.dU += self.W_hat

        return

###############################################################################
