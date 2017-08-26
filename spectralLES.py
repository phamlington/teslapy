"""
module spectralLES

Description:
============

Notes:
======

Indexing convention:
--------------------
Since TESLa has mostly worked in MATLAB and Fortran, it is common for us to
think in terms of column-major index order, i.e., [x1, x2, x3], where x1 is
contiguous in memory and x3 is always the inhomogenous dimension in the
Athena-RFX flame geometry.
However, Python and C/C++ are natively row-major index order, i.e.
[x3, x2, x1], where x1 remains contiguous in memory and x3 remains the
inhomogenous dimension.

The TESLaCU package adheres to row-major order for indexing data grids and when
indexing variables associated with the grid dimensions (e.g. nx, ixs, etc.),
however all vector fields retain standard Einstein notation indexing order,
(e.g. u[0] == u1, u[1] == u2, u[2] == u3).

Coding Style Guide:
-------------------
This module generally adheres to the Python style guide published in PEP 8,
with the following notable exceptions:
- Warning W503 (line break occurred before a binary operator) is ignored
- Error E129 (visually indented line with same indent as next logical line)
  is ignored
- Error E225 (missing whitespace around operator) is ignored

For more information see <http://pep8.readthedocs.org/en/latest/intro.html>

Definitions:
============
self.K - wavevector - the Fourier-space spatial-frequency vector,
                          K = [k3, k2, k1].
kmag   - wavenumber - the wavevector magnitude, k = |kvec|

Authors:
========
Colin Towery

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
"""

from mpi4py import MPI
import numpy as np
from math import *
from teslacu.fft_mpi4py_numpy import *          # FFT transforms
from teslacu.stats_mpi4py_numpy import *        # statistical functions

world_comm = MPI.COMM_WORLD


# def psum(data):
#     """
#     input argument data can be any n-dimensional array-like object, including
#     a 0D scalar value or 1D array.
#     """
#     psum = np.asarray(data)
#     for n in range(data.ndim):
#         psum.sort(axis=-1)
#         psum = np.sum(psum, axis=-1)
#     return psum


class spectralLES(object):
    def __init__(self, comm, L, nx, nu, Gtype='spectral', les_scale=None,
                 test_scale=None, eps_inj=None):

        self.comm = comm

        if np.iterable(nx):
            if len(nx) == 1:
                self.nx = np.array(list(nx)*3, dtype=int)
            elif len(nx) == 3:
                self.nx = np.array(nx, dtype=int)  # "analysis nx"
            else:
                raise IndexError("The length of nx must be either 1 or 3")
        else:
            self.nx = np.array([int(nx)]*3, dtype=int)

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

        # MPI Global domain variables (1D Decomposition)
        self.dx = self.L/self.nx
        self.Nx = self.nx.prod()
        self.Nxinv = 1.0/self.Nx

        self.nk = self.nx.copy()
        self.nk[2] = self.nx[2]//2+1
        self.dk = 1.0/self.L

        # MPI Local physical-space subdomain variables (1D Decomposition)
        self.nnx = self.nx.copy()
        self.ixs = np.zeros(3, dtype=int)
        self.ixe = self.nx.copy()

        self.nnx[0] = self.nx[0]//self.comm.size
        self.ixs[0] = self.nnx[0]*self.comm.rank
        self.ixe[0] = self.ixs[0]+self.nnx[0]

        self.X = np.mgrid[self.ixs[0]:self.ixe[0],
                          self.ixs[1]:self.ixe[1],
                          self.ixs[2]:self.ixe[2]].astype(float)
        for i in range(3):
            self.X[i] *= self.dx[i]

        # MPI Local spectral-space subdomain variables (1D Decomposition)
        self.nnk = self.nk.copy()
        self.nnk[1] = self.nx[1]//self.comm.size

        self.iks = np.zeros(3, dtype=int)
        self.iks[1] = self.nnk[1]*self.comm.rank
        self.ike = self.iks+self.nnk

        # !WARNING!
        # these following wavevectors and wavenumbers are in units of dki,
        # and each dki can be different, such that equal integer values of ki
        # do not correspond to equal dimensionalized spatial frequencies.
        # I kept K as integer values to match the results of
        # spectralDNS32_short.py, for now.
        # This will give incorrect derivatives for domains that are not
        # homogeneous with L = 2*pi! (Leray-Hopf projection is unaffected)
        k1 = np.fft.rfftfreq(self.nx[2])*self.nx[2]
        k2 = np.fft.fftfreq(self.nx[1])*self.nx[1]
        k2 = k2[self.iks[1]:self.ike[1]].copy()
        k3 = np.fft.fftfreq(self.nx[0])*self.nx[0]
        self.K = np.array(np.meshgrid(k3, k2, k1, indexing='ij'), dtype=int)

        self.Ksq = np.sum(np.square(self.K), axis=0)
        # kmag = np.sqrt(self.Ksq.astype(float))
        self.K_Ksq = (self.K.astype(float)
                      /np.where(self.Ksq==0, 1, self.Ksq).astype(float))

        # standard dealias filter
        # this is neither isotropic nor strictly correct for 3D FFTs
        # can be replaced in the the calling program, of course.
        kmax_dealias = self.nx//3
        self.dealias_filter = np.array((np.abs(self.K[0]) < kmax_dealias[0])
                                       *(np.abs(self.K[1]) < kmax_dealias[1])
                                       *(np.abs(self.K[2]) < kmax_dealias[2]),
                                       dtype=bool)

        if les_scale is None:
            self.les_scale = kmax_dealias.min()
        else:
            self.les_scale = les_scale

        self.les_filter = self.filter_kernel(self.les_scale, Gtype)

        if test_scale is None:
            self.test_scale = 0.5*kmax_dealias.min()
        else:
            self.test_scale = test_scale

        self.test_filter = self.filter_kernel(self.test_scale, Gtype)

        self.forcing_filter = np.ones_like(self.test_filter)
        self.forcing_rate = eps_inj

        Ck = 1.6
        Cs = sqrt((pi**-2)*((3*Ck)**-1.5))  # == 0.098...
        # Cs = 0.2
        # so long as K, kmag, scales, etc. are integer, need to dimensionalize
        D = self.L.min()/self.les_scale
        self.smag_coef = 2.0*(Cs*D)**2
        self.nuTmax = 0.0

        # MPI Local subdomain data arrays (1D Decomposition)
        nnz, ny, nx = self.nnx
        nz, nny, nk = self.nnk

        self.U = np.empty((3, nnz, ny, nx))     # solution vector
        self.omga = np.empty_like(self.U)       # vorticity and vector memory
        self.A = np.empty((3, 3, nnz, ny, nx))  # Tensor memory
        # P = np.empty((nnz, ny, nx))

        self.U_hat = np.empty((3, nz, nny, nk), dtype=complex)
        self.U_hat0= np.empty_like(self.U_hat)
        self.U_hat1= np.empty_like(self.U_hat)
        self.S_hat = np.zeros_like(self.U_hat)  # source-term vector memory
        self.dU = np.empty_like(self.U_hat)     # RHS accumulator

    # Class Properities -------------------------------------------------------

    def __enter__(self):
        # with-statement initialization
        return self

    def __exit__(self, type, value, tb):
        # with-statement finalization
        pass

    # Class Methods -----------------------------------------------------------

    def filter_kernel(self, kf, Gtype='comp_exp', k_kf=None,
                      dtype=np.complex128):
        """
        kf    - input cutoff wavenumber for ensured isotropic filtering
        Gtype - (Default='comp_exp') filter kernel type
        k_kf  - (Default=None) spectral-space wavenumber field pre-normalized
                by filter cutoff wavenumber. Pass this into
                filter_kernel(), for anisotropic filtering, since this kernel
                computes isotropic filter kernels by default.
                If not None, kf is ignored.
        """
        if k_kf is None:
            A = self.L/self.L.min()  # domain size aspect ratios
            A.resize((3, 1, 1, 1))   # ensure proper array broadcasting
            kmag = np.sqrt(np.sum(np.square(self.K.astype(float)/A), axis=0))
            k_kf = kmag/kf

        Ghat = np.empty(k_kf.shape, dtype=dtype)

        if Gtype == 'tophat':
            Ghat[:] = np.sin(pi*k_kf)/(pi*k_kf**2)

        elif Gtype == 'comp_exp':
            """
            A 'COMPact EXPonential' filter which:
                1) has compact support in a ball of spectral (physical) radius
                   kf (2/kf)
                2) is strictly positive, and
                3) is smooth (infinitely differentiable)
            in _both_ physical and spectral space!
            """
            Ghat[:] = np.where(k_kf < 0.5,
                               np.exp(-k_kf**2/(0.25-k_kf**2)),
                               0.0).astype(dtype)
            G = irfft3(self.comm, Ghat)
            G[:] = np.square(G)
            rfft3(self.comm, G, Ghat)
            Ghat *= 1.0/self.comm.allreduce(Ghat[0, 0, 0], op=MPI.MAX)
            Ghat -= 1j*np.imag(Ghat)

            # elif Gtype == 'inv_comp_exp':
            #     """
            #     Same as 'comp_exp' but the physical-space and spectral-space
            #     kernels are swapped so that the physical-space kernel has
            #     only a central lobe of support.
            #     """
            #     H = np.exp(-r_rf**2/(1.0-r_rf**2))
            #     G = np.where(r_rf < 1.0, H, 0.0)
            #     rfft3(self.comm, G, Ghat)
            #     Ghat[:] = Ghat**2
            #     G[:] = irfft3(self.comm, Ghat)
            #     G /= self.comm.allreduce(psum(G), op=MPI.SUM)
            #     rfft3(self.comm, G, Ghat)

        elif Gtype == 'spectral':
            Ghat[:] = (np.abs(k_kf) < 1.0).astype(dtype)

        else:
            raise ValueError('did not understand filter type')

        return Ghat

    def Initialize_Taylor_Green_vortex(self):
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

    def get_random_HIT_spectrum(self, k_exp, k_peak, rseed=None):
        """
        Generates a random, incompressible, velocity field with an unnormalized
        Gamie-Ostriker isotropic turbulence spectrum
        """
        if type(rseed) is int and rseed > 0:
            np.random.seed(rseed)

        # Give each wavevector component a random phase and random magnitude
        # where magnitude is normally-distributed with variance 1 and mean 0
        # This will give RMS magnitude of 1.0
        q1 = np.random.rand(*self.S_hat.shape)   # standard uniform samples
        q2 = np.random.randn(*self.S_hat.shape)  # standard normal samples
        self.S_hat[:] = q2*(np.cos(2*pi*q1)+1j*np.sin(2*pi*q1))

        # Rescale to give desired spectrum

        # - First ensure that the wavenumber magnitudes are isotropic
        A = self.L/self.L.min()  # domain size aspect ratios
        A.resize((3, 1, 1, 1))   # ensure proper array broadcasting
        kmag = np.sqrt(np.sum(np.square(self.K.astype(float)/A), axis=0))

        # - Second, scale to Gamie-Ostriker spectrum with k_exp and k_peak
        #   and do not excite modes smaller than dk along the shortest
        #   dimension L (kmag < 1.0).
        self.S_hat *= np.where(kmag<1.0, 0.0, np.power(kmag, k_exp-1.0))
        self.S_hat *= self.dealias_filter*np.exp(-kmag/k_peak)

        return

    def Initialize_HIT_random_spectrum(self, Urms, k_exp, k_peak, rseed=None):
        """
        Generates a random, incompressible, velocity initial condition with a
        Gamie-Ostriker isotropic turbulence spectrum
        """
        self.get_random_HIT_spectrum(k_exp, k_peak, rseed)

        # Solenoidally-project, U_hat*(1-ki*kj/k^2)
        self.S_hat -= np.sum(self.S_hat*self.K_Ksq, axis=0)*self.K

        # - Third, scale to Urms
        self.U[0] = irfft3(self.comm, self.S_hat[0])
        self.U[1] = irfft3(self.comm, self.S_hat[1])
        self.U[2] = irfft3(self.comm, self.S_hat[2])

        self.U *= Urms*sqrt(self.Nx/self.comm.allreduce(psum(self.U**2)))

        # transform to finish initial conditions
        rfft3(self.comm, self.U[0], self.U_hat[0])
        rfft3(self.comm, self.U[1], self.U_hat[1])
        rfft3(self.comm, self.U[2], self.U_hat[2])

        return

    def computeSource_HIT_random_forcing(self, **kwargs):
        """
        Source function to be added to spectralLES solver instance
        all **kwargs are ignored
        """
        rseed = kwargs['random_seed']
        self.get_random_HIT_spectrum(-5./3., self.nk[-1], rseed)
        self.S_hat *= self.forcing_filter

        self.omga[0] = irfft3(self.comm, self.S_hat[0])
        self.omga[1] = irfft3(self.comm, self.S_hat[1])
        self.omga[2] = irfft3(self.comm, self.S_hat[2])
        dvScale = self.forcing_rate/(self.comm.allreduce(
                                     psum(self.omga*self.U))*self.dx.prod())
        self.dU += dvScale*self.S_hat

        # if self.comm.rank == 0:
        #     print("dvScale = {}".format(dvScale))

        return

    def computeSource_HIT_linear_forcing(self, **kwargs):
        """
        Source function to be added to spectralLES solver instance
        all **kwargs are ignored
        """
        # Update the HIT forcing function, use omga as vector memory
        self.S_hat[:] = self.U_hat*self.forcing_filter
        self.omga[0] = irfft3(self.comm, self.S_hat[0])
        self.omga[1] = irfft3(self.comm, self.S_hat[1])
        self.omga[2] = irfft3(self.comm, self.S_hat[2])
        dvScale = self.forcing_rate/(self.comm.allreduce(
                                     psum(self.omga*self.U))*self.dx.prod())
        self.S_hat *= dvScale
        self.dU += self.S_hat

        # self.omga[0] = irfft3(self.comm, self.S_hat[0])
        # self.omga[1] = irfft3(self.comm, self.S_hat[1])
        # self.omga[2] = irfft3(self.comm, self.S_hat[2])
        # eps_inj = 0.5*psum(self.S_hat*np.conj(self.U_hat)
        #                    +np.conj(self.S_hat)*self.U_hat)
        # eps_inj = self.comm.allreduce(eps_inj)
        # eps_ratio = eps_inj/self.forcing_rate
        # if self.comm.rank == 0:
        #     print("---- inj_ratio = %15.8f ----" % eps_ratio)

        return

    def computeSource_Smagorinksy_SGS(self, **kwargs):
        """
        Standard Smagorinsky model (fixed C_s for now)
        Note that self.A is the generic Tensor memory space, with the letter
        S reserved for 'Source', as in self.S_hat.
        all **kwargs are ignored for now, but you could pass in a user-defined
        Cs here, or some other kinds of parameters.
        """
        for i in range(3):
            for j in range(3):
                self.A[j, i] = 0.5*irfft3(self.comm,
                                          1j*(self.K[2-j]*self.U_hat[i]
                                              +self.K[2-i]*self.U_hat[j]))

        # compute SGS flux tensor, nuT = 2|S|(Cs*D)**2 uses omga as
        # working memory
        self.omga[0] = np.sqrt(2.0*np.sum(np.square(self.A), axis=(0, 1)))
        self.omga[0]*= self.smag_coef  # self.omga[0] == 2*nu_T

        # m1, c2, c3, c4, c5, c6, gmin, gmax = \
        #     central_moments(self.comm, self.Nx, self.omga[0])
        # if self.comm.rank == 0:
        #     print("---- nu_T moments: mean = %8.4f\tvar = %8.4f\t"
        #           "min = %8.4f\tmax = %8.4f" % (m1, c2, gmin, gmax))

        self.nuTmax = self.comm.allreduce(np.max(self.omga[0]), op=MPI.MAX)

        self.S_hat[:] = 0.0
        for i in range(3):
            for j in range(3):
                self.S_hat[i]+= 1j*self.K[2-j]*rfft3(self.comm,
                                                     self.A[j, i]*self.omga[0])
        self.dU += self.S_hat

        # self.omga[0] = irfft3(self.comm, self.S_hat[0])
        # self.omga[1] = irfft3(self.comm, self.S_hat[1])
        # self.omga[2] = irfft3(self.comm, self.S_hat[2])

        # eps_ratio = self.forcing_rate/(self.comm.allreduce(
        #                                psum(self.omga*self.U))*self.dx.prod())
        # if self.comm.rank == 0:
        #     print("---- SGS_ratio = %15.8f ----" % eps_ratio)

        return

    def computeAD_vorticity_formulation(self, **kwargs):
        """
        Computes right-hand-side (RHS) advection and diffusion (AD) terms of
        the incompressible Navier-Stokes equations using a
        vorticity formulation of the advection term.
        """

        # take curl of velocity to get vorticity and inverse transform
        self.omga[2] = irfft3(self.comm,
                              1j*(self.K[0]*self.U_hat[1]
                                  -self.K[1]*self.U_hat[0]))
        self.omga[1] = irfft3(self.comm,
                              1j*(self.K[2]*self.U_hat[0]
                                  -self.K[0]*self.U_hat[2]))
        self.omga[0] = irfft3(self.comm,
                              1j*(self.K[1]*self.U_hat[2]
                                  -self.K[2]*self.U_hat[1]))

        # compute convective transport as the physical-space cross-product of
        # vorticity and velocity and forward transform
        rfft3(self.comm, self.U[1]*self.omga[2]-self.U[2]*self.omga[1],
              self.dU[0])
        rfft3(self.comm, self.U[2]*self.omga[0]-self.U[0]*self.omga[2],
              self.dU[1])
        rfft3(self.comm, self.U[0]*self.omga[1]-self.U[1]*self.omga[0],
              self.dU[2])

        # Compute the diffusive transport term and add to the LH-projected
        # convective transport term
        self.dU -= self.nu*self.Ksq*self.U_hat

        return

    def new_dt_const_nu(self, cfl):
        u1m = u2m = u3m = 0.0
        u1m = self.comm.allreduce(np.max(self.U[0]), op=MPI.MAX)
        u2m = self.comm.allreduce(np.max(self.U[1]), op=MPI.MAX)
        u3m = self.comm.allreduce(np.max(self.U[2]), op=MPI.MAX)

        dtMinHydro = cfl*min(self.dx[0]/u1m, self.dx[1]/u2m, self.dx[2]/u3m)
        dtMinDiff = min(self.dx)**2/max(2.0*self.nu, self.nuTmax/cfl)
        dtMin = min(dtMinHydro, dtMinDiff)

        if self.comm.rank == 0:
            if dtMinDiff < dtMinHydro:
                print("timestep limited by diffusion! {} {}"
                      .format(dtMinHydro, dtMinDiff))

        return dtMin

    def RK4_integrate(self, dt, *Sources, **kwargs):
        """
        Nth order Runge-Kutta time integrator for spectralLES
        Olga/Peter: I didn't really pay attention to the paper, is this
                    actually 4th order, or is it only 3rd order?

        Arguments:
        ----------
        dt            - current timestep
        *Source_terms - (Optional) User-supplied source terms. This is a
                        special Python syntax, basically any argument you
                        feed RK4_integrate() after dt will be stored in the
                        list Source_terms. If no arguments are given,
                        Source_terms = [], and Python will accept the null
                        list ([]) in it's for loops, in which case the
                        loop region is skipped. This is equivalent to
                        pre-bundling function handles into a list and then
                        explicitly requiring a list as the second argument.
        **kwargs      - the keyword arguments to be passed to all Sources
        Note: The source terms Colin coded accept any number of extra arguments
        and ignore them, that way if you need to pass a computeSource()
        function an argument each call, those source terms are forwards
        compatible with the change inside the Source loop.
        """

        a = [1./6., 1./3., 1./3., 1./6.]
        b = [0.5, 0.5, 1.]

        self.U_hat1[:] = self.U_hat0[:] = self.U_hat

        for rk in range(4):

            self.U[0] = irfft3(self.comm, self.U_hat[0])
            self.U[1] = irfft3(self.comm, self.U_hat[1])
            self.U[2] = irfft3(self.comm, self.U_hat[2])

            self.computeAD()
            for Source in Sources:
                Source(**kwargs)

            # Dealias/filter the nonlinear and pseudospectral terms
            self.dU *= self.les_filter

            # Apply the Leray-Hopf projection operator (1 - Helmholtz operator)
            # to dealiased pseudospectral terms in order to enforce the
            # divergence-free continuity condition.
            # This operation is equivalent to computing the pressure field
            # using a physical-space pressure-Poisson solver and then adding
            # the pressure-gradient transport term to the RHS.
            self.dU -= np.sum(self.dU*self.K_Ksq, axis=0)*self.K

            if rk < 3:
                self.U_hat[:] = self.U_hat0 + b[rk]*dt*self.dU
            self.U_hat1[:] += a[rk]*dt*self.dU

        self.U[0] = irfft3(self.comm, self.U_hat[0])
        self.U[1] = irfft3(self.comm, self.U_hat[1])
        self.U[2] = irfft3(self.comm, self.U_hat[2])

        return
