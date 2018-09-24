"""
Description:
============
This module contains the mpiAnalyzer object classes for the TESLaCU package.
It should not be imported unless "__main__" has been executed with MPI.

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
This module generally adheres to the Python style guide published in
PEP 8, with the following exceptions:
- Warning W503 (line break occurred before a binary operator) is
  ignored, since this warning is a mistake and PEP 8 recommends breaking
  before operators
- Error E225 (missing whitespace around operator) is ignored

For more information see <http://pep8.readthedocs.org/en/latest/intro.html>

Additionally, I have not yet, but plan to eventually get all of these
docstrings whipped into shape and in compliance with the Numpy/Scipy
style guide for documentation:
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>

Finally, this module should always strive to achieve enlightenment by
following the Zen of Python (PEP 20, just `import this` in a Python
shell) and using idiomatic Python (i.e. 'Pythonic') concepts and design
patterns.

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
# from math import sqrt
import os
import sys
# from memory_profiler import profile

from . import fft as tcfft          # FFT transforms and math functions
from . import stats as tcstats      # statistical functions
from .diff import central as tcfd   # finite difference functions
from .diff import akima as tcas     # Akima spline approximation functions

__all__ = []


def mpiAnalyzer(comm=MPI.COMM_WORLD, odir='./analysis/', pid='test', ndims=3,
                L=[2*np.pi]*3, N=[512]*3, config='hit',
                method='spline_flux_diff', **kwargs):
    """
    The mpiAnalyzer() function is a "class factory" which returns the
    appropriate mpi-parallel analyzer class instance based upon the
    inputs. Each subclass specializes the BaseAnalyzer for a different
    problem configuration that can be added to the if/elif branch of this
    factory.

    Arguments:
    ----------
    comm: MPI communicator for the analyzer

    odir: output directory of analysis products

    pid: problem ID (file prefix)

    ndims: number of spatial dimensions of global data (not subdomain)

    L: scalar or tuple of domain dimensions

    N: scalar or tuple of mesh dimensions

    config: problem configuration (switch)

    kwargs: additional arguments to be handled by the subclasses

    Output:
    -------
    Single instance of _baseAnalyzer or one of its subclasses
    """

    if config == 'hit':
        analyzer = _hitAnalyzer(comm, odir, pid, ndims, L, N, method)
    elif config is None:
        analyzer = _baseAnalyzer(comm, odir, pid, ndims, L, N, method)
    else:
        if comm.rank == 0:
            print("mpiAnalyzer.factory configuration arguments not recognized!"
                  "\nDefaulting to base analysis class: _baseAnalyzer().")
        analyzer = _baseAnalyzer(comm, odir, pid, ndims, L, N)

    return analyzer


class _baseAnalyzer(object):
    """
    class _baseAnalyzer(object) ...
    """

    def __init__(self, comm, odir, pid, ndims, L, N, method):

        # "Protected" variables masked by property method
        self._odir = odir
        self._pid = pid
        self._comm = comm
        self._ndims = ndims
        self._config = "Unknown (Base Configuration)"
        self._periodic = [False]*ndims

        # "Public" variables
        self.tol = 1.0e-6
        self.prefix = pid+'-'

        # Global domain variables
        if np.iterable(N):
            if len(N) == 1:
                self._nx = np.array(list(N)*ndims, dtype=np.int)
            elif len(N) == ndims:
                self._nx = np.array(N, dtype=np.int)
            else:
                raise IndexError("The length of N must be either 1 or ndims")
        else:
            self._nx = np.array([N]*ndims, dtype=np.int)

        if np.iterable(L):
            if len(L) == 1:
                self._L = np.array(list(L)*ndims)
            elif len(L) == ndims:
                self._L = np.array(L)
            else:
                raise IndexError("The length of L must be either 1 or ndims")
        else:
            self._L = np.array([L]*ndims, dtype=np.float)

        self.dx = self._L/self._nx
        self.Nx = self._nx.prod()

        # Local subdomain variables (1D Decomposition)
        self.nnx = self._nx.copy()
        self.ixs = np.zeros(ndims, dtype=np.int)
        self.ixe = self._nx.copy()

        self.nnx[0] = self._nx[0]//self.comm.size
        self.ixs[0] = self.nnx[0]*self.comm.rank
        self.ixe[0] = self.ixs[0]+self.nnx[0]

        # eventually add other subdomain decompositions

        # MAKE ODIR, CHECKING IF IT IS A VALID PATH.
        if comm.rank == 0:
            try:
                os.makedirs(odir)
            except OSError as e:
                if not os.path.isdir(odir):
                    raise e
                else:
                    status = e
            finally:
                if os.path.isdir(odir):
                    status = 0
        else:
            status = None

        status = comm.bcast(status)
        if status != 0:
            MPI.Finalize()
            sys.exit(999)

        self.mpi_moments_file = '%s%s.moments' % (self.odir, self.prefix)

        if method == 'central_diff':
            self.deriv = self._centdiff_deriv
        elif method == 'spline_flux_diff':
            self.deriv = self._akima_deriv
        elif method == 'ignore':
            self.deriv = None
        else:
            if comm.rank == 0:
                print("mpiAnalyzer._baseAnalyzer.__init__(): "
                      "'method' argument not recognized!\n"
                      "Defaulting to Akima spline flux differencing.")
            self.deriv = self._akima_deriv

    # Class Properities -------------------------------------------------------
    def __enter__(self):
        # with statement initialization
        return self

    def __exit__(self, type, value, tb):
        # with statement finalization
        return False

    @property
    def comm(self):
        return self._comm

    @property
    def odir(self):
        return self._odir

    @property
    def pid(self):
        return self._pid

    @property
    def ndims(self):
        return self._ndims

    @property
    def config(self):
        return self._config

    @property
    def periodic(self):
        return self._periodic

    @property
    def L(self):
        return self._L

    @property
    def nx(self):
        return self._nx

    # Statistical Moments -----------------------------------------------------
    def psum(self, data):
        return tcstats.psum(data)

    def central_moments(self, data, w=None, wbar=None, m1=None, norm=1.0):
        """
        Computes global min, max, and 1st to 6th biased central moments of
        assigned spatial field. To get raw moments, simply pass in m1=0.
        """
        return tcstats.central_moments(
                                self.comm, self.Nx*norm, data, w, wbar, m1)

    def write_mpi_moments(self, data, label, w=None, wbar=None,
                          m1=None, norm=1.0):
        """Compute min, max, mean, and 2nd-6th (biased) central
        moments for assigned spatial field"""
        m1, c2, c3, c4, gmin, gmax = tcstats.central_moments(
                            self.comm, self.Nx*norm, data, w, wbar, m1)

        if self.comm.rank == 0:
            with open(self.mpi_moments_file, 'a') as fh:
                fh.write(('{:s}\t%s\n' % '  '.join(['{:14.8e}']*6))
                         .format(label, m1, c2, c3, c4, gmin, gmax))

        return m1, c2, c3, c4, gmin, gmax

    def mpi_mean(self, data, w=None, wbar=None, norm=1.0):
        N = self.Nx*norm
        if w is None:
            u1 = self.comm.allreduce(self.psum(data), op=MPI.SUM)/N
        else:
            if wbar is None:
                wbar = self.comm.allreduce(self.psum(w), op=MPI.SUM)/N
            N *= wbar
            u1 = self.comm.allreduce(self.psum(w*data), op=MPI.SUM)/N

        return u1

    def mpi_rms(self, data, w=None, wbar=None, norm=1.0):
        """
        Spatial root-mean-square (RMS)
        """
        N = self.Nx*norm
        if w is None:
            m2 = self.comm.allreduce(self.psum(data**2), op=MPI.SUM)/N
        else:
            if wbar is None:
                wbar = self.comm.allreduce(self.psum(w), op=MPI.SUM)/N
            N *= wbar
            m2 = self.comm.allreduce(self.psum(w*data**2), op=MPI.SUM)/N

        return np.sqrt(m2)

    def mpi_min_max(self, data):
        gmin = self.comm.allreduce(data.min(), op=MPI.MIN)
        gmax = self.comm.allreduce(data.max(), op=MPI.MAX)

        return (gmin, gmax)

    # Histograms --------------------------------------------------------------
    def mpi_histogram1(self, var, fname, metadata='', range=None,
                       bins=100, w=None, wbar=None, m1=None, norm=1.0):
        """MPI-distributed univariate spatial histogram."""

        # get histogram and statistical moments (every task gets the results)
        # result = (hist, u1, c2, g3, g4, g5, g6, gmin, gmax, width)
        results = tcstats.histogram1(self.comm, self.Nx*norm,
                                     var, range, bins, w, wbar, m1)
        hist = results[0]
        gmin, gmax, width = results[-3:]
        m = results[1:-3]

        # write histogram from root task
        if self.comm.rank == 0:
            fh = open('%s%s%s.hist' % (self.odir, self.prefix, fname), 'w')
            fh.write('%s\n' % metadata)
            fmt = ('%s\n' % '  '.join(['{:14.8e}']*len(m))).format
            fh.write(fmt(*m))
            fmt = ('{:d}  %s\n' % '  '.join(['{:14.8e}']*3)).format
            fh.write(fmt(bins, width, gmin, gmax))
            hist.tofile(fh, sep='\n', format='%14.8e')
            fh.close()

        return m

    def mpi_histogram2(self, var1, var2, fname, metadata='',
                       xrange=None, yrange=None, bins=100, w=None):
        """MPI-distributed bivariate spatial histogram."""

        # get histogram and statistical moments (every task gets the results)
        # result = (jhist, (min1, max1), width1, (min2, max2), width2)
        if w is not None:
            w = np.ravel(w)

        result = tcstats.histogram2(
                    self.comm, np.ravel(var1), np.ravel(var2),
                    xrange, yrange, bins, w)
        hist = result[0]
        m = result[1:]

        # write histogram from root task
        if self.comm.rank == 0:
            fmt = ('%d  %s\n' % '  '.join(['{:14.8e}']*len(m))).format
            fh = open('%s%s%s.hist2d' % (self.odir, self.prefix, fname), 'w')
            fh.write('%s\n' % metadata)
            fh.write(fmt(bins, *m))
            hist.tofile(fh, sep='\n', format='%14.8e')
            fh.close()

        return m

    # Data Transposing --------------------------------------------------------
    def z2y_slab_exchange(self, var):
        """
        Domain decomposition 'transpose' of MPI-distributed scalar array.
        Assumes 1D domain decomposition
        """

        nnz, ny, nx = var.shape
        nz = nnz*self.comm.size
        nny = ny//self.comm.size

        temp = np.empty([self.comm.size, nnz, nny, nx], dtype=var.dtype)

        temp[:] = np.rollaxis(var.reshape([nnz, self.comm.size, nny, nx]), 1)
        self.comm.Alltoall(MPI.IN_PLACE, temp)  # send, receive
        temp.resize([nz, nny, nx])

        return temp

    def y2z_slab_exchange(self, varT):
        """
        Domain decomposition 'transpose' of MPI-distributed scalar array.
        Assumes 1D domain decomposition
        """

        nz, nny, nx = varT.shape
        nnz = nz//self.comm.size
        ny = nny*self.comm.size

        temp = np.empty([self.comm.size, nnz, nny, nx], dtype=varT.dtype)

        self.comm.Alltoall(varT.reshape(temp.shape), temp)  # send, receive
        temp.resize([nnz, ny, nx])
        temp[:] = np.rollaxis(temp, 1).reshape(temp.shape)

        return temp

    # Scalar and Vector Derivatives -------------------------------------------
    def div(self, var):
        """
        Calculate and return the divergence of a vector field.

        Currently the slab_exchange routines limit this function to vector
        fields.
        """
        div = self.deriv(var[0], dim=0)   # axis=2
        div+= self.deriv(var[1], dim=1)   # axis=1
        div+= self.deriv(var[2], dim=2)   # axis=0
        return div

    def curl(self, var):
        """
        Calculate and return the curl of a vector field.
        """

        if var.ndim == 5:   # var is the gradient tensor field
            e = np.zeros((3, 3, 3))
            e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
            e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
            omega = np.einsum('ijk,jk...->i...', e, var)

        elif var.ndim == 4:     # var is the vector field
            omega = np.empty_like(var)
            omega[0] = self.deriv(var[1], dim=2)
            omega[0]-= self.deriv(var[2], dim=1)

            omega[1] = self.deriv(var[2], dim=0)
            omega[1]-= self.deriv(var[0], dim=2)

            omega[2] = self.deriv(var[0], dim=1)
            omega[2]-= self.deriv(var[1], dim=0)
        else:
            raise

        return omega

    def scl_grad(self, var):
        """
        Calculate and return the gradient vector field of a scalar field.
        """

        shape = list(var.shape)
        shape.insert(0, 3)
        grad = np.empty(shape, dtype=var.dtype)

        grad[0] = self.deriv(var, dim=0)
        grad[1] = self.deriv(var, dim=1)
        grad[2] = self.deriv(var, dim=2)

        return grad

    def grad(self, var):
        """
        Calculate and return the gradient tensor field of a vector field.
        """

        shape = list(var.shape)
        shape.insert(0, 3)
        A = np.empty(shape, dtype=var.dtype)

        for j in range(3):
            for i in range(3):
                A[j, i] = self.deriv(var[i], dim=j)

        return A

    def grad_curl_div(self, u):
        """
        Uses numpy.einsum which can be dramatically faster than
        alternative routines for many use cases
        """

        A = self.grad(u)

        e = np.zeros((3, 3, 3))
        e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
        e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
        omega = np.einsum('ijk,jk...->i...', e, A)

        Aii = np.einsum('ii...', A)

        return A, omega, Aii

    # Underlying Linear Algebra Routines --------------------------------------
    def _centdiff_deriv(self, var, dim=0, k=1):
        """
        Calculate and return the specified derivative of a 3D scalar field at
        the specified order of accuracy.
        While k is passed on to the central_deriv function in the teslacu
        finite difference module, central_deriv may not honor a request for
        anything but the first derivative, depending on it's state of
        development.
        """
        dim = dim % 3
        axis = 2-dim
        if axis == 0:
            var = self.z2y_slab_exchange(var)

        deriv = tcfd.central_deriv(var, self.dx[axis], bc='periodic',
                                   k=k, order=4, axis=axis)
        if axis == 0:
            deriv = self.y2z_slab_exchange(deriv)

        return deriv

    def _akima_deriv(self, var, dim=0, k=1):
        """
        Calculate and return the _first_ derivative of a 3D scalar field.
        The k parameter is ignored, a first derivative is _always_ returned.
        """
        dim = dim % 3
        axis = 2-dim
        if axis == 0:
            var = self.z2y_slab_exchange(var)

        deriv = tcas.deriv(var, self.dx[axis], axis=axis)

        if axis == 0:
            deriv = self.y2z_slab_exchange(deriv)

        return deriv

    def _fft_deriv(self, var, dim=0, k=1):
        """
        Calculate and return the specified derivative of a 3D scalar field.
        This function uses 1D FFTs and MPI-decomposed transposing instead of
        MPI-decomposed 3D FFTs.
        """
        dim = dim % 3
        axis = 2-dim
        s = [1]*var.ndim
        s[axis] = self.k1.shape[0]
        K = self.k1.reshape(s)

        if axis == 0:
            var = self.z2y_slab_exchange(var)

        deriv = np.fft.irfft(
                    np.power(1j*K, k)*np.fft.rfft(var, axis=axis), axis=axis)

        if axis == 0:
            deriv = self.y2z_slab_exchange(deriv)

        return deriv


class _hitAnalyzer(_baseAnalyzer):
    """
    class _hitAnalyzer(_baseAnalyzer)
        ...
    """

    def __init__(self, comm, odir, pid, ndims, L, N, method):

        super().__init__(comm, odir, pid, ndims, L, N, 'ignore')

        self._config = "Homogeneous Isotropic Turbulence"
        self._periodic = [True]*ndims

        # Spectral variables (1D Decomposition)
        self.nk = self.nx.copy()
        self.nk[-1] = self.nx[-1]//2+1
        self.nnk = self.nk.copy()
        self.nnk[1] = self.nnx[0]
        self.dk = 1.0/self.L[0]

        nx = self.nx[-1]
        dk = self.dk

        # The teslacu.fft.rfft3 and teslacu.fft.irfft3 functions currently
        # transpose Z and Y in the forward fft (rfft3) and inverse the
        # tranpose in the inverse fft (irfft3).
        # These FFT routines and these variables below assume that ndims=3
        # which ruins the generality I so carefully crafted in the base class
        k1 = np.fft.rfftfreq(self.nx[2])*dk*nx
        k2 = np.fft.fftfreq(self.nx[1])*dk*nx
        k2 = k2[self.ixs[0]:self.ixe[0]].copy()
        k3 = np.fft.fftfreq(self.nx[0])*dk*nx

        # MPI local 3D wavemode index
        self.K = np.array(np.meshgrid(k3, k2, k1, indexing='ij'))
        self.Ksq = np.sum(np.square(self.K), axis=0)
        self.k = np.sqrt(self.Ksq)
        self.km = (self.k//dk).astype(int)
        self.k1 = k1

        if method == 'central_diff':
            self.deriv = self._centdiff_deriv
        elif method == 'spline_flux_diff':
            self.deriv = self._akima_deriv
        elif method == 'spectral':
            self.deriv = self._fft_deriv
        else:
            if comm.rank == 0:
                print("mpiAnalyzer._hitAnalyzer.__init__(): "
                      "'method' argument not recognized!\n"
                      "Defaulting to Akima spline flux differencing.")
            self.deriv = self._akima_deriv

    # Spectra -----------------------------------------------------------------

    def spectral_density(self, var, fname, metadata=''):
        """
        Write the 1D power spectral density of var to text file. Method
        assumes a real input is in physical space and a complex input is
        in Fourier space.
        """
        if np.iscomplexobj(var) is True:
            cdata = var
        else:
            if var.ndim == 3:
                cdata = tcfft.rfft3(self.comm, var)
            elif var.ndim == 4:
                cdata = self.vec_fft(var)
            else:
                raise AttributeError('Input is {}D, '.format(var.ndim)
                                     +'spectral_density expects 3D or 4D!')

        # get spectrum (each task will receive the full spectrum)
        spect3d = np.real(cdata*np.conj(cdata))
        if var.ndim == 4:
            spect3d = np.sum(spect3d, axis=0)
        spect3d[..., 0] *= 0.5

        spect1d = tcfft.shell_average(self.comm, spect3d, self.km)

        if self.comm.rank == 0:
            fh = open('%s%s%s.spectra' % (self.odir, self.prefix, fname), 'w')
            fh.write('%s\n' % metadata)
            spect1d.tofile(fh, sep='\n', format='% .8e')
            fh.close()

        return spect1d

    def integral_scale(self, Ek):
        """
        Computes the integral scale from the standard formula,
        where u'^2 = 2/3*Int{Ek}
        ell = (pi/2)*(1/u'^2)*Int{Ek/k}
            = 3*pi/4*Int{Ek/k}/Int{Ek}
        """
        return 0.75*np.pi*self.psum(Ek[1:]/self.k1[1:])/self.psum(Ek[1:])

    def scl_fft(self, var):
        """
        Convenience function for MPI-distributed 3D r2c FFT of scalar.
        """
        return tcfft.rfft3(self.comm, var)

    def scl_ifft(self, var):
        """
        Convenience function for MPI-distributed 3D c2r IFFT of scalar.
        """
        return tcfft.irfft3(self.comm, var)

    def vec_fft(self, var):
        """
        Convenience function for MPI-distributed 3D r2c FFT of vector.
        """
        nnz, ny, nx = var.shape[1:]
        nk = nx//2+1
        nny = ny//self.comm.size
        nz = nnz*self.comm.size

        if var.dtype.itemsize == 8:
            fft_complex = np.complex128
        elif var.dtype.itemsize == 4:
            fft_complex = np.complex64
        else:
            raise AttributeError("cannot detect dataype of u")

        fvar = np.empty([3, nz, nny, nk], dtype=fft_complex)
        fvar[0] = tcfft.rfft3(self.comm, var[0])
        fvar[1] = tcfft.rfft3(self.comm, var[1])
        fvar[2] = tcfft.rfft3(self.comm, var[2])

        return fvar

    def vec_ifft(self, fvar):
        """
        Convenience function for MPI-distributed 3D c2r IFFT of vector.
        """
        nz, nny, nk = fvar.shape[1:]
        nx = (nk-1)*2
        ny = nny*self.comm.size
        nnz = nz//self.comm.size

        if fvar.dtype.itemsize == 16:
            fft_real = np.float64
        elif fvar.dtype.itemsize == 8:
            fft_real = np.float32
        else:
            raise AttributeError("cannot detect dataype of u")

        var = np.empty([3, nnz, ny, nx], dtype=fft_real)
        var[0] = tcfft.irfft3(self.comm, fvar[0])
        var[1] = tcfft.irfft3(self.comm, fvar[1])
        var[2] = tcfft.irfft3(self.comm, fvar[2])

        return var

    def shell_average(self, E3):
        """
        Convenience function for shell averaging
        """
        return tcfft.shell_average(self.comm, E3, self.km)

    def filter_kernel(self, ell, gtype='comp_exp', dtype=np.complex128):
        """
        ell - filter width
        G - user-supplied filter kernel array
        """
        kl = self.k*ell

        Ghat = np.zeros(self.k.shape, dtype=dtype)

        if gtype == 'tophat':
            Ghat = np.sin(np.pi*kl)/(np.pi*kl**2)

        elif gtype == 'comp_exp':
            """
            A 'COMPact EXPonential' filter which has
            1) compact support in a ball of radius ell (or 1/ell)
            2) is strictly positive, and
            3) is smooth (infinitely differentiable)
            in _both_ physical and spectral space!
            """
            Hhat = np.exp(-kl**2/(0.25-kl**2))
            ball = kl <= 0.5
            Ghat = np.where(ball, Hhat, Ghat)
            G0 = tcfft.irfft3(self.comm, Ghat)
            G = G0**2
            Gbar = self.comm.allreduce(self.psum(G), op=MPI.SUM)
            G = G/Gbar
            Ghat = tcfft.rfft3(self.comm, G)

        elif gtype == 'spectral':
            Ghat = (kl < 1.0).astype(dtype)

        else:
            raise ValueError('did not understand filter type')

        # elif gtype == 'inv_comp_exp':
        #     H = np.exp(-kl**2/(0.25-kl**2))
        #     ball = (kl <= 0.5).astype(np.int8)
        #     Ghat[ball] = Hhat[ball]
        #     G0 = tcfft.irfft3(self.comm, Ghat)
        #     G = G0**2
        #     Gbar = self.comm.allreduce(self.psum(G), op=MPI.SUM)
        #     G = G/Gbar
        #     Ghat = tcfft.rfft3(self.comm, G)

        return Ghat

    def scalar_filter(self, phi, Ghat):
        return tcfft.irfft3(self.comm, Ghat*tcfft.rfft3(self.comm, phi))

    def vector_filter(self, u, Ghat):
        return self.vec_ifft(Ghat*self.vec_fft(u))
