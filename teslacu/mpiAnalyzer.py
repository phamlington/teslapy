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
This module generally adheres to the Python style guide published in PEP 8,
with the following notable exceptions:
- Warning W503 (line break occurred before a binary operator) is ignored
- Error E129 (visually indented line with same indent as next logical line)
  is ignored
- Error E225 (missing whitespace around operator) is ignored

For more information see <http://pep8.readthedocs.org/en/latest/intro.html>

Definitions:
============
wavevector  - the Fourier-space, dimensionalized, spatial-frequency
                vector, kvec = [k3, k2, k1].
wavenumber  - the wavevector magnitude, k = |kvec|
wavemode    - the integer mode index of the wavenumber, km = nint(k)/dk,
                from 0:nk

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
import os
import sys
# from memory_profiler import profile

from . import fft as tcfft          # FFT transforms and math functions
from . import stats as tcstats      # statistical functions
from .diff import central as tcfd   # finite difference functions
from .diff import akima as tcas     # Akima spline approximation functions

__all__ = ['mpiAnalyzer']

world_comm = MPI.COMM_WORLD


###############################################################################
def mpiAnalyzer(comm=world_comm, idir='./data/', odir='./analysis/',
                probID='athena', ndims=3, L=[2*np.pi]*3, nx=[512]*3, nh=None,
                geo=None, rct=None, method='central_diff'):
    """
    The mpiAnalyzer() function is a "class factory" which returns the
    appropriate mpi-parallel analyzer class instance based upon the
    inputs. Each subclass contains a different ...

    Arguments:

    Output:
    """

    if geo == 'hit' and rct is None:
        newAnalyzer = hitAnalyzer(comm, idir, odir, probID, ndims, L, nx,
                                  method)
    elif geo is None and rct is None:
        newAnalyzer = mpiBaseAnalyzer(comm, idir, odir, probID, ndims, L, nx)
    else:
        if comm.rank == 0:
            print("mpiAnalyzer.factory configuration arguments not recognized!"
                  "\nDefaulting to base analysis class: mpiBaseAnalyzer().")
        newAnalyzer = mpiBaseAnalyzer(comm, idir, odir, probID, ndims, L, nx)

    return newAnalyzer


###############################################################################
class mpiBaseAnalyzer(object):
    """
    class mpiBaseAnalyzer(object) ...
    """

    def __init__(self, comm, idir, odir, probID, ndims, L, nx):

        # DEFINE THE INSTANCE VARIABLES

        # "Protected" variables masked by property method
        self.__idir = idir
        self.__odir = odir
        self.__probID = probID
        self.prefix = probID+'-'
        self.__comm = comm
        self.__ndims = ndims
        self.__sim_geom = "Unknown (Base Configuration)"
        self.__sim_rctn = None
        self.__periodic = [False]*ndims
        self.__reacting = False

        self.tol = 1.0e-6

        # "Public" variables
        # Global domain variables
        if np.iterable(nx):
            if len(nx) == 1:
                self.__nx = np.array(list(nx)*ndims, dtype=int)
            elif len(nx) == ndims:
                self.__nx = np.array(nx, dtype=int)  # "analysis nx"
            else:
                raise IndexError("The length of nx must be either 1 or ndims")
        else:
            self.__nx = np.array([int(nx)]*ndims, dtype=int)

        if np.iterable(L):
            if len(L) == 1:
                self.__L = np.array(list(L)*ndims)
            elif len(L) == ndims:
                self.__L = np.array(L)  # "analysis nx"
            else:
                raise IndexError("The length of L must be either 1 or ndims")
        else:
            self.__L = np.array([float(L)]*ndims)

        self.dx = self.__L/self.__nx
        self.Nx = self.__nx.prod()
        self.Nxinv = 1.0/self.Nx

        # Local subdomain variables (1D Decomposition)
        self.nnx = self.__nx.copy()
        self.ixs = np.zeros(ndims, dtype=int)
        self.ixe = self.__nx.copy()

        self.nnx[0] = self.__nx[0]//self.comm.size
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

    # Class Properities -------------------------------------------------------

    def __enter__(self):
        # with statement initialization
        return self

    def __exit__(self, type, value, tb):
        # with statement finalization
        return False

    @property
    def comm(self):
        return self.__comm

    @property
    def idir(self):
        return self.__idir

    @property
    def odir(self):
        return self.__odir

    @property
    def probID(self):
        return self.__probID

    @property
    def ndims(self):
        return self.__ndims

    @property
    def sim_geom(self):
        return self.__sim_geom

    @property
    def sim_rctn(self):
        return self.__sim_rctn

    @property
    def reacting(self):
        return self.__reacting

    @property
    def periodic(self):
        return self.__periodic

    @property
    def L(self):
        return self.__L

    @property
    def nx(self):
        return self.__nx

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

    def write_mpi_moments(self, data, title, sym, w=None, wbar=None,
                          m1=None, norm=1.0):
        """Compute min, max, mean, and 2nd-6th (biased) central
        moments for assigned spatial field"""
        m1, c2, c3, c4, c5, c6, gmin, gmax = tcstats.central_moments(
                            self.comm, self.Nx*norm, data, w, wbar, m1)

        if self.comm.rank == 0:
            fh = open(self.mpi_moments_file, 'a')
            fh.write(('{:s}\n{:s}\t%s\n' % '\t'.join(['{:.8e}']*6))
                     .format(title, sym, m1, c2, c3, c4, gmin, gmax))

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

# Histograms ------------------------------------------------------------------

    def mpi_histogram1(self, var, fname, xlabel='', ylabel='', range=None,
                       bins=100, w=None, wbar=None, m1=None, norm=1.0):
        """MPI-distributed univariate spatial histogram."""

        # get histogram and statistical moments (every task gets the results)
        # result = (hist, u1, c2, g3, g4, g5, g6, gmin, gmax, width)
        result = tcstats.histogram1(self.comm, self.Nx*norm,
                                    var, range, bins, w, wbar, m1)
        hist = result[0]
        m = result[1:]

        # write histogram from root task
        if self.comm.rank == 0:
            fmt = ''.join(['{:.8e}\n']*len(m)).format
            fh = open('%s%s%s.hist' % (self.odir, self.prefix, fname), 'w')
            fh.write(xlabel+'\n'+ylabel+'\n'+fmt(*m)+str(bins)+'\n')
            hist.tofile(fh, sep='\n', format='%.8e')
            fh.close()

        return m

    def mpi_histogram2(self, var1, var2, fname, xlabel='', ylabel='',
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
            fmt = ''.join(['{:.8e}\n']*len(m)).format
            fh = open('%s%s%s.hist2d' % (self.odir, self.prefix, fname), 'w')
            fh.write(xlabel+'\n'+ylabel+'\n'+fmt(*m)+str(bins)+'\n')
            hist.tofile(fh, sep='\n', format='%.8e')
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

        self.comm.Alltoall(temp, varT.reshape(temp.shape))  # send, receive
        temp.resize([nnz, ny, nx])
        temp[:] = np.rollaxis(temp, 1).reshape(temp.shape)

        return temp


###############################################################################
class hitAnalyzer(mpiBaseAnalyzer):
    """
    class hitAnalyzer(mpiBaseAnalyzer)
        ...
    """

    def __init__(self, comm, idir, odir, probID, ndims, L, nx,
                 method='central_diff'):

        super(hitAnalyzer, self).__init__(comm, idir, odir, probID,
                                          ndims, L, nx)

        self.__sim_geom = "Homogeneous Isotropic Turbulence"
        self.__periodic = [True]*ndims

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
        self.km = (self.k/dk).astype(int)
        self.k1 = k1

        if method == 'central_diff':
            self.deriv = self._centdiff_deriv
        elif method == 'akima':
            self.deriv = self._akima_deriv
        elif method == 'fourier':
            self.deriv = self._fft_deriv
        else:
            if comm.rank == 0:
                print("mpiAnalyzer.hitAnalyzer.__init__(): "
                      "'method' argument not recognized!\n"
                      "Defaulting to Akima spline flux differencing.")
            self.deriv = self._akima_deriv

    # Spectra -----------------------------------------------------------------

    def spectral_density(self, var, fname, title, ylabel):
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
            fh.write(title+'\n'+ylabel+'\n')
            spect1d.tofile(fh, sep='\n', format='% .8e')
            fh.close()

            self.comm.Barrier()
        else:
            self.comm.Barrier()

        return spect1d

    def integral_scale(self, Ek):
        """
        Computes the integral scale from the standard formula,
        ell = (pi/2*u'^2)*Int{Ek/k} (angular wavenumbers)
            = (3/8)*Int{Ek/k}/Int{Ek} (ordinary wavenumbers)
        """
        return 0.375*self.psum(Ek[1:]/self.k1[1:])/self.psum(Ek[1:])

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

# Scalar and Vector Derivatives -----------------------------------------------

    def div(self, var):
        """
        Calculate and return the divergence of a vector field.

        Currently the slab_exchange routines limit this function to vector
        fields.
        """
        div = self.deriv(var[0], dim=0)  # axis=2
        div+= self.deriv(var[1], dim=1)  # axis=1
        div+= self.deriv(var[2], dim=2)  # axis=0
        return div

    def curl(self, var):
        """
        Calculate and return the curl of a vector field.
        """

        if var.ndim == 5:   # var is the gradient tensor field
            e = np.zeros((3, 3, 3))
            e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
            e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
            omga = np.einsum('ijk,jk...->i...', e, var)

        elif var.ndim == 4:     # var is the vector field
            omga = np.empty_like(var)
            omga[0] = self.deriv(var[1], dim=2)
            omga[0]-= self.deriv(var[2], dim=1)

            omga[1] = self.deriv(var[2], dim=0)
            omga[1]-= self.deriv(var[0], dim=2)

            omga[2] = self.deriv(var[0], dim=1)
            omga[2]-= self.deriv(var[1], dim=0)
        else:
            raise

        return omga

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

        A = self.grad(u, order=order)

        e = np.zeros((3, 3, 3))
        e[0, 1, 2] = e[1, 2, 0] = e[2, 0, 1] = 1
        e[0, 2, 1] = e[2, 1, 0] = e[1, 0, 2] = -1
        omga = np.einsum('ijk,jk...->i...', e, A)

        Aii = np.einsum('ii...', A)

        return A, omga, Aii

# Underlying Linear Algebra Routines ------------------------------------------
    # Note that the FFT derivative does not rely on the teslacu FFT package
    # This allows FFT-based derivatives to conform to a consistent deriv
    # template for all types of numerical differentiation.
    # The 'k' parameter asks for the order of the derivative, but only the FFT
    # derivative can provide any order derivative to the user.

    # @profile
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
