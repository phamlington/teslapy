"""
Description:
============
This module contains limited-functionality MPI-distributed FFT transforms
and associated routines for the TESLaCU Python package. These routines assume
1D or 'slab' domain decomposition along 0th axis ("z-direction") of a 3D
variable and that the non-contiguous dimensions of the mesh are divisible by
the number of MPI tasks, i.e. nz % ntasks == 0 and ny % ntasks == 0, where
mesh is shape [nz, ny, nx].

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

Original Author Attribution:
----------------------------
The FFT transforms included here were derived from pure Python routines
included with the spectralDNS code written by Mikael Mortensen and Diako Darian
(<https://github.com/spectralDNS/spectralDNS/spectralDNS3D_short.py>) and
published in [Mortensen & Langtangen (JCPC 2016)]
(<http://dx.doi.org/10.1016/j.cpc.2016.02.005>).

spectralDNS is licensed under the GNU GPL, version 3.
spectralDNS is Copyright (2014-2016) by the authors.

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

__all__ = ['psum', 'rfft3', 'irfft3', 'shell_average', 'y2z_slab_exchange',
           'z2y_slab_exchange']


# 3D real-valued FFTs ---------------------------------------------------------
def rfft3(comm, u, fu=None):
    """
    Compute MPI-distributed, real-to-complex 3D FFT.
    Input array must have only three dimensions (not curently checked)
    temp and fu are complex data work arrays where the array view for fu can
    be passed in from the calling function
    """
    ntasks = comm.size
    nnz, ny, nx = u.shape
    nk = nx//2+1
    nny = ny//ntasks
    nz = nnz*ntasks

    if fu is None:
        fu = np.empty([nz, nny, nk], dtype=np.complex128)

    temp1 = np.empty([nnz, ny, nk], dtype=np.complex128)

    temp1[:] = np.fft.rfft2(u, axes=(1, 2))
    fu[:] = np.rollaxis(temp1.reshape([nnz, ntasks, nny, nk]),
                        1).reshape(fu.shape)
    comm.Alltoall(MPI.IN_PLACE, [fu, MPI.DOUBLE_COMPLEX])  # send, receive
    fu[:] = np.fft.fft(fu, axis=0)

    return fu


def irfft3(comm, fu, u=None):
    """
    compute MPI-distributed, complex-to-real 3D FFT.
    Input array must have only three dimensions (not curently checked)
    temp1 and temp2 are complex data work arrays
    """
    ntasks = comm.size
    nz, nny, nk = fu.shape
    nnz = nz//ntasks
    ny = nny*ntasks

    temp1 = np.empty_like(fu)
    temp2 = np.empty([nnz, ny, nk], dtype=fu.dtype)

    temp1[:] = np.fft.ifft(fu, axis=0)
    temp1.resize([ntasks, nnz, nny, nk])
    comm.Alltoall(MPI.IN_PLACE, [temp1, MPI.DOUBLE_COMPLEX])  # send, receive
    temp2[:] = np.rollaxis(temp1, 1).reshape([nnz, ny, nk])

    if u is None:
        u = np.fft.irfft2(temp2, axes=(1, 2))
    else:
        u[:] = np.fft.irfft2(temp2, axes=(1, 2))

    return u


# Auxiliary functions ---------------------------------------------------------
def psum(data):
    """
    Array summation by heirarchical partial summation.
    Input argument data can be any n-dimensional array-like object,
    even a (n=0) scalar. That is, psum() is safe to use on any
    numeric data type.
    """
    return np.sum(np.sum(np.sum(data, axis=-1), axis=-1))


def shell_average(comm, E3, km):
    """
    Compute the 1D, shell-averaged, spectrum of the 3D Fourier-space
    variable E3.

    Arguments:
        comm - MPI intracommunicator
        nk   - scalar length of 1-D spectrum
        km   - wavemode of each n-D wavevector
        E3   - 3-dimensional complex or real Fourier-space scalar
    """
    nz, nny, nk = E3.shape
    E1 = np.zeros(nk, dtype=E3.dtype)
    zeros = np.zeros_like(E3)

    if km[0, 0, 0] == 0:
        # this mpi task has the DC mode, only works for 1D domain decomp
        E1[0] = E3[0, 0, 0]

    for k in range(1, nk):
        E1[k] = psum(np.where(km==k, E3, zeros))

    comm.Allreduce(MPI.IN_PLACE, E1, op=MPI.SUM)

    return E1


# Data Transposing ------------------------------------------------------------
def z2y_slab_exchange(comm, var, varT=None):
    """
    Domain decomposition 'transpose' of MPI-distributed scalar array.
    Assumes 1D domain decomposition
    """

    nnz, ny, nx = var.shape
    nz = nnz*comm.size
    nny = ny//comm.size

    if varT is None:
        varT = np.empty([comm.size, nnz, nny, nx], dtype=var.dtype)
    else:
        varT.resize([comm.size, nnz, nny, nx])

    varT[:] = np.rollaxis(var.reshape([nnz, comm.size, nny, nx]), 1)
    comm.Alltoall(MPI.IN_PLACE, varT)  # send, receive
    varT.resize([nz, nny, nx])

    return varT


def y2z_slab_exchange(comm, varT, var=None):
    """
    Domain decomposition 'transpose' of MPI-distributed scalar array.
    Assumes 1D domain decomposition
    """

    nz, nny, nx = varT.shape
    nnz = nz//comm.size
    ny = nny*comm.size

    if var is None:
        var = np.empty([comm.size, nnz, nny, nx], dtype=varT.dtype)
    else:
        var.resize([comm.size, nnz, nny, nx])

    comm.Alltoall(var, varT.reshape(var.shape))  # send, receive
    var.resize([nnz, ny, nx])
    var[:] = np.rollaxis(var, 1).reshape(var.shape)

    return var


# Package testing functions ---------------------------------------------------
def fft3_unit_test(comm, u):
    fu = rfft3(comm, u)
    up = irfft3(comm, fu)
    check = u/up
    print('fft3 checksum:', check.min(), check.max(), check.sum())

    return


def shift_theorem_test(comm, u):
    """
    NOTE that if you are using FFTs to shift data in a periodic domain
    of dimensions L = [2pi, 2pi, 2pi], then the shift can be formed
    using either
        2j*pi * k * n/N
    or
        1j * k * dx*n
    since dx == 2*pi/N.

    NOTE that the shift theorem describes negative shifts, i -> i-n, by
    exp(-2j*pi * k * n/N), therefore a positive shift, i -> i+n, should
    not have a negative sign!
    """

    N = u.shape[-1]             # length of x-direction
    k = np.fft.rfftfreq(N)*N    # integer wavenumbers in x-direction
    n = 2                       # number of positions to shift along x
    shift = np.exp(2j*np.pi*k*n/N)
    fu = rfft3(comm, u)
    u_shift = irfft3(comm, fu*shift)
    assert np.allclose(u[:, :, 2:], u_shift[:, :, :-2])

    return
