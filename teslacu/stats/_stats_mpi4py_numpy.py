"""
MPI-distributed statistics methods. Does not require information about
domain decomposition or dimensionality.
"""

from mpi4py import MPI
import numpy as np

__all__ = ['psum', 'central_moments', 'histogram1', 'histogram2']


def psum(data):
    """
    input argument data can be any n-dimensional array-like object, including
    a 0D scalar value or 1D array.
    """

    psum = np.array(data)  # np.array hard-copies, np.asarray might just point
    for n in range(psum.ndim):
        psum.sort(axis=-1)
        psum = np.sum(psum, axis=-1)
    return psum


def central_moments(comm, N, data, w=None, wbar=None, m1=None):
    """
    Computes global min, max, and 1st to 6th central moments of
    MPI-decomposed data.

    To get raw moments, simply pass in m1=0.
    To get weighted moments, bass in the weighting coefficients, w.
    If you've already calculated the mean of w, save some computations and pass
    it in as wbar.
    """

    gmin = comm.allreduce(np.nanmin(data), op=MPI.MIN)
    gmax = comm.allreduce(np.nanmax(data), op=MPI.MAX)

    if w is None:   # unweighted moments
        w = 1.0
        wbar = 1.0
    elif wbar is None:
            wbar = comm.allreduce(psum(w), op=MPI.SUM)/N

    N = N*wbar

    # 1st raw moment
    if m1 is None:
        m1 = comm.allreduce(psum(w*data), op=MPI.SUM)/N

    # 2nd-4th centered moments
    cdata = data - m1
    c2 = comm.allreduce(psum(w*np.power(cdata, 2)), op=MPI.SUM)/N
    c3 = comm.allreduce(psum(w*np.power(cdata, 3)), op=MPI.SUM)/N
    c4 = comm.allreduce(psum(w*np.power(cdata, 4)), op=MPI.SUM)/N
    # c5 = comm.allreduce(psum(w*np.power(cdata, 5)), op=MPI.SUM)/N
    # c6 = comm.allreduce(psum(w*np.power(cdata, 6)), op=MPI.SUM)/N

    return m1, c2, c3, c4, gmin, gmax


def histogram1(comm, N, data, range=None, bins=50, w=None, wbar=None, m1=None):
    """
    Constructs the histogram (probability mass function) of an MPI-
    decomposed data.
    """

    if w is None:   # unweighted moments
        w = np.ones_like(data)
        wbar = 1
    elif wbar is None:
        wbar = comm.allreduce(psum(w), op=MPI.SUM)/N

    N = N*wbar

    # 1st raw moment
    if m1 is None:
        m1 = comm.allreduce(psum(w*data), op=MPI.SUM)/N

    # 2nd raw moment
    m2 = comm.allreduce(psum(w*np.power(data, 2)), op=MPI.SUM)/N

    if range is None:
        gmin = comm.allreduce(np.nanmin(data), op=MPI.MIN)
        gmax = comm.allreduce(np.nanmax(data), op=MPI.MAX)
    else:
        (gmin, gmax) = range

    width = (gmax-gmin)/bins

    temp = np.histogram(data, bins=bins, range=(gmin, gmax), weights=w)[0]
    hist = temp.astype(data.dtype, order='C')
    comm.Allreduce(MPI.IN_PLACE, hist, op=MPI.SUM)
    hist *= 1.0/hist.sum()  # makes this a probability mass function

    return hist, m1, m2, gmin, gmax, width


def histogram2(comm, var1, var2, xrange=None, yrange=None, bins=50, w=None):
    """
    Constructs the 2D histogram (probability mass function) of two MPI-
    decomposed data sets.
    """

    if xrange is None:
        gmin1 = comm.allreduce(np.min(var1), op=MPI.MIN)
        gmax1 = comm.allreduce(np.max(var1), op=MPI.MAX)
    else:
        (gmin1, gmax1) = xrange

    width1 = (gmax1-gmin1)/bins

    if yrange is None:
        gmin2 = comm.allreduce(np.min(var2), op=MPI.MIN)
        gmax2 = comm.allreduce(np.max(var2), op=MPI.MAX)
    else:
        (gmin2, gmax2) = yrange

    width2 = (gmax2-gmin2)/bins

    xy_range = [[gmin1, gmax1], [gmin2, gmax2]]

    temp = np.histogram2d(var1, var2, bins=bins, range=xy_range, weights=w)[0]
    hist = temp.astype(var1.dtype, order='C')
    comm.Allreduce(MPI.IN_PLACE, hist, op=MPI.SUM)
    hist *= 1.0/hist.sum()  # makes this a probability mass function

    return hist, gmin1, gmax1, width1, gmin2, gmax2, width2


# def alt_local_moments(data, w=None, wbar=None, N=None, unbias=True):
#     """
#     Returns the mean and 2nd-4th central moments of a memory-local
#     numpy array as a list. Default behavior is to return unbiased
#     sample moments for 1st-3rd order and a partially-corrected
#     sample 4th central moment.
#     """

#     if w is None:
#         u1 = psum(data)/N
#         if unbias:
#             c2 = psum(np.power(data-u1, 2))/(N-1)
#             c3 = psum(np.power(data-u1, 3))*N/(N**2-3*N+2)
#             c4 = psum(np.power(data-u1, 4))*N**2/(N**3-4*N**2+5*N-1)
#             c4+= (3/(N**2-3*N+3)-6/(N-1))*c2**2
#         else:
#             c2 = psum(np.power(data-u1, 2))/N
#             c3 = psum(np.power(data-u1, 3))/N
#             c4 = psum(np.power(data-u1, 4))/N
#     else:
#         if wbar is None:
#             wbar = psum(w)/N

#         u1 = psum(w*data)/(N*wbar)
#         if unbias:
#             c2 = psum(w*np.power(data-u1, 2))/(wbar*(N-1))
#             c3 = psum(w*np.power(data-u1, 3))*N/((N**2-3*N+2)*wbar)
#             c4 = psum(w*np.power(data-u1, 4))*N**2
#             c4/= (N**3-4*N**2+5*N-1)*wbar
#             c4+= (3/(N**2-3*N+3)-6/(N-1))*c2**2
#         else:
#             c2 = psum(w*np.power(data-u1, 2))/(N*wbar)
#             c3 = psum(w*np.power(data-u1, 3))/(N*wbar)
#             c4 = psum(w*np.power(data-u1, 4))/(N*wbar)

#     return u1, c2, c3, c4
