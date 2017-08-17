"""
MPI-distributed statistics methods. Does not require information about
domain decomposition or dimensionality.
"""

from mpi4py import MPI
import numpy as np


def psum(data):
    """
    input argument data can be any n-dimensional array-like object, including
    a 0D scalar value or 1D array.
    """
    psum = np.asarray(data)
    for n in range(data.ndim):
        psum.sort(axis=-1)
        psum = np.sum(psum, axis=-1)
    return psum


def central_moments(comm, N, data, w=None, wbar=None, m1=None):
    """
    Computes global min, max, and 1st to 6th biased central moments of
    MPI-decomposed data. To get raw moments, simply pass in m1=0.
    """
    gmin = comm.allreduce(np.nanmin(data), op=MPI.MIN)
    gmax = comm.allreduce(np.nanmax(data), op=MPI.MAX)

    if w is None:   # unweighted moments
        # 1st raw moment
        if m1 is None:
            m1 = comm.allreduce(psum(data), op=MPI.SUM)/N
        # 2nd-6th centered moments
        cdata = data-m1
        c2 = comm.allreduce(psum(np.power(cdata, 2)), op=MPI.SUM)/N
        c3 = comm.allreduce(psum(np.power(cdata, 3)), op=MPI.SUM)/N
        c4 = comm.allreduce(psum(np.power(cdata, 4)), op=MPI.SUM)/N
        c5 = comm.allreduce(psum(np.power(cdata, 5)), op=MPI.SUM)/N
        c6 = comm.allreduce(psum(np.power(cdata, 6)), op=MPI.SUM)/N

    else:           # weighted moments
        if wbar is None:
            wbar = comm.allreduce(psum(w), op=MPI.SUM)/N

        N = N*wbar
        # 1st raw moment
        if m1 is None:
            m1 = comm.allreduce(psum(w*data), op=MPI.SUM)/N
        # 2nd-6th centered moments
        cdata = data-m1
        c2 = comm.allreduce(psum(w*np.power(cdata, 2)), op=MPI.SUM)/N
        c3 = comm.allreduce(psum(w*np.power(cdata, 3)), op=MPI.SUM)/N
        c4 = comm.allreduce(psum(w*np.power(cdata, 4)), op=MPI.SUM)/N
        c5 = comm.allreduce(psum(w*np.power(cdata, 5)), op=MPI.SUM)/N
        c6 = comm.allreduce(psum(w*np.power(cdata, 6)), op=MPI.SUM)/N

    return m1, c2, c3, c4, c5, c6, gmin, gmax


def histogram1(comm, N, data, range=None, bins=50, w=None, wbar=None, m1=None):
    """
    Constructs the histogram (probability mass function) of an MPI-
    decomposed data.
    """

    # is_finite = np.all(np.isfinite(data))
    # is_finite = comm.allreduce(is_finite, op=MPI.LAND)
    # if not is_finite:
    #     if comm.rank == 0:
    #         raise ValueError('Histogram data contains non-finite values!')
    #     MPI.Finalize()
    #     sys.exit(999)

    if w is None:
        if m1 is None:
            m1 = comm.allreduce(psum(data), op=MPI.SUM)/N
        m2 = comm.allreduce(psum(np.power(data, 2)), op=MPI.SUM)/N
    else:
        if wbar is None:
            wbar = comm.allreduce(psum(w), op=MPI.SUM)/N
        N = N*wbar
        if m1 is None:
            m1 = comm.allreduce(psum(w*data), op=MPI.SUM)/N
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

    # is_finite = np.all(np.isfinite(var1))
    # is_finite = comm.allreduce(is_finite, op=MPI.LAND)
    # if not is_finite:
    #     if comm.rank == 0:
    #         raise ValueError('Histogram2 var1 contains non-finite values!')
    #     MPI.Finalize()
    #     sys.exit(999)

    # is_finite = np.all(np.isfinite(var2))
    # is_finite = comm.allreduce(is_finite, op=MPI.LAND)
    # if not is_finite:
    #     if comm.rank == 0:
    #         raise ValueError('Histogram2 var2 contains non-finite values!')
    #     MPI.Finalize()
    #     sys.exit(999)

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
