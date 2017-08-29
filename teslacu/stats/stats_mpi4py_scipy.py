"""
MPI-distributed statistics methods. Does not require information about
domain decomposition or dimensionality.
"""

from mpi4py import MPI
import numpy as np
from math import sqrt
from scipy import stats
import sys


def psum(data):
    return np.array(np.sum(np.sum(np.sum(data, axis=-1), axis=-1)))


def moments(comm, N, data):
    """
    Computes global min, max, mean, and 2nd-6th (population) central
    moments of MPI-decomposed data
    """
    # TODO update this with the pairwise-decomposed 'on-line' formulas

    gmin = np.empty(1)
    gmax = np.empty(1)
    gsum = np.empty(1)

    comm.Allreduce(np.array([np.min(data)]), gmin, op=MPI.MIN)
    comm.Allreduce(np.array([np.max(data)]), gmax, op=MPI.MAX)

    comm.Allreduce(psum(data), gsum, op=MPI.SUM)
    u1 = gsum[0]/N                              # 1st raw moment
    comm.Allreduce(psum(np.power(data-u1, 2)), gsum, op=MPI.SUM)
    c2 = gsum[0]/N                              # 2nd centered moment
    comm.Allreduce(psum(np.power(data-u1, 3)), gsum, op=MPI.SUM)
    c3 = gsum[0]/N                              # 3rd centered moment
    comm.Allreduce(psum(np.power(data-u1, 4)), gsum, op=MPI.SUM)
    c4 = gsum[0]/N                              # 4th centered moment
    comm.Allreduce(psum(np.power(data-u1, 5)), gsum, op=MPI.SUM)
    c5 = gsum[0]/N                              # 5th centered moment
    comm.Allreduce(psum(np.power(data-u1, 6)), gsum, op=MPI.SUM)
    c6 = gsum[0]/N                              # 6th centered moment

    return (gmin[0], gmax[0]), u1, c2, c3, c4, c5, c6


def histogram1(comm, N, data, bins=50):
    """
    Constructs the histogram (probability mass function) of an MPI-
    decomposed data.
    """

    if not np.all(np.isfinite(data)):
        data[np.isfinite(data) is False] = np.nan

    (gmin, gmax), u1, c2, c3, c4, c5, c6 = moments(comm, N, data)

    try:
        g3 = c3/sqrt(c2**3)     # 3rd standardized moment
        g4 = c4/(c2**2)         # 4th standardized moment
        g5 = c5/sqrt(c2**5)     # 5th standardized moment
        g6 = c6/(c2**3)         # 6th standardized moment

    except (RuntimeError, FloatingPointError, ValueError) as e:
        if comm.Get_rank() == 0:
                print ('---------------------------------------'
                       '---------------------------------------')
                print str(e), e.message()
                print 'min, mean, max:\n'
                print ('min: {}, u1: {}, max {}\n'
                       .format(gmin, u1, gmax))
                print 'moments from two-pass algorithm:\n'
                print ('c2: {}, c3: {}, c4: {}, c5: {}, c6: {}\n'
                       .format(c2, c3, c4, c5, c6))
                print ('---------------------------------------'
                       '---------------------------------------')
        MPI.Finalize()
        sys.exit(1)

    hist, low, width, extra = stats.histogram(
                                data, numbins=bins, defaultlimits=(gmin, gmax),
                                printextras=True)

    comm.Allreduce(MPI.IN_PLACE, hist, op=MPI.SUM)
    hist *= 1/psum(hist)  # makes this a probability mass function

    return hist, u1, c2, g3, g4, g5, g6, gmin, gmax, width


def histogram2(comm, nx, var1, var2, bins=50):
    pass

# psum = np.nansum(data,axis=-1)
# comm.Allreduce(np.array(np.nansum(psum)), gsum, op=MPI.SUM)
# u1 = gsum[0]/N
# 1st uncentered moment
# psum = np.nansum(np.power(data,2),axis=-1)
# comm.Allreduce(np.array(np.nansum(psum)), gsum, op=MPI.SUM)
# u2 = gsum[0]/N
# 2nd uncentered moment
# psum = np.nansum(np.power(data,3),axis=-1)
# comm.Allreduce(np.array(np.nansum(psum)), gsum, op=MPI.SUM)
# u3 = gsum[0]/N
# 3rd uncentered moment
# psum = np.nansum(np.power(data,4),axis=-1)
# comm.Allreduce(np.array(np.nansum(psum)), gsum, op=MPI.SUM)
# u4 = gsum[0]/N
# 4th uncentered moment
# psum = np.nansum(np.power(data,5),axis=-1)
# comm.Allreduce(np.array(np.nansum(psum)), gsum, op=MPI.SUM)
# u5 = gsum[0]/N
# 5th uncentered moment
# psum = np.nansum(np.power(data,6),axis=-1)
# comm.Allreduce(np.array(np.nansum(psum)), gsum, op=MPI.SUM)
# u6 = gsum[0]/N
# 6th uncentered moment

# c2 = u2 - u1**2
# 2nd centered moment
# c3 = u3 - 3*u1*u2 +  2*u1**3
# 3rd centered moment
# c4 = u4 - 4*u1*u3 +  6*(u1**2)*u2 -  3*u1**4
# 4th centered moment
# c5 = u5 - 5*u1*u4 + 10*(u1**2)*u3 - 10*(u1**3)*u2 +  4*u1**5
# 5th centered moment
# c6 = u6 - 6*u1*u5 + 15*(u1**2)*u4 - 20*(u1**3)*u3 + 15*(u1**4)*u2 - 5*u1**6
# 6th centered moment
