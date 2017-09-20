"""
Description:
============
This module contains the mpiWriter object classes for the TESLaCU package.
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

Coding Style Guide:
-------------------
This module generally adheres to the Python style guide published in
PEP 8, with the following exceptions:
- Warning W503 (line break occurred before a binary operator) is
  ignored, since this warning is a mistake and PEP 8 recommends breaking
  before operators
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
import sys
import os
# from vtk import vtkStructuredPointsReader
# from vtk.util import numpy_support as vn

__all__ = ['mpiWriter']


def mpiWriter(comm=MPI.COMM_WORLD, odir='./', N=512, ftype='binary', ndims=3,
              nh=None, decomp=None, byteswap=False):
    """
    The mpiWriter() function is a "class factory" which returns the
    appropriate mpi-parallel writer class instance based upon the
    inputs. Each subclass contains a different ...

    Arguments:

    Output:
    """

    if ftype == 'binary':
        newWriter = _binaryWriter(comm, odir, ndims, decomp, N, nh, byteswap)
    else:
        newWriter = _binaryWriter(comm, odir, ndims, decomp, N, nh, byteswap)

    return newWriter
# -----------------------------------------------------------------------------


class _binaryWriter(object):
    """
    """

    def __init__(self, comm, odir, ndims, decomp, N, nh, byteswap):

        # DEFINE THE INSTANCE VARIABLES

        # "Protected" variables masked by property method
        #  Global variables
        self._odir = odir
        self._comm = comm
        self._ndims = ndims
        self._byteswap = byteswap

        if decomp is None:
            decomp = list([True, ])
            decomp.extend([False]*(ndims-1))
            self._decomp = decomp
        elif len(decomp) == ndims:
            self._decomp = decomp
        else:
            raise IndexError("Either len(decomp) must be ndims or "
                             "decomp must be None")

        if np.iterable(N):
            if len(N) == 1:
                self._nx = np.array(list(N)*ndims, dtype=int)
            elif len(N) == ndims:
                self._nx = np.array(N, dtype=int)
            else:
                raise IndexError("The length of N must be either 1 or ndims")
        else:
            self._nx = np.array([int(N)]*ndims, dtype=int)

        if nh is None:
            self._nh = np.zeros(ndims, dtype=int)
        elif len(nh) == ndims:
            self._nh = np.array(nh, dtype=int)
        else:
            raise IndexError("Either len(nh) must be ndims or nh "
                             "must be None")

        # Local subdomain variables
        self._nnx = self._nx.copy()
        self._ixs = np.zeros(ndims, dtype=int)
        self._ixe = self._nx.copy()

        if sum(self._decomp) == 1:
            # 1D domain decomposition (plates in 3D, pencils in 2D)
            self._nnx[0] = self._nx[0]/comm.size
            self._ixs[0] = self._nnx[0]*comm.rank
            self._ixe[0] = self._ixs[0]+self._nnx[0]
        else:
            raise AssertionError("mpiReader can't yet handle anything "
                                 "but 1D Decomposition.")

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

        return

    @property
    def comm(self):
        return self._comm

    @property
    def ndims(self):
        return self._ndims

    @property
    def decomp(self):
        return self._decomp

    @property
    def nx(self):
        return self._nx

    @property
    def nh(self):
        return self._nh

    @property
    def nnx(self):
        return self._nnx

    @property
    def ixs(self):
        return self._ixs

    @property
    def ixe(self):
        return self._ixe

    @property
    def byteswap(self):
        return self._byteswap

    def write_scalar(self, filename, data, dtype=np.float32):
        """
        Currently hard coded to 1D domain decomposition.
        """
        status = MPI.Status()
        if self.byteswap:
            stmp = data.byteswap(True).astype(dtype)
        else:
            stmp = data.astype(dtype)

        fhandle = MPI.File.Open(self.comm, self._odir+filename,
                                MPI.MODE_WRONLY | MPI.MODE_CREATE)
        offset = self.comm.rank*stmp.nbytes
        fhandle.Write_at_all(offset, stmp, status)
        fhandle.Close()

        return status

    def write_vector(self, filename, data, dtype=np.float32):
        """
        Currently hard coded to 1D domain decomposition.
        Vector can be arbitrary length.
        """
        status = MPI.Status()
        if self.byteswap:
            stmp = data.byteswap(True).astype(dtype)
        else:
            stmp = data.astype(dtype)

        fhandle = MPI.File.Open(self.comm, self._odir+filename,
                                MPI.MODE_WRONLY | MPI.MODE_CREATE)

        displ = self.comm.size*stmp[0].nbytes
        offset = self.comm.rank*stmp[0].nbytes
        for i in range(stmp.shape[0]):
            fhandle.Write_at_all(offset, stmp[i], status)
            offset += displ

        fhandle.Close()

        return status

###############################################################################
