"""
Description:
============
This module contains the mpiReader object classes for the TESLaCU package.
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
import os
# from vtk import vtkStructuredPointsReader
# from vtk.util import numpy_support as vn

__all__ = []


def mpiReader(comm=MPI.COMM_WORLD, idir='./', ftype='binary', N=512, ndims=3,
              decomp=None, periodic=None, byteswap=False):
    """
    The mpiReader() function is a "class factory" which returns the
    appropriate mpi-parallel reader class instance based upon the
    inputs. Each subclass contains a different ...

    Arguments:

    Output:
    """

    if ftype == 'binary':
        newReader = _binaryReader(comm, idir, N, ndims, decomp,
                                  periodic, byteswap)
    else:
        newReader = _binaryReader(comm, idir, N, ndims, decomp,
                                  periodic, byteswap)

    return newReader
# -----------------------------------------------------------------------------


class _binaryReader(object):
    """
    class _binaryReader

    Development notes:
        - this thing is pretty much not developed at all.
    """

    def __init__(self, comm, idir, N, ndims, decomp, periodic, byteswap):

        # DEFINE THE INSTANCE VARIABLES

        # "Protected" variables masked by property method
        #  Global variables
        self._idir = idir
        self._ndims = ndims
        self.byteswap = byteswap

        if decomp is None:
            self._decomp = 1
        elif decomp in [1, 2, 3] and decomp <= ndims:
            self._decomp = decomp
        else:
            raise IndexError("decomp must be 1, 2, 3 or None")

        if np.iterable(N):
            if len(N) == 1:
                self._nx = np.array(list(N)*ndims, dtype=int)
            elif len(N) == ndims:
                self._nx = np.array(N, dtype=int)
            else:
                raise IndexError("The length of N must be either 1 or ndims")
        else:
            self._nx = np.array([N]*ndims, dtype=int)

        if periodic is None:
            self._periodic = tuple([False]*ndims)
        elif len(periodic) == ndims:
            self._periodic = tuple(periodic)
        else:
            raise IndexError("Either len(periodic) must be ndims or "
                             "periodic must be None")

        # Local subdomain variables
        self._nnx = self._nx.copy()
        self._ixs = np.zeros(ndims, dtype=int)
        self._ixe = self._nx.copy()

        if self._decomp == 1:
            # 1D domain decomposition (plates in 3D, pencils in 2D)
            self.comm = comm
            self._nnx[0] = self._nx[0]/comm.size
            self._ixs[0] = self._nnx[0]*comm.rank
            self._ixe[0] = self._ixs[0]+self._nnx[0]
            self.Read_all = self._read_all_1d

        elif self._decomp == 2:
            raise ValueError("mpiReader can't yet handle 2D domain "
                             "decomposition.")

        elif self._decomp == 3:
            comm_orig = comm
            dims = MPI.Compute_dims(comm_orig.size, ndims)
            comm = comm_orig.Create_cart(dims, self._periodic)
            self._coords = comm.Get_coords(comm.rank)
            self.comm = comm

            assert np.all(np.mod(self._nx, dims) == 0)
            self._nnx = self._nx//dims
            self._ixs = self._nnx*self._coords
            self._ixe = self._ixs + self._nnx
            self.Read_all = self._read_all_3d

        else:
            raise ValueError("mpiReader can't yet handle anything but 1D and"
                             " 3D domain decomposition.")

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
    def coords(self):
        return self._coords

    @property
    def nnx(self):
        return self._nnx

    @property
    def ixs(self):
        return self._ixs

    @property
    def ixe(self):
        return self._ixe

    def simulation_time(self, filename):
        if self.taskid==0:
            with open(self._idir+filename) as fh:
                t = float(fh.readline())
        else:
            t = None

        t = self.comm.bcast(t, root=0)

        return t

    def _read_all_1d(self, filename, ftype=np.float32, mtype=np.float64):
        """
        1D domain decomposition
        """
        status = MPI.Status()
        temp = np.zeros(self.nnx, dtype=ftype)
        offset = self.comm.rank*temp.nbytes

        fpath = os.path.join(self._idir, filename)
        fhandle = MPI.File.Open(self.comm, fpath)

        fhandle.Read_at_all(offset, temp, status)

        fhandle.Close()

        if self.byteswap:
            temp.byteswap(inplace=True)

        return temp.astype(mtype, casting='safe', copy=False)

    def _read_all_3d(self, filename, ftype=np.float32, mtype=np.float64):
        """
        3D domain decomposition
        """
        disp = 0
        nbits = np.finfo(ftype).bits
        if nbits == 32:
            MPI_ETYPE = MPI.FLOAT
        elif nbits == 64:
            MPI_ETYPE = MPI.DOUBLE
        else:
            raise ValueError("ftype must be a 32-bit or 64-bit floating point"
                             " datatype.")

        status = MPI.Status()
        temp = np.zeros(self._nnx, dtype=ftype)

        filepath = os.path.join(self._idir, filename)
        fh = MPI.File.Open(self.comm, filepath)

        filetype = MPI_ETYPE.Create_subarray(self._nx, self._nnx, self._ixs)
        filetype.Commit()
        fh.Set_view(disp, MPI_ETYPE, filetype, 'native')
        fh.Read_all(temp, status)

        fh.Close()
        filetype.Free()

        if self.byteswap:
            temp.byteswap(inplace=True)

        return temp.astype(mtype, casting='safe', copy=False)
