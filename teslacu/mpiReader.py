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
# from vtk import vtkStructuredPointsReader
# from vtk.util import numpy_support as vn

__all__ = []


def mpiReader(comm=MPI.COMM_WORLD, idir='./', ftype='binary', ndims=3,
              decomp=None, N=512, nh=None, periodic=None, byteswap=True):
    """
    The mpiReader() function is a "class factory" which returns the
    appropriate mpi-parallel reader class instance based upon the
    inputs. Each subclass contains a different ...

    Arguments:

    Output:
    """

    if ftype == 'binary':
        newReader = _binaryReader(comm, idir, N, nh, ndims, decomp,
                                  periodic, byteswap)
    else:
        newReader = _binaryReader(comm, idir, N, nh, ndims, decomp,
                                  periodic, byteswap)

    return newReader
# -----------------------------------------------------------------------------


class _binaryReader(object):
    """
    class _binaryReader

    Development notes:
        - this thing is pretty much not developed at all.
    """

    def __init__(self, comm, idir, N, nh, ndims, decomp, periodic, byteswap):

        # DEFINE THE INSTANCE VARIABLES

        # "Protected" variables masked by property method
        #  Global variables
        self._idir = idir
        self._comm = comm
        self._ndims = ndims
        self._byteswap = byteswap

        if decomp is None:
            decomp = list([True])
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
            self._nx = np.array([N]*ndims, dtype=int)

        if nh is None:
            self._nh = np.zeros(ndims, dtype=int)
        elif len(nh) == ndims:
            self._nh = np.array(nh, dtype=int)
        else:
            raise IndexError("Either len(nh) must be ndims or nh "
                             "must be None")

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

        if sum(self._decomp) == 1:
            # 1D domain decomposition (plates in 3D, pencils in 2D)
            self._nnx[0] = self._nx[0]/comm.size
            self._ixs[0] = self._nnx[0]*comm.rank
            self._ixe[0] = self._ixs[0]+self._nnx[0]
        else:
            raise AssertionError("mpiReader can't yet handle anything "
                                 "but 1D Decomposition.")

    @property
    def comm(self):
        return self._comm

    @property
    def taskid(self):
        return self._taskid

    @property
    def ntasks(self):
        return self._ntasks

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

    def simulation_time(self, filename):
        if self.taskid==0:
            with open(self._idir+filename) as fh:
                t = float(fh.readline())
        else:
            t = None

        t = self.comm.bcast(t, root=0)

        return t

    def read_variable(self, filename, ftype=np.float32, mtype=np.float64):
        """Currently hard coded to 1D domain decomposition."""
        status = MPI.Status()
        temp = np.zeros(self.nnx, dtype=ftype)
        fpath = self._idir+filename
        fhandle = MPI.File.Open(self.comm, fpath)
        offset = self._comm.rank*temp.nbytes
        fhandle.Read_at_all(offset, temp, status)
        fhandle.Close()

        if self.byteswap:
            var = temp.byteswap(True).astype(mtype)
        else:
            var = temp.astype(mtype)
        return var

    def read_variable_ghost_cells(self, filename, dtype=np.float64):
        """Currently hard coded to 1D domain decomposition."""
        status = MPI.Status()
        shape = np.array([self.nh[0]*2, self.nnx[1], self.nnx[2]])
        temp = np.zeros(shape, dtype=np.float32)
        hsize = shape.prod()*2     # 1/2 of temp size * 4 bytes
        dsize = self.nnx.prod()*4  # subdomain size * 4 bytes

        fpath = self._idir+filename
        fhandle = MPI.File.Open(self.comm, fpath)

        # read in the -z ghost zones
        if self.taskid==0:
            idx = self.ntasks
        else:
            idx = self.taskid
        offset = dsize*idx - hsize
        fhandle.Read_at_all(offset, temp[:self.nh[0], ...], status)

        # read in the +z ghost zones
        if self.taskid==self.ntasks-1:
            idx = 0
        else:
            idx = self.taskid+1
        offset = dsize*idx
        fhandle.Read_at_all(offset, temp[self.nh[0]:, ...], status)

        fhandle.Close()

        if self.byteswap:
            var = temp.byteswap(True).astype(dtype)
        else:
            var = temp.astype(dtype)
        return var


###############################################################################
# class mpiVtkReader(object):
    """
    reader = vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.ReadAllVectorsOn()
    reader.ReadAllScalarsOn()
    reader.Update()

    data = reader.GetOutput()

    dim = data.GetDimensions()
    vec = list(dim)
    vec = [i-1 for i in dim]
    vec.append(3)

    u    = vn.vtk_to_numpy(data.GetCellData().GetArray('velocity'))
    rho  = vn.vtk_to_numpy(data.GetCellData().GetArray('density'))
    Etot = vn.vtk_to_numpy(data.GetCellData().GetArray('total_energy'))
    Y    = vn.vtk_to_numpy(data.GetCellData().GetArray('scalar'))
    byte_mask = vn.vtk_to_numpy(data.GetCellData().GetArray('avtGhostZones'))

    x = zeros(data.GetNumberOfPoints())
    y = zeros(data.GetNumberOfPoints())
    z = zeros(data.GetNumberOfPoints())

    for i in range(data.GetNumberOfPoints()):
        x[i],y[i],z[i] = data.GetPoint(i)

    u    = u.reshape(vec,order='F')
    rho  = rho.reshape(dim,order='F')
    Etot = Etot.reshape(dim,order='F')
    Y    = Y.reshape(dim,order='F')
    x    = x.reshape(dim,order='F')
    y    = y.reshape(dim,order='F')
    z    = z.reshape(dim,order='F')
    byte_mask = byte_mask.reshape(dim,order='F')
    """
