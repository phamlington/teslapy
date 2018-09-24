"""
Limited functionality Akima-spline-based numerical analysis methods.
Package is shared-memory only. User must wrap this package with an MPI-parallel
data analysis class for distributed memory computing.
Assumes 1-D domain decomposition.

Notes:
------

Definitions:
------------

Authors:
--------
Colin Towery

Turbulence and Energy Systems Laboratory
Department of Mechanical Engineering
University of Colorado Boulder
http://tesla.colorado.edu
"""
import numpy as _np
from scipy.interpolate import Akima1DInterpolator as _interp

__all__ = ['deriv']


def deriv(phi, h, axis=0):
    """
    deriv(phi, h, axis=0):

    deriv computes the k'th derivative of a uniform gridded array along the
    prescribed axis using Akima spline approximation.

    Arguments
    ---------
    phi   - input array
    h     - uniform grid spacing
    axis  -

    Output
    ------
    f - d^k/dx^k(phi)
    """
    if phi.ndim != 3:
        print("ERROR: phi.ndim not equal to 3!")

    axis = axis % phi.ndim
    if axis != 2:
        phi = _np.swapaxes(phi, axis, 2)

    s = list(phi.shape)
    x = _np.arange(-3, s[2]+3, dtype=phi.dtype)
    xh = _np.arange(-0.5, s[2]+0.5, 1.0, dtype=phi.dtype)
    deriv = _np.empty_like(phi)

    nx = s[2] + 6
    tmp = _np.empty(nx, dtype=phi.dtype)

    for k in range(s[0]):
        for j in range(s[1]):
            tmp[3:-3] = phi[k, j, :]
            tmp[:3] = phi[k, j, -3:]
            tmp[-3:] = phi[k, j, :3]

            spline = _interp(x, tmp)
            phih = spline(xh)
            deriv[k, j, :] = (1.0/h)*(phih[1:] - phih[:-1])

    if axis != 2:
        deriv = _np.swapaxes(deriv, axis, 2)

    return deriv


def deriv_bak(phi, h, axis=0):
    """
    deriv(phi, h, axis=0):

    deriv computes the k'th derivative of a uniform gridded array along the
    prescribed axis using Akima spline approximation.

    Arguments
    ---------
    phi   - input array
    h     - uniform grid spacing
    bc    -
    k     - order of the derivative
    axis  -

    Output
    ------
    f - d^k/dx^k(phi)
    """
    axis = axis % phi.ndim

    if axis != 0:
        phi = _np.swapaxes(phi, axis, 0)

    s = list(phi.shape)
    x = _np.arange(-3, s[0]+3, dtype=_np.float64)
    xi = _np.arange(-0.5, s[0]+0.5, 1.0, dtype=_np.float64)
    deriv = _np.empty_like(phi)

    s[0] += 6
    tmp = _np.empty(s, dtype=phi.dtype)

    tmp[3:-3] = phi
    tmp[:3] = phi[-3:]
    tmp[-3:] = phi[:3]

    for j in range(s[-2]):
        for i in range(s[-1]):
            spline = _interp(x, tmp[..., j, i])
            phi2 = spline(xi)
            deriv[..., j, i] = (phi2[1:] - phi2[:-1])*(1.0/h)

    if axis != 0:
        deriv = _np.swapaxes(deriv, axis, 0)

    return deriv
