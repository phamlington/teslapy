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
import numpy as np
from scipy.interpolate import Akima1DInterpolator as interp


def deriv(phi, h, axis=0):
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
        phi = np.swapaxes(phi, axis, 0)

    s = list(phi.shape)
    x = np.arange(-3, s[0]+3, dtype=np.float64)
    xi = np.arange(-0.5, s[0]+0.5, 1.0, dtype=np.float64)
    deriv = np.empty_like(phi)

    s[0] += 6
    tmp = np.empty(s, dtype=phi.dtype)

    tmp[3:-3] = phi
    tmp[:3] = phi[-3:]
    tmp[-3:] = phi[:3]

    for j in xrange(s[-2]):
        for i in xrange(s[-1]):
            spline = interp(x, tmp[..., j, i])
            phi2 = spline(xi)
            deriv[..., j, i] = (phi2[1:] - phi2[:-1])*(1.0/h)

    if axis != 0:
        deriv = np.swapaxes(deriv, axis, 0)

    return deriv
