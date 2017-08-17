"""
Linear algebra based finite difference methods. Package is shared-memory only.
User must wrap this package with an MPI-parallel data analysis class for
distributed memory computing.
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
from scipy.linalg import circulant


def central_matrix(n, d=1, o=4, dtype=np.float64):
    '''
    central_matrix(n,d=1,o=4):

    central_matrix() explicitly constructs the central finite difference
    operator matrix.

    Arguments
    ---------
    n - length of the solution vector, u
    d - order of the derivative to be computed
    o - order of the finite difference approximation

    Output
    ------
    A - coefficient matrix, shape (n,n)

    Example of A (n=6, d=1, o=4):
    1./12 * [
     0   8  -1   0   0   0
    -8   0   8  -1   0   0
     1  -8   0   8  -1   0
     0   1  -8   0   8  -1
     0   0   1  -8   0   8
     0   0   0   1  -8   0
    ]
    '''

    if d==1:
        if o==2:
            d1 = np.ones(n-1, dtype)
            A = np.diag(d1, 1) - np.diag(d1, -1)

        elif o==4:
            d1 = (8./12)*np.ones(n-1, dtype)
            d2 = (1./12)*np.ones(n-2, dtype)
            A = np.diag(d2, -2) - np.diag(d1, -1) + np.diag(d1, 1) \
                - np.diag(d2, 2)

        elif o==6:
            d1 = (3./4)*np.ones(n-1, dtype)
            d2 = (3./20)*np.ones(n-2, dtype)
            d3 = (1./60)*np.ones(n-3, dtype)
            A = -np.diag(d3, -3) + np.diag(d2, -2) - np.diag(d1, -1) \
                + np.diag(d1, 1) - np.diag(d2, 2) + np.diag(d3, 3)

        elif o==8:
            d1 = (4./5)*np.ones(n-1, dtype)
            d2 = (1./5)*np.ones(n-2, dtype)
            d3 = (4./105)*np.ones(n-3, dtype)
            d4 = (1./280)*np.ones(n-4, dtype)
            A = np.diag(d4, -4) - np.diag(d3, -3) + np.diag(d2, -2) \
                - np.diag(d1, -1) + np.diag(d1, 1) - np.diag(d2, 2) \
                + np.diag(d3, 3) - np.diag(d4, 4)

        else:
            raise

    elif d==2:
        if o==2:
            c = 2*np.ones(n, dtype)
            d1 = np.ones(n-1, dtype)
            A = np.diag(d1, -1) - np.diag(c, 0) + np.diag(d1, 1)

        elif o==4:
            c = (30./12)*np.ones(n, dtype)
            d1 = (16./12)*np.ones(n-1, dtype)
            d2 = (1./12)*np.ones(n-2, dtype)
            A = -np.diag(d2, -2) + np.diag(d1, -1) - np.diag(c, 0) \
                - np.diag(d2, 2) + np.diag(d1, 1)

        elif o==6:
            c = (49./18)*np.ones(n, dtype)
            d1 = (3./2)*np.ones(n-1, dtype)
            d2 = (3./20)*np.ones(n-2, dtype)
            d3 = (1./90)*np.ones(n-3, dtype)
            A = np.diag(d3, -3) - np.diag(d2, -2) + np.diag(d1, -1) \
                - np.diag(c, 0) + np.diag(d3, 3) - np.diag(d2, 2) \
                + np.diag(d1, 1)

        elif o==8:
            c = (205./72)*np.ones(n, dtype)
            d1 = (8./5)*np.ones(n-1, dtype)
            d2 = (1./5)*np.ones(n-2, dtype)
            d3 = (8./315)*np.ones(n-3, dtype)
            d4 = (1./560)*np.ones(n-4, dtype)
            A = -np.diag(d4, -4) + np.diag(d3, -3) - np.diag(d2, -2) \
                + np.diag(d1, -1) - np.diag(c, 0) -np.diag(d4, 4) \
                + np.diag(d3, 3) - np.diag(d2, 2) + np.diag(d1, 1)

        else:
            raise ValueError

    else:
        raise ValueError

    return A


def central_deriv(phi, h, bc='periodic', k=1, order=4, axis=-1):
    """
    central_deriv(phi, h, bc='periodic', k=1, order=4, axis=-1):

    central_deriv computes the k'th derivative of an array along the
    prescribed axis using the central finite difference approximation with the
    prescribed order of accuracy.

    Arguments
    ---------
    phi   - input array
    h     - uniform grid spacing
    bc    -
    k     - order of the derivative
    order - order of the finite difference approximation
    axis  -

    Output
    ------
    f - d^k/dx^k(phi)
    """
    axis = axis % phi.ndim

    nx= phi.shape[axis]
    c = np.zeros(nx)
    c[1:3] = [-8./12, 1./12]
    c[-2:] = [-1./12, 8./12]
    c /= h
    A = circulant(c)

    # A = (1.0/h**k)*central_matrix(nx, k, order, phi.dtype)

    # if bc=='periodic':
    #     for d in range(1, order/2+1):
    #         for i, j in enumerate(range(-d, 0)):
    #             A[i, j] = A[d, 0]
    #             A[j, i] = A[0, d]

    # elif bc=='ghost_zones':
    #     raise ValueError('findiff_numpy_scipy.central_deriv(): '
    #                      'only periodic bc implemented')

    # else:
    #     raise ValueError('findiff_numpy_scipy.central_deriv(): '
    #                      'only periodic bc implemented')

    if axis != phi.ndim-1:
        phi = np.swapaxes(phi, axis, -1)

    deriv = np.einsum('ijk,mk->ijm', phi, A)

    if axis != phi.ndim-1:
        deriv = np.swapaxes(deriv, axis, -1)

    return deriv
