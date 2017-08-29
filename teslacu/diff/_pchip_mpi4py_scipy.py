"""
Limited functionality MPI-distributed Akima spline based finite difference methods.
Assumes ghost zones are 2 cells wide and 1-D domain decomposition.
"""
import numpy as _np
from scipy.interpolate import PchipInterpolator as _interp
# import time
# from memory_profiler import profile


def deriv(x, phi, order=1, axis=0):
    """compute the derivative of a scalar field"""
    deriv = _np.zeros_like(phi)
    copy = _np.ascontiguousarray

    if axis == 0:
        for x1 in xrange(phi.shape[1]):
            for x2 in xrange(phi.shape[2]):
                polynomial     = _interp(X[0], copy(phi[:,x1,x2]))
                deriv[:,x1,x2] = polynomial(X[0], nu=order)
    elif axis == 1:
        for x0 in xrange(phi.shape[0]):
            for x2 in xrange(phi.shape[2]):
                polynomial     = _interp(X[1], copy(phi[x0,:,x2]))
                deriv[x0,:,x2] = polynomial(X[1], nu=order)
    else:
        for x0 in xrange(phi.shape[0]):
            for x1 in xrange(phi.shape[1]):
                polynomial     = _interp(X[2], copy(phi[x0,x1,:]))
                deriv[x0,x1,:] = polynomial(X[2], nu=order)

    return deriv

def div(X, u):
    """compute the divergence for a vector field"""
    div  = _np.zeros(u.shape[1:],dtype=u.dtype)

    poly = _interp(X[0], u[0], axis=0)
    div += poly(X[0], nu=1)

    poly = _interp(X[1], u[1], axis=1)
    div += poly(X[1], nu=1)

    poly = _interp(X[2], u[2], axis=2)
    div += poly(X[2], nu=1)

    return div

def curl(X, u): pass

def scl_grad(X, phi):
    """compute the gradient vector for a scalar field"""
    shape = list(phi.shape)
    shape.insert(0,3)
    grad = _np.zeros(shape,dtype=phi.dtype)

    polynomial  = _interp(X[0], phi, axis=0)
    grad[0,...] = polynomial(X[0], nu=1)

    polynomial  = _interp(X[1], phi, axis=1)
    grad[1,...] = polynomial(X[1], nu=1)

    polynomial  = _interp(X[2], phi, axis=2)
    grad[2,...] = polynomial(X[2], nu=1)

    return grad

def grad(X, u):
    """compute the gradient tensor (Jacobian) for a vector field"""
    Jshape = list(u.shape)
    Jshape[1] -= 4  #subtract ghost cell count from shape
    Jshape.insert(0,3)
    J = _np.zeros(Jshape,dtype=u.dtype)

    for i in range(3):
        polynomial = _interp(X[0], u[i,:,:,:], axis=0)
        J[0,i,...] = polynomial(X[0][2:-2], nu=1)

        polynomial = _interp(X[1], u[i,2:-2,:,:], axis=1)
        J[1,i,...] = polynomial(X[1], nu=1)

        polynomial = _interp(X[2], u[i,2:-2,:,:], axis=2)
        J[2,i,...] = polynomial(X[2], nu=1)

    return J

def helmholtz(X, u):
    """Computes the Helmholtz decomposition of a vector field into a dilatational (zero curl) and
    solenoidal (zero div) component by solving the poisson equation for the dilatational component."""
    ud   = _np.zeros_like(u)
    copy = _np.ascontiguousarray

    theta = div(X, u)
    gth   = scl_grad(X, theta)

    for i in range(3):
        poly1      = _interp(X[0], gth[i,:,:,:], axis=0)
        poly2      = poly1.antiderivative(nu=2)
        ud[i,...] += poly2(X[0])

        poly1      = _interp(X[1], gth[i,:,:,:], axis=1)
        poly2      = poly1.antiderivative(nu=2)
        ud[i,...] += poly2(X[1])

        poly1      = _interp(X[2], gth[i,:,:,:], axis=2)
        poly2      = poly1.antiderivative(nu=2)
        ud[i,...] += poly2(X[2])

    return ud

