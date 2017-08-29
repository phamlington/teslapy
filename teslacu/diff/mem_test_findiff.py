import numpy as np
import findiff_numpy_scipy as tcfd
from memory_profiler import profile
import gc

@profile
def foo(phi):
    A = np.empty((3,3,4,1024,1024))

    for j in range(3):
        for i in range(3):
            A[j,i] = tcfd.central_deriv(phi[i], 1, axis=j)

    return A

@profile
def bar():
    phi = np.random.randn(3,4,1024,1024)
    A = foo(phi)
    A = A + np.rollaxis(A, 1)
    A = None
    A = foo(phi)
    A[:] = A + np.rollaxis(A, 1)
    A = None

if __name__ == '__main__':
    bar()
