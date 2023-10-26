# custom_distance_cython.pyx
import numpy as np
cimport numpy as np
from cython.parallel import prange

# Cython directives
cython: boundscheck=False
cython: wraparound=False

cpdef double custom_distance_cython(np.ndarray[np.float64_t, ndim=1] a, np.ndarray[np.float64_t, ndim=1] b):
    cdef int i, n
    cdef double distance = 0.0
    n = a.shape[0]

    if a[-1] == b[-1]:
        return np.inf

    for i in prange(n - 1, nogil=True):
        distance += (a[i] - b[i]) * (a[i] - b[i])

    return distance**0.5