import numpy as np
from libc.math cimport sin
from libc.math cimport abs
from libc.math cimport sqrt
cimport cython

DTYPE = np.single

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def custom_hake_pairwise_distances(float[:, :] u, float lam, float lam2):
#     dist_vec_mod = custom_mod_pairwise_distances(u[:, :int(u.shape[1] / 2)], lam2=lam2)
#     dist_vec_phase = custom_phase_pairwise_distances(u[:, int(u.shape[1] / 2):], lam=lam)
    
#     dist_vec = dist_vec_mod + dist_vec_phase

    cdef Py_ssize_t n = u.shape[0]
    

    cdef float[:, :] dist_mat = np.zeros((n, n), dtype=DTYPE)
    cdef float[:, :] u1 = u[:, :int(u.shape[1] / 2)]
    cdef float[:, :] u2 = u[:, int(u.shape[1] / 2):]
    cdef Py_ssize_t d = u2.shape[1]
    cdef float u1_summed = 0
    cdef float u2_summed = 0
    
    cdef Py_ssize_t i, j, k


    for i in range(n):
        for j in range(i+1, n):
#             dist_vec[c] = lam * np.sum(np.abs(np.sin((u[i] - u[j]) / 2.0)))
#             dist_mat[i, j] = lam * np.sum(np.abs(np.sin((u2[i] - u2[j]) / 2.0))) + lam2 * np.sqrt(np.sum((u1[i] - u1[j]) ** 2))
            u1_summed = 0
            u2_summed = 0
            for k in range(d):
                u1_summed += (u1[i, k] - u1[j, k]) ** 2
                u2_summed += abs(sin((u2[i, k] - u2[j, k]) / 2.0))
                #dist_mat[i, j] = lam2 * np.sqrt(np.sum((u1[i, :] - u1[j, :]) ** 2))
                #dist_mat[j, i] = dist_mat[i, j]
            
            dist_mat[i, j] = lam2 * sqrt(u1_summed) + lam * u2_summed
            dist_mat[j, i] = dist_mat[i, j]
    
#     dist_mat = scipy.spatial.distance.squareform(dist_vec)
    
    return np.array(dist_mat)