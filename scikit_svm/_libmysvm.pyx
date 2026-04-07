# _libmysvm.pyx – Cython wrapper for the mySVM library
# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

# ── C++ wrapper declaration ────────────────────────────────────────────────
cdef extern from "mysvm_wrapper.hpp":
    cdef cppclass MySVMWrapper:
        MySVMWrapper() except +
        void set_svm_type(int t)
        void set_C(double C)
        void set_nu(double nu)
        void set_epsilon(double eps)
        void set_biased(int b)
        void set_verbosity(int v)
        void set_cache_mb(int mb)
        void set_max_iter(int n)
        void set_working_set_size(int n)
        void set_convergence_epsilon(double eps)
        void set_balance_cost(int b)
        void set_kernel_type(int kt)
        void set_kernel_gamma(double g)
        void set_kernel_degree(int d)
        void set_kernel_coef0(double c)
        void train(const double* X, const double* y,
                   int n_samples, int n_features) except +
        void predict(const double* X, double* out,
                     int n_test, int n_features) except +
        int get_n_sv()


# ── Python extension type ──────────────────────────────────────────────────
cdef class LibMySVMModel:
    """Thin Cython wrapper around MySVMWrapper.

    SVM types
    ---------
    0 : C-SVM classification  (svm_pattern_c)
    1 : C-SVM regression      (svm_regression_c)
    2 : nu-SVM classification (svm_nu_pattern_c)
    3 : nu-SVM regression     (svm_nu_regression_c)

    Kernel types
    ------------
    0 : linear     K = x·y
    1 : polynomial K = (1 + x·y)^degree
    2 : rbf        K = exp(-gamma·||x-y||²)
    3 : sigmoid    K = tanh(gamma·x·y + coef0)
    """

    cdef MySVMWrapper* _model

    def __cinit__(self):
        self._model = new MySVMWrapper()

    def __dealloc__(self):
        if self._model != NULL:
            del self._model
            self._model = NULL

    # ── setters ─────────────────────────────────────────────────────────
    def set_svm_type(self, int t):          self._model.set_svm_type(t)
    def set_C(self, double C):              self._model.set_C(C)
    def set_nu(self, double nu):            self._model.set_nu(nu)
    def set_epsilon(self, double eps):      self._model.set_epsilon(eps)
    def set_biased(self, int b):            self._model.set_biased(b)
    def set_verbosity(self, int v):         self._model.set_verbosity(v)
    def set_cache_mb(self, int mb):         self._model.set_cache_mb(mb)
    def set_max_iter(self, int n):          self._model.set_max_iter(n)
    def set_working_set_size(self, int n):  self._model.set_working_set_size(n)
    def set_convergence_epsilon(self, double eps):
        self._model.set_convergence_epsilon(eps)
    def set_balance_cost(self, int b):      self._model.set_balance_cost(b)
    def set_kernel_type(self, int kt):      self._model.set_kernel_type(kt)
    def set_kernel_gamma(self, double g):   self._model.set_kernel_gamma(g)
    def set_kernel_degree(self, int d):     self._model.set_kernel_degree(d)
    def set_kernel_coef0(self, double c):   self._model.set_kernel_coef0(c)

    # ── train ─────────────────────────────────────────────────────────
    def train(self,
              np.ndarray[np.float64_t, ndim=2, mode='c'] X,
              np.ndarray[np.float64_t, ndim=1, mode='c'] y):
        cdef int n = X.shape[0]
        cdef int d = X.shape[1]
        self._model.train(&X[0, 0], &y[0], n, d)

    # ── predict ───────────────────────────────────────────────────────
    def predict(self,
                np.ndarray[np.float64_t, ndim=2, mode='c'] X):
        cdef int n = X.shape[0]
        cdef int d = X.shape[1]
        cdef np.ndarray[np.float64_t, ndim=1, mode='c'] out = \
            np.empty(n, dtype=np.float64)
        self._model.predict(&X[0, 0], &out[0], n, d)
        return out

    # ── info ──────────────────────────────────────────────────────────
    property n_sv:
        def __get__(self):
            return self._model.get_n_sv()
