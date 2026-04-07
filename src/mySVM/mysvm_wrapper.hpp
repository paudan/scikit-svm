#pragma once

/*
 * Opaque C++ wrapper around the mySVM library.
 *
 * Only plain C++ types are used here so that the Cython extension can include
 * this header without pulling in any mySVM-internal types (which use the
 * non-standard <iostream.h> header and the `abs` macro).
 *
 * Supported SVM types (svm_type parameter):
 *   0  C-SVM classification   (svm_pattern_c)
 *   1  C-SVM regression       (svm_regression_c)
 *   2  nu-SVM classification  (svm_nu_pattern_c)
 *   3  nu-SVM regression      (svm_nu_regression_c)
 *
 * Supported kernels (kernel_type parameter):
 *   0  linear     K(x,y) = x·y
 *   1  polynomial K(x,y) = (1 + x·y)^degree
 *   2  rbf        K(x,y) = exp(-gamma * ||x-y||²)
 *   3  sigmoid    K(x,y) = tanh(gamma * x·y + coef0)
 */

class MySVMWrapper {
public:
    MySVMWrapper();
    ~MySVMWrapper();

    /* ── parameter setters ──────────────────────────────────────────────── */
    void set_svm_type(int t);          // 0=C_SVC, 1=EPSILON_SVR, 2=NU_SVC, 3=NU_SVR
    void set_C(double C);
    void set_nu(double nu);
    void set_epsilon(double eps);      // epsilon tube for regression
    void set_biased(int b);            // 1=biased hyperplane (default), 0=unbiased
    void set_verbosity(int v);         // 0=silent … 5=flood
    void set_cache_mb(int mb);
    void set_max_iter(int n);
    void set_working_set_size(int n);
    void set_convergence_epsilon(double eps);
    void set_balance_cost(int b);      // class-weight balancing

    void set_kernel_type(int kt);      // 0=linear,1=poly,2=rbf,3=sigmoid
    void set_kernel_gamma(double g);
    void set_kernel_degree(int d);
    void set_kernel_coef0(double c);   // coef0 / b for sigmoid/polynomial

    /* ── main API ───────────────────────────────────────────────────────── */
    /* X is row-major, shape (n_samples, n_features), C-contiguous doubles.
       y is shape (n_samples,).  Both arrays are read-only.               */
    void train(const double* X, const double* y, int n_samples, int n_features);

    /* X is row-major, shape (n_test, n_features).
       out is pre-allocated shape (n_test,) and will be filled.           */
    void predict(const double* X, double* out, int n_test, int n_features);

    int  get_n_sv() const;

private:
    /* opaque pointers – actual mySVM types only in mysvm_wrapper.cpp */
    void* m_params;
    void* m_kernel;
    void* m_svm;
    void* m_train_set;

    int    m_n_features;
    int    m_trained;

    /* stored parameters */
    int    m_svm_type;
    int    m_kernel_type;
    int    m_degree;
    int    m_biased;
    int    m_verbosity;
    int    m_cache_mb;
    int    m_max_iter;
    int    m_wss;
    int    m_balance_cost;
    double m_C;
    double m_nu;
    double m_eps;      /* epsilon tube (regression) */
    double m_conv_eps;
    double m_gamma;
    double m_coef0;

    void cleanup();
};
