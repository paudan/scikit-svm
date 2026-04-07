/*
 * mysvm_wrapper.cpp
 *
 * C++ implementation of the MySVMWrapper facade.  All mySVM-internal headers
 * are included here and never exposed to Cython.
 */

/* ── compat shims (must come before any mySVM header) ─────────────────────── */
#include "iostream.h"  /* maps <iostream.h> → <iostream> + using namespace std */
#include "fstream.h"   /* maps <fstream.h>  → <fstream>  + using namespace std */

/* ── mySVM headers ─────────────────────────────────────────────────────────── */
#include "globals.h"
#include "parameters.h"
#include "kernel.h"
#include "example_set.h"
#include "svm_c.h"
#include "svm_nu.h"

/* ── standard headers ──────────────────────────────────────────────────────── */
#include "mysvm_wrapper.hpp"
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

/* ============================================================================
 * helpers
 * ========================================================================== */

static kernel_c* make_kernel(int kernel_type, double gamma, int degree, double coef0)
{
    switch (kernel_type) {
    case 0: /* linear */
        return new kernel_dot_c();

    case 1: { /* polynomial: K(x,y) = (1 + x·y)^degree */
        kernel_polynomial_c* k = new kernel_polynomial_c();
        std::ostringstream oss;
        oss << "degree " << degree << "\n";
        std::istringstream iss(oss.str());
        k->input(iss);
        return k;
    }

    case 2: { /* radial: K(x,y) = exp(-gamma*||x-y||²) */
        kernel_radial_c* k = new kernel_radial_c();
        std::ostringstream oss;
        oss << "gamma " << gamma << "\n";
        std::istringstream iss(oss.str());
        k->input(iss);
        return k;
    }

    case 3: { /* sigmoid: K(x,y) = tanh(gamma*x·y + coef0) */
        kernel_neural_c* k = new kernel_neural_c();
        std::ostringstream oss;
        oss << "a " << gamma << "\nb " << coef0 << "\n";
        std::istringstream iss(oss.str());
        k->input(iss);
        return k;
    }

    default:
        throw std::invalid_argument("Unknown kernel_type");
    }
}


static svm_c* make_svm(int svm_type)
{
    switch (svm_type) {
    case 0: return new svm_pattern_c();
    case 1: return new svm_regression_c();
    case 2: return new svm_nu_pattern_c();
    case 3: return new svm_nu_regression_c();
    default:
        throw std::invalid_argument("Unknown svm_type");
    }
}


/* Fill example_set_c from a dense row-major matrix X (n x d) and label vector y (n).
 * The put_example(pos, SVMFLOAT*) API expects a buffer of size d+2:
 *   buf[0..d-1] = feature values, buf[d] = y, buf[d+1] = alpha (0). */
static void fill_example_set(example_set_c* es,
                             const double* X, const double* y,
                             int n, int d)
{
    std::vector<double> buf(static_cast<size_t>(d + 2), 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j)
            buf[j] = X[static_cast<size_t>(i) * d + j];
        buf[d]   = (y != 0) ? y[i] : 0.0;
        buf[d+1] = 0.0;
        es->put_example(static_cast<SVMINT>(i), buf.data());
    }
    if (y != 0)
        es->set_initialised_y();
    es->set_initialised_alpha();
}


/* ============================================================================
 * MySVMWrapper
 * ========================================================================== */

MySVMWrapper::MySVMWrapper()
    : m_params(0), m_kernel(0), m_svm(0), m_train_set(0),
      m_n_features(0), m_trained(0),
      m_svm_type(0), m_kernel_type(2), m_degree(3),
      m_biased(1), m_verbosity(0), m_cache_mb(256),
      m_max_iter(100000), m_wss(10), m_balance_cost(0),
      m_C(1.0), m_nu(0.5), m_eps(0.1),
      m_conv_eps(1e-3), m_gamma(1.0), m_coef0(1.0)
{}


MySVMWrapper::~MySVMWrapper()
{
    cleanup();
}


void MySVMWrapper::cleanup()
{
    if (m_svm) {
        /* Delete via the concrete type to avoid non-virtual-dtor UB */
        switch (m_svm_type) {
        case 0: delete static_cast<svm_pattern_c*>(m_svm);     break;
        case 1: delete static_cast<svm_regression_c*>(m_svm);  break;
        case 2: delete static_cast<svm_nu_pattern_c*>(m_svm);  break;
        case 3: delete static_cast<svm_nu_regression_c*>(m_svm); break;
        default: delete static_cast<svm_c*>(m_svm);            break;
        }
        m_svm = 0;
    }
    if (m_kernel)    { delete static_cast<kernel_c*>(m_kernel);        m_kernel    = 0; }
    if (m_params)    { delete static_cast<parameters_c*>(m_params);    m_params    = 0; }
    if (m_train_set) { delete static_cast<example_set_c*>(m_train_set);m_train_set = 0; }
    m_trained = 0;
}


/* ── setters ──────────────────────────────────────────────────────────────── */

void MySVMWrapper::set_svm_type(int t)              { m_svm_type    = t; }
void MySVMWrapper::set_C(double C)                  { m_C           = C; }
void MySVMWrapper::set_nu(double nu)                { m_nu          = nu; }
void MySVMWrapper::set_epsilon(double eps)          { m_eps         = eps; }
void MySVMWrapper::set_biased(int b)                { m_biased      = b; }
void MySVMWrapper::set_verbosity(int v)             { m_verbosity   = v; }
void MySVMWrapper::set_cache_mb(int mb)             { m_cache_mb    = mb; }
void MySVMWrapper::set_max_iter(int n)              { m_max_iter    = n; }
void MySVMWrapper::set_working_set_size(int n)      { m_wss         = n; }
void MySVMWrapper::set_convergence_epsilon(double e){ m_conv_eps    = e; }
void MySVMWrapper::set_balance_cost(int b)          { m_balance_cost= b; }
void MySVMWrapper::set_kernel_type(int kt)          { m_kernel_type = kt; }
void MySVMWrapper::set_kernel_gamma(double g)       { m_gamma       = g; }
void MySVMWrapper::set_kernel_degree(int d)         { m_degree      = d; }
void MySVMWrapper::set_kernel_coef0(double c)       { m_coef0       = c; }


/* ── train ────────────────────────────────────────────────────────────────── */

void MySVMWrapper::train(const double* X, const double* y, int n_samples, int n_features)
{
    cleanup();

    m_n_features = n_features;

    /* parameters */
    parameters_c* params = new parameters_c();
    params->realC          = m_C;
    params->nu             = m_nu;
    params->epsilon_pos    = m_eps;
    params->epsilon_neg    = m_eps;
    params->biased         = m_biased;
    params->verbosity      = m_verbosity;
    params->kernel_cache   = m_cache_mb;
    params->max_iterations = static_cast<SVMINT>(m_max_iter);
    params->working_set_size = static_cast<SVMINT>(m_wss);
    params->convergence_epsilon = m_conv_eps;
    params->balance_cost   = m_balance_cost;
    params->do_scale       = 0;
    params->do_scale_y     = 0;

    /* SVM type flags */
    bool is_nu      = (m_svm_type == 2 || m_svm_type == 3);
    bool is_pattern = (m_svm_type == 0 || m_svm_type == 2);
    params->is_nu      = is_nu ? 1 : 0;
    params->is_pattern = is_pattern ? 1 : 0;

    /* kernel */
    kernel_c* kern = make_kernel(m_kernel_type, m_gamma, m_degree, m_coef0);

    /* example set */
    example_set_c* es = new example_set_c(static_cast<SVMINT>(n_samples),
                                          static_cast<SVMINT>(n_features));
    fill_example_set(es, X, y, n_samples, n_features);

    /* SVM */
    svm_c* svm = make_svm(m_svm_type);
    svm->init(kern, params);
    kern->init(static_cast<SVMINT>(m_cache_mb), es);
    svm->train(es);

    m_params    = params;
    m_kernel    = kern;
    m_train_set = es;
    m_svm       = svm;
    m_trained   = 1;
}


/* ── predict ──────────────────────────────────────────────────────────────── */

void MySVMWrapper::predict(const double* X, double* out, int n_test, int n_features)
{
    if (!m_trained)
        throw std::runtime_error("MySVMWrapper: model not trained");

    example_set_c* test_set = new example_set_c(static_cast<SVMINT>(n_test),
                                                 static_cast<SVMINT>(n_features));
    fill_example_set(test_set, X, 0, n_test, n_features);

    svm_c* svm = static_cast<svm_c*>(m_svm);
    svm->predict(test_set);

    for (int i = 0; i < n_test; ++i)
        out[i] = static_cast<double>(test_set->get_y(static_cast<SVMINT>(i)));

    delete test_set;
}


/* ── get_n_sv ─────────────────────────────────────────────────────────────── */

int MySVMWrapper::get_n_sv() const
{
    if (!m_trained) return 0;

    example_set_c* es = static_cast<example_set_c*>(m_train_set);
    SVMFLOAT* alphas  = es->get_alphas();
    int n = static_cast<int>(es->size());
    int count = 0;
    for (int i = 0; i < n; ++i)
        if (alphas[i] != 0.0) ++count;
    return count;
}
