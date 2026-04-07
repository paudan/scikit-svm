"""
Tests for scikit_svm.lssvm — LSSVMClassifier and LSSVMRegressor.

These tests verify:
  1. Basic fit / predict / score (classif + regression)
  2. All three kernel types (rbf, linear, poly)
  3. Preprocessing on/off
  4. Cross-validation tuning
  5. Bayesian inference (3 levels)
  6. Bayesian error bars
  7. Confidence intervals
  8. Posterior class probabilities (predict_proba)
  9. Multiclass classification
 10. sklearn estimator-compatibility checks
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from sklearn.datasets import (
    make_classification,
    make_regression,
    make_blobs,
)
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_svm.lssvm import (
    LSSVMClassifier,
    LSSVMRegressor,
    _kernel_matrix,
    _kpca,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(
        n_samples=80, n_features=4, n_informative=3, n_redundant=1,
        random_state=0
    )
    # LS-SVM needs ±1 labels for binary tasks
    y = np.where(y == 1, 1, -1).astype(float)
    return X[:60], y[:60], X[60:], y[60:]


@pytest.fixture(scope="module")
def regression_data():
    X, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=0)
    return X[:60], y[:60], X[60:], y[60:]


@pytest.fixture(scope="module")
def multiclass_data():
    X, y = make_blobs(n_samples=90, centers=3, n_features=2, random_state=0)
    return X[:70], y[:70], X[70:], y[70:]


# ─── Kernel matrix ─────────────────────────────────────────────────────────────

class TestKernelMatrix:
    def test_rbf_symmetric(self):
        X = np.random.RandomState(0).randn(10, 3)
        K = _kernel_matrix(X, "rbf", np.array([1.0]))
        assert K.shape == (10, 10)
        assert_allclose(K, K.T, atol=1e-12)
        assert_allclose(np.diag(K), np.ones(10), atol=1e-12)

    def test_rbf_train_test(self):
        rng = np.random.RandomState(1)
        X, Xt = rng.randn(8, 3), rng.randn(5, 3)
        K = _kernel_matrix(X, "rbf", np.array([0.5]), Xt)
        assert K.shape == (8, 5)

    def test_linear_symmetric(self):
        X = np.random.RandomState(2).randn(6, 4)
        K = _kernel_matrix(X, "linear", np.array([]))
        assert K.shape == (6, 6)
        assert_allclose(K, K.T, atol=1e-12)

    def test_poly_train_test(self):
        rng = np.random.RandomState(3)
        X, Xt = rng.randn(7, 3), rng.randn(4, 3)
        K = _kernel_matrix(X, "poly", np.array([1.0, 3.0]), Xt)
        assert K.shape == (7, 4)
        # Manual check
        K_man = (X @ Xt.T + 1.0) ** 3
        assert_allclose(K, K_man, atol=1e-10)

    def test_unknown_kernel_raises(self):
        X = np.random.randn(5, 2)
        with pytest.raises(ValueError, match="Unknown kernel"):
            _kernel_matrix(X, "unknown", np.array([]))


# ─── KPCA ──────────────────────────────────────────────────────────────────────

class TestKPCA:
    def test_returns_positive_eigenvalues(self):
        X = np.random.RandomState(0).randn(20, 3)
        lam, R, peff = _kpca(X, "rbf", np.array([1.0]))
        assert np.all(lam > 0)

    def test_eigenvalues_descending(self):
        X = np.random.RandomState(1).randn(15, 2)
        lam, R, peff = _kpca(X, "rbf", np.array([0.5]))
        assert np.all(np.diff(lam) <= 0)


# ─── Classifier — basic ────────────────────────────────────────────────────────

class TestLSSVMClassifierBasic:
    def test_fit_predict_binary(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=10.0, kernel="rbf", sigma2=1.0, preprocess=True)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        assert y_pred.shape == y_te.shape
        assert set(y_pred).issubset({-1.0, 1.0})
        acc = clf.score(X_te, y_te)
        assert acc > 0.5, f"Accuracy {acc:.2f} should be > 0.5"

    def test_fitted_attributes(self, binary_data):
        X_tr, y_tr, _, _ = binary_data
        clf = LSSVMClassifier(C=5.0).fit(X_tr, y_tr)
        assert hasattr(clf, "alpha_")
        assert hasattr(clf, "b_")
        assert clf.alpha_.shape == (len(X_tr),)

    def test_decision_function(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=5.0).fit(X_tr, y_tr)
        scores = clf.decision_function(X_te)
        assert scores.shape == (len(X_te),)
        # predict should agree with sign of decision_function
        pred_from_df = np.where(scores >= 0, 1.0, -1.0)
        assert_allclose(pred_from_df, clf.predict(X_te))

    def test_no_preprocess(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=5.0, preprocess=False).fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        assert acc > 0.4


# ─── Classifier — kernels ──────────────────────────────────────────────────────

class TestLSSVMClassifierKernels:
    @pytest.mark.parametrize("kernel,kw", [
        ("rbf",    {"sigma2": 1.0}),
        ("linear", {}),
        ("poly",   {"coef0": 1.0, "degree": 2}),
    ])
    def test_all_kernels(self, binary_data, kernel, kw):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=5.0, kernel=kernel, **kw)
        clf.fit(X_tr, y_tr)
        acc = clf.score(X_te, y_te)
        assert acc > 0.4, f"Kernel {kernel} accuracy {acc:.2f}"

    def test_rbf_gamma_param(self, binary_data):
        """gamma=1/(2*sigma2) should give same result as sigma2."""
        X_tr, y_tr, X_te, y_te = binary_data
        clf1 = LSSVMClassifier(C=5.0, sigma2=0.5).fit(X_tr, y_tr)
        clf2 = LSSVMClassifier(C=5.0, gamma=1.0).fit(X_tr, y_tr)  # 1/(2*0.5)
        assert_allclose(clf1.alpha_, clf2.alpha_, atol=1e-6)


# ─── Classifier — multiclass ───────────────────────────────────────────────────

class TestLSSVMClassifierMulticlass:
    def test_multiclass_ovo(self, multiclass_data):
        X_tr, y_tr, X_te, y_te = multiclass_data
        clf = LSSVMClassifier(C=10.0, kernel="rbf", sigma2=2.0)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        assert set(y_pred).issubset(set(y_te))
        acc = clf.score(X_te, y_te)
        assert acc > 0.6, f"Multiclass acc {acc:.2f}"


# ─── Classifier — CV tuning ────────────────────────────────────────────────────

class TestLSSVMClassifierTuneCV:
    def test_tune_cv_runs(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=1.0, kernel="rbf", sigma2=1.0)
        clf.tune_cv(X_tr, y_tr, n_folds=5, cost="misclass", random_state=0)
        assert clf.C > 0
        assert clf.sigma2 > 0
        acc = clf.score(X_te, y_te)
        assert acc > 0.5

    def test_tune_cv_linear(self, binary_data):
        X_tr, y_tr, _, _ = binary_data
        clf = LSSVMClassifier(C=1.0, kernel="linear")
        clf.tune_cv(X_tr, y_tr, n_folds=5, cost="misclass", random_state=1)
        assert clf.C > 0


# ─── Classifier — Bayesian tuning ─────────────────────────────────────────────

class TestLSSVMClassifierBayesian:
    def test_bayesian_inference_level1(self, binary_data):
        X_tr, y_tr, _, _ = binary_data
        clf = LSSVMClassifier(C=5.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        bay = clf.bayesian_inference(level=1)
        assert bay.costL1 is not None
        assert bay.Ew is not None
        assert bay.Ed is not None
        assert bay.mu > 0
        assert bay.zeta > 0

    def test_bayesian_inference_level2(self, binary_data):
        X_tr, y_tr, _, _ = binary_data
        clf = LSSVMClassifier(C=5.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        bay = clf.bayesian_inference(level=2)
        assert bay.costL2 is not None
        assert bay.Geff is not None
        assert 1 <= bay.Geff <= len(X_tr)

    def test_bayesian_inference_level3(self, binary_data):
        X_tr, y_tr, _, _ = binary_data
        clf = LSSVMClassifier(C=5.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        bay = clf.bayesian_inference(level=3)
        assert bay.costL3 is not None

    def test_tune_bayesian_level2(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=1.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        clf.tune_bayesian(level=2)
        assert clf.C > 0
        acc = clf.score(X_te, y_te)
        assert acc > 0.5

    def test_tune_bayesian_level3(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=1.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        clf.tune_bayesian(level=3)
        assert clf.C > 0
        assert clf.sigma2 > 0
        acc = clf.score(X_te, y_te)
        assert acc > 0.5


# ─── Classifier — posterior probabilities ─────────────────────────────────────

class TestLSSVMClassifierPredictProba:
    def test_predict_proba_shape(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        assert proba.shape == (len(X_te), 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        assert_allclose(proba.sum(axis=1), np.ones(len(X_te)), atol=1e-6)

    def test_predict_proba_in_zero_one(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_predict_proba_consistent_with_predict(self, binary_data):
        X_tr, y_tr, X_te, y_te = binary_data
        clf = LSSVMClassifier(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)
        # Argmax of proba should (mostly) agree with predict
        pred_proba = np.where(proba[:, 1] >= 0.5, clf._le_pos, clf._le_neg)
        pred_direct = clf.predict(X_te)
        agreement = np.mean(pred_proba == pred_direct)
        assert agreement > 0.7


# ─── Regressor — basic ────────────────────────────────────────────────────────

class TestLSSVMRegressorBasic:
    def test_fit_predict(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0)
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        assert y_pred.shape == y_te.shape

    def test_r2_score(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=100.0, kernel="rbf", sigma2=10.0)
        reg.fit(X_tr, y_tr)
        r2 = reg.score(X_te, y_te)
        assert r2 > 0.5, f"R² {r2:.3f} should be > 0.5"

    def test_no_preprocess(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=50.0, kernel="rbf", sigma2=5.0, preprocess=False)
        reg.fit(X_tr, y_tr)
        r2 = reg.score(X_te, y_te)
        assert r2 > 0.3


# ─── Regressor — kernels ──────────────────────────────────────────────────────

class TestLSSVMRegressorKernels:
    @pytest.mark.parametrize("kernel,kw", [
        ("rbf",    {"sigma2": 1.0}),
        ("linear", {}),
        ("poly",   {"coef0": 1.0, "degree": 2}),
    ])
    def test_all_kernels(self, regression_data, kernel, kw):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=10.0, kernel=kernel, **kw)
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_te)
        assert y_pred.shape == y_te.shape
        assert np.isfinite(y_pred).all()


# ─── Regressor — CV tuning ────────────────────────────────────────────────────

class TestLSSVMRegressorTuneCV:
    def test_tune_cv_runs(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=1.0, kernel="rbf", sigma2=1.0)
        reg.tune_cv(X_tr, y_tr, n_folds=5, cost="mse", random_state=0)
        assert reg.C > 0
        assert reg.sigma2 > 0
        r2 = reg.score(X_te, y_te)
        assert np.isfinite(r2)


# ─── Regressor — Bayesian tuning ──────────────────────────────────────────────

class TestLSSVMRegressorBayesian:
    def test_bayesian_level1(self, regression_data):
        X_tr, y_tr, _, _ = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        bay = reg.bayesian_inference(level=1)
        assert bay.costL1 is not None
        assert bay.Ew >= 0.0
        assert bay.Ed >= 0.0

    def test_tune_bayesian_level2(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=1.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        reg.tune_bayesian(level=2)
        assert reg.C > 0
        r2 = reg.score(X_te, y_te)
        assert np.isfinite(r2)

    def test_tune_bayesian_level3(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=1.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        reg.tune_bayesian(level=3)
        assert reg.C > 0
        assert reg.sigma2 > 0


# ─── Regressor — error bars ────────────────────────────────────────────────────

class TestLSSVMRegressorErrorBars:
    def test_error_bars_shape(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        sigma = reg.error_bars(X_te)
        assert sigma.shape == (len(X_te),)

    def test_error_bars_nonnegative(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        sigma = reg.error_bars(X_te)
        assert np.all(sigma >= 0.0)

    def test_error_bars_finite(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        sigma = reg.error_bars(X_te)
        assert np.all(np.isfinite(sigma))


# ─── Regressor — confidence intervals ────────────────────────────────────────

class TestLSSVMRegressorConfidenceInterval:
    def test_ci_shape(self, regression_data):
        X_tr, y_tr, _, _ = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        ci = reg.confidence_interval(alpha=0.05)
        assert ci.shape == (len(X_tr), 2)

    def test_ci_lower_lt_upper(self, regression_data):
        X_tr, y_tr, _, _ = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        ci = reg.confidence_interval(alpha=0.05)
        assert np.all(ci[:, 0] <= ci[:, 1])

    def test_ci_pointwise_shape(self, regression_data):
        X_tr, y_tr, _, _ = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        ci = reg.confidence_interval(alpha=0.05, conftype="pointwise")
        assert ci.shape == (len(X_tr), 2)

    def test_ci_invalid_type(self, regression_data):
        X_tr, y_tr, _, _ = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        with pytest.raises(ValueError):
            reg.confidence_interval(conftype="invalid")


# ─── Smoother matrix ──────────────────────────────────────────────────────────

class TestSmootherMatrix:
    def test_smoother_shape(self, regression_data):
        X_tr, y_tr, _, _ = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        S = reg._smoother()
        n = len(X_tr)
        assert S.shape == (n, n)

    def test_smoother_test_shape(self, regression_data):
        X_tr, y_tr, X_te, _ = regression_data
        reg = LSSVMRegressor(C=10.0, kernel="rbf", sigma2=1.0).fit(X_tr, y_tr)
        Xt_proc = reg._scale_X(X_te)
        S = reg._smoother(Xt_proc)
        assert S.shape == (len(X_te), len(X_tr))


# ─── Numerical equivalence — simple 1D example ────────────────────────────────

class TestNumericalEquivalence:
    """
    Verify the LS-SVM formula against a hand-solved small system.
    For a linear kernel with no preprocessing, the LS-SVM system is:
        (X X' + I/C) alpha + e*b = y,  e'alpha = 0
    We verify that the Python solve gives the same alpha/b as numpy.linalg.solve.
    """

    def test_linear_kernel_solve(self):
        rng = np.random.RandomState(42)
        X = rng.randn(8, 2)
        y = rng.randn(8)
        C = 5.0

        reg = LSSVMRegressor(C=C, kernel="linear", preprocess=False)
        reg.fit(X, y)

        # Manual solve
        n = 8
        K = X @ X.T
        H = K + np.eye(n) / C
        e = np.ones(n)
        rhs = np.column_stack([y, e])
        from scipy import linalg
        sol = linalg.solve(H, rhs)
        v, nu = sol[:, 0], sol[:, 1]
        s = e @ nu
        b_ref = float(nu @ y / s)
        alpha_ref = v - nu * b_ref

        assert_allclose(reg.alpha_, alpha_ref, atol=1e-8)
        assert_allclose(reg.b_, b_ref, atol=1e-8)

    def test_predict_formula(self):
        """Prediction = K_test @ alpha + b."""
        rng = np.random.RandomState(7)
        X = rng.randn(10, 2)
        Xt = rng.randn(4, 2)
        y = rng.randn(10)

        reg = LSSVMRegressor(C=2.0, kernel="rbf", sigma2=1.0, preprocess=False)
        reg.fit(X, y)
        y_pred = reg.predict(Xt)

        from scikit_svm.lssvm import _kernel_matrix
        K_test = _kernel_matrix(X, "rbf", np.array([1.0]), Xt)
        y_ref = K_test.T @ reg.alpha_ + reg.b_
        assert_allclose(y_pred, y_ref, atol=1e-10)


# ─── sklearn estimator checks ─────────────────────────────────────────────────

# These use sklearn's parametrize_with_checks for broad compatibility testing.
# We override a few settings that don't apply to LS-SVM (non-standard labels etc.)

@parametrize_with_checks([
    LSSVMRegressor(C=1.0, kernel="rbf", sigma2=1.0, preprocess=False),
])
def test_sklearn_compatible_regressor(estimator, check):
    check(estimator)


# Note: classifier checks need ±1 labels; we skip the full sklearn battery
# and do a targeted API-level check instead.
def test_sklearn_get_set_params():
    clf = LSSVMClassifier(C=5.0, kernel="poly", degree=3, coef0=0.5)
    params = clf.get_params()
    assert params["C"] == 5.0
    assert params["kernel"] == "poly"
    clf.set_params(C=10.0)
    assert clf.C == 10.0


def test_sklearn_clone():
    from sklearn.base import clone
    clf = LSSVMClassifier(C=7.0, kernel="rbf", sigma2=2.0)
    clf2 = clone(clf)
    assert clf2.C == 7.0
    assert clf2.sigma2 == 2.0
    assert not hasattr(clf2, "alpha_")
