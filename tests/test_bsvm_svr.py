"""
Tests for BSVMRegressor (Bound-Constrained SVM regressor).
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression

from scikit_svm.bsvm import BSVMRegressor


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def regression_data():
    """100-sample noisy linear regression problem."""
    X, y = make_regression(
        n_samples=100, n_features=4, noise=10.0, random_state=0,
    )
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture(scope="module")
def simple_regression_data():
    """50-sample near-noiseless linear problem: easy for SVR."""
    rng = np.random.RandomState(7)
    X = rng.randn(50, 2)
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.05 * rng.randn(50)
    return X.astype(np.float64), y.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Basic fit / predict
# ─────────────────────────────────────────────────────────────────────────────

class TestBSVMRegressorBasic:

    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        reg = BSVMRegressor(verbose=False)
        assert reg.fit(X, y) is reg

    def test_fitted_attributes_exist(self, regression_data):
        X, y = regression_data
        reg = BSVMRegressor(verbose=False).fit(X, y)
        assert hasattr(reg, 'model_')
        assert hasattr(reg, 'support_vectors_')
        assert hasattr(reg, 'dual_coef_')
        assert hasattr(reg, 'n_sv_')
        assert hasattr(reg, 'n_features_in_')
        assert hasattr(reg, 'time_')

    def test_n_sv_positive(self, regression_data):
        X, y = regression_data
        reg = BSVMRegressor(verbose=False).fit(X, y)
        assert reg.n_sv_ > 0

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        reg = BSVMRegressor(verbose=False).fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_is_float(self, regression_data):
        X, y = regression_data
        reg = BSVMRegressor(verbose=False).fit(X, y)
        preds = reg.predict(X)
        assert preds.dtype == np.float64

    def test_mse_reasonable(self, simple_regression_data):
        """MSE on training data should be less than the variance of y."""
        X, y = simple_regression_data
        reg = BSVMRegressor(kernel='rbf', C=100.0, verbose=False).fit(X, y)
        preds = reg.predict(X)
        mse = np.mean((preds - y) ** 2)
        assert mse < np.var(y), (
            f"MSE {mse:.4f} should be less than variance {np.var(y):.4f}"
        )

    def test_decision_function_close_to_predict(self, regression_data):
        """decision_function and predict should return very similar values."""
        X, y = regression_data
        reg = BSVMRegressor(verbose=False).fit(X, y)
        preds  = reg.predict(X)
        scores = reg.decision_function(X)
        assert scores.shape == (len(X),)
        # They use different code paths but the same model; allow some tolerance
        # due to floating-point differences between C++ and NumPy kernel eval.
        np.testing.assert_allclose(scores, preds, rtol=0.05, atol=1.0)

    def test_n_features_in_(self, regression_data):
        X, y = regression_data
        reg = BSVMRegressor(verbose=False).fit(X, y)
        assert reg.n_features_in_ == X.shape[1]

    def test_time_non_negative(self, regression_data):
        X, y = regression_data
        reg = BSVMRegressor(verbose=False).fit(X, y)
        assert reg.time_ >= 0.0

    def test_dual_coef_shape(self, regression_data):
        """dual_coef_ must have shape (1, n_sv) for SVR."""
        X, y = regression_data
        reg = BSVMRegressor(verbose=False).fit(X, y)
        assert reg.dual_coef_.shape == (1, reg.n_sv_)


# ─────────────────────────────────────────────────────────────────────────────
# Kernels
# ─────────────────────────────────────────────────────────────────────────────

class TestBSVMRegressorKernels:

    @pytest.mark.parametrize("kernel", ["rbf", "linear"])
    def test_named_kernels(self, regression_data, kernel):
        X, y = regression_data
        reg = BSVMRegressor(kernel=kernel, verbose=False).fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (len(X),)
        assert preds.dtype == np.float64


# ─────────────────────────────────────────────────────────────────────────────
# scikit-learn API compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestBSVMRegressorSklearnAPI:

    def test_get_params(self):
        reg = BSVMRegressor(C=5.0, kernel='linear', gamma=0.1)
        p = reg.get_params()
        assert p['C']       == 5.0
        assert p['kernel']  == 'linear'
        assert p['gamma']   == 0.1

    def test_set_params(self):
        reg = BSVMRegressor()
        reg.set_params(C=10.0, kernel='poly', epsilon=0.5)
        assert reg.C       == 10.0
        assert reg.kernel  == 'poly'
        assert reg.epsilon == 0.5

    def test_clone(self, regression_data):
        X, y = regression_data
        reg  = BSVMRegressor(C=2.0, verbose=False).fit(X, y)
        reg2 = clone(reg)
        assert not hasattr(reg2, 'model_')
        assert reg2.C == reg.C

    def test_score(self, simple_regression_data):
        """R² score on the near-noiseless problem should be high."""
        X, y = simple_regression_data
        reg = BSVMRegressor(kernel='rbf', C=100.0, verbose=False).fit(X, y)
        score = reg.score(X, y)
        assert score > 0.9, f"R² = {score:.4f} is unexpectedly low"

    def test_predict_before_fit_raises(self, regression_data):
        X, _ = regression_data
        with pytest.raises(Exception):
            BSVMRegressor().predict(X)
