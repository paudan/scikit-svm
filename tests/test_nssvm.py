"""Tests for NSSVM (nonlinear / kernel Smooth SVM)."""

import numpy as np
import pytest
from sklearn.base import clone

from scikit_svm import NSSVM
from scikit_svm._ssvm_base import _est_mu, _rec_kernel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separable_data():
    rng = np.random.RandomState(0)
    X_pos = rng.randn(20, 2) + np.array([3.0, 3.0])
    X_neg = rng.randn(20, 2) + np.array([-3.0, -3.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * 20 + [-1.0] * 20)
    return X, y


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------

class TestNSSVMBasic:
    def test_fit_returns_self(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0)
        assert clf.fit(X, y) is clf

    def test_fitted_attributes_exist(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        for attr in ("w_", "gamma_", "Abar_", "mu_", "nu_", "n_iter_", "time_", "classes_"):
            assert hasattr(clf, attr), f"missing: {attr}"

    def test_w_shape_full_basis(self, separable_data):
        X, y = separable_data
        clf = NSSVM(reduce_rate=1.0, random_state=0).fit(X, y)
        # With reduce_rate=1, n_basis = m
        assert clf.w_.shape == (len(y),)

    def test_Abar_shape(self, separable_data):
        X, y = separable_data
        clf = NSSVM(reduce_rate=0.5, random_state=0).fit(X, y)
        n_basis = int(np.floor(0.5 * len(y)))
        assert clf.Abar_.shape == (n_basis, X.shape[1])

    def test_classes(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [-1, 1])

    def test_predict_shape(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        assert clf.predict(X).shape == (len(y),)

    def test_predict_values_are_pm1(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        assert set(clf.predict(X)).issubset({1.0, -1.0})

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_decision_function_shape(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        assert clf.decision_function(X).shape == (len(y),)

    def test_decision_function_sign_matches_predict(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        scores = clf.decision_function(X)
        pred   = clf.predict(X)
        assert np.all((scores > 0) == (pred == 1.0))


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------

class TestNSSVMLabelValidation:
    def test_invalid_labels_raise(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="classes must be all 1 or -1"):
            NSSVM().fit(X, np.array([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Mu parameter
# ---------------------------------------------------------------------------

class TestNSSVMMu:
    def test_mu_auto_estimated(self, separable_data):
        X, y = separable_data
        clf = NSSVM(mu=None, random_state=0).fit(X, y)
        assert clf.mu_ > 0

    def test_mu_explicit(self, separable_data):
        X, y = separable_data
        clf = NSSVM(mu=0.5, random_state=0).fit(X, y)
        assert clf.mu_ == 0.5

    def test_mu_estimate_helper(self, separable_data):
        X, y = separable_data
        mu = _est_mu(X, y)
        assert mu > 0


# ---------------------------------------------------------------------------
# Nu estimation
# ---------------------------------------------------------------------------

class TestNSSVMNuEstimation:
    def test_nu_none(self, separable_data):
        X, y = separable_data
        clf = NSSVM(nu=None, random_state=0).fit(X, y)
        assert clf.nu_ > 0

    def test_nu_easy(self, separable_data):
        X, y = separable_data
        clf = NSSVM(nu='easy', random_state=0).fit(X, y)
        assert clf.nu_ > 0

    def test_nu_explicit(self, separable_data):
        X, y = separable_data
        clf = NSSVM(nu=1.0, random_state=0).fit(X, y)
        assert clf.nu_ == 1.0


# ---------------------------------------------------------------------------
# reduce_rate (basis selection)
# ---------------------------------------------------------------------------

class TestNSSVMReduceRate:
    def test_reduce_rate_half(self, separable_data):
        X, y = separable_data
        clf = NSSVM(reduce_rate=0.5, random_state=0).fit(X, y)
        expected_basis = int(np.floor(0.5 * len(y)))
        assert clf.Abar_.shape[0] == expected_basis
        assert clf.score(X, y) >= 0.8

    def test_reduce_rate_full(self, separable_data):
        X, y = separable_data
        clf = NSSVM(reduce_rate=1.0, random_state=0).fit(X, y)
        assert clf.Abar_.shape[0] == len(y)


# ---------------------------------------------------------------------------
# rec_kernel helper
# ---------------------------------------------------------------------------

class TestRecKernel:
    def test_output_shape(self):
        rng = np.random.RandomState(0)
        A = rng.randn(10, 3)
        B = rng.randn(5, 3)
        K = _rec_kernel(A, B, mu=0.5)
        assert K.shape == (10, 5)

    def test_self_kernel_diagonal_is_one(self):
        rng = np.random.RandomState(0)
        A = rng.randn(8, 3)
        K = _rec_kernel(A, A, mu=1.0)
        np.testing.assert_allclose(np.diag(K), np.ones(8))

    def test_kernel_values_in_0_1(self):
        rng = np.random.RandomState(0)
        A = rng.randn(10, 3)
        B = rng.randn(6, 3)
        K = _rec_kernel(A, B, mu=1.0)
        assert np.all(K >= 0) and np.all(K <= 1.0)

    def test_symmetry_square(self):
        rng = np.random.RandomState(0)
        A = rng.randn(8, 3)
        K = _rec_kernel(A, A, mu=0.5)
        np.testing.assert_allclose(K, K.T)


# ---------------------------------------------------------------------------
# Step size
# ---------------------------------------------------------------------------

class TestNSSVMStepSize:
    def test_armijo(self, separable_data):
        X, y = separable_data
        clf = NSSVM(use_armijo=True, random_state=0).fit(X, y)
        assert clf.score(X, y) >= 0.8

    def test_newton(self, separable_data):
        X, y = separable_data
        clf = NSSVM(use_armijo=False, random_state=0).fit(X, y)
        assert clf.score(X, y) >= 0.8


# ---------------------------------------------------------------------------
# Other parameters
# ---------------------------------------------------------------------------

class TestNSSVMParameters:
    def test_max_iter_respected(self, separable_data):
        X, y = separable_data
        clf = NSSVM(max_iter=2, random_state=0).fit(X, y)
        assert clf.n_iter_ <= 2

    def test_n_iter_positive(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        assert clf.n_iter_ >= 1

    def test_time_non_negative(self, separable_data):
        X, y = separable_data
        clf = NSSVM(random_state=0).fit(X, y)
        assert clf.time_ >= 0.0

    def test_random_state_reproducible(self, separable_data):
        X, y = separable_data
        clf1 = NSSVM(random_state=7).fit(X, y)
        clf2 = NSSVM(random_state=7).fit(X, y)
        np.testing.assert_array_equal(clf1.w_, clf2.w_)


# ---------------------------------------------------------------------------
# Verbose
# ---------------------------------------------------------------------------

class TestNSSVMVerbose:
    def test_verbose_true_prints(self, separable_data, capsys):
        X, y = separable_data
        NSSVM(verbose=True, random_state=0).fit(X, y)
        out = capsys.readouterr().out
        assert "Number of Iterations" in out
        assert "Elapse time" in out

    def test_verbose_false_silent(self, separable_data, capsys):
        X, y = separable_data
        NSSVM(verbose=False, random_state=0).fit(X, y)
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# scikit-learn API
# ---------------------------------------------------------------------------

class TestNSSVMSklearnAPI:
    def test_get_params(self):
        clf = NSSVM(nu=1.0, mu=0.5, reduce_rate=0.8)
        p = clf.get_params()
        assert p["nu"] == 1.0
        assert p["mu"] == 0.5
        assert p["reduce_rate"] == 0.8

    def test_set_params(self):
        clf = NSSVM()
        clf.set_params(nu=2.0, mu=0.1)
        assert clf.nu == 2.0
        assert clf.mu == 0.1

    def test_clone(self, separable_data):
        X, y = separable_data
        clf    = NSSVM(nu=1.0, mu=0.5, random_state=0).fit(X, y)
        cloned = clone(clf)
        assert cloned.nu == clf.nu
        assert cloned.mu == clf.mu
        assert not hasattr(cloned, "w_")

    def test_score_range(self, separable_data):
        X, y = separable_data
        s = NSSVM(random_state=0).fit(X, y).score(X, y)
        assert 0.0 <= s <= 1.0

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            NSSVM().predict(np.array([[1.0, 2.0]]))
