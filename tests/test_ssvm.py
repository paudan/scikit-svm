"""Tests for SSVM (linear Smooth SVM)."""

import numpy as np
import pytest
from sklearn.base import clone

from scikit_svm import SSVM


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

class TestSSVMBasic:
    def test_fit_returns_self(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0)
        assert clf.fit(X, y) is clf

    def test_fitted_attributes_exist(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        for attr in ("w_", "gamma_", "nu_", "n_iter_", "time_", "classes_"):
            assert hasattr(clf, attr), f"missing attribute: {attr}"

    def test_w_shape(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        assert clf.w_.shape == (X.shape[1],)

    def test_gamma_is_scalar(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        assert np.ndim(clf.gamma_) == 0

    def test_classes(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [-1, 1])

    def test_predict_shape(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        assert clf.predict(X).shape == (len(y),)

    def test_predict_values_are_pm1(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        assert set(clf.predict(X)).issubset({1.0, -1.0})

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_decision_function_shape(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        assert clf.decision_function(X).shape == (len(y),)

    def test_decision_function_sign_matches_predict(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        scores = clf.decision_function(X)
        pred   = clf.predict(X)
        assert np.all((scores > 0) == (pred == 1.0))


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------

class TestSSVMLabelValidation:
    def test_invalid_labels_raise(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="classes must be all 1 or -1"):
            SSVM().fit(X, np.array([0.0, 1.0]))

    def test_integer_labels_accepted(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y.astype(int))
        assert clf.score(X, y.astype(int)) == 1.0


# ---------------------------------------------------------------------------
# Nu estimation
# ---------------------------------------------------------------------------

class TestSSVMNuEstimation:
    def test_nu_none_runs(self, separable_data):
        X, y = separable_data
        clf = SSVM(nu=None, random_state=0).fit(X, y)
        assert clf.nu_ > 0
        assert clf.score(X, y) >= 0.9

    def test_nu_easy_runs(self, separable_data):
        X, y = separable_data
        clf = SSVM(nu='easy', random_state=0).fit(X, y)
        assert clf.nu_ > 0
        assert clf.score(X, y) >= 0.9

    def test_nu_explicit_runs(self, separable_data):
        X, y = separable_data
        clf = SSVM(nu=1.0, random_state=0).fit(X, y)
        assert clf.nu_ == 1.0
        assert clf.score(X, y) >= 0.9

    def test_nu_stored_as_float(self, separable_data):
        X, y = separable_data
        clf = SSVM(nu=2.5, random_state=0).fit(X, y)
        assert clf.nu_ == 2.5


# ---------------------------------------------------------------------------
# Step size
# ---------------------------------------------------------------------------

class TestSSVMStepSize:
    def test_armijo_step(self, separable_data):
        X, y = separable_data
        clf = SSVM(use_armijo=True, random_state=0).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_newton_step(self, separable_data):
        X, y = separable_data
        clf = SSVM(use_armijo=False, random_state=0).fit(X, y)
        assert clf.score(X, y) >= 0.9


# ---------------------------------------------------------------------------
# Other parameters
# ---------------------------------------------------------------------------

class TestSSVMParameters:
    def test_max_iter_respected(self, separable_data):
        X, y = separable_data
        clf = SSVM(max_iter=3, random_state=0).fit(X, y)
        assert clf.n_iter_ <= 3

    def test_n_iter_positive(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        assert clf.n_iter_ >= 1

    def test_time_non_negative(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        assert clf.time_ >= 0.0

    def test_random_state_reproducible(self, separable_data):
        X, y = separable_data
        clf1 = SSVM(random_state=42).fit(X, y)
        clf2 = SSVM(random_state=42).fit(X, y)
        np.testing.assert_array_equal(clf1.w_, clf2.w_)
        assert clf1.gamma_ == clf2.gamma_


# ---------------------------------------------------------------------------
# Verbose
# ---------------------------------------------------------------------------

class TestSSVMVerbose:
    def test_verbose_true_prints(self, separable_data, capsys):
        X, y = separable_data
        SSVM(verbose=True, random_state=0).fit(X, y)
        out = capsys.readouterr().out
        assert "Number of Iterations" in out
        assert "Elapse time" in out

    def test_verbose_false_silent(self, separable_data, capsys):
        X, y = separable_data
        SSVM(verbose=False, random_state=0).fit(X, y)
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# scikit-learn API
# ---------------------------------------------------------------------------

class TestSSVMSklearnAPI:
    def test_get_params(self):
        clf = SSVM(nu=0.5, tol=1e-6, max_iter=50)
        p = clf.get_params()
        assert p["nu"] == 0.5
        assert p["tol"] == 1e-6
        assert p["max_iter"] == 50

    def test_set_params(self):
        clf = SSVM()
        clf.set_params(nu=0.3, use_armijo=False)
        assert clf.nu == 0.3
        assert clf.use_armijo is False

    def test_clone(self, separable_data):
        X, y = separable_data
        clf    = SSVM(nu=1.0, random_state=0).fit(X, y)
        cloned = clone(clf)
        assert cloned.nu == clf.nu
        assert not hasattr(cloned, "w_")

    def test_score_range(self, separable_data):
        X, y = separable_data
        clf = SSVM(random_state=0).fit(X, y)
        s = clf.score(X, y)
        assert 0.0 <= s <= 1.0

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            SSVM().predict(np.array([[1.0, 2.0]]))
