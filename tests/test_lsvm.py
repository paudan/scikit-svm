"""Tests for LSVM (linear Lagrangian SVM)."""

import numpy as np
import pytest
from sklearn.base import clone

from scikit_svm import LSVM


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separable_data():
    """Two well-separated Gaussian clusters — should yield 100 % train acc."""
    rng = np.random.RandomState(0)
    X_pos = rng.randn(20, 2) + np.array([3.0, 3.0])
    X_neg = rng.randn(20, 2) + np.array([-3.0, -3.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * 20 + [-1.0] * 20)
    return X, y


# ---------------------------------------------------------------------------
# Basic fit / predict
# ---------------------------------------------------------------------------

class TestLSVMBasic:
    def test_fit_returns_self(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False)
        assert clf.fit(X, y) is clf

    def test_fitted_attributes_exist(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        assert hasattr(clf, "w_")
        assert hasattr(clf, "gamma_")
        assert hasattr(clf, "n_iter_")
        assert hasattr(clf, "opt_cond_")
        assert hasattr(clf, "time_")
        assert hasattr(clf, "classes_")

    def test_w_shape(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        assert clf.w_.shape == (X.shape[1],)

    def test_gamma_is_scalar(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        assert np.ndim(clf.gamma_) == 0

    def test_classes(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [-1, 1])

    def test_predict_shape(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (len(y),)

    def test_predict_values_are_pm1(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        pred = clf.predict(X)
        assert set(pred).issubset({1.0, -1.0})

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_decision_function_shape(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(y),)

    def test_decision_function_sign_matches_predict(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        pred = clf.predict(X)
        # positive score → +1, negative score → -1
        assert np.all((scores > 0) == (pred == 1.0))


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------

class TestLSVMLabelValidation:
    def test_invalid_labels_raise(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_bad = np.array([0.0, 1.0])   # 0 is not valid
        with pytest.raises(ValueError, match="classes must be all 1 or -1"):
            LSVM(verbose=False).fit(X, y_bad)

    def test_valid_labels_integers(self, separable_data):
        X, y = separable_data
        # sklearn check_X_y converts int labels; ensure they are accepted
        y_int = y.astype(int)
        clf = LSVM(verbose=False).fit(X, y_int)
        assert clf.score(X, y_int) == 1.0


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestLSVMParameters:
    def test_explicit_nu(self, separable_data):
        X, y = separable_data
        clf = LSVM(nu=0.1, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_explicit_alpha(self, separable_data):
        X, y = separable_data
        nu = 0.1
        clf = LSVM(nu=nu, alpha=1.5 / nu, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_max_iter_respected(self, separable_data):
        X, y = separable_data
        clf = LSVM(max_iter=5, verbose=False).fit(X, y)
        assert clf.n_iter_ <= 5

    def test_normalize(self, separable_data):
        X, y = separable_data
        clf = LSVM(normalize=True, verbose=False).fit(X, y)
        assert clf.score(X, y) == 1.0
        # normalisation parameters should be stored
        assert clf._avg_ is not None
        assert clf._dev_ is not None

    def test_normalize_applied_in_predict(self, separable_data):
        X, y = separable_data
        clf_norm = LSVM(normalize=True, verbose=False).fit(X, y)
        clf_plain = LSVM(normalize=False, verbose=False).fit(X, y)
        # Both should predict correctly; the test is that predict doesn't crash
        assert clf_norm.score(X, y) >= 0.9
        assert clf_plain.score(X, y) >= 0.9

    def test_perturb(self, separable_data):
        X, y = separable_data
        clf = LSVM(perturb=0.01, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_alpha_warning(self, separable_data, capsys):
        X, y = separable_data
        nu = 1.0 / len(y)
        # alpha > 2/nu should trigger a warning print
        LSVM(nu=nu, alpha=3.0 / nu, verbose=False).fit(X, y)
        captured = capsys.readouterr()
        assert "Alpha is larger than 2/nu" in captured.out

    def test_n_iter_positive(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        assert clf.n_iter_ >= 1

    def test_opt_cond_non_negative(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        assert clf.opt_cond_ >= 0.0


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------

class TestLSVMVerbose:
    def test_verbose_true_prints(self, separable_data, capsys):
        X, y = separable_data
        LSVM(verbose=True).fit(X, y)
        out = capsys.readouterr().out
        assert "Running time" in out
        assert "Number of iterations" in out
        assert "Training accuracy" in out

    def test_verbose_false_silent(self, separable_data, capsys):
        X, y = separable_data
        LSVM(verbose=False).fit(X, y)
        out = capsys.readouterr().out
        assert out == ""


# ---------------------------------------------------------------------------
# scikit-learn API compatibility
# ---------------------------------------------------------------------------

class TestLSVMSklearnAPI:
    def test_get_params(self):
        clf = LSVM(nu=0.5, tol=1e-4, max_iter=50)
        params = clf.get_params()
        assert params["nu"] == 0.5
        assert params["tol"] == 1e-4
        assert params["max_iter"] == 50

    def test_set_params(self):
        clf = LSVM()
        clf.set_params(nu=0.2, tol=1e-3)
        assert clf.nu == 0.2
        assert clf.tol == 1e-3

    def test_clone(self, separable_data):
        X, y = separable_data
        clf = LSVM(nu=0.1, verbose=False).fit(X, y)
        cloned = clone(clf)
        # cloned should have same params but not be fitted
        assert cloned.nu == clf.nu
        assert not hasattr(cloned, "w_")

    def test_score_method(self, separable_data):
        X, y = separable_data
        clf = LSVM(verbose=False).fit(X, y)
        s = clf.score(X, y)
        assert 0.0 <= s <= 1.0

    def test_predict_before_fit_raises(self):
        clf = LSVM()
        with pytest.raises(Exception):
            clf.predict(np.array([[1.0, 2.0]]))

    def test_default_nu_is_1_over_m(self, separable_data):
        """Default nu = 1/m is used internally — verify fit succeeds."""
        X, y = separable_data
        clf = LSVM(nu=None, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9
