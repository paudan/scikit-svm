"""Tests for LSVMK (kernel Lagrangian SVM)."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel

from scikit_svm import LSVMK


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def separable_data():
    """Two well-separated Gaussian clusters."""
    rng = np.random.RandomState(0)
    X_pos = rng.randn(20, 2) + np.array([3.0, 3.0])
    X_neg = rng.randn(20, 2) + np.array([-3.0, -3.0])
    X = np.vstack([X_pos, X_neg])
    y = np.array([1.0] * 20 + [-1.0] * 20)
    return X, y


@pytest.fixture
def precomputed_data(separable_data):
    """Pre-computed RBF kernel matrix for the separable dataset."""
    X, y = separable_data
    KM = rbf_kernel(X)
    return KM, y, X


# ---------------------------------------------------------------------------
# Basic fit / predict (default RBF kernel)
# ---------------------------------------------------------------------------

class TestLSVMKBasic:
    def test_fit_returns_self(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False)
        assert clf.fit(X, y) is clf

    def test_fitted_attributes_exist(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        assert hasattr(clf, "dual_coef_")
        assert hasattr(clf, "d_")
        assert hasattr(clf, "X_fit_")
        assert hasattr(clf, "n_iter_")
        assert hasattr(clf, "opt_cond_")
        assert hasattr(clf, "time_")
        assert hasattr(clf, "classes_")

    def test_dual_coef_shape(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        assert clf.dual_coef_.shape == (len(y),)

    def test_d_matches_y(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        np.testing.assert_array_equal(clf.d_, y)

    def test_X_fit_stored(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        assert clf.X_fit_ is not None
        assert clf.X_fit_.shape == X.shape

    def test_classes(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [-1, 1])

    def test_predict_shape(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        assert clf.predict(X).shape == (len(y),)

    def test_predict_values_are_pm1(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        assert set(clf.predict(X)).issubset({1.0, -1.0})

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_decision_function_shape(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        assert clf.decision_function(X).shape == (len(y),)

    def test_decision_function_sign_matches_predict(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        pred = clf.predict(X)
        assert np.all((scores > 0) == (pred == 1.0))


# ---------------------------------------------------------------------------
# Label validation
# ---------------------------------------------------------------------------

class TestLSVMKLabelValidation:
    def test_invalid_labels_raise(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_bad = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="classes must be all 1 or -1"):
            LSVMK(verbose=False).fit(X, y_bad)


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

class TestLSVMKKernels:
    @pytest.mark.parametrize("kernel", ["rbf", "linear", "poly"])
    def test_named_kernels(self, separable_data, kernel):
        X, y = separable_data
        clf = LSVMK(kernel=kernel, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_linear_kernel_consistent_with_lsvm(self, separable_data):
        """Linear kernel LSVMK and LSVM should both perfectly classify."""
        from scikit_svm import LSVM
        X, y = separable_data
        lsvmk = LSVMK(kernel="linear", verbose=False).fit(X, y)
        lsvm = LSVM(verbose=False).fit(X, y)
        assert lsvmk.score(X, y) == 1.0
        assert lsvm.score(X, y) == 1.0

    def test_callable_kernel(self, separable_data):
        X, y = separable_data

        def my_linear(A, B):
            return A @ B.T

        clf = LSVMK(kernel=my_linear, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_gamma_parameter(self, separable_data):
        X, y = separable_data
        clf = LSVMK(kernel="rbf", gamma=0.5, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_poly_degree_coef0(self, separable_data):
        X, y = separable_data
        clf = LSVMK(kernel="poly", degree=2, coef0=1.0, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9


# ---------------------------------------------------------------------------
# Precomputed kernel
# ---------------------------------------------------------------------------

class TestLSVMKPrecomputed:
    def test_precomputed_fit(self, precomputed_data):
        KM, y, _ = precomputed_data
        clf = LSVMK(kernel="precomputed", verbose=False).fit(KM, y)
        assert clf.X_fit_ is None

    def test_precomputed_predict(self, precomputed_data):
        KM, y, X = precomputed_data
        clf = LSVMK(kernel="precomputed", verbose=False).fit(KM, y)
        # For training data, K_test == K_train
        KM_test = rbf_kernel(X, X)
        pred = clf.predict(KM_test)
        assert pred.shape == (len(y),)
        assert set(pred).issubset({1.0, -1.0})

    def test_precomputed_accuracy(self, precomputed_data):
        KM, y, X = precomputed_data
        clf = LSVMK(kernel="precomputed", verbose=False).fit(KM, y)
        KM_test = rbf_kernel(X, X)
        pred = clf.predict(KM_test)
        assert np.mean(pred == y) == 1.0

    def test_precomputed_matches_rbf(self, separable_data):
        """Precomputed RBF and kernel='rbf' should agree on training data."""
        X, y = separable_data
        KM = rbf_kernel(X)
        clf_pre = LSVMK(kernel="precomputed", verbose=False).fit(KM, y)
        clf_rbf = LSVMK(kernel="rbf", verbose=False).fit(X, y)

        pred_pre = clf_pre.predict(rbf_kernel(X, X))
        pred_rbf = clf_rbf.predict(X)
        np.testing.assert_array_equal(pred_pre, pred_rbf)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

class TestLSVMKParameters:
    def test_explicit_nu(self, separable_data):
        X, y = separable_data
        clf = LSVMK(nu=0.1, verbose=False).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_max_iter_respected(self, separable_data):
        X, y = separable_data
        clf = LSVMK(max_iter=5, verbose=False).fit(X, y)
        assert clf.n_iter_ <= 5

    def test_alpha_warning(self, separable_data, capsys):
        X, y = separable_data
        nu = 1.0 / len(y)
        LSVMK(nu=nu, alpha=3.0 / nu, verbose=False).fit(X, y)
        assert "Alpha is larger than 2/nu" in capsys.readouterr().out

    def test_opt_cond_non_negative(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        assert clf.opt_cond_ >= 0.0


# ---------------------------------------------------------------------------
# Verbose output
# ---------------------------------------------------------------------------

class TestLSVMKVerbose:
    def test_verbose_true_prints(self, separable_data, capsys):
        X, y = separable_data
        LSVMK(verbose=True).fit(X, y)
        out = capsys.readouterr().out
        assert "Running time" in out
        assert "Number of iterations" in out
        assert "Training accuracy" in out

    def test_verbose_false_silent(self, separable_data, capsys):
        X, y = separable_data
        LSVMK(verbose=False).fit(X, y)
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# scikit-learn API compatibility
# ---------------------------------------------------------------------------

class TestLSVMKSklearnAPI:
    def test_get_params(self):
        clf = LSVMK(kernel="poly", nu=0.5, degree=2)
        params = clf.get_params()
        assert params["kernel"] == "poly"
        assert params["nu"] == 0.5
        assert params["degree"] == 2

    def test_set_params(self):
        clf = LSVMK()
        clf.set_params(nu=0.2, kernel="linear")
        assert clf.nu == 0.2
        assert clf.kernel == "linear"

    def test_clone(self, separable_data):
        X, y = separable_data
        clf = LSVMK(kernel="rbf", nu=0.1, verbose=False).fit(X, y)
        cloned = clone(clf)
        assert cloned.nu == clf.nu
        assert cloned.kernel == clf.kernel
        assert not hasattr(cloned, "dual_coef_")

    def test_score_method(self, separable_data):
        X, y = separable_data
        clf = LSVMK(verbose=False).fit(X, y)
        s = clf.score(X, y)
        assert 0.0 <= s <= 1.0

    def test_predict_before_fit_raises(self):
        clf = LSVMK()
        with pytest.raises(Exception):
            clf.predict(np.array([[1.0, 2.0]]))
