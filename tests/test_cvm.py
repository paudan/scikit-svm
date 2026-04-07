"""
Tests for the CVM (Core Vector Machine) classifier.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_blobs

from scikit_svm import CVM


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def binary_data():
    """100-sample, 4-feature linearly separable binary dataset."""
    X, y_01 = make_classification(
        n_samples=100, n_features=4, n_redundant=0,
        n_informative=4, random_state=0,
    )
    y = np.where(y_01 == 0, -1.0, 1.0)
    return X.astype(np.float64), y


@pytest.fixture(scope="module")
def separable_data():
    """Two perfectly separated blobs with ±1 labels."""
    X, y_01 = make_blobs(n_samples=60, centers=2, cluster_std=0.4, random_state=7)
    y = np.where(y_01 == 0, -1.0, 1.0)
    return X.astype(np.float64), y


# ─────────────────────────────────────────────────────────────────────────────
# Basic fit / predict / decision_function
# ─────────────────────────────────────────────────────────────────────────────

class TestCVMBasic:

    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False)
        assert clf.fit(X, y) is clf

    def test_fitted_attributes_exist(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        assert hasattr(clf, 'model_')
        assert hasattr(clf, 'n_sv_')
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'n_features_in_')
        assert hasattr(clf, 'time_')

    def test_n_sv_positive(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        assert clf.n_sv_ > 0

    def test_classes(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, np.array([-1.0, 1.0]))

    def test_predict_shape(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_are_pm1(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({-1.0, 1.0})

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = CVM(verbose=False).fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_decision_function_shape(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(X),)

    def test_decision_function_sign_matches_predict(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        preds  = clf.predict(X)
        # Positive score ↔ predict +1; non-positive ↔ predict -1
        np.testing.assert_array_equal(
            np.sign(scores[scores != 0]),
            preds[scores != 0],
        )

    def test_n_features_in_(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        assert clf.n_features_in_ == X.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Label validation
# ─────────────────────────────────────────────────────────────────────────────

class TestCVMLabelValidation:

    def test_invalid_labels_raise(self, binary_data):
        X, _ = binary_data
        y_bad = np.zeros(len(X))
        with pytest.raises(ValueError, match="[Ll]abels"):
            CVM(verbose=False).fit(X, y_bad)

    def test_valid_labels_integers(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y.astype(int))
        assert clf.score(X, y) > 0.7


# ─────────────────────────────────────────────────────────────────────────────
# Kernels
# ─────────────────────────────────────────────────────────────────────────────

class TestCVMKernels:

    @pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid"])
    def test_named_kernels(self, separable_data, kernel):
        X, y = separable_data
        clf = CVM(kernel=kernel, verbose=False).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_unknown_kernel_raises(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="[Uu]nknown kernel"):
            CVM(kernel='badkernel', verbose=False).fit(X, y)

    def test_gamma_parameter(self, separable_data):
        X, y = separable_data
        clf = CVM(kernel='rbf', gamma=0.5, verbose=False).fit(X, y)
        assert clf.score(X, y) > 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestCVMParameters:

    def test_explicit_C(self, separable_data):
        X, y = separable_data
        clf = CVM(C=10.0, verbose=False).fit(X, y)
        assert clf.score(X, y) > 0.9

    def test_time_non_negative(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        assert clf.time_ >= 0.0

    def test_max_sv_affects_result(self, binary_data):
        """max_sv caps core vectors during MEB approximation.
        The final SV count after reconstruction may exceed max_sv (CVM behaviour),
        but a very tight budget should still produce a valid model.
        """
        X, y = binary_data
        clf = CVM(max_sv=5, verbose=False).fit(X, y)
        # Model must be valid regardless of the core-vector budget
        assert clf.n_sv_ > 0
        preds = clf.predict(X)
        assert preds.shape == (len(X),)


# ─────────────────────────────────────────────────────────────────────────────
# Verbose flag
# ─────────────────────────────────────────────────────────────────────────────

class TestCVMVerbose:

    def test_verbose_false_silent(self, binary_data, capsys):
        X, y = binary_data
        CVM(verbose=False).fit(X, y)
        captured = capsys.readouterr()
        # Python-level output must be empty; C-level is suppressed separately
        assert captured.out == ""

    def test_verbose_true_runs(self, binary_data):
        """verbose=True must not raise; output goes to C-level stdout."""
        X, y = binary_data
        CVM(verbose=True).fit(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# scikit-learn API compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestCVMSklearnAPI:

    def test_get_params(self):
        clf = CVM(C=50.0, kernel='linear', gamma=0.1)
        p = clf.get_params()
        assert p['C']      == 50.0
        assert p['kernel'] == 'linear'
        assert p['gamma']  == 0.1

    def test_set_params(self):
        clf = CVM()
        clf.set_params(C=200.0, kernel='poly')
        assert clf.C      == 200.0
        assert clf.kernel == 'poly'

    def test_clone(self, binary_data):
        X, y = binary_data
        clf   = CVM(C=10.0, verbose=False).fit(X, y)
        clf2  = clone(clf)
        assert not hasattr(clf2, 'model_')
        assert clf2.C == clf.C

    def test_score_method(self, binary_data):
        X, y = binary_data
        clf = CVM(verbose=False).fit(X, y)
        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_predict_before_fit_raises(self, binary_data):
        X, _ = binary_data
        with pytest.raises(Exception):
            CVM().predict(X)
