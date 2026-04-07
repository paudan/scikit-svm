"""
Tests for the BVM (Ball Vector Machine) classifier.
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs, make_classification

from scikit_svm import BVM


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def binary_data():
    X, y_01 = make_classification(
        n_samples=100, n_features=4, n_redundant=0,
        n_informative=4, random_state=0,
    )
    y = np.where(y_01 == 0, -1.0, 1.0)
    return X.astype(np.float64), y


@pytest.fixture(scope="module")
def separable_data():
    X, y_01 = make_blobs(n_samples=60, centers=2, cluster_std=0.4, random_state=7)
    y = np.where(y_01 == 0, -1.0, 1.0)
    return X.astype(np.float64), y


# ─────────────────────────────────────────────────────────────────────────────
# Basic fit / predict / decision_function
# ─────────────────────────────────────────────────────────────────────────────

class TestBVMBasic:

    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False)
        assert clf.fit(X, y) is clf

    def test_fitted_attributes_exist(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        assert hasattr(clf, 'model_')
        assert hasattr(clf, 'n_sv_')
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'n_features_in_')
        assert hasattr(clf, 'time_')

    def test_n_sv_positive(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        assert clf.n_sv_ > 0

    def test_classes(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, np.array([-1.0, 1.0]))

    def test_predict_shape(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_are_pm1(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({-1.0, 1.0})

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = BVM(verbose=False).fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_decision_function_shape(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(X),)

    def test_decision_function_sign_matches_predict(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        preds  = clf.predict(X)
        np.testing.assert_array_equal(
            np.sign(scores[scores != 0]),
            preds[scores != 0],
        )

    def test_n_features_in_(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        assert clf.n_features_in_ == X.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# Label validation
# ─────────────────────────────────────────────────────────────────────────────

class TestBVMLabelValidation:

    def test_invalid_labels_raise(self, binary_data):
        X, _ = binary_data
        y_bad = np.zeros(len(X))
        with pytest.raises(ValueError, match="[Ll]abels"):
            BVM(verbose=False).fit(X, y_bad)


# ─────────────────────────────────────────────────────────────────────────────
# Kernels – BVM requires isotropic kernels
# ─────────────────────────────────────────────────────────────────────────────

class TestBVMKernels:

    @pytest.mark.parametrize("kernel", ["rbf", "exp", "inv_sqdist"])
    def test_isotropic_kernels(self, separable_data, kernel):
        X, y = separable_data
        clf = BVM(kernel=kernel, verbose=False).fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_non_isotropic_kernel_raises(self, binary_data):
        """linear, poly, sigmoid are NOT supported by BVM."""
        X, y = binary_data
        with pytest.raises(ValueError, match="[Ii]sotropic"):
            BVM(kernel='linear', verbose=False).fit(X, y)

    def test_poly_kernel_raises(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError, match="[Ii]sotropic"):
            BVM(kernel='poly', verbose=False).fit(X, y)

    def test_gamma_parameter(self, separable_data):
        X, y = separable_data
        clf = BVM(kernel='rbf', gamma=0.5, verbose=False).fit(X, y)
        assert clf.score(X, y) > 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

class TestBVMParameters:

    def test_explicit_C(self, separable_data):
        X, y = separable_data
        clf = BVM(C=10.0, verbose=False).fit(X, y)
        assert clf.score(X, y) > 0.9

    def test_time_non_negative(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        assert clf.time_ >= 0.0

    def test_n_sv_capped_by_max_sv(self, binary_data):
        X, y = binary_data
        clf = BVM(max_sv=5, verbose=False).fit(X, y)
        assert clf.n_sv_ <= 5

    def test_sample_size_parameter(self, separable_data):
        X, y = separable_data
        clf = BVM(sample_size=30, verbose=False).fit(X, y)
        assert clf.score(X, y) > 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Verbose flag
# ─────────────────────────────────────────────────────────────────────────────

class TestBVMVerbose:

    def test_verbose_false_silent(self, binary_data, capsys):
        X, y = binary_data
        BVM(verbose=False).fit(X, y)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true_runs(self, binary_data):
        X, y = binary_data
        BVM(verbose=True).fit(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# scikit-learn API compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestBVMSklearnAPI:

    def test_get_params(self):
        clf = BVM(C=50.0, kernel='rbf', gamma=0.1, sample_size=30)
        p = clf.get_params()
        assert p['C']           == 50.0
        assert p['kernel']      == 'rbf'
        assert p['gamma']       == 0.1
        assert p['sample_size'] == 30

    def test_set_params(self):
        clf = BVM()
        clf.set_params(C=200.0, sample_size=120)
        assert clf.C           == 200.0
        assert clf.sample_size == 120

    def test_clone(self, binary_data):
        X, y = binary_data
        clf  = BVM(C=10.0, verbose=False).fit(X, y)
        clf2 = clone(clf)
        assert not hasattr(clf2, 'model_')
        assert clf2.C == clf.C

    def test_score_method(self, binary_data):
        X, y = binary_data
        clf = BVM(verbose=False).fit(X, y)
        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_predict_before_fit_raises(self, binary_data):
        X, _ = binary_data
        with pytest.raises(Exception):
            BVM().predict(X)


# ─────────────────────────────────────────────────────────────────────────────
# Numerical consistency: BVM should give results close to CVM
# ─────────────────────────────────────────────────────────────────────────────

class TestBVMvsCVM:

    def test_similar_accuracy_to_cvm(self, separable_data):
        """BVM and CVM should agree on a clean separable dataset."""
        from scikit_svm import CVM
        X, y = separable_data
        bvm = BVM(kernel='rbf', C=100.0, gamma=0.5, verbose=False).fit(X, y)
        cvm = CVM(kernel='rbf', C=100.0, gamma=0.5, verbose=False).fit(X, y)
        # Both should achieve high accuracy
        assert bvm.score(X, y) >= 0.95
        assert cvm.score(X, y) >= 0.95
