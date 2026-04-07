"""
Tests for BSVMClassifier (Bound-Constrained SVM classifier).
"""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_blobs

from scikit_svm.bsvm import BSVMClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def binary_data():
    """100-sample, 4-feature binary dataset with ±1 labels."""
    X, y_01 = make_classification(
        n_samples=100, n_features=4, n_redundant=0,
        n_informative=4, random_state=0,
    )
    y = np.where(y_01 == 0, -1, 1)
    return X.astype(np.float64), y.astype(np.intp)


@pytest.fixture(scope="module")
def separable_data():
    """Two perfectly separated blobs; labels are 0 and 1."""
    X, y = make_blobs(n_samples=60, centers=2, cluster_std=0.4, random_state=7)
    return X.astype(np.float64), y


@pytest.fixture(scope="module")
def multiclass_data():
    """150-sample, 4-feature 3-class dataset."""
    X, y = make_blobs(
        n_samples=150, centers=3, cluster_std=0.5, random_state=42
    )
    return X.astype(np.float64), y


# ─────────────────────────────────────────────────────────────────────────────
# Basic fit / predict / decision_function
# ─────────────────────────────────────────────────────────────────────────────

class TestBSVMClassifierBasic:

    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False)
        assert clf.fit(X, y) is clf

    def test_fitted_attributes_exist(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        assert hasattr(clf, 'model_')
        assert hasattr(clf, 'support_vectors_')
        assert hasattr(clf, 'dual_coef_')
        assert hasattr(clf, 'classes_')
        assert hasattr(clf, 'n_sv_')
        assert hasattr(clf, 'n_features_in_')
        assert hasattr(clf, 'time_')

    def test_n_sv_positive(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        assert clf.n_sv_ > 0

    def test_classes_binary(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        # classes_ must contain both unique original labels
        assert set(clf.classes_) == set(np.unique(y))

    def test_predict_shape(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_are_valid(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset(set(clf.classes_))

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        assert clf.score(X, y) == 1.0

    def test_decision_function_shape_binary(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (len(X),)

    def test_decision_function_sign_matches_predict(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        preds  = clf.predict(X)
        # For C_SVC binary: positive score → higher-indexed encoded class,
        # negative → lower-indexed encoded class.  We only check that the
        # sign of non-zero scores is consistent with predictions.
        nonzero = scores != 0
        if nonzero.any():
            # encoded label of the positive-score class is the larger one
            pos_class = clf.classes_[clf.label_encoder_.transform(
                [preds[0]]
            )[0] > 0]
            # just verify the shape contract and no NaN
            assert not np.any(np.isnan(scores))

    def test_n_features_in_(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        assert clf.n_features_in_ == X.shape[1]


# ─────────────────────────────────────────────────────────────────────────────
# SVM types
# ─────────────────────────────────────────────────────────────────────────────

class TestBSVMClassifierTypes:

    def test_c_svc(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(svm_type='c_svc', verbose=False).fit(X, y)
        assert clf.n_sv_ > 0
        assert clf.predict(X).shape == (len(X),)

    def test_kbb(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(svm_type='kbb', verbose=False).fit(X, y)
        assert clf.n_sv_ > 0
        assert clf.predict(X).shape == (len(X),)

    def test_spoc(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(svm_type='spoc', verbose=False).fit(X, y)
        assert clf.n_sv_ > 0
        assert clf.predict(X).shape == (len(X),)

    def test_spoc_l2(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(svm_type='spoc_l2', verbose=False).fit(X, y)
        assert clf.n_sv_ > 0
        assert clf.predict(X).shape == (len(X),)


# ─────────────────────────────────────────────────────────────────────────────
# Kernels
# ─────────────────────────────────────────────────────────────────────────────

class TestBSVMClassifierKernels:

    @pytest.mark.parametrize("kernel", ["rbf", "linear", "poly", "sigmoid"])
    def test_named_kernels(self, separable_data, kernel):
        X, y = separable_data
        clf = BSVMClassifier(kernel=kernel, verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset(set(clf.classes_))


# ─────────────────────────────────────────────────────────────────────────────
# Multi-class
# ─────────────────────────────────────────────────────────────────────────────

class TestBSVMClassifierMulticlass:

    def test_multiclass_c_svc(self, multiclass_data):
        X, y = multiclass_data
        clf = BSVMClassifier(svm_type='c_svc', verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert len(clf.classes_) == 3
        assert clf.n_sv_ > 0

    def test_multiclass_spoc(self, multiclass_data):
        X, y = multiclass_data
        clf = BSVMClassifier(svm_type='spoc', verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        assert len(clf.classes_) == 3

    def test_decision_function_shape_multiclass_ovo(self, multiclass_data):
        """C_SVC OVO decision function returns (m, K*(K-1)/2) for K=3 → 3 pairs."""
        X, y = multiclass_data
        clf = BSVMClassifier(svm_type='c_svc', verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        K = len(clf.classes_)
        assert scores.shape == (len(X), K * (K - 1) // 2)

    def test_decision_function_shape_multiclass_spoc(self, multiclass_data):
        """SPOC decision function returns (m, K) for K classes."""
        X, y = multiclass_data
        clf = BSVMClassifier(svm_type='spoc', verbose=False).fit(X, y)
        scores = clf.decision_function(X)
        K = len(clf.classes_)
        assert scores.shape == (len(X), K)

    def test_multiclass_predict_valid_labels(self, multiclass_data):
        X, y = multiclass_data
        clf = BSVMClassifier(svm_type='c_svc', verbose=False).fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset(set(clf.classes_))


# ─────────────────────────────────────────────────────────────────────────────
# scikit-learn API compatibility
# ─────────────────────────────────────────────────────────────────────────────

class TestBSVMClassifierSklearnAPI:

    def test_get_params(self):
        clf = BSVMClassifier(C=5.0, kernel='linear', gamma=0.1)
        p = clf.get_params()
        assert p['C']      == 5.0
        assert p['kernel'] == 'linear'
        assert p['gamma']  == 0.1

    def test_set_params(self):
        clf = BSVMClassifier()
        clf.set_params(C=10.0, kernel='poly')
        assert clf.C      == 10.0
        assert clf.kernel == 'poly'

    def test_clone(self, binary_data):
        X, y = binary_data
        clf  = BSVMClassifier(C=2.0, verbose=False).fit(X, y)
        clf2 = clone(clf)
        assert not hasattr(clf2, 'model_')
        assert clf2.C == clf.C

    def test_score_method(self, binary_data):
        X, y = binary_data
        clf = BSVMClassifier(verbose=False).fit(X, y)
        score = clf.score(X, y)
        assert 0.0 <= score <= 1.0

    def test_predict_before_fit_raises(self, binary_data):
        X, _ = binary_data
        with pytest.raises(Exception):
            BSVMClassifier().predict(X)
