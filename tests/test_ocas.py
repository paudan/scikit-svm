"""
Tests for SVMOCASClassifier and MSVMOCASClassifier.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_classification
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_svm.ocas import SVMOCASClassifier, MSVMOCASClassifier

# ── shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def separable_binary():
    """Linearly separable binary dataset."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 4)
    y = np.sign(X[:, 0] - X[:, 1])
    return X, y


@pytest.fixture
def iris():
    data = load_iris()
    return data.data, data.target


# ── SVMOCASClassifier ────────────────────────────────────────────────────────

class TestSVMOCASClassifier:

    def _clf(self, **kw):
        return SVMOCASClassifier(C=1.0, **kw)

    # ── fit / attributes ──────────────────────────────────────────────────

    def test_fit_returns_self(self, separable_binary):
        X, y = separable_binary
        clf = self._clf()
        assert clf.fit(X, y) is clf

    def test_fitted_attributes(self, separable_binary):
        X, y = separable_binary
        clf = self._clf().fit(X, y)
        assert hasattr(clf, "coef_")
        assert hasattr(clf, "intercept_")
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "n_iter_")
        assert hasattr(clf, "train_time_")
        assert hasattr(clf, "n_features_in_")

    def test_coef_shape(self, separable_binary):
        X, y = separable_binary
        clf = self._clf().fit(X, y)
        assert clf.coef_.shape == (1, X.shape[1])
        assert clf.intercept_.shape == (1,)

    def test_predict_shape(self, separable_binary):
        X, y = separable_binary
        clf = self._clf().fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (len(X),)

    def test_predict_from_classes(self, separable_binary):
        X, y = separable_binary
        clf = self._clf().fit(X, y)
        pred = clf.predict(X)
        assert set(pred).issubset(set(clf.classes_))

    def test_decision_function_shape(self, separable_binary):
        X, y = separable_binary
        clf = self._clf().fit(X, y)
        df = clf.decision_function(X)
        assert df.shape == (len(X),)

    def test_predict_sign_consistent(self, separable_binary):
        X, y = separable_binary
        clf = self._clf().fit(X, y)
        scores = clf.decision_function(X)
        pred = clf.predict(X)
        le_pos = clf._le_pos
        le_neg = clf._le_neg
        for i in range(len(X)):
            if scores[i] > 0:
                assert pred[i] == le_pos
            else:
                assert pred[i] == le_neg

    # ── accuracy ──────────────────────────────────────────────────────────

    def test_high_accuracy_separable(self, separable_binary):
        X, y = separable_binary
        clf = self._clf(tol=1e-4).fit(X, y)
        assert clf.score(X, y) > 0.90

    # ── parameter variants ────────────────────────────────────────────────

    def test_method_cp(self, separable_binary):
        X, y = separable_binary
        clf = SVMOCASClassifier(C=1.0, method="cp").fit(X, y)
        assert clf.score(X, y) > 0.80

    def test_no_intercept(self, separable_binary):
        X, y = separable_binary
        clf = SVMOCASClassifier(C=1.0, fit_intercept=False).fit(X, y)
        assert clf.intercept_[0] == 0.0

    def test_sklearn_labels(self):
        """Accepts 0/1 labels and arbitrary class values."""
        X = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 0.0],
                      [2.0, 0.0], [2.0, 1.0], [-2.0, 0.0]])
        y = np.array([1, 1, 0, 1, 1, 0])
        clf = SVMOCASClassifier(C=0.5).fit(X, y)
        assert set(clf.predict(X)).issubset({0, 1})

    def test_large_C_overfits(self, separable_binary):
        X, y = separable_binary
        clf = SVMOCASClassifier(C=1000.0, tol=1e-5).fit(X, y)
        assert clf.score(X, y) >= 0.95

    def test_convergence(self, separable_binary):
        X, y = separable_binary
        clf = SVMOCASClassifier(C=1.0, tol=1e-4, buf_size=500).fit(X, y)
        stats = clf._stats
        # exitflag 1 (tol met) or 2 (abs met) means convergence
        assert stats["exitflag"] in (1, 2) or clf.score(X, y) > 0.85

    # ── error handling ────────────────────────────────────────────────────

    def test_one_class_raises(self):
        X = np.ones((5, 2))
        y = np.ones(5)
        with pytest.raises(ValueError, match="at least 2"):
            SVMOCASClassifier().fit(X, y)

    def test_wrong_feature_count_raises(self, separable_binary):
        X, y = separable_binary
        clf = SVMOCASClassifier().fit(X, y)
        with pytest.raises(ValueError):
            clf.predict(X[:, :2])

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            SVMOCASClassifier().predict(np.ones((5, 2)))

    def test_multiclass_raises(self):
        X = np.eye(3)
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="binary"):
            SVMOCASClassifier().fit(X, y)

    def test_continuous_y_raises(self):
        X = np.eye(4)
        y = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError):
            SVMOCASClassifier().fit(X, y)

    def test_invalid_C_raises(self):
        with pytest.raises(ValueError, match="C must be"):
            SVMOCASClassifier(C=-1.0).fit(np.ones((4, 2)), np.array([1, -1, 1, -1]))

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            SVMOCASClassifier(method="rbf").fit(np.ones((4, 2)), np.array([1, -1, 1, -1]))

    # ── sklearn interface ─────────────────────────────────────────────────

    def test_get_params(self):
        clf = SVMOCASClassifier(C=2.0, method="cp")
        p = clf.get_params()
        assert p["C"] == 2.0
        assert p["method"] == "cp"

    def test_set_params(self):
        clf = SVMOCASClassifier()
        clf.set_params(C=5.0)
        assert clf.C == 5.0

    def test_verbose(self, separable_binary, capsys):
        X, y = separable_binary
        SVMOCASClassifier(C=1.0, verbose=True).fit(X, y)
        out = capsys.readouterr().out
        assert "SVMOCASClassifier" in out

    def test_random_state_no_effect(self, separable_binary):
        """Binary OCAS is deterministic — no random_state needed."""
        X, y = separable_binary
        clf1 = SVMOCASClassifier(C=1.0).fit(X, y)
        clf2 = SVMOCASClassifier(C=1.0).fit(X, y)
        np.testing.assert_array_equal(clf1.coef_, clf2.coef_)


# ── MSVMOCASClassifier ───────────────────────────────────────────────────────

class TestMSVMOCASClassifier:

    def _clf(self, **kw):
        return MSVMOCASClassifier(C=1.0, **kw)

    # ── fit / attributes ──────────────────────────────────────────────────

    def test_fit_returns_self(self, iris):
        X, y = iris
        clf = self._clf()
        assert clf.fit(X, y) is clf

    def test_fitted_attributes(self, iris):
        X, y = iris
        clf = self._clf().fit(X, y)
        assert hasattr(clf, "coef_")
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "n_iter_")
        assert hasattr(clf, "train_time_")
        assert hasattr(clf, "n_features_in_")

    def test_coef_shape(self, iris):
        X, y = iris
        clf = self._clf().fit(X, y)
        nY = len(np.unique(y))
        assert clf.coef_.shape == (nY, X.shape[1])

    def test_predict_shape(self, iris):
        X, y = iris
        clf = self._clf().fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (len(X),)

    def test_predict_from_classes(self, iris):
        X, y = iris
        clf = self._clf().fit(X, y)
        pred = clf.predict(X)
        assert set(pred).issubset(set(clf.classes_))

    def test_decision_function_shape(self, iris):
        X, y = iris
        clf = self._clf().fit(X, y)
        df = clf.decision_function(X)
        assert df.shape == (len(X), 3)

    # ── accuracy ──────────────────────────────────────────────────────────

    def test_iris_accuracy(self, iris):
        X, y = iris
        clf = self._clf(tol=1e-3).fit(X, y)
        assert clf.score(X, y) > 0.90

    def test_binary_via_msvm(self):
        """MSVMOCASClassifier also handles binary problems."""
        rng = np.random.RandomState(1)
        X = rng.randn(80, 2)
        y = (X[:, 0] > 0).astype(int)
        clf = MSVMOCASClassifier(C=1.0).fit(X, y)
        assert clf.score(X, y) > 0.80
        assert len(clf.classes_) == 2

    def test_method_cp(self, iris):
        X, y = iris
        clf = MSVMOCASClassifier(C=1.0, method="cp").fit(X, y)
        assert clf.score(X, y) > 0.85

    def test_convergence_iris(self, iris):
        X, y = iris
        clf = MSVMOCASClassifier(C=1.0, tol=1e-3).fit(X, y)
        stats = clf._stats
        assert stats["exitflag"] in (1, 2) or clf.score(X, y) > 0.85

    # ── non-integer labels ────────────────────────────────────────────────

    def test_string_labels(self):
        X, _ = load_iris(return_X_y=True)
        y_str = np.repeat(["setosa", "versicolor", "virginica"], 50)
        clf = MSVMOCASClassifier(C=1.0).fit(X, y_str)
        pred = clf.predict(X)
        assert set(pred).issubset({"setosa", "versicolor", "virginica"})
        assert clf.score(X, y_str) > 0.85

    def test_non_zero_based_labels(self):
        X, _ = load_iris(return_X_y=True)
        y_offset = np.repeat([10, 20, 30], 50)
        clf = MSVMOCASClassifier(C=1.0).fit(X, y_offset)
        assert clf.score(X, y_offset) > 0.85

    # ── error handling ────────────────────────────────────────────────────

    def test_one_class_raises(self):
        X = np.ones((5, 2))
        y = np.ones(5)
        with pytest.raises(ValueError, match="at least 2"):
            MSVMOCASClassifier().fit(X, y)

    def test_wrong_feature_count_raises(self, iris):
        X, y = iris
        clf = MSVMOCASClassifier().fit(X, y)
        with pytest.raises(ValueError):
            clf.predict(X[:, :2])

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            MSVMOCASClassifier().predict(np.ones((5, 2)))

    def test_invalid_C_raises(self):
        with pytest.raises(ValueError, match="C must be"):
            MSVMOCASClassifier(C=0).fit(np.ones((4, 2)), np.array([0, 0, 1, 1]))

    def test_continuous_y_raises(self):
        X = np.eye(4)
        y = np.array([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError):
            MSVMOCASClassifier().fit(X, y)

    # ── sklearn interface ─────────────────────────────────────────────────

    def test_get_set_params(self):
        clf = MSVMOCASClassifier(C=3.0, method="cp")
        p = clf.get_params()
        assert p["C"] == 3.0
        clf.set_params(C=4.0)
        assert clf.C == 4.0

    def test_verbose(self, iris, capsys):
        X, y = iris
        MSVMOCASClassifier(C=1.0, verbose=True).fit(X, y)
        out = capsys.readouterr().out
        assert "MSVMOCASClassifier" in out

    def test_deterministic(self, iris):
        X, y = iris
        clf1 = self._clf().fit(X, y)
        clf2 = self._clf().fit(X, y)
        np.testing.assert_array_almost_equal(clf1.coef_, clf2.coef_)


# ── sklearn compatibility checks ─────────────────────────────────────────────

def _make_svm():
    return SVMOCASClassifier(C=0.1)

def _make_msvm():
    return MSVMOCASClassifier(C=0.1)


@parametrize_with_checks([_make_svm(), _make_msvm()])
def test_sklearn_compatible(estimator, check):
    check(estimator)
