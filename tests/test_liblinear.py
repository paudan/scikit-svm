"""
Tests for LibLinearSVC and LibLinearSVR.

Covers:
 - Fit/predict/decision_function shapes and types
 - Attribute presence and shapes after fit
 - Accuracy on separable and overlapping data
 - All supported solver types
 - Multi-class classification
 - fit_intercept=False
 - class_weight (balanced and dict)
 - Parameter validation errors
 - Not-fitted / wrong-shape errors
 - sklearn estimator compatibility (parametrize_with_checks)
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_svm import LibLinearSVC, LibLinearSVR


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def separable_binary():
    rng = np.random.RandomState(0)
    X = np.vstack([rng.randn(60, 4) + 3, rng.randn(60, 4) - 3])
    y = np.array([0] * 60 + [1] * 60)
    return X, y


@pytest.fixture
def binary_blobs():
    X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.0,
                      random_state=42)
    return X, y


@pytest.fixture
def multiclass_data():
    X, y = make_blobs(n_samples=150, centers=3, cluster_std=1.0,
                      random_state=7)
    return X, y


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1,
                            random_state=0)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVC — fit / attributes
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVCFit:

    def test_fit_returns_self(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC()
        assert clf.fit(X, y) is clf

    def test_fitted_attributes_binary(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        assert hasattr(clf, "coef_")
        assert hasattr(clf, "intercept_")
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "train_time_")
        assert hasattr(clf, "n_features_in_")

    def test_coef_shape_binary(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        assert clf.coef_.shape == (1, X.shape[1])
        assert clf.intercept_.shape == (1,)

    def test_coef_shape_multiclass(self, multiclass_data):
        X, y = multiclass_data
        clf = LibLinearSVC().fit(X, y)
        n_classes = len(np.unique(y))
        assert clf.coef_.shape == (n_classes, X.shape[1])
        assert clf.intercept_.shape == (n_classes,)

    def test_classes_binary(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        assert clf.classes_.shape == (2,)
        assert set(clf.classes_) == set(np.unique(y))

    def test_classes_multiclass(self, multiclass_data):
        X, y = multiclass_data
        clf = LibLinearSVC().fit(X, y)
        assert len(clf.classes_) == 3
        assert set(clf.classes_) == set(np.unique(y))

    def test_n_features_in(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        assert clf.n_features_in_ == X.shape[1]

    def test_train_time_positive(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        assert clf.train_time_ >= 0.0

    def test_no_intercept_binary(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC(fit_intercept=False).fit(X, y)
        assert clf.intercept_[0] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVC — predict / decision_function shapes and types
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVCPredict:

    def test_predict_shape(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_predict_from_classes(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset(set(clf.classes_))

    def test_decision_function_shape_binary(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        scores = clf.decision_function(X)
        assert scores.ndim == 1
        assert scores.shape == (len(X),)

    def test_decision_function_shape_multiclass(self, multiclass_data):
        X, y = multiclass_data
        clf = LibLinearSVC().fit(X, y)
        scores = clf.decision_function(X)
        assert scores.ndim == 2
        assert scores.shape == (len(X), len(clf.classes_))

    def test_decision_sign_consistent_binary(self, binary_blobs):
        """Positive decision score → predicts classes_[1]."""
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        scores = clf.decision_function(X)
        preds = clf.predict(X)
        assert np.all(preds[scores > 0] == clf.classes_[1])
        assert np.all(preds[scores <= 0] == clf.classes_[0])

    def test_predict_multiclass_argmax_consistent(self, multiclass_data):
        """predict(X) == classes_[argmax(decision_function(X), axis=1)]."""
        X, y = multiclass_data
        clf = LibLinearSVC().fit(X, y)
        scores = clf.decision_function(X)
        expected = clf.classes_[np.argmax(scores, axis=1)]
        assert np.array_equal(clf.predict(X), expected)

    def test_string_labels(self):
        """Classifier should accept non-numeric labels."""
        rng = np.random.RandomState(1)
        X = rng.randn(80, 3)
        y = np.array(["cat"] * 40 + ["dog"] * 40)
        clf = LibLinearSVC().fit(X, y)
        preds = clf.predict(X)
        assert set(preds) <= {"cat", "dog"}


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVC — accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVCAccuracy:

    def test_high_accuracy_separable(self, separable_binary):
        X, y = separable_binary
        clf = LibLinearSVC(C=10.0).fit(X, y)
        assert clf.score(X, y) > 0.95

    def test_high_accuracy_multiclass(self, multiclass_data):
        X, y = multiclass_data
        clf = LibLinearSVC(C=1.0).fit(X, y)
        assert clf.score(X, y) > 0.90

    def test_larger_C_fits_tighter(self, binary_blobs):
        X, y = binary_blobs
        clf_small = LibLinearSVC(C=0.001).fit(X, y)
        clf_large = LibLinearSVC(C=100.0).fit(X, y)
        assert clf_large.score(X, y) >= clf_small.score(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVC — solver variants
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVCSolvers:

    @pytest.mark.parametrize("solver", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_all_solvers_fit_predict(self, solver, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC(solver=solver, C=1.0).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    @pytest.mark.parametrize("solver", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_all_solvers_multiclass(self, solver, multiclass_data):
        X, y = multiclass_data
        clf = LibLinearSVC(solver=solver, C=1.0).fit(X, y)
        assert clf.predict(X).shape == (len(X),)


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVC — class_weight
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVCClassWeight:

    def test_balanced_weight_fits(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC(class_weight="balanced").fit(X, y)
        assert clf.coef_.shape == (1, X.shape[1])

    def test_dict_weight_fits(self, binary_blobs):
        X, y = binary_blobs
        classes = np.unique(y)
        cw = {classes[0]: 1.0, classes[1]: 2.0}
        clf = LibLinearSVC(class_weight=cw).fit(X, y)
        assert clf.coef_.shape == (1, X.shape[1])

    def test_invalid_class_weight_raises(self, binary_blobs):
        X, y = binary_blobs
        with pytest.raises(ValueError):
            LibLinearSVC(class_weight="invalid").fit(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVC — sklearn interface
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVCSklearnInterface:

    def test_get_params(self):
        clf = LibLinearSVC(solver=2, C=0.5, tol=1e-3)
        params = clf.get_params()
        assert params["solver"] == 2
        assert params["C"] == 0.5
        assert params["tol"] == 1e-3

    def test_set_params(self):
        clf = LibLinearSVC()
        clf.set_params(C=5.0, solver=3)
        assert clf.C == 5.0
        assert clf.solver == 3

    def test_verbose_output(self, binary_blobs, capfd):
        X, y = binary_blobs
        LibLinearSVC(verbose=True).fit(X, y)
        out = capfd.readouterr().out
        # liblinear prints to C-level stdout when verbose=True
        assert len(out) > 0


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVC — error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVCErrors:

    def test_invalid_solver_raises(self, binary_blobs):
        X, y = binary_blobs
        with pytest.raises(ValueError, match="solver must be"):
            LibLinearSVC(solver=99).fit(X, y)

    def test_negative_C_raises(self, binary_blobs):
        X, y = binary_blobs
        with pytest.raises(ValueError, match="C must be"):
            LibLinearSVC(C=-1.0).fit(X, y)

    def test_zero_tol_raises(self, binary_blobs):
        X, y = binary_blobs
        with pytest.raises(ValueError, match="tol must be"):
            LibLinearSVC(tol=0.0).fit(X, y)

    def test_one_class_raises(self):
        X = np.ones((10, 2))
        y = np.zeros(10)
        with pytest.raises(ValueError):
            LibLinearSVC().fit(X, y)

    def test_predict_before_fit_raises(self, binary_blobs):
        X, _ = binary_blobs
        with pytest.raises(Exception):
            LibLinearSVC().predict(X)

    def test_wrong_feature_count_raises(self, binary_blobs):
        X, y = binary_blobs
        clf = LibLinearSVC().fit(X, y)
        X_bad = np.ones((5, X.shape[1] + 3))
        with pytest.raises(ValueError):
            clf.predict(X_bad)


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVR — fit / attributes
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVRFit:

    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        assert LibLinearSVR().fit(X, y) is LibLinearSVR().fit(X, y) or True
        reg = LibLinearSVR()
        assert reg.fit(X, y) is reg

    def test_fitted_attributes(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR().fit(X, y)
        assert hasattr(reg, "coef_")
        assert hasattr(reg, "intercept_")
        assert hasattr(reg, "train_time_")
        assert hasattr(reg, "n_features_in_")

    def test_coef_shape(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR().fit(X, y)
        assert reg.coef_.shape == (1, X.shape[1])
        assert reg.intercept_.shape == (1,)

    def test_n_features_in(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR().fit(X, y)
        assert reg.n_features_in_ == X.shape[1]

    def test_no_intercept(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR(fit_intercept=False).fit(X, y)
        assert reg.intercept_[0] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVR — predict
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVRPredict:

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR().fit(X, y)
        assert reg.predict(X).shape == (len(X),)

    def test_predict_continuous(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR().fit(X, y)
        preds = reg.predict(X)
        assert preds.dtype.kind == "f"

    def test_reasonable_r2(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR(C=1.0).fit(X, y)
        assert reg.score(X, y) > 0.5

    def test_predict_equals_coef_dot(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR().fit(X, y)
        expected = X @ reg.coef_[0] + reg.intercept_[0]
        assert np.allclose(reg.predict(X), expected)


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVR — solver variants
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVRSolvers:

    @pytest.mark.parametrize("solver", [11, 12, 13])
    def test_all_solvers_fit_predict(self, solver, regression_data):
        X, y = regression_data
        reg = LibLinearSVR(solver=solver).fit(X, y)
        assert reg.predict(X).shape == (len(X),)


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVR — error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVRErrors:

    def test_invalid_solver_raises(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="solver must be"):
            LibLinearSVR(solver=1).fit(X, y)

    def test_negative_C_raises(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="C must be"):
            LibLinearSVR(C=-1.0).fit(X, y)

    def test_negative_p_raises(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="p must be"):
            LibLinearSVR(p=-0.1).fit(X, y)

    def test_predict_before_fit_raises(self, regression_data):
        X, _ = regression_data
        with pytest.raises(Exception):
            LibLinearSVR().predict(X)

    def test_wrong_feature_count_raises(self, regression_data):
        X, y = regression_data
        reg = LibLinearSVR().fit(X, y)
        X_bad = np.ones((5, X.shape[1] + 2))
        with pytest.raises(ValueError):
            reg.predict(X_bad)


# ─────────────────────────────────────────────────────────────────────────────
# LibLinearSVR — sklearn interface
# ─────────────────────────────────────────────────────────────────────────────

class TestLibLinearSVRSklearnInterface:

    def test_get_params(self):
        reg = LibLinearSVR(solver=12, C=2.0, p=0.2)
        params = reg.get_params()
        assert params["solver"] == 12
        assert params["C"] == 2.0
        assert params["p"] == 0.2

    def test_set_params(self):
        reg = LibLinearSVR()
        reg.set_params(C=3.0, solver=13)
        assert reg.C == 3.0
        assert reg.solver == 13


# ─────────────────────────────────────────────────────────────────────────────
# sklearn estimator compatibility
# ─────────────────────────────────────────────────────────────────────────────

def _make_svc():
    return LibLinearSVC(solver=1, C=1.0)

def _make_svr():
    return LibLinearSVR(solver=11, C=1.0)


@parametrize_with_checks([_make_svc(), _make_svr()])
def test_sklearn_compatible(estimator, check):
    check(estimator)
