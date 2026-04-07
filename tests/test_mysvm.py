"""Tests for MySVMClassifier, MySVMRegressor,
MySVMNuClassifier and MySVMNuRegressor."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from scikit_svm import (MySVMClassifier, MySVMRegressor,
                        MySVMNuClassifier, MySVMNuRegressor)


# ── fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def binary_data():
    """200 samples, 10 features, binary labels 0/1."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    return StandardScaler().fit_transform(X), y


@pytest.fixture(scope="module")
def separable_data():
    """60 clearly separable binary samples (±1 labels)."""
    X, y = make_blobs(n_samples=60, centers=2, cluster_std=0.5, random_state=0)
    y = np.where(y == 0, -1, 1)
    return StandardScaler().fit_transform(X), y


@pytest.fixture(scope="module")
def regression_data():
    """100-sample regression dataset."""
    X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=7)
    return StandardScaler().fit_transform(X), y


# ── MySVMClassifier ────────────────────────────────────────────────────────

class TestMySVMClassifierBasic:
    def test_fit_predict_binary(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, kernel="rbf", verbose=False)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(preds).issubset(set(y))

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = MySVMClassifier(C=100.0, kernel="rbf", gamma=1.0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.95

    def test_classes_attribute(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier().fit(X, y)
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == set(y)

    def test_decision_function_shape(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier().fit(X, y)
        df = clf.decision_function(X)
        assert df.shape == (X.shape[0],)
        assert df.dtype == np.float64

    def test_decision_function_consistent_with_predict(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, kernel="rbf").fit(X, y)
        df = clf.decision_function(X)
        preds = clf.predict(X)
        pos_class = clf.classes_[1]
        neg_class = clf.classes_[0]
        assert np.all(preds[df > 0] == pos_class)
        assert np.all(preds[df <= 0] == neg_class)

    def test_rejects_multiclass(self, binary_data):
        X, _ = binary_data
        y_mc = np.array([0, 1, 2] * (len(binary_data[0]) // 3) + [0])[:len(binary_data[0])]
        clf = MySVMClassifier()
        with pytest.raises(ValueError, match="binary"):
            clf.fit(X, y_mc)

    def test_arbitrary_labels(self):
        rng = np.random.RandomState(1)
        X = rng.randn(80, 4)
        y = np.array(["cat", "dog"] * 40)
        clf = MySVMClassifier(C=1.0, kernel="linear").fit(X, y)
        preds = clf.predict(X)
        assert set(preds) == {"cat", "dog"}

    def test_pm1_labels(self, separable_data):
        X, y = separable_data   # already ±1
        clf = MySVMClassifier(C=10.0, kernel="rbf").fit(X, y)
        assert clf.score(X, y) > 0.90

    def test_get_set_params(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=2.0, kernel="linear")
        params = clf.get_params()
        assert params["C"] == 2.0
        assert params["kernel"] == "linear"
        clf.set_params(C=5.0)
        assert clf.get_params()["C"] == 5.0


class TestMySVMClassifierKernels:
    @pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
    def test_kernel_fits(self, binary_data, kernel):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, kernel=kernel, gamma="auto")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_kernel_accuracy(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, kernel="linear").fit(X, y)
        assert clf.score(X, y) > 0.70

    def test_rbf_gamma_scale(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, kernel="rbf", gamma="scale").fit(X, y)
        assert clf.score(X, y) > 0.70

    def test_rbf_gamma_float(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, kernel="rbf", gamma=0.1).fit(X, y)
        assert clf.score(X, y) > 0.70

    def test_invalid_kernel(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError):
            MySVMClassifier(kernel="unknown").fit(X, y)


class TestMySVMClassifierClassWeight:
    def test_balanced_fits(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, class_weight="balanced").fit(X, y)
        assert clf.score(X, y) > 0.60

    def test_invalid_class_weight(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError):
            MySVMClassifier(class_weight="invalid").fit(X, y)


class TestMySVMClassifierUnbiased:
    def test_unbiased_hyperplane(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, kernel="linear",
                              biased=False).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)


# ── MySVMRegressor ─────────────────────────────────────────────────────────

class TestMySVMRegressorBasic:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        reg = MySVMRegressor(C=1.0, kernel="rbf")
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (X.shape[0],)
        assert preds.dtype == np.float64

    def test_r2_score(self, regression_data):
        X, y = regression_data
        reg = MySVMRegressor(C=100.0, kernel="rbf", gamma="auto")
        reg.fit(X, y)
        assert reg.score(X, y) > 0.70

    def test_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError
        reg = MySVMRegressor()
        with pytest.raises(NotFittedError):
            reg.predict(np.zeros((5, 3)))

    def test_get_set_params(self):
        reg = MySVMRegressor(C=3.0, epsilon=0.5)
        params = reg.get_params()
        assert params["C"] == 3.0
        assert params["epsilon"] == 0.5
        reg.set_params(C=7.0)
        assert reg.get_params()["C"] == 7.0


class TestMySVMRegressorKernels:
    @pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
    def test_kernel_fits(self, regression_data, kernel):
        X, y = regression_data
        reg = MySVMRegressor(C=1.0, kernel=kernel)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_kernel_r2(self, regression_data):
        X, y = regression_data
        reg = MySVMRegressor(C=10.0, kernel="linear").fit(X, y)
        assert reg.score(X, y) > 0.80


# ── MySVMNuClassifier ──────────────────────────────────────────────────────

class TestMySVMNuClassifierBasic:
    def test_fit_predict_binary(self, binary_data):
        X, y = binary_data
        clf = MySVMNuClassifier(nu=0.5, kernel="rbf", verbose=False)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(preds).issubset(set(y))

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = MySVMNuClassifier(nu=0.1, kernel="rbf", gamma=1.0)
        clf.fit(X, y)
        assert clf.score(X, y) >= 0.5

    def test_classes_attribute(self, binary_data):
        X, y = binary_data
        clf = MySVMNuClassifier().fit(X, y)
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == set(y)

    def test_decision_function(self, binary_data):
        X, y = binary_data
        clf = MySVMNuClassifier().fit(X, y)
        df = clf.decision_function(X)
        assert df.shape == (X.shape[0],)
        assert df.dtype == np.float64

    def test_rejects_multiclass(self, binary_data):
        X, _ = binary_data
        y_mc = np.array([0, 1, 2] * (len(binary_data[0]) // 3) + [0])[:len(binary_data[0])]
        clf = MySVMNuClassifier()
        with pytest.raises(ValueError, match="binary"):
            clf.fit(X, y_mc)

    def test_get_set_params(self):
        clf = MySVMNuClassifier(nu=0.3, kernel="linear")
        assert clf.get_params()["nu"] == 0.3
        clf.set_params(nu=0.2)
        assert clf.get_params()["nu"] == 0.2


# ── MySVMNuRegressor ───────────────────────────────────────────────────────

class TestMySVMNuRegressorBasic:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        reg = MySVMNuRegressor(nu=0.5, kernel="rbf")
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (X.shape[0],)
        assert preds.dtype == np.float64

    def test_r2_score(self, regression_data):
        X, y = regression_data
        reg = MySVMNuRegressor(nu=0.3, kernel="rbf", gamma="auto")
        reg.fit(X, y)
        assert reg.score(X, y) > 0.50

    def test_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError
        reg = MySVMNuRegressor()
        with pytest.raises(NotFittedError):
            reg.predict(np.zeros((5, 3)))

    def test_get_set_params(self):
        reg = MySVMNuRegressor(nu=0.3)
        assert reg.get_params()["nu"] == 0.3
        reg.set_params(nu=0.4)
        assert reg.get_params()["nu"] == 0.4


# ── sklearn API ────────────────────────────────────────────────────────────

class TestMySVMSklearnAPI:
    def test_clone_classifier(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=2.0, kernel="linear")
        clf2 = clone(clf)
        assert clf2.C == 2.0
        assert clf2.kernel == "linear"
        assert not hasattr(clf2, "_model")

    def test_clone_regressor(self, regression_data):
        X, y = regression_data
        reg = MySVMRegressor(C=3.0, epsilon=0.2)
        reg2 = clone(reg)
        assert reg2.C == 3.0
        assert reg2.epsilon == 0.2

    def test_set_params_chain_classifier(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier()
        clf.set_params(C=5.0, kernel="linear")
        clf.fit(X, y)
        assert clf.C == 5.0

    def test_refit_classifier(self, binary_data):
        X, y = binary_data
        clf = MySVMClassifier(C=1.0, kernel="linear").fit(X, y)
        score1 = clf.score(X, y)
        clf.fit(X, y)
        score2 = clf.score(X, y)
        # Refitting should give consistent results
        assert abs(score1 - score2) < 0.05

    def test_nu_clf_clone(self):
        clf = MySVMNuClassifier(nu=0.3, kernel="rbf")
        clf2 = clone(clf)
        assert clf2.nu == 0.3

    def test_nu_reg_clone(self):
        reg = MySVMNuRegressor(nu=0.4, kernel="linear")
        reg2 = clone(reg)
        assert reg2.nu == 0.4
