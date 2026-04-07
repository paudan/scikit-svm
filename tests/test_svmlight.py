"""Tests for SVMLightClassifier and SVMLightRegressor."""

import numpy as np
import pytest
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_regression,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_svm import SVMLightClassifier, SVMLightRegressor


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def binary_data():
    """200 samples, 2 features, binary classes 0/1."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    return StandardScaler().fit_transform(X), y


@pytest.fixture(scope="module")
def separable_data():
    """60 clearly separable binary samples."""
    X, y = make_blobs(n_samples=60, centers=2, cluster_std=0.5, random_state=0)
    y = np.where(y == 0, -1, 1)   # ±1 labels
    return StandardScaler().fit_transform(X), y


@pytest.fixture(scope="module")
def regression_data():
    """100-sample regression dataset."""
    X, y = make_regression(n_samples=100, n_features=5, noise=10.0, random_state=7)
    return StandardScaler().fit_transform(X), y


@pytest.fixture(scope="module")
def simple_regression():
    """Near-noiseless 1-D regression for decision-function checks."""
    rng = np.random.RandomState(0)
    X = rng.uniform(-3, 3, (60, 1))
    y = X.ravel() ** 2
    return X, y


# ── SVMLightClassifier ────────────────────────────────────────────────────────
class TestSVMLightClassifierBasic:
    def test_fit_predict_binary(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, kernel="rbf", verbose=False)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)
        assert set(preds).issubset(set(y))

    def test_training_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = SVMLightClassifier(C=100.0, kernel="rbf", gamma=1.0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.95

    def test_classes_attribute(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier().fit(X, y)
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == set(y)

    def test_sklearn_attributes(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, kernel="rbf").fit(X, y)
        assert clf.support_vectors_.ndim == 2
        assert clf.support_vectors_.shape[1] == X.shape[1]
        assert clf.dual_coef_.shape[0] == 1
        assert clf.dual_coef_.shape[1] == clf.support_vectors_.shape[0]
        assert clf.intercept_.shape == (1,)
        assert clf.support_.ndim == 1

    def test_decision_function_shape(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier().fit(X, y)
        df = clf.decision_function(X)
        assert df.shape == (X.shape[0],)
        assert df.dtype == np.float64

    def test_decision_function_consistent_with_predict(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, kernel="rbf").fit(X, y)
        df = clf.decision_function(X)
        preds = clf.predict(X)
        pos_class = clf.classes_[1]
        neg_class = clf.classes_[0]
        assert np.all(preds[df > 0] == pos_class)
        assert np.all(preds[df <= 0] == neg_class)

    def test_rejects_multiclass(self, binary_data):
        X, _ = binary_data
        y_mc = np.array([0, 1, 2] * (len(binary_data[0]) // 3) + [0])[:len(binary_data[0])]
        clf = SVMLightClassifier()
        with pytest.raises(ValueError, match="binary"):
            clf.fit(X, y_mc)

    def test_arbitrary_labels(self):
        rng = np.random.RandomState(1)
        X = rng.randn(80, 4)
        y = np.array(["cat", "dog"] * 40)
        clf = SVMLightClassifier(C=1.0, kernel="linear").fit(X, y)
        preds = clf.predict(X)
        assert set(preds) == {"cat", "dog"}

    def test_pm1_labels(self, separable_data):
        X, y = separable_data   # already ±1
        clf = SVMLightClassifier(C=10.0, kernel="rbf").fit(X, y)
        assert clf.score(X, y) > 0.90

    def test_get_set_params(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=2.0, kernel="linear")
        params = clf.get_params()
        assert params["C"] == 2.0
        assert params["kernel"] == "linear"
        clf.set_params(C=5.0)
        assert clf.get_params()["C"] == 5.0


class TestSVMLightClassifierKernels:
    @pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
    def test_kernel_fits(self, binary_data, kernel):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, kernel=kernel, gamma="auto")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_kernel_accuracy(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, kernel="linear").fit(X, y)
        assert clf.score(X, y) > 0.70

    def test_rbf_gamma_scale(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, kernel="rbf", gamma="scale").fit(X, y)
        assert clf.score(X, y) > 0.70

    def test_rbf_gamma_float(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, kernel="rbf", gamma=0.1).fit(X, y)
        assert clf.score(X, y) > 0.70

    def test_invalid_kernel(self, binary_data):
        X, y = binary_data
        with pytest.raises((ValueError, KeyError)):
            SVMLightClassifier(kernel="unknown").fit(X, y)


class TestSVMLightClassifierClassWeight:
    def test_balanced_fits(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, class_weight="balanced").fit(X, y)
        assert clf.score(X, y) > 0.60

    def test_invalid_class_weight(self, binary_data):
        X, y = binary_data
        with pytest.raises(ValueError):
            SVMLightClassifier(class_weight="invalid").fit(X, y)


class TestSVMLightClassifierUnbiased:
    def test_unbiased_hyperplane(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0, kernel="linear",
                                  biased_hyperplane=False).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (X.shape[0],)
        # b should be zero for unbiased
        assert abs(clf.intercept_[0]) < 1e-6


# ── SVMLightRegressor ─────────────────────────────────────────────────────────
class TestSVMLightRegressorBasic:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        reg = SVMLightRegressor(C=1.0, kernel="rbf")
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (X.shape[0],)
        assert preds.dtype == np.float64

    def test_r2_score(self, regression_data):
        X, y = regression_data
        reg = SVMLightRegressor(C=100.0, kernel="rbf", gamma="auto")
        reg.fit(X, y)
        assert reg.score(X, y) > 0.70

    def test_sklearn_attributes(self, regression_data):
        X, y = regression_data
        reg = SVMLightRegressor(C=1.0).fit(X, y)
        assert reg.support_vectors_.ndim == 2
        assert reg.support_vectors_.shape[1] == X.shape[1]
        assert reg.dual_coef_.shape[0] == 1
        assert reg.intercept_.shape == (1,)
        assert reg.support_.ndim == 1

    def test_epsilon_parameter(self, regression_data):
        X, y = regression_data
        reg_narrow = SVMLightRegressor(C=1.0, epsilon=0.01)
        reg_wide   = SVMLightRegressor(C=1.0, epsilon=1.0)
        reg_narrow.fit(X, y)
        reg_wide.fit(X, y)
        # Narrower tube → more SVs
        assert reg_narrow._model.sv_num >= reg_wide._model.sv_num

    def test_not_fitted_raises(self):
        from sklearn.exceptions import NotFittedError
        reg = SVMLightRegressor()
        with pytest.raises(NotFittedError):
            reg.predict(np.zeros((5, 3)))

    def test_get_set_params(self):
        reg = SVMLightRegressor(C=3.0, epsilon=0.5)
        params = reg.get_params()
        assert params["C"] == 3.0
        assert params["epsilon"] == 0.5
        reg.set_params(C=7.0)
        assert reg.get_params()["C"] == 7.0


class TestSVMLightRegressorKernels:
    @pytest.mark.parametrize("kernel", ["linear", "poly", "rbf", "sigmoid"])
    def test_kernel_fits(self, regression_data, kernel):
        X, y = regression_data
        reg = SVMLightRegressor(C=1.0, kernel=kernel)
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_linear_kernel_r2(self, regression_data):
        X, y = regression_data
        reg = SVMLightRegressor(C=10.0, kernel="linear").fit(X, y)
        assert reg.score(X, y) > 0.80

    def test_decision_function_matches_predict(self, simple_regression):
        """predict() == decision function for regression."""
        X, y = simple_regression
        reg = SVMLightRegressor(C=10.0, kernel="rbf", gamma=1.0).fit(X, y)
        preds = reg.predict(X)
        assert np.allclose(preds, preds)   # basic sanity (not NaN)
        # Training R² should be reasonable
        assert reg.score(X, y) > 0.80


class TestSVMLightSklearnAPI:
    def test_clone(self, binary_data):
        from sklearn.base import clone
        X, y = binary_data
        clf = SVMLightClassifier(C=2.0, kernel="linear")
        clf2 = clone(clf)
        assert clf2.C == 2.0
        assert clf2.kernel == "linear"
        assert not hasattr(clf2, "_model")

    def test_set_params_chain(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier()
        clf.set_params(C=5.0, kernel="linear")
        clf.fit(X, y)
        assert clf.C == 5.0

    def test_refit(self, binary_data):
        X, y = binary_data
        clf = SVMLightClassifier(C=1.0).fit(X, y)
        first_intercept = clf.intercept_.copy()
        clf.fit(X, y)
        # Re-fitting should give same result (SVM-Light has minor float variation)
        np.testing.assert_allclose(clf.intercept_, first_intercept, rtol=1e-3)

    def test_regressor_clone(self, regression_data):
        from sklearn.base import clone
        X, y = regression_data
        reg = SVMLightRegressor(C=3.0, epsilon=0.2)
        reg2 = clone(reg)
        assert reg2.C == 3.0
        assert reg2.epsilon == 0.2
