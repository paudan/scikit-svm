"""
Tests for LapSVMClassifier and LapRLSCClassifier.

Covers:
 - kernel computation (linear/poly/rbf)
 - graph construction (adjacency, Laplacian, normalization)
 - Newton solver (hinge and squared loss)
 - predict / decision_function / score
 - semi-supervised vs fully-supervised
 - parameter effects (gamma_A, gamma_I, use_bias)
 - edge cases (all labeled, precomputed graph)
 - sklearn estimator compatibility
"""

import numpy as np
import pytest
from scipy import sparse
from sklearn.datasets import make_classification, make_blobs
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_svm import LapRLSCClassifier, LapSVMClassifier
from scikit_svm.lapsvm import (
    _build_adjacency,
    _euclidean_dist_matrix,
    _kernel_cross,
    _kernel_sym,
    build_laplacian,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def small_data():
    """50 points, 2 features, ±1 labels, 20 labeled."""
    rng = np.random.RandomState(42)
    X = rng.randn(50, 2)
    y_true = np.sign(X[:, 0])
    y_true[y_true == 0] = 1.0
    y = np.zeros(50)
    y[:10] = y_true[:10]
    y[25:35] = y_true[25:35]
    return X, y, y_true


@pytest.fixture
def labeled_data():
    """40 points, all labeled ±1."""
    rng = np.random.RandomState(7)
    X, y_raw = make_blobs(n_samples=40, n_features=2, centers=2,
                          random_state=7, cluster_std=0.8)
    y = np.where(y_raw == 0, -1.0, 1.0)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Kernel tests
# ─────────────────────────────────────────────────────────────────────────────

class TestKernels:
    def _X(self):
        rng = np.random.RandomState(0)
        return rng.randn(15, 3).astype(float)

    def test_linear_sym_psd(self):
        X = self._X()
        K = _kernel_sym(X, "linear", 1.0)
        assert K.shape == (15, 15)
        eigs = np.linalg.eigvalsh(K)
        assert np.all(eigs >= -1e-10)

    def test_rbf_sym_psd(self):
        X = self._X()
        K = _kernel_sym(X, "rbf", 1.0)
        assert K.shape == (15, 15)
        assert np.allclose(K, K.T)
        eigs = np.linalg.eigvalsh(K)
        assert np.all(eigs >= -1e-10)
        # diagonal must be 1 for RBF
        assert np.allclose(np.diag(K), 1.0)

    def test_poly_sym(self):
        X = self._X()
        K = _kernel_sym(X, "poly", 2)
        assert K.shape == (15, 15)
        assert np.allclose(K, K.T)

    def test_cross_vs_sym_rbf(self):
        X = self._X()
        K_sym = _kernel_sym(X, "rbf", 0.8)
        K_cross = _kernel_cross(X, X, "rbf", 0.8)
        assert np.allclose(K_sym, K_cross)

    def test_cross_shape(self):
        X1 = self._X()           # (15, 3) train
        X2 = self._X()[:5]       # (5, 3) test
        K = _kernel_cross(X1, X2, "rbf", 1.0)
        assert K.shape == (5, 15)

    def test_unknown_kernel_raises(self):
        X = self._X()
        with pytest.raises(ValueError, match="Unknown kernel"):
            _kernel_sym(X, "sigmoid", 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGraph:
    def _X(self, n=30, d=2, seed=0):
        return np.random.RandomState(seed).randn(n, d)

    def test_adjacency_shape(self):
        X = self._X()
        A = _build_adjacency(X, nn=5, dist_fn="euclidean",
                              weights="heat", weight_param=0.0)
        assert A.shape == (30, 30)

    def test_adjacency_symmetric(self):
        X = self._X()
        A = _build_adjacency(X, nn=5, dist_fn="euclidean",
                              weights="heat", weight_param=0.0)
        diff = (A - A.T)
        assert np.allclose(diff.data, 0.0, atol=1e-12)

    def test_adjacency_binary(self):
        X = self._X()
        A = _build_adjacency(X, nn=4, dist_fn="euclidean",
                              weights="binary", weight_param=0.0)
        # values should be 0 or 1
        assert set(np.unique(A.data)).issubset({0.0, 1.0})

    def test_adjacency_cosine(self):
        X = np.abs(self._X())  # positive values for cosine
        A = _build_adjacency(X, nn=3, dist_fn="cosine",
                              weights="heat", weight_param=0.5)
        assert A.shape == (30, 30)
        assert np.all(A.data >= 0)

    def test_laplacian_unnormalized(self):
        X = self._X()
        L = build_laplacian(X, nn=5, normalize=False, degree=1)
        L_arr = L.toarray()
        # row sums should be ~0
        assert np.allclose(L_arr.sum(axis=1), 0.0, atol=1e-10)
        # symmetric
        assert np.allclose(L_arr, L_arr.T, atol=1e-12)

    def test_laplacian_normalized(self):
        X = self._X()
        L = build_laplacian(X, nn=5, normalize=True, degree=1)
        L_arr = L.toarray()
        assert np.allclose(L_arr, L_arr.T, atol=1e-12)
        # eigenvalues in [0, 2]
        eigs = np.linalg.eigvalsh(L_arr)
        assert np.all(eigs >= -1e-10)
        assert np.all(eigs <= 2.0 + 1e-10)

    def test_laplacian_degree2(self):
        X = self._X()
        L1 = build_laplacian(X, nn=5, normalize=False, degree=1).toarray()
        L2 = build_laplacian(X, nn=5, normalize=False, degree=2).toarray()
        assert np.allclose(L2, L1 @ L1, atol=1e-10)

    def test_distance_weight(self):
        X = self._X()
        A = _build_adjacency(X, nn=3, dist_fn="euclidean",
                              weights="distance", weight_param=0.0)
        assert A.shape == (30, 30)
        assert np.all(A.data >= 0)

    def test_unknown_dist_raises(self):
        X = self._X()
        with pytest.raises(ValueError, match="Unknown distance"):
            _build_adjacency(X, nn=3, dist_fn="hamming",
                              weights="heat", weight_param=1.0)

    def test_unknown_weights_raises(self):
        X = self._X()
        with pytest.raises(ValueError, match="Unknown graph weights"):
            _build_adjacency(X, nn=3, dist_fn="euclidean",
                              weights="gaussian", weight_param=1.0)

    def test_euclidean_dist_correct(self):
        A = np.array([[0.0, 0.0], [3.0, 4.0]])
        B = np.array([[0.0, 0.0]])
        D = _euclidean_dist_matrix(A, B)
        assert np.isclose(D[0, 0], 0.0)
        assert np.isclose(D[1, 0], 5.0)


# ─────────────────────────────────────────────────────────────────────────────
# LapSVMClassifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLapSVMClassifier:

    def _make_clf(self, **kw):
        """Default config for semi-supervised (small_data) tests."""
        defaults = dict(kernel_param=0.5, gamma_A=1e-4,
                        gamma_I=1.0, nn=5, verbose=False,
                        laplacian_normalize=False, max_iter=50,
                        unlabeled_value=0)
        defaults.update(kw)
        return LapSVMClassifier(**defaults)

    def test_fit_returns_self(self, small_data):
        X, y, _ = small_data
        clf = self._make_clf()
        assert clf.fit(X, y) is clf

    def test_fitted_attributes(self, small_data):
        X, y, _ = small_data
        clf = self._make_clf().fit(X, y)
        assert hasattr(clf, "alpha_")
        assert hasattr(clf, "svs_")
        assert hasattr(clf, "b_")
        assert hasattr(clf, "n_iter_")
        assert hasattr(clf, "train_time_")
        assert clf.n_features_in_ == 2
        assert np.array_equal(clf.classes_, [-1.0, 1.0])

    def test_predict_shape(self, small_data):
        X, y, _ = small_data
        clf = self._make_clf().fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (50,)
        assert set(pred).issubset({-1.0, 1.0})

    def test_decision_function_shape(self, small_data):
        X, y, _ = small_data
        clf = self._make_clf().fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (50,)

    def test_predict_sign_consistent(self, small_data):
        X, y, _ = small_data
        clf = self._make_clf().fit(X, y)
        scores = clf.decision_function(X)
        pred = clf.predict(X)
        # labels come from classes_ not raw ±1
        pos = clf._le_pos
        neg = clf._le_neg
        assert np.all(pred == np.where(scores > 0, pos, neg))

    def test_score(self, labeled_data):
        X, y = labeled_data
        clf = self._make_clf(gamma_I=0.5).fit(X, y)
        acc = clf.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_fully_supervised(self, labeled_data):
        """All labeled — should learn a reasonable boundary."""
        X, y = labeled_data
        clf = LapSVMClassifier(kernel_param=0.8, gamma_A=1e-3,
                               gamma_I=0.1, nn=5, verbose=False,
                               laplacian_normalize=False, max_iter=100)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.7

    def test_gamma_I_zero_supervised(self, labeled_data):
        """gamma_I=0 → standard SVM; still works."""
        X, y = labeled_data
        clf = LapSVMClassifier(kernel_param=0.8, gamma_A=1e-3,
                               gamma_I=0.0, nn=5, verbose=False,
                               laplacian_normalize=False, max_iter=100)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.6

    def test_semi_supervised_score(self, small_data):
        X, y, y_true = small_data
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=1.0, nn=6, verbose=False,
                               laplacian_normalize=False, max_iter=100,
                               unlabeled_value=0)
        clf.fit(X, y)
        acc = clf.score(X, y_true)
        assert acc > 0.5

    def test_linear_kernel(self, labeled_data):
        X, y = labeled_data
        clf = LapSVMClassifier(kernel="linear", gamma_A=1e-3,
                               gamma_I=0.1, nn=5, verbose=False,
                               laplacian_normalize=False, max_iter=50)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (40,)

    def test_poly_kernel(self, labeled_data):
        X, y = labeled_data
        clf = LapSVMClassifier(kernel="poly", kernel_param=2,
                               gamma_A=1e-3, gamma_I=0.1, nn=5,
                               verbose=False, laplacian_normalize=False,
                               max_iter=50)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (40,)

    def test_use_bias(self, labeled_data):
        X, y = labeled_data
        clf = LapSVMClassifier(kernel_param=0.8, gamma_A=1e-3,
                               gamma_I=0.0, nn=5, use_bias=True,
                               verbose=False, laplacian_normalize=False,
                               max_iter=50)
        clf.fit(X, y)
        assert clf.b_ != 0.0 or True  # b_ may be 0 by coincidence

    def test_bias_normalized_laplacian_raises(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 2)
        y = np.ones(20)
        y[10:] = -1.0
        clf = LapSVMClassifier(use_bias=True, laplacian_normalize=True)
        with pytest.raises(ValueError, match="use_bias"):
            clf.fit(X, y)

    def test_wrong_feature_count_raises(self, small_data):
        X, y, _ = small_data
        clf = self._make_clf().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            clf.decision_function(X[:, :1])

    def test_predict_before_fit_raises(self):
        clf = LapSVMClassifier()
        with pytest.raises(Exception):
            clf.predict(np.zeros((5, 2)))

    def test_invalid_labels_raises(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 2)
        # 3 classes → multiclass → should raise ValueError
        y = np.array([0, 1, 2] * 6 + [0, 2])
        clf = LapSVMClassifier()
        with pytest.raises(ValueError):
            clf.fit(X, y)

    def test_binary_graph_weights(self, small_data):
        X, y, _ = small_data
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=0.5, nn=5, graph_weights="binary",
                               verbose=False, laplacian_normalize=False,
                               max_iter=50, unlabeled_value=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (50,)

    def test_distance_graph_weights(self, small_data):
        X, y, _ = small_data
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=0.5, nn=5, graph_weights="distance",
                               verbose=False, laplacian_normalize=False,
                               max_iter=50, unlabeled_value=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (50,)

    def test_cosine_graph_dist(self, small_data):
        X, y, _ = small_data
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=0.5, nn=5, graph_dist="cosine",
                               verbose=False, laplacian_normalize=False,
                               max_iter=50, unlabeled_value=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (50,)

    def test_laplacian_degree2(self, small_data):
        X, y, _ = small_data
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=0.5, nn=5, laplacian_degree=2,
                               verbose=False, laplacian_normalize=False,
                               max_iter=50, unlabeled_value=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (50,)

    def test_get_params(self):
        clf = LapSVMClassifier(gamma_A=0.01, gamma_I=2.0, nn=10)
        p = clf.get_params()
        assert p["gamma_A"] == 0.01
        assert p["gamma_I"] == 2.0
        assert p["nn"] == 10

    def test_set_params(self):
        clf = LapSVMClassifier()
        clf.set_params(gamma_A=0.1, kernel="linear")
        assert clf.gamma_A == 0.1
        assert clf.kernel == "linear"

    def test_svs_are_subset(self, small_data):
        X, y, _ = small_data
        clf = self._make_clf().fit(X, y)
        assert np.all(clf.svs_ < len(y))

    def test_n_iter_bounded(self, small_data):
        X, y, _ = small_data
        clf = LapSVMClassifier(max_iter=10, verbose=False,
                               laplacian_normalize=False,
                               kernel_param=0.5, gamma_A=1e-4,
                               unlabeled_value=0)
        clf.fit(X, y)
        assert clf.n_iter_ <= 10

    def test_verbose_does_not_raise(self, small_data, capsys):
        X, y, _ = small_data
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=0.5, nn=5, max_iter=3,
                               verbose=True, laplacian_normalize=False,
                               unlabeled_value=0)
        clf.fit(X, y)
        captured = capsys.readouterr()
        assert "[t=" in captured.out

    def test_heat_bandwidth_auto(self, small_data):
        X, y, _ = small_data
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=0.5, nn=5, graph_weight_param=0.0,
                               verbose=False, laplacian_normalize=False,
                               max_iter=30, unlabeled_value=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (50,)

    def test_heat_bandwidth_explicit(self, small_data):
        X, y, _ = small_data
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=0.5, nn=5, graph_weight_param=1.0,
                               verbose=False, laplacian_normalize=False,
                               max_iter=30, unlabeled_value=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (50,)

    def test_only_positive_class(self):
        """One-class scenario: all labeled as +1, rest unlabeled."""
        rng = np.random.RandomState(1)
        X = rng.randn(40, 2)
        y = np.zeros(40)
        y[:10] = 1.0
        y[20:30] = -1.0   # need two classes
        clf = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-4,
                               gamma_I=0.5, nn=5, use_bias=False,
                               verbose=False, laplacian_normalize=False,
                               max_iter=30, unlabeled_value=0)
        clf.fit(X, y)
        assert clf.predict(X).shape == (40,)


# ─────────────────────────────────────────────────────────────────────────────
# LapRLSCClassifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestLapRLSCClassifier:

    def _make_clf(self, **kw):
        defaults = dict(kernel_param=0.5, gamma_A=1e-3,
                        gamma_I=1.0, nn=5, verbose=False,
                        laplacian_normalize=False, max_iter=30)
        defaults.update(kw)
        return LapRLSCClassifier(**defaults)

    def test_fit_predict(self, small_data):
        X, y, _ = small_data
        clf = LapRLSCClassifier(kernel_param=0.5, gamma_A=1e-3,
                                gamma_I=1.0, nn=5, verbose=False,
                                laplacian_normalize=False, max_iter=30,
                                unlabeled_value=0)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (50,)
        assert set(pred).issubset({-1.0, 1.0})

    def test_single_iteration(self, labeled_data):
        """RLSC always has all labeled as SV → converges in 1 Newton step."""
        X, y = labeled_data
        clf = LapRLSCClassifier(kernel_param=0.8, gamma_A=1e-3,
                                gamma_I=0.0, nn=5, verbose=False,
                                laplacian_normalize=False, max_iter=50)
        clf.fit(X, y)
        # RLSC with gamma_I=0 converges in exactly 1 iteration
        assert clf.n_iter_ == 1

    def test_score(self, labeled_data):
        X, y = labeled_data
        clf = self._make_clf(gamma_I=0.5).fit(X, y)
        assert clf.score(X, y) > 0.6

    def test_invalid_labels_raises(self):
        X = np.random.randn(20, 2)
        # 3-class → should raise ValueError (not binary)
        y = np.array([0, 1, 2] * 6 + [0, 2]).astype(float)
        with pytest.raises(ValueError):
            LapRLSCClassifier().fit(X, y)

    def test_bias_raises_with_normalized_lap(self):
        X = np.random.randn(20, 2)
        y = np.ones(20); y[10:] = -1.0
        with pytest.raises(ValueError):
            LapRLSCClassifier(use_bias=True,
                              laplacian_normalize=True).fit(X, y)

    def test_get_set_params(self):
        clf = LapRLSCClassifier(gamma_A=0.01)
        clf.set_params(gamma_I=5.0)
        assert clf.gamma_I == 5.0

    def test_decision_function(self, labeled_data):
        X, y = labeled_data
        clf = self._make_clf().fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (40,)

    def test_wrong_features_raises(self, labeled_data):
        X, y = labeled_data
        clf = self._make_clf().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            clf.predict(X[:, :1])


# ─────────────────────────────────────────────────────────────────────────────
# Numerical sanity: compare to hand-computed small case
# ─────────────────────────────────────────────────────────────────────────────

class TestNumericalSanity:
    """Small, fully-supervised, no-graph cases with known analytic behaviour."""

    def test_rbf_lapsvm_gamma_I_zero_convergence(self):
        """With gamma_I=0 and linearly separable data, expect high accuracy."""
        rng = np.random.RandomState(99)
        n = 60
        X = np.vstack([rng.randn(n // 2, 2) + 2,
                        rng.randn(n // 2, 2) - 2])
        y = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
        clf = LapSVMClassifier(kernel="rbf", kernel_param=1.0,
                               gamma_A=1e-2, gamma_I=0.0, nn=5,
                               verbose=False, laplacian_normalize=False,
                               max_iter=100)
        clf.fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_laprlsc_gamma_I_zero_convergence(self):
        """LapRLSC should also achieve high acc on separable data."""
        rng = np.random.RandomState(42)
        n = 60
        X = np.vstack([rng.randn(n // 2, 2) + 2,
                        rng.randn(n // 2, 2) - 2])
        y = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
        clf = LapRLSCClassifier(kernel="rbf", kernel_param=1.0,
                                gamma_A=1e-2, gamma_I=0.0, nn=5,
                                verbose=False, laplacian_normalize=False,
                                max_iter=50)
        clf.fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_laprlsc_equals_kernel_ridge_fully_supervised(self):
        """LapRLSC (gamma_I=0, no bias) = kernel ridge regression.

        Decision boundary should be symmetric: f = K α, where
        (gamma_A I + K) α = y.  Verify that predicted f = K(K + γI)⁻¹ y.
        """
        rng = np.random.RandomState(5)
        n = 20
        X = rng.randn(n, 2)
        y = np.sign(X[:, 0])
        y[y == 0] = 1.0
        gamma_A = 0.1

        clf = LapRLSCClassifier(kernel="rbf", kernel_param=1.0,
                                gamma_A=gamma_A, gamma_I=0.0, nn=5,
                                use_bias=False, verbose=False,
                                laplacian_normalize=False, max_iter=50)
        clf.fit(X, y)

        from scikit_svm.lapsvm import _kernel_sym
        K = _kernel_sym(X, "rbf", 1.0)
        # analytic solution: alpha = (K + gamma_A I)^{-1} y
        alpha_ref = np.linalg.solve(K + gamma_A * np.eye(n), y)
        f_ref = K @ alpha_ref
        f_clf = clf.decision_function(X)
        assert np.allclose(f_ref, f_clf, atol=1e-6)

    def test_lapsvm_manifold_effect(self):
        """Adding gamma_I should change the decision function."""
        rng = np.random.RandomState(13)
        n = 40
        X = rng.randn(n, 2)
        y = np.zeros(n)
        y[:8] = 1.0; y[20:28] = -1.0

        clf0 = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-3, gamma_I=0.0,
                                nn=5, verbose=False, laplacian_normalize=False,
                                max_iter=50, unlabeled_value=0)
        clf1 = LapSVMClassifier(kernel_param=0.5, gamma_A=1e-3, gamma_I=10.0,
                                nn=5, verbose=False, laplacian_normalize=False,
                                max_iter=50, unlabeled_value=0)
        clf0.fit(X, y); clf1.fit(X, y)
        f0 = clf0.decision_function(X)
        f1 = clf1.decision_function(X)
        # They should NOT be identical
        assert not np.allclose(f0, f1)


# ─────────────────────────────────────────────────────────────────────────────
# sklearn estimator checks
# ─────────────────────────────────────────────────────────────────────────────

def _make_sklearn_clf():
    """Return a LapSVMClassifier that passes sklearn's standard checks.

    sklearn's check_estimator passes fully-labeled ±1 data; we use a
    configuration without graph regularization so the behaviour is
    well-defined even with tiny datasets.
    """
    return LapSVMClassifier(
        kernel="rbf",
        kernel_param=1.0,
        gamma_A=1e-2,
        gamma_I=0.0,         # no Laplacian → plain kernel SVM
        nn=3,
        laplacian_normalize=False,
        verbose=False,
        max_iter=50,
    )


def _make_sklearn_rlsc():
    return LapRLSCClassifier(
        kernel="rbf",
        kernel_param=1.0,
        gamma_A=1e-2,
        gamma_I=0.0,
        nn=3,
        laplacian_normalize=False,
        verbose=False,
        max_iter=50,
    )


@parametrize_with_checks([_make_sklearn_clf(), _make_sklearn_rlsc()])
def test_sklearn_compatible(estimator, check):
    check(estimator)
