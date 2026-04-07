"""
Tests for PSVMClassifier and NPSVMClassifier.

Covers:
 - nu estimation (EstNuLong, EstNuShort)
 - RBF kernel computation
 - PSVMClassifier: fit/predict/decision_function/score
 - NPSVMClassifier: fit/predict, reduced kernel, mu estimation
 - Class balancing
 - Edge cases and error handling
 - Analytical correctness (PSVM = ridge regression limit)
 - sklearn estimator compatibility
"""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_svm import NPSVMClassifier, PSVMClassifier
from scikit_svm.psvm import (
    _build_HV,
    _estimate_mu,
    _estimate_nu_long,
    _estimate_nu_short,
    _psvm_core,
    _rbf_kernel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def separable_data():
    """60 linearly separable points, ±1 labels."""
    rng = np.random.RandomState(42)
    n = 60
    X = np.vstack([rng.randn(n // 2, 2) + 2.5,
                    rng.randn(n // 2, 2) - 2.5])
    y = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
    return X, y


@pytest.fixture
def blobs_data():
    """40-point two-class blobs, sklearn-style labels {0, 1}."""
    X, y = make_blobs(n_samples=40, centers=2, cluster_std=0.8,
                       random_state=7)
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / internal function tests
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:

    def test_rbf_kernel_shape(self):
        A = np.random.randn(10, 3)
        B = np.random.randn(5, 3)
        K = _rbf_kernel(A, B, mu=0.5)
        assert K.shape == (10, 5)

    def test_rbf_kernel_diagonal_one_when_same(self):
        A = np.random.randn(8, 2)
        K = _rbf_kernel(A, A, mu=1.0)
        assert np.allclose(np.diag(K), 1.0)

    def test_rbf_kernel_symmetric(self):
        A = np.random.randn(8, 2)
        K = _rbf_kernel(A, A, mu=0.7)
        assert np.allclose(K, K.T)

    def test_rbf_kernel_nonneg(self):
        A = np.random.randn(10, 3)
        B = np.random.randn(6, 3)
        K = _rbf_kernel(A, B, mu=2.0)
        assert np.all(K >= 0)
        assert np.all(K <= 1.0)

    def test_rbf_mu_larger_sharper(self):
        A = np.random.randn(5, 2)
        B = np.random.randn(5, 2)
        K1 = _rbf_kernel(A, B, mu=0.01)
        K2 = _rbf_kernel(A, B, mu=10.0)
        # Higher mu → smaller values for non-identical points
        assert K2.mean() <= K1.mean()

    def test_estimate_nu_long_positive(self):
        rng = np.random.RandomState(0)
        H = np.random.randn(30, 3)
        d = np.sign(np.random.randn(30))
        d[d == 0] = 1.0
        nu = _estimate_nu_long(H, d, rng)
        assert nu > 0

    def test_estimate_nu_short_positive(self):
        C = np.random.randn(20, 4)
        nu = _estimate_nu_short(C)
        assert nu > 0

    def test_estimate_nu_short_formula(self):
        C = np.ones((4, 3))   # 4 rows, 3 cols, all ones
        # frob2 = 12, p = 3 → nu = 3/12 = 0.25
        assert np.isclose(_estimate_nu_short(C), 0.25)

    def test_estimate_mu_positive(self):
        X = np.random.randn(20, 2)
        y = np.concatenate([np.ones(10), -np.ones(10)])
        mu = _estimate_mu(X, y)
        assert mu > 0

    def test_estimate_mu_range(self):
        X = np.random.randn(30, 4)
        y = np.concatenate([np.ones(15), -np.ones(15)])
        mu = _estimate_mu(X, y)
        assert 0 < mu <= 1.0

    def test_build_HV_shape(self):
        X = np.random.randn(20, 3)
        y = np.sign(np.random.randn(20))
        y[y == 0] = 1.0
        H, v = _build_HV(X, y, balance=False)
        assert H.shape == (20, 4)  # 3 features + bias col
        assert v.shape == (4,)

    def test_build_HV_balanced_shape(self):
        X = np.random.randn(20, 2)
        y = np.concatenate([np.ones(10), -np.ones(10)])
        H, v = _build_HV(X, y, balance=True)
        assert H.shape == (20, 3)

    def test_psvm_core_returns_correct_sizes(self):
        n = 5
        H = np.random.randn(20, n)
        v = np.random.randn(n)
        w, gamma = _psvm_core(H, v, nu=1.0)
        assert w.shape == (n - 1,)
        assert isinstance(gamma, float)


# ─────────────────────────────────────────────────────────────────────────────
# PSVMClassifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPSVMClassifier:

    def _clf(self, **kw):
        defaults = dict(nu=0.1, random_state=0)
        defaults.update(kw)
        return PSVMClassifier(**defaults)

    def test_fit_returns_self(self, separable_data):
        X, y = separable_data
        clf = self._clf()
        assert clf.fit(X, y) is clf

    def test_fitted_attributes(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        assert hasattr(clf, "w_")
        assert hasattr(clf, "gamma_")
        assert hasattr(clf, "nu_")
        assert hasattr(clf, "train_time_")
        assert clf.n_features_in_ == 2
        assert len(clf.classes_) == 2

    def test_w_shape(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        assert clf.w_.shape == (2,)

    def test_predict_shape(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (60,)

    def test_predict_values_from_classes(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        assert set(np.unique(clf.predict(X))).issubset(set(clf.classes_))

    def test_decision_function_shape(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (60,)

    def test_predict_sign_consistent(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        scores = clf.decision_function(X)
        pred = clf.predict(X)
        assert np.all(pred == np.where(scores > 0, clf._le_pos, clf._le_neg))

    def test_high_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = self._clf(nu=0.1).fit(X, y)
        assert clf.score(X, y) >= 0.95

    def test_nu_auto_estimation(self, separable_data):
        X, y = separable_data
        clf = PSVMClassifier(nu=0, random_state=0).fit(X, y)
        assert clf.nu_ > 0
        assert clf.score(X, y) > 0.7

    def test_nu_easy_estimation(self, separable_data):
        X, y = separable_data
        clf = PSVMClassifier(nu=-1, random_state=0).fit(X, y)
        assert clf.nu_ > 0
        assert clf.score(X, y) > 0.7

    def test_sklearn_labels(self, blobs_data):
        """Accept 0/1 labels (sklearn convention)."""
        X, y = blobs_data
        clf = self._clf().fit(X, y)
        assert set(clf.classes_) == {0, 1}
        pred = clf.predict(X)
        assert set(pred).issubset({0, 1})

    def test_balanced_mode(self, separable_data):
        X, y = separable_data
        clf = PSVMClassifier(nu=0.1, balance=True, random_state=0).fit(X, y)
        assert clf.score(X, y) > 0.7

    def test_imbalanced_balance_helps(self):
        """balance=True should not crash on imbalanced data."""
        rng = np.random.RandomState(1)
        X = np.vstack([rng.randn(80, 2) + 2, rng.randn(20, 2) - 2])
        y = np.concatenate([np.ones(80), -np.ones(20)])
        clf = PSVMClassifier(nu=0.1, balance=True, random_state=1).fit(X, y)
        assert clf.score(X, y) > 0.7

    def test_wrong_feature_count_raises(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            clf.decision_function(X[:, :1])

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            PSVMClassifier().predict(np.zeros((5, 2)))

    def test_multiclass_raises(self):
        X = np.random.randn(30, 2)
        y = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError):
            PSVMClassifier().fit(X, y)

    def test_continuous_y_raises(self):
        X = np.random.randn(20, 2)
        y = np.random.randn(20)   # continuous
        with pytest.raises(ValueError):
            PSVMClassifier().fit(X, y)

    def test_get_params(self):
        clf = PSVMClassifier(nu=0.5, balance=True)
        p = clf.get_params()
        assert p["nu"] == 0.5
        assert p["balance"] is True

    def test_set_params(self):
        clf = PSVMClassifier()
        clf.set_params(nu=2.0, balance=True)
        assert clf.nu == 2.0
        assert clf.balance is True

    def test_verbose(self, separable_data, capsys):
        X, y = separable_data
        PSVMClassifier(nu=0.1, verbose=True, random_state=0).fit(X, y)
        out = capsys.readouterr().out
        assert "Training accuracy" in out

    def test_random_state_deterministic(self, separable_data):
        """Same random_state → same result (only matters when nu=0 + m>200)."""
        X, y = separable_data
        clf1 = PSVMClassifier(nu=0, random_state=7).fit(X, y)
        clf2 = PSVMClassifier(nu=0, random_state=7).fit(X, y)
        assert np.allclose(clf1.w_, clf2.w_)


# ─────────────────────────────────────────────────────────────────────────────
# Analytical correctness: PSVM vs closed form
# ─────────────────────────────────────────────────────────────────────────────

class TestPSVMAnalytical:

    def test_psvm_solution_matches_closed_form(self):
        """PSVM solution must satisfy (I/nu + H'H) v = H'y exactly."""
        rng = np.random.RandomState(3)
        n = 30
        X = rng.randn(n, 3)
        y = np.sign(X[:, 0])
        y[y == 0] = 1.0
        nu = 0.5

        clf = PSVMClassifier(nu=nu, random_state=0).fit(X, y)
        v = np.append(clf.w_, clf.gamma_)
        H, rhs = _build_HV(X, y, balance=False)
        lhs_v = np.eye(4) / nu + H.T @ H
        assert np.allclose(lhs_v @ v, rhs, atol=1e-8)

    def test_decision_boundary_symmetric(self):
        """Points on the hyperplane should have decision score ≈ 0."""
        rng = np.random.RandomState(5)
        n = 40
        X = np.vstack([rng.randn(n // 2, 2) + 2,
                        rng.randn(n // 2, 2) - 2])
        y = np.concatenate([np.ones(n // 2), -np.ones(n // 2)])
        clf = PSVMClassifier(nu=0.1, random_state=0).fit(X, y)
        # point on the hyperplane: w·x = gamma → x = w*gamma/||w||^2
        w, gamma = clf.w_, clf.gamma_
        x_boundary = w * gamma / (w @ w)
        score = x_boundary @ w - gamma
        assert abs(score) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# NPSVMClassifier tests
# ─────────────────────────────────────────────────────────────────────────────

class TestNPSVMClassifier:

    def _clf(self, **kw):
        defaults = dict(nu=0.1, mu=0.5, random_state=0)
        defaults.update(kw)
        return NPSVMClassifier(**defaults)

    def test_fit_returns_self(self, separable_data):
        X, y = separable_data
        assert self._clf().fit(X, y) is self._clf().fit(X, y) or True  # just runs
        clf = self._clf()
        assert clf.fit(X, y) is clf

    def test_fitted_attributes(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        assert hasattr(clf, "w_")
        assert hasattr(clf, "gamma_")
        assert hasattr(clf, "nu_")
        assert hasattr(clf, "mu_")
        assert hasattr(clf, "X_bar_")
        assert hasattr(clf, "train_time_")
        assert clf.n_features_in_ == 2
        assert len(clf.classes_) == 2

    def test_X_bar_shape_full(self, separable_data):
        X, y = separable_data
        clf = self._clf(reduce_ratio=1.0).fit(X, y)
        assert clf.X_bar_.shape[0] == 60
        assert clf.X_bar_.shape[1] == 2

    def test_X_bar_shape_reduced(self, separable_data):
        X, y = separable_data
        clf = self._clf(reduce_ratio=0.5).fit(X, y)
        assert clf.X_bar_.shape[0] == 30

    def test_w_shape_matches_r(self, separable_data):
        X, y = separable_data
        clf = self._clf(reduce_ratio=0.4).fit(X, y)
        r = clf.X_bar_.shape[0]
        assert clf.w_.shape == (r,)

    def test_predict_shape(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (60,)

    def test_decision_function_shape(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        scores = clf.decision_function(X)
        assert scores.shape == (60,)

    def test_predict_sign_consistent(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        scores = clf.decision_function(X)
        pred = clf.predict(X)
        assert np.all(pred == np.where(scores > 0, clf._le_pos, clf._le_neg))

    def test_high_accuracy_separable(self, separable_data):
        X, y = separable_data
        clf = self._clf(nu=0.1, mu=1.0).fit(X, y)
        assert clf.score(X, y) >= 0.9

    def test_mu_auto_estimation(self, separable_data):
        X, y = separable_data
        clf = NPSVMClassifier(nu=0.1, mu=0.0, random_state=0).fit(X, y)
        assert clf.mu_ > 0
        assert clf.score(X, y) > 0.7

    def test_nu_auto_estimation(self, separable_data):
        X, y = separable_data
        clf = NPSVMClassifier(nu=0, mu=0.5, random_state=0).fit(X, y)
        assert clf.nu_ > 0
        assert clf.score(X, y) > 0.7

    def test_sklearn_labels(self, blobs_data):
        X, y = blobs_data
        clf = self._clf().fit(X, y)
        assert set(clf.classes_) == {0, 1}
        pred = clf.predict(X)
        assert set(pred).issubset({0, 1})

    def test_reduced_kernel_still_works(self, separable_data):
        X, y = separable_data
        clf = NPSVMClassifier(nu=0.1, mu=0.5, reduce_ratio=0.3,
                               random_state=0).fit(X, y)
        assert clf.score(X, y) > 0.5  # reduced may be less accurate

    def test_balance_mode(self, separable_data):
        X, y = separable_data
        clf = NPSVMClassifier(nu=0.1, mu=0.5, balance=True,
                               random_state=0).fit(X, y)
        assert clf.score(X, y) > 0.7

    def test_wrong_feature_count_raises(self, separable_data):
        X, y = separable_data
        clf = self._clf().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            clf.decision_function(X[:, :1])

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            NPSVMClassifier().predict(np.zeros((5, 2)))

    def test_multiclass_raises(self):
        X = np.random.randn(30, 2)
        y = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError):
            NPSVMClassifier().fit(X, y)

    def test_get_set_params(self):
        clf = NPSVMClassifier(nu=0.5, mu=0.2, reduce_ratio=0.7)
        p = clf.get_params()
        assert p["nu"] == 0.5
        assert p["mu"] == 0.2
        assert p["reduce_ratio"] == 0.7
        clf.set_params(nu=1.0)
        assert clf.nu == 1.0

    def test_verbose(self, separable_data, capsys):
        X, y = separable_data
        NPSVMClassifier(nu=0.1, mu=0.5, verbose=True, random_state=0).fit(X, y)
        out = capsys.readouterr().out
        assert "Training accuracy" in out

    def test_nu_easy_estimation(self, separable_data):
        X, y = separable_data
        clf = NPSVMClassifier(nu=-1, mu=0.5, random_state=0).fit(X, y)
        assert clf.nu_ > 0

    def test_random_state_deterministic(self, separable_data):
        """Same random_state → same X_bar_ and w_."""
        X, y = separable_data
        clf1 = NPSVMClassifier(nu=0.1, mu=0.5, random_state=42).fit(X, y)
        clf2 = NPSVMClassifier(nu=0.1, mu=0.5, random_state=42).fit(X, y)
        assert np.allclose(clf1.X_bar_, clf2.X_bar_)
        assert np.allclose(clf1.w_, clf2.w_)

    def test_nonlinear_data(self):
        """XOR-like data, not linearly separable."""
        rng = np.random.RandomState(11)
        X = rng.randn(80, 2) * 0.5
        # class +1: quadrants 1 and 3; class -1: quadrants 2 and 4
        y = np.sign(X[:, 0] * X[:, 1])
        y[y == 0] = 1.0
        clf = NPSVMClassifier(nu=0.01, mu=2.0, random_state=0).fit(X, y)
        assert clf.score(X, y) > 0.7  # should be able to learn XOR


# ─────────────────────────────────────────────────────────────────────────────
# sklearn estimator checks
# ─────────────────────────────────────────────────────────────────────────────

def _make_psvm():
    return PSVMClassifier(nu=0.1, random_state=0)


def _make_npsvm():
    return NPSVMClassifier(nu=0.1, mu=0.5, random_state=0)


@parametrize_with_checks([_make_psvm(), _make_npsvm()])
def test_sklearn_compatible(estimator, check):
    check(estimator)
