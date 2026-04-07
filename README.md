# scikit-svm

A collection of **Support Vector Machine** classifiers and regressors with a
**scikit-learn compatible interface**.  The library brings together multiple SVM
algorithm families — from classic research implementations and vendored C/C++
solvers to modern Python-package backends — all under a single, consistent
`fit` / `predict` / `score` API.

---

## Estimator families

| Family | Classes | Backend | Task |
|--------|---------|---------|------|
| Lagrangian SVM | `LSVM`, `LSVMK` | Pure Python (NumPy) | Binary classification |
| Smooth SVM | `SSVM`, `NSSVM` | Pure Python (NumPy) | Binary classification |
| Least Squares SVM | `LSSVMClassifier`, `LSSVMRegressor` | Pure Python (NumPy) | Binary classification, regression |
| Laplacian SVM | `LapSVMClassifier`, `LapRLSCClassifier` | Pure Python (NumPy) | Semi-supervised binary classification |
| Proximal SVM | `PSVMClassifier`, `NPSVMClassifier` | Pure Python (NumPy) | Binary classification |
| OCAS linear SVM | `SVMOCASClassifier`, `MSVMOCASClassifier` | Cython → libocas C | Binary and multi-class classification |
| Core/Ball Vector Machine | `CVM`, `BVM` | Cython → libCVM C++ | Binary classification |
| Bound-Constrained SVM | `BSVMClassifier`, `BSVMRegressor` | Cython → BSVM C++ | Binary classification, regression |
| SVM-Light | `SVMLightClassifier`, `SVMLightRegressor` | Cython → SVM-Light C | Binary classification, regression |
| mySVM | `MySVMClassifier`, `MySVMRegressor`, `MySVMNuClassifier`, `MySVMNuRegressor` | Cython → mySVM C++ | Binary classification, regression |
| LIBLINEAR | `LibLinearSVC`, `LibLinearSVR` | Python → liblinear-official | Linear classification, regression |

Pure-Python estimators are available immediately after install.  Cython-backed
estimators require building the C/C++ extensions (see [Building extensions](#building-extensions)).
`LibLinearSVC` and `LibLinearSVR` require `pip install liblinear-official` but
no compilation step.

---

## Installation

```bash
# Install runtime dependencies and pure-Python estimators
pip install .

# Editable install for development
pip install -e ".[dev]"

# Also build Cython extensions (required for OCAS, CVM, BVM, BSVM, SVM-Light, mySVM)
python setup.py build_ext --inplace
```

**Requirements**: Python ≥ 3.8 · NumPy ≥ 1.20 · scikit-learn ≥ 1.0

**Optional**: `liblinear-official` ≥ 2.47 (for `LibLinearSVC` / `LibLinearSVR`)

---

## Quick start

### LIBLINEAR — linear SVM and SVR

```python
from sklearn.datasets import load_iris, load_diabetes
from scikit_svm import LibLinearSVC, LibLinearSVR

# Classification (binary and multi-class)
X, y = load_iris(return_X_y=True)
clf = LibLinearSVC(solver=1, C=1.0)   # L2-regularised L2-loss SVM (dual)
clf.fit(X, y)
print(clf.score(X, y))                # e.g. 0.967
print(clf.coef_.shape)                # (3, 4)  — one row per class

# Regression
X_r, y_r = load_diabetes(return_X_y=True)
reg = LibLinearSVR(C=1.0)
reg.fit(X_r, y_r)
print(reg.score(X_r, y_r))
```

### Lagrangian SVM

> Labels must be **±1** (not 0/1) for `LSVM` and `LSVMK`.

```python
import numpy as np
from scikit_svm import LSVM, LSVMK

X = np.array([[1, 2], [2, 3], [-1, -2], [-2, -3]], dtype=float)
y = np.array([1., 1., -1., -1.])

# Linear LSVM
clf = LSVM(nu=0.5, verbose=False)
clf.fit(X, y)
print(clf.predict(X))            # [ 1.  1. -1. -1.]
print(clf.score(X, y))           # 1.0

# Kernel LSVM (RBF)
clf = LSVMK(kernel='rbf', nu=0.5)
clf.fit(X, y)
print(clf.predict(X))
```

### Smooth SVM

```python
from scikit_svm import SSVM, NSSVM

# Linear Smooth SVM (arbitrary binary labels)
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, random_state=0)

clf = SSVM(nu='easy')
clf.fit(X, y)
print(clf.score(X, y))

# Nonlinear Smooth SVM (reduced RBF kernel)
clf = NSSVM(reduce_rate=0.3)
clf.fit(X, y)
print(clf.score(X, y))
```

### Proximal SVM

```python
from scikit_svm import PSVMClassifier, NPSVMClassifier

clf = PSVMClassifier(nu=0)     # nu=0: auto-estimate via largest eigenvalue
clf.fit(X, y)
print(clf.score(X, y))
```

### Least Squares SVM

```python
from scikit_svm import LSSVMClassifier, LSSVMRegressor

clf = LSSVMClassifier(C=10.0, kernel='rbf')
clf.fit(X, y)
print(clf.score(X, y))
```

### Laplacian SVM (semi-supervised)

```python
import numpy as np
from scikit_svm import LapSVMClassifier

X_all = np.vstack([X_labeled, X_unlabeled])
# Mark unlabeled samples with a sentinel value (default unlabeled_value=2)
y_semi = np.concatenate([y_labeled, np.full(len(X_unlabeled), 2)])

clf = LapSVMClassifier(gamma_A=0.1, gamma_I=0.01, nn=6)
clf.fit(X_all, y_semi)
print(clf.score(X_labeled, y_labeled))
```

### OCAS linear SVM (requires Cython build)

```python
from scikit_svm import SVMOCASClassifier, MSVMOCASClassifier

# Binary
clf = SVMOCASClassifier(C=1.0)
clf.fit(X, y)
print(clf.score(X, y))

# Multi-class (Crammer-Singer formulation)
from sklearn.datasets import load_iris
X_iris, y_iris = load_iris(return_X_y=True)
clf = MSVMOCASClassifier(C=1.0)
clf.fit(X_iris, y_iris)
print(clf.score(X_iris, y_iris))
```

### Core / Ball Vector Machine (requires Cython build)

```python
from scikit_svm import CVM, BVM

clf = CVM(C=1.0, kernel='rbf', gamma=0.5)
clf.fit(X, y)
print(clf.score(X, y))
```

### SVM-Light (requires Cython build)

```python
from scikit_svm import SVMLightClassifier, SVMLightRegressor

clf = SVMLightClassifier(C=1.0, kernel='rbf')
clf.fit(X, y)
print(clf.score(X, y))
```

---

## Estimator reference

### `LibLinearSVC`

Linear SVM / logistic regression classifier backed by [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `solver` | `1` | Solver: 0=L2R_LR, **1=L2R_L2LOSS_SVC_DUAL**, 2=L2R_L2LOSS_SVC, 3=L2R_L1LOSS_SVC_DUAL, 4=MCSVM_CS, 5=L1R_L2LOSS_SVC, 6=L1R_LR, 7=L2R_LR_DUAL |
| `C` | `1.0` | Regularisation parameter |
| `tol` | `1e-4` | Solver tolerance |
| `fit_intercept` | `True` | Fit bias term |
| `class_weight` | `None` | `None`, `'balanced'`, or dict |
| `verbose` | `False` | Print solver output |

Fitted attributes: `coef_` `(1, n_features)` binary / `(n_classes, n_features)` multi-class,
`intercept_`, `classes_`, `train_time_`.

---

### `LibLinearSVR`

Linear SVR backed by [LIBLINEAR](https://www.csie.ntu.edu.tw/~cjlin/liblinear/).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `solver` | `11` | Solver: **11=L2R_L2LOSS_SVR**, 12=L2R_L2LOSS_SVR_DUAL, 13=L2R_L1LOSS_SVR_DUAL |
| `C` | `1.0` | Regularisation parameter |
| `p` | `0.1` | Epsilon in the epsilon-insensitive loss |
| `tol` | `1e-4` | Solver tolerance |
| `fit_intercept` | `True` | Fit bias term |
| `verbose` | `False` | Print solver output |

Fitted attributes: `coef_` `(1, n_features)`, `intercept_` `(1,)`, `train_time_`.

---

### `LSVM` — Lagrangian SVM (linear)

Port of `lsvm.m` (Mangasarian & Musicant, 2000).  Requires labels **±1**.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nu` | `1/m` | Regularisation (positive float; `None` or `0` → auto) |
| `tol` | `1e-5` | Convergence tolerance |
| `max_iter` | `100` | Maximum iterations |
| `alpha` | `1.9/nu` | Step size; must satisfy `0 < α < 2/ν` |
| `perturb` | `0.0` | Random perturbation scale |
| `normalize` | `False` | Column standardisation |
| `verbose` | `True` | Print iteration log |

Fitted attributes: `w_`, `gamma_`, `n_iter_`, `opt_cond_`.

---

### `LSVMK` — Lagrangian SVM (kernel)

Port of `lsvmk.m`.  Requires labels **±1**.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel` | `'rbf'` | `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'`, `'precomputed'`, or callable |
| `nu` | `1/m` | Regularisation |
| `gamma` | `None` | Kernel coefficient (auto if `None`) |
| `degree` | `3` | Polynomial degree |
| `coef0` | `1.0` | Kernel zero coefficient |
| `tol` | `1e-5` | Convergence tolerance |
| `max_iter` | `100` | Maximum iterations |
| `verbose` | `True` | Print iteration log |

Fitted attributes: `dual_coef_`, `d_`, `X_fit_`, `n_iter_`, `opt_cond_`.

---

### `SSVM` / `NSSVM` — Smooth SVM

`SSVM`: linear smooth SVM (Newton iterations).
`NSSVM`: nonlinear variant using a reduced RBF kernel map.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nu` | `'easy'` | Regularisation (`'easy'`=auto, `None`=auto, or positive float) |
| `use_armijo` | `True` | Armijo line-search |
| `tol` | `1e-5` | Convergence tolerance |
| `max_iter` | `200` | Maximum iterations |
| `random_state` | `None` | RNG seed |
| `verbose` | `False` | Print iteration log |
| `mu` *(NSSVM)* | `None` | RBF bandwidth |
| `reduce_rate` *(NSSVM)* | `0.5` | Fraction of training points used as kernel basis |

---

### `LSSVMClassifier` / `LSSVMRegressor` — Least Squares SVM

Port of LSSVMlab v1.8.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `1.0` | Regularisation |
| `kernel` | `'rbf'` | `'linear'`, `'rbf'`, `'poly'` |
| `sigma2` / `gamma` | `1.0` | RBF bandwidth / kernel coefficient |
| `degree` | `3` | Polynomial degree |
| `coef0` | `1.0` | Kernel zero coefficient |
| `preprocess` | `True` | Standardise features |

---

### `LapSVMClassifier` / `LapRLSCClassifier` — Laplacian SVM

Semi-supervised classifiers with graph manifold regularisation.
Unlabeled samples must be included in `X` during `fit` with a sentinel label
(`unlabeled_value`, default `2`).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kernel` | `'rbf'` | Ambient-space kernel |
| `kernel_param` | `1.0` | Kernel bandwidth / degree |
| `gamma_A` | `1.0` | Ambient regularisation |
| `gamma_I` | `1.0` | Manifold regularisation |
| `nn` | `6` | Nearest neighbours for graph |
| `dist_fn` | `'euclidean'` | Distance metric |
| `weights` | `'binary'` | Graph edge weights |
| `unlabeled_value` | `2` | Sentinel value for unlabeled samples |

---

### `PSVMClassifier` / `NPSVMClassifier` — Proximal SVM

Solves a linear system instead of a QP; no iterative solver required.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nu` | `0` | Regularisation (`0`=eigenvalue estimate, `-1`=Frobenius norm, positive=direct) |
| `balance` | `False` | Balance class weights by inverse frequency |
| `random_state` | `None` | RNG seed |
| `mu` *(NPSVM)* | `None` | RBF bandwidth |
| `reduce_ratio` *(NPSVM)* | `0.5` | Fraction of training points used as kernel basis |

---

### `SVMOCASClassifier` / `MSVMOCASClassifier` — OCAS SVM

Large-scale linear SVM via the OCAS cutting-plane solver (Franc & Sonnenburg, 2008).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `1.0` | Regularisation |
| `method` | `'ocas'` | `'ocas'` or `'cp'` (cutting-plane / BMRM) |
| `tol` | `1e-3` | Relative duality-gap tolerance |
| `buf_size` | `2000` | Maximum buffered cutting planes |
| `max_time` | `inf` | Wall-clock time budget (seconds) |
| `fit_intercept` | `True` | Fit bias term (binary only) |
| `verbose` | `False` | Print solver statistics |

Fitted attributes: `coef_`, `intercept_`, `classes_`, `n_iter_`, `train_time_`.

---

### `CVM` / `BVM` — Core / Ball Vector Machine

Kernel SVM via Minimum Enclosing Ball core-set approximation (Tsang et al., 2004).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `1.0` | Regularisation |
| `kernel` | `'rbf'` | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'`, `'exp'`, `'normal_poly'`, `'inv_dist'`, `'inv_sqdist'` |
| `gamma` | `None` | Kernel coefficient |
| `degree` | `3` | Polynomial degree |
| `coef0` | `0.0` | Kernel zero coefficient |
| `eps` | `1e-6` | Convergence tolerance |
| `max_sv` | `0` | Maximum support vectors (`0` = unlimited) |
| `cache_size` | `256` | Kernel cache (MB) |
| `verbose` | `False` | Print solver output |

`BVM` only supports isotropic kernels (`rbf`, `exp`, `normal_poly`, `inv_dist`, `inv_sqdist`).

---

### `BSVMClassifier` / `BSVMRegressor` — Bound-Constrained SVM

Bound-constrained SVM 2.09 with successive over-relaxation solver.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `1.0` | Regularisation |
| `kernel` | `'rbf'` | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| `gamma` | `'scale'` | `'scale'`, `'auto'`, or float |
| `degree` | `3` | Polynomial degree |
| `coef0` | `0.0` | Kernel zero coefficient |
| `tol` | `1e-3` | Convergence tolerance |
| `max_iter` | `1000` | Maximum iterations |
| `cache_size` | `256` | Kernel cache (MB) |
| `verbose` | `False` | Print solver output |

---

### `SVMLightClassifier` / `SVMLightRegressor` — SVM-Light

Wrapper around SVM-Light V6.02 (Joachims, 1999).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `1.0` | Regularisation |
| `kernel` | `'rbf'` | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| `gamma` | `'scale'` | `'scale'`, `'auto'`, or float |
| `degree` | `3` | Polynomial degree |
| `coef0` | `1.0` | Kernel zero coefficient |
| `epsilon_crit` | `1e-3` | Convergence criterion |
| `cache_size` | `40` | Kernel cache (MB) |
| `biased_hyperplane` | `True` | Fit bias term |
| `class_weight` | `None` | Per-class weight adjustment |
| `verbose` | `False` | Print solver output |

---

### `MySVMClassifier` / `MySVMRegressor` / `MySVMNuClassifier` / `MySVMNuRegressor` — mySVM

C-SVM and ν-SVM variants wrapping the mySVM library (Rüping, 2000).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | `1.0` | Regularisation (`MySVMClassifier` / `MySVMRegressor`) |
| `nu` | `0.5` | ν parameter (`MySVMNuClassifier` / `MySVMNuRegressor`) |
| `kernel` | `'rbf'` | `'linear'`, `'poly'`, `'rbf'`, `'sigmoid'` |
| `gamma` | `'scale'` | `'scale'`, `'auto'`, or float |
| `degree` | `3` | Polynomial degree |
| `coef0` | `0.0` | Kernel zero coefficient |
| `biased` | `True` | Fit bias term |
| `class_weight` | `None` | Per-class weight (`MySVMClassifier` / `MySVMNuClassifier`) |
| `convergence_epsilon` | `1e-3` | Convergence tolerance |
| `max_iter` | `1000` | Maximum iterations |
| `cache_size` | `256` | Kernel cache (MB) |
| `verbose` | `False` | Print solver output |

---

## scikit-learn compatibility

All estimators inherit from `BaseEstimator` and either `ClassifierMixin` or
`RegressorMixin`, providing the full sklearn interface:

```python
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cross-validation
from scikit_svm import LibLinearSVC
scores = cross_val_score(LibLinearSVC(C=1.0), X, y, cv=5)

# Grid search
grid = GridSearchCV(LibLinearSVC(), {"C": [0.1, 1, 10]}, cv=3)
grid.fit(X, y)

# Pipeline
pipe = Pipeline([("scaler", StandardScaler()), ("svm", LibLinearSVC())])
pipe.fit(X, y)

# Clone / get_params / set_params
clf = LibLinearSVC(C=5.0)
clf.set_params(solver=2)
clone(clf)
```

**Note**: `LSVM` and `LSVMK` require labels **exactly ±1**.  All other
classifiers accept arbitrary binary labels and perform encoding internally.

---

## Building extensions

The Cython-backed estimators (`SVMOCASClassifier`, `MSVMOCASClassifier`,
`CVM`, `BVM`, `BSVMClassifier`, `BSVMRegressor`, `SVMLightClassifier`,
`SVMLightRegressor`, `MySVMClassifier`, etc.) require compiling the bundled
C/C++ libraries:

```bash
# One-step install + compile
pip install -e ".[dev]"

# Or compile in-place without full install
python setup.py build_ext --inplace
```

A C/C++ compiler (GCC ≥ 9 or Clang) and Cython ≥ 3.0 are required.  The
`__init__.py` wraps each Cython import in `try/except ImportError` so the
package loads gracefully when extensions are not yet built.

---

## Running tests

```bash
pip install -e ".[dev]"
pytest               # full suite
pytest tests/test_liblinear.py -v   # single file
pytest tests/test_ocas.py::TestSVMOCASClassifier::test_high_accuracy_separable -v
```

---

## References

- Mangasarian, O. L., & Musicant, D. R. (2001). Lagrangian support vector machines. *JMLR*, 1, 161–177.
- Lee, Y.-J., & Mangasarian, O. L. (2001). SSVM: A smooth support vector machine for classification. *Computational Optimization and Applications*, 20(1), 5–22.
- Suykens, J. A. K., & Vandewalle, J. (1999). Least squares support vector machine classifiers. *Neural Processing Letters*, 9(3), 293–300.
- Melacci, S., & Belkin, M. (2011). Laplacian support vector machines trained in the primal. *JMLR*, 12, 1149–1184.
- Franc, V., & Sonnenburg, S. (2008). Optimized cutting plane algorithm for support vector machines. *ICML*.
- Tsang, I. W., Kwok, J. T., & Cheung, P.-M. (2005). Core vector machines: Fast SVM training on very large data sets. *JMLR*, 6, 363–392.
- Joachims, T. (1999). Making large-scale SVM learning practical. In *Advances in Kernel Methods* (pp. 169–184). MIT Press.
- Rüping, S. (2000). mySVM — another one of those support vector machines. [Software]. University of Dortmund.
- Fan, R.-E., Chang, K.-W., Hsieh, C.-J., Wang, X.-R., & Lin, C.-J. (2008). LIBLINEAR: A library for large linear classification. *JMLR*, 9, 1871–1874.

---

## License

Copyright © 2026 Paulius Danenas. See [LICENSE](LICENSE) for full terms.
