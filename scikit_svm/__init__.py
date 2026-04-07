"""
scikit-svm: Lagrangian, Smooth, Core-Vector, Ball-Vector and Bound-Constrained
SVM classifiers with a scikit-learn compatible interface.

Pure-Python classifiers (LSVM, LSVMK, SSVM, NSSVM) are ports of MATLAB code
by Olvi L. Mangasarian and David R. Musicant, University of Wisconsin-Madison,
2000.

Cython-backed classifiers (CVM, BVM) wrap the libCVM C++ library by Ivor W.
Tsang, Andras Kocsor and James T. Kwok (LibCVM Toolkit v2.2).

Cython-backed classifiers (BSVMClassifier, BSVMRegressor) wrap the BSVM 2.09
C++ library (Bound-Constrained SVM).
"""

from .lsvm  import LSVM
from .lsvmk import LSVMK
from .ssvm  import SSVM
from .nssvm import NSSVM
from .lssvm  import LSSVMClassifier, LSSVMRegressor
from .lapsvm import LapSVMClassifier, LapRLSCClassifier
from .psvm  import PSVMClassifier, NPSVMClassifier

try:
    from .ocas import SVMOCASClassifier, MSVMOCASClassifier
    _HAS_LIBOCAS = True
except ImportError:
    _HAS_LIBOCAS = False

try:
    from .cvm import CVM
    from .bvm import BVM
    _HAS_LIBCVM = True
except ImportError:
    _HAS_LIBCVM = False

try:
    from .bsvm import BSVMClassifier, BSVMRegressor
    _HAS_LIBBSVM = True
except ImportError:
    _HAS_LIBBSVM = False

try:
    from .svmlight import SVMLightClassifier, SVMLightRegressor
    _HAS_LIBSVMLIGHT = True
except ImportError:
    _HAS_LIBSVMLIGHT = False

try:
    from .mysvm import (MySVMClassifier, MySVMRegressor,
                        MySVMNuClassifier, MySVMNuRegressor)
    _HAS_LIBMYSVM = True
except ImportError:
    _HAS_LIBMYSVM = False

try:
    from .liblinear import LibLinearSVC, LibLinearSVR
    _HAS_LIBLINEAR = True
except ImportError:
    _HAS_LIBLINEAR = False

__version__ = "1.0.0"
__all__ = ["LSVM", "LSVMK", "SSVM", "NSSVM",
           "LSSVMClassifier", "LSSVMRegressor",
           "LapSVMClassifier", "LapRLSCClassifier",
           "PSVMClassifier", "NPSVMClassifier",
           "SVMOCASClassifier", "MSVMOCASClassifier",
           "CVM", "BVM",
           "BSVMClassifier", "BSVMRegressor",
           "SVMLightClassifier", "SVMLightRegressor",
           "MySVMClassifier", "MySVMRegressor",
           "MySVMNuClassifier", "MySVMNuRegressor",
           "LibLinearSVC", "LibLinearSVR"]
