"""
setup.py – Cython extension build configuration for scikit-svm.

Build:
    pip install -e .                        # editable install (rebuilds .so)
    python setup.py build_ext --inplace     # build .so in-place only
"""

import os
import sys
import numpy as np

# Change to the project root so all relative source paths resolve correctly
# (required when setup.py is invoked from pip's isolated build environment)
# os.chdir(os.path.dirname(os.path.abspath(__file__)))
from Cython.Build import cythonize
from setuptools import Extension, setup

# Define your flags for Linux/macOS (GCC/Clang)
unix_cflags = [
    "-O3", 
    "-fPIC", 
    "-Wno-conversion", 
    "-Wno-implicit-function-declaration", 
    "-Wno-write-strings",
    "-fpermissive",
    "-fcommon"
]
unix_cppflags = [
    "-std=c++11", 
]

# Define equivalent flags for Windows (MSVC)
# Note: MSVC handles PIC automatically and uses different warning suppression
msvc_cflags = [
    "/Ox",            # Optimization
    "/TP",            # Force compile as C++ (for extensions with mixed C/C++ sources)
    "/wd4244",        # Disable conversion warnings (equiv to -Wno-conversion)
    "/wd4047",        # Disable pointer level warnings
    "/D_CRT_SECURE_NO_WARNINGS"
]

# For pure-C extensions: do NOT use /TP (it breaks legacy C sources when compiled as C++).
# Instead, use /std:c17 so MSVC accepts C99 features in Cython-generated code
# (C99 for-loop declarations, const locals with non-constant initializers, etc.).
msvc_cflags_c = [
    "/Ox",
    "/std:c17",       # C17 mode: allows C99 declarations used by Cython-generated C code
    "/wd4244",
    "/wd4047",
    "/D_CRT_SECURE_NO_WARNINGS"
]

def get_compile_args(use_cpp=True):
    if sys.platform == "win32":
        return msvc_cflags if use_cpp else msvc_cflags_c
    flags = unix_cflags
    if use_cpp:
        flags += unix_cppflags
    return flags

# ── libCVM extension ──────────────────────────────────────────────────────────
# All paths must be relative to the project root (setuptools manifest requirement).
_libcvm_cpp_sources = [
    os.path.join("src", "libCVM", name)
    for name in (
        "svm.cpp",
        "cvm.cpp",
        "cvm_core.cpp",
        "bvm.cpp",
        "utility.cpp",
        "sgraph.cpp",
        "random.cpp",
    )
]

ext_libcvm = Extension(
    name="scikit_svm._libcvm",
    sources=[os.path.join("scikit_svm", "_libcvm.pyx")] + _libcvm_cpp_sources,
    include_dirs=[os.path.join("src", "libCVM"), np.get_include()],
    extra_compile_args=get_compile_args(),
    extra_link_args=["-lm"],
    language="c++",
)

# ── BSVM extension ────────────────────────────────────────────────────────────
# C/C++ sources live in src/bsvm/, dtron/ and f2c/ sub-directories.
# Headers are also in src/bsvm/ (svm.h), src/bsvm/f2c/ and src/bsvm/dtron/.
_bsvm_base = os.path.join("src", "bsvm")
_bsvm_cpp_sources = [
    os.path.join(_bsvm_base, "bsvm.cpp"),
    os.path.join(_bsvm_base, "solvebqp.c"),
]
_bsvm_dtron_sources = [
    os.path.join(_bsvm_base, "dtron", name)
    for name in (
        "dtron.c",
        "dcauchy.c",
        "dgpnrm.c",
        "dgpstep.c",
        "dprecond.c",
        "dprsrch.c",
        "dspcg.c",
        "dbreakpt.c",
        "dtrpcg.c",
        "dtrqsol.c",
        "misc.c",
    )
]
_bsvm_f2c_sources = [
    os.path.join(_bsvm_base, "f2c", name)
    for name in (
        "dasum.c",
        "daxpy.c",
        "dcopy.c",
        "ddot.c",
        "dgemv.c",
        "dnrm2.c",
        "dpotf2.c",
        "dscal.c",
        "dsymv.c",
        "dtrsv.c",
        "lsame.c",
        "xerbla.c",
    )
]

ext_libbsvm = Extension(
    name="scikit_svm._libbsvm",
    sources=(
        [os.path.join("scikit_svm", "_libbsvm.pyx")]
        + _bsvm_cpp_sources
        + _bsvm_dtron_sources
        + _bsvm_f2c_sources
    ),
    include_dirs=[
        _bsvm_base,
        os.path.join(_bsvm_base, "f2c"),
        os.path.join(_bsvm_base, "dtron"),
        np.get_include(),
    ],
    extra_compile_args=get_compile_args(),
    extra_link_args=["-lm"],
    language="c++",
)

# ── SVM-Light extension ───────────────────────────────────────────────────────
# Sources: svm_common.c, svm_learn.c, svm_hideo.c (QP solver).
_svmlight_base = os.path.join("src", "svm_light")
_svmlight_sources = [
    os.path.join(_svmlight_base, name)
    for name in (
        "svm_common.c",
        "svm_learn.c",
        "svm_hideo.c",
    )
]

ext_libsvmlight = Extension(
    name="scikit_svm._libsvmlight",
    sources=[os.path.join("scikit_svm", "_libsvmlight.pyx")] + _svmlight_sources,
    include_dirs=[_svmlight_base, np.get_include()],
    extra_compile_args=get_compile_args(use_cpp=False),
    extra_link_args=["-lm"],
    language="c",
)

# ── mySVM extension ───────────────────────────────────────────────────────
# Sources: mysvm_wrapper.cpp + mySVM core C++ files.
_mysvm_base = os.path.join("src", "mySVM")
_mysvm_sources = [
    os.path.join(_mysvm_base, "mysvm_wrapper.cpp"),
    os.path.join(_mysvm_base, "smo.cpp"),
    os.path.join(_mysvm_base, "svm_nu.cpp"),
    os.path.join(_mysvm_base, "svm_c.cpp"),
    os.path.join(_mysvm_base, "globals.cpp"),
    os.path.join(_mysvm_base, "example_set.cpp"),
    os.path.join(_mysvm_base, "parameters.cpp"),
    os.path.join(_mysvm_base, "kernel.cpp"),
]

ext_libmysvm = Extension(
    name="scikit_svm._libmysvm",
    sources=[os.path.join("scikit_svm", "_libmysvm.pyx")] + _mysvm_sources,
    include_dirs=[_mysvm_base, np.get_include()],
    extra_compile_args=get_compile_args(),
    extra_link_args=["-lm"],
    language="c++",
)

# ── libocas extension ─────────────────────────────────────────────────────────
# Pure C sources: libocas solver, inner QP solver, and our callback wrapper.
_libocas_base = os.path.join("src", "libocas")

_libocas_sources = [
    os.path.join("scikit_svm", "_libocas.pyx"),
    os.path.join(_libocas_base, "libocas.c"),
    os.path.join(_libocas_base, "libqp_splx.c"),
    os.path.join(_libocas_base, "ocas_wrapper.c"),
]

ext_libocas = Extension(
    name="scikit_svm._libocas",
    sources=_libocas_sources,
    include_dirs=[_libocas_base, np.get_include()],
    extra_compile_args=get_compile_args(use_cpp=False),
    extra_link_args=["-lm"],
    language="c",
)

setup(
    ext_modules=cythonize(
        [ext_libcvm, ext_libbsvm, ext_libsvmlight, ext_libmysvm, ext_libocas],
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound":  False,
            "cdivision":   True,
        },
    ),
)
