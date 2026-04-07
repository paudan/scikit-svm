"""
Internal utilities shared across the scikit-svm package.
"""

import contextlib
import io
import os
import sys


@contextlib.contextmanager
def _suppress_c_stdout():
    """
    Context manager that redirects C-level stdout to /dev/null.

    Used to silence libCVM's internal ``info()`` print calls when the
    caller sets ``verbose=False``.  Falls back silently when stdout has no
    real file descriptor (e.g. in some notebook / test environments).
    """
    sys.stdout.flush()
    try:
        fd = sys.stdout.fileno()
    except (AttributeError, io.UnsupportedOperation):
        yield
        return

    old_fd = os.dup(fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, fd)
    os.close(devnull)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(old_fd, fd)
        os.close(old_fd)
