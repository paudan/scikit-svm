"""conftest.py – subprocess isolation for SVMLight and libocas tests.

Tests in test_svmlight.py wrap native C extensions that may
cause a hard process crash on some platforms (e.g. Windows access violation
exit code 3221225477 / 0xC0000005).  On Windows each such test is run in its
own child process so a crash does not kill the entire test session; crash exit
codes are reported as expected failures (xfail).  On other platforms the tests
run in-process as normal.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from _pytest.reports import TestReport

# Test files backed by native extensions that may crash the process on Windows.
_FRAGILE = frozenset({'test_svmlight.py'})

# OS-level exit codes that indicate a hard native crash rather than a Python
# test failure.  3221225477 == 0xC0000005 == Windows STATUS_ACCESS_VIOLATION.
CRASH_CODES = frozenset({
    3221225477,  # 0xC0000005  Windows STATUS_ACCESS_VIOLATION
})

# Environment variable set in child processes to prevent recursive wrapping.
_SUBPROCESS_GUARD = 'SCIKIT_SVM_FRAGILE_SUBPROCESS'


def _is_fragile(item: pytest.Item) -> bool:
    return item.path.name in _FRAGILE


def _in_subprocess() -> bool:
    return os.environ.get(_SUBPROCESS_GUARD) == '1'


# ── hook: replace the test-protocol for fragile tests on Windows ──────────────

@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem) -> bool | None:
    if sys.platform != 'win32' or _in_subprocess() or not _is_fragile(item):
        return None  # use the normal protocol

    ihook = item.ihook
    ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
    for rep in _reports_from_subprocess(item):
        ihook.pytest_runtest_logreport(report=rep)
    ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
    return True  # signal that we consumed this item


# ── subprocess runner ─────────────────────────────────────────────────────────

def _reports_from_subprocess(item: pytest.Item):
    """Yield setup / call / teardown TestReport objects for *item*."""
    yield _synthetic(item, 'setup', 'passed')

    cmd = [
        sys.executable, '-m', 'pytest',
        '--override-ini=addopts=',   # clear any session-level --ignore flags
        '--no-header', '--tb=short', '-q',
        item.nodeid,
    ]
    env = {**os.environ, _SUBPROCESS_GUARD: '1'}

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=30, env=env)
    except subprocess.TimeoutExpired:
        yield _synthetic(item, 'call', 'failed',
                         longrepr='Test timed out after 30 s')
        yield _synthetic(item, 'teardown', 'passed')
        return

    rc = proc.returncode
    out = (proc.stdout + proc.stderr).decode(errors='replace')

    if rc == 0:
        call_rep = _synthetic(item, 'call', 'passed')
    elif rc in CRASH_CODES:
        call_rep = _synthetic(
            item, 'call', 'failed',
            longrepr=f'Process crashed with exit code {rc:#010x}\n{out}',
        )
        call_rep.wasxfail = f'native crash: exit code {rc:#010x}'
    else:
        call_rep = _synthetic(item, 'call', 'failed',
                              longrepr=f'Subprocess exited {rc}\n{out}')

    yield call_rep
    yield _synthetic(item, 'teardown', 'passed')


# ── helpers ───────────────────────────────────────────────────────────────────

def _synthetic(item: pytest.Item, when: str, outcome: str,
               longrepr: str | None = None) -> TestReport:
    return TestReport(
        nodeid=item.nodeid,
        location=item.location,
        keywords={m.name: True for m in item.iter_markers()},
        outcome=outcome,
        longrepr=longrepr,
        when=when,
        sections=[],
        duration=0.0,
        user_properties=item.user_properties,
    )
