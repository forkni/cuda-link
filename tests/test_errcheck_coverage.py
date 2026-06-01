"""
Test that errcheck is installed on every cudart function whose restype is c_int,
and that the explicit errcheck list (_strict_funcs inside _install_errcheck) stays
in sync with the argtypes block (_setup_function_signatures).

Background
----------
_install_errcheck (cuda_ipc_wrapper.py) attaches _strict_errcheck to a
hand-maintained tuple of 44 function names.  The argtypes block registers ~50
functions.  If a developer adds a new cudart function to _setup_function_signatures
but forgets to add it to _strict_funcs (or vice versa), CUDA errors from that
function are silently swallowed.  This test converts that drift risk into a CI
failure.

These tests are GPU-free: they use object.__new__ + MagicMock to exercise the
signature/errcheck setup logic without loading an actual CUDA DLL.
"""

from __future__ import annotations

from ctypes import c_int
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_api() -> object:
    """Return a CUDARuntimeAPI with a real MagicMock cudart and full setup run."""
    from tests.fakes import make_bare_runtime_api

    return make_bare_runtime_api(install_errcheck=True)


# ---------------------------------------------------------------------------
# Documented exemptions — functions whose non-zero return is NOT always fatal.
# Keep this in sync with the comment at the top of _install_errcheck.
# ---------------------------------------------------------------------------

_ERRCHECK_EXEMPTIONS = frozenset(
    {
        "cudaGetErrorString",  # restype is c_char_p, not c_int
        "cudaEventQuery",  # cudaErrorNotReady (600) is a valid non-error sentinel
        "cudaStreamQuery",  # cudaErrorNotReady (600) is a valid non-error sentinel
        "cudaPeekAtLastError",  # reads the sticky error; raising would defeat the purpose
        # cudaGetLastError is intentionally NOT bound (see test_cudaGetLastError_not_bound).
    }
)


# ---------------------------------------------------------------------------
# Main drift-prevention test
# ---------------------------------------------------------------------------


def test_every_c_int_cudart_func_has_errcheck() -> None:
    """All c_int-restype cudart functions must have a real errcheck callable installed.

    The test enumerates every MagicMock child created on api.cudart during
    _setup_function_signatures.  For each child whose restype was explicitly set
    to c_int (stored in child.__dict__['restype']), it asserts that errcheck was
    also set to a real callable (not a MagicMock default) — unless the function
    appears in _ERRCHECK_EXEMPTIONS.
    """
    api = _make_api()

    missing_errcheck: list[str] = []
    unexpected_errcheck: list[str] = []

    for name, child in api.cudart._mock_children.items():
        if name.startswith("_"):
            continue
        # Check via __dict__ so we distinguish an explicit assignment (c_int) from
        # the lazy MagicMock child that __getattr__ would produce.
        restype = child.__dict__.get("restype")
        if restype is not c_int:
            continue

        errcheck = child.__dict__.get("errcheck")
        is_real_callable = errcheck is not None and not isinstance(errcheck, MagicMock)

        if name in _ERRCHECK_EXEMPTIONS:
            if is_real_callable:
                unexpected_errcheck.append(name)
        else:
            if not is_real_callable:
                missing_errcheck.append(name)

    assert not missing_errcheck, (
        "These c_int-restype cudart functions are missing errcheck:\n"
        + "".join(f"  {n}\n" for n in sorted(missing_errcheck))
        + "Add them to _strict_funcs inside _install_errcheck, "
        "or document why they are exempt and add them to _ERRCHECK_EXEMPTIONS here."
    )
    assert not unexpected_errcheck, (
        "These _ERRCHECK_EXEMPTIONS functions unexpectedly have errcheck set:\n"
        + "".join(f"  {n}\n" for n in sorted(unexpected_errcheck))
        + "Either remove them from _ERRCHECK_EXEMPTIONS or from _strict_funcs."
    )


def test_exemptions_do_not_have_errcheck() -> None:
    """Each function in _ERRCHECK_EXEMPTIONS must NOT have errcheck installed."""
    api = _make_api()

    has_errcheck: list[str] = []
    for name in _ERRCHECK_EXEMPTIONS:
        child = api.cudart._mock_children.get(name)
        if child is None:
            continue  # not even bound — also fine
        errcheck = child.__dict__.get("errcheck")
        if errcheck is not None and not isinstance(errcheck, MagicMock):
            has_errcheck.append(name)

    assert not has_errcheck, "Exempted functions that unexpectedly have errcheck:\n" + "".join(
        f"  {n}\n" for n in sorted(has_errcheck)
    )


def test_every_bound_func_has_argtypes() -> None:
    """Every explicitly-bound cudart function must have argtypes set.

    Background
    ----------
    The ctypes reference manual stresses that *always* specifying argtypes is
    critical on 64-bit platforms: without it, a pointer argument is silently
    truncated to C int (32 bits), producing silent data corruption rather than
    a clear error.  This test catches the "added restype but forgot argtypes"
    mistake before it reaches production.

    A function is considered explicitly bound when its restype was assigned
    (stored in child.__dict__['restype']), which is the same signal used by the
    errcheck coverage test.  We then require that argtypes was also explicitly
    set to a non-MagicMock list/tuple.
    """
    api = _make_api()

    missing_argtypes: list[str] = []

    for name, child in api.cudart._mock_children.items():
        if name.startswith("_"):
            continue
        # Only check functions that were explicitly bound (restype was assigned).
        # Unbound functions have no __dict__ restype entry.
        restype = child.__dict__.get("restype")
        if restype is None:
            continue  # never explicitly set — not a bound function

        argtypes = child.__dict__.get("argtypes")
        is_real_argtypes = argtypes is not None and not isinstance(argtypes, MagicMock)

        if not is_real_argtypes:
            missing_argtypes.append(name)

    assert not missing_argtypes, (
        "These explicitly-bound cudart functions are missing argtypes:\n"
        + "".join(f"  {n}\n" for n in sorted(missing_argtypes))
        + "Add argtypes to _setup_function_signatures for each listed function.  "
        "Without argtypes, pointer arguments are silently truncated to 32-bit int "
        "on 64-bit platforms (ctypes reference, 'Specifying the required argument types')."
    )


def test_every_errcheck_func_is_typed() -> None:
    """Reverse of test_every_c_int_cudart_func_has_errcheck.

    Every cudart function that received a real errcheck (i.e. appears in
    _strict_funcs inside _install_errcheck) must ALSO have been typed with
    restype=c_int in _setup_function_signatures.

    An errcheck'd-but-untyped function runs with default argtypes — pointer
    arguments are silently truncated to 32-bit on 64-bit platforms (the exact
    hazard the forward test guards against, only mirror-imaged).  This closes
    the reverse drift path: adding a name to _strict_funcs without first adding
    it to the signature table.
    """
    api = _make_api()

    errcheck_without_restype: list[str] = []
    for name, child in api.cudart._mock_children.items():
        if name.startswith("_"):
            continue
        errcheck = child.__dict__.get("errcheck")
        is_real_callable = errcheck is not None and not isinstance(errcheck, MagicMock)
        if not is_real_callable:
            continue
        if child.__dict__.get("restype") is not c_int:
            errcheck_without_restype.append(name)

    assert not errcheck_without_restype, (
        "These _strict_funcs entries have errcheck installed but were never typed "
        "(restype=c_int) in _setup_function_signatures:\n"
        + "".join(f"  {n}\n" for n in sorted(errcheck_without_restype))
        + "Add argtypes+restype=c_int to _setup_function_signatures for each, or "
        "remove them from _strict_funcs in _install_errcheck."
    )


def test_cudaGetLastError_not_bound() -> None:
    """cudaGetLastError must NOT be bound on cudart.

    It was intentionally removed as a dead binding: it destructively clears
    the sticky CUDA error, making it unsafe to call in non-fatal poll paths.
    cudaPeekAtLastError (non-destructive) is the correct API for sticky-error
    reads.  If someone re-adds cudaGetLastError, the sticky-error check logic
    in check_sticky_error / peek_last_error needs to be revisited.
    """
    api = _make_api()
    assert "cudaGetLastError" not in api.cudart._mock_children, (
        "cudaGetLastError was re-bound on cudart.  It was intentionally removed as a "
        "dead binding (never called).  Use cudaPeekAtLastError for non-destructive "
        "sticky-error reads, or explicitly document why cudaGetLastError is needed."
    )
