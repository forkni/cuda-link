"""The one table of cudart signatures (cuda_ipc_wrapper._CUDART_SIGNATURES)."""

from __future__ import annotations

from ctypes import c_int

from cuda_link.cuda_ipc_wrapper import _CUDART_SIGNATURES

# The functions whose non-zero return is not an error to raise on (see the table comment).
_NOT_STRICT = {
    "cudaGetErrorString",
    "cudaEventQuery",
    "cudaStreamQuery",
    "cudaGetLastError",
    "cudaPeekAtLastError",
}


def test_each_function_is_listed_once() -> None:
    names = [name for name, _argtypes, _restype, _strict in _CUDART_SIGNATURES]

    assert len(names) == len(set(names))


def test_the_not_strict_set_is_exactly_the_documented_one() -> None:
    not_strict = {name for name, _argtypes, _restype, strict in _CUDART_SIGNATURES if not strict}

    assert not_strict == _NOT_STRICT


def test_a_strict_function_returns_a_cuda_error_code() -> None:
    """errcheck compares the return value with 0, so a strict row must return cudaError_t."""
    wrong = [name for name, _argtypes, restype, strict in _CUDART_SIGNATURES if strict and restype is not c_int]

    assert wrong == []
