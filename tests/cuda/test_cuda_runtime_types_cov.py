"""Coverage-focused tests for CUDAError.get_name() in cuda_link.cuda_runtime_types.

test_cuda_runtime_types.py only covers _abi_guard(); this fills the
CUDAError.get_name() gap. Pure lookup logic — no GPU required.
"""

from __future__ import annotations

from cuda_link.cuda_runtime_types import CUDAError


def test_get_name_known_codes():
    assert CUDAError.get_name(CUDAError.SUCCESS) == "SUCCESS"
    assert CUDAError.get_name(CUDAError.INVALID_VALUE) == "INVALID_VALUE"
    assert CUDAError.get_name(CUDAError.MEMORY_ALLOCATION) == "MEMORY_ALLOCATION"
    assert CUDAError.get_name(CUDAError.INVALID_DEVICE_POINTER) == "INVALID_DEVICE_POINTER"
    assert CUDAError.get_name(CUDAError.INVALID_DEVICE) == "INVALID_DEVICE"
    assert CUDAError.get_name(CUDAError.INVALID_CONTEXT) == "INVALID_CONTEXT"
    assert CUDAError.get_name(CUDAError.NOT_READY) == "NOT_READY"
    assert CUDAError.get_name(CUDAError.PEER_ACCESS_ALREADY_ENABLED) == "PEER_ACCESS_ALREADY_ENABLED"


def test_get_name_unknown_code_falls_back_to_generic_label():
    assert CUDAError.get_name(9999) == "UNKNOWN_ERROR_9999"
    assert CUDAError.get_name(-1) == "UNKNOWN_ERROR_-1"
