"""The native loader must fail with actionable guidance when _spout_bridge is absent.

In CI / on non-Windows machines the compiled extension does not exist; loading it
should raise a clear RuntimeError (not a bare ImportError), and the high-level API
should still default to requiring it only when no backend is injected.
"""

from __future__ import annotations

import pytest
from cuda_link_spout import SpoutSenderSpec
from cuda_link_spout._native import load_native_backend
from cuda_link_spout.sender import SpoutSender


def _native_module_present() -> bool:
    try:
        from cuda_link_spout import _spout_bridge  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(_native_module_present(), reason="native _spout_bridge is built")
def test_load_native_backend_raises_actionable_error():
    with pytest.raises(RuntimeError, match="native module"):
        load_native_backend(0)


@pytest.mark.skipif(_native_module_present(), reason="native _spout_bridge is built")
def test_sender_open_without_backend_surfaces_native_error():
    with pytest.raises(RuntimeError, match="native module"):
        SpoutSender.open(SpoutSenderSpec("o", 8, 8))  # no backend injected -> tries native
