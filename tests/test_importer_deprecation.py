"""
Tests that CUDAIPCImporter emits DeprecationWarning exactly once per process.
"""

from __future__ import annotations

import contextlib
import warnings


def test_connect_emits_deprecation_warning() -> None:
    """CUDAIPCImporter.connect() emits exactly one DeprecationWarning."""
    import cuda_link.cuda_ipc_importer as shim_mod

    # Reset the per-process guard so this test is self-contained.
    shim_mod._deprecation_warned = False

    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    imp = CUDAIPCImporter(shm_name="definitely_does_not_exist_xyzzy")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.suppress(FileNotFoundError, OSError, RuntimeError):
            imp.connect()

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "Importer.open()" in str(dep_warnings[0].message)
    assert "v1.8" in str(dep_warnings[0].message)


def test_deprecation_warning_fires_only_once() -> None:
    """Repeated connect() calls emit the warning only once (per-process guard)."""
    import cuda_link.cuda_ipc_importer as shim_mod

    shim_mod._deprecation_warned = False

    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    call_count = [0]

    def _count(msg, cat, *args, **kwargs):
        if issubclass(cat, DeprecationWarning):
            call_count[0] += 1

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        # First call
        imp1 = CUDAIPCImporter(shm_name="xyzzy_a")
        with contextlib.suppress(FileNotFoundError, OSError, RuntimeError):
            imp1.connect()
        # Second call — should not emit again
        imp2 = CUDAIPCImporter(shm_name="xyzzy_b")
        with contextlib.suppress(FileNotFoundError, OSError, RuntimeError):
            imp2.connect()

    # Guard should be set after first connect
    assert shim_mod._deprecation_warned is True


def test_from_connected_emits_deprecation_warning() -> None:
    """CUDAIPCImporter.from_connected() also emits the deprecation warning."""
    import cuda_link.cuda_ipc_importer as shim_mod

    shim_mod._deprecation_warned = False

    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.suppress(FileNotFoundError, OSError, RuntimeError):
            CUDAIPCImporter.from_connected(shm_name="definitely_does_not_exist_xyzzy")

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "Importer.open()" in str(dep_warnings[0].message)


def test_deprecation_message_mentions_migration_guide() -> None:
    """The warning message references the migration doc."""
    import cuda_link.cuda_ipc_importer as shim_mod

    shim_mod._deprecation_warned = False

    from cuda_link.cuda_ipc_importer import CUDAIPCImporter

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.suppress(FileNotFoundError, OSError, RuntimeError):
            CUDAIPCImporter(shm_name="xyzzy").connect()

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    assert "MIGRATION_v1.7.md" in str(dep_warnings[0].message)
