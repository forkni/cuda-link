"""Test to verify that TD-exporter copies stay in sync with canonical sources.

Duplicated pairs (byte-identical):
  src/cuda_link/cuda_ipc_wrapper.py  <-> td_exporter/CUDAIPCWrapper.py
  src/cuda_link/nvml_observer.py     <-> td_exporter/NVMLObserver.py

Run scripts/sync_td_wrapper.py to regenerate the TD-exporter copies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent

_SYNC_PAIRS = [
    (
        _PROJECT_ROOT / "src" / "cuda_link" / "cuda_ipc_wrapper.py",
        _PROJECT_ROOT / "td_exporter" / "CUDAIPCWrapper.py",
    ),
    (
        _PROJECT_ROOT / "src" / "cuda_link" / "nvml_observer.py",
        _PROJECT_ROOT / "td_exporter" / "NVMLObserver.py",
    ),
]


@pytest.mark.parametrize("canonical,derived", _SYNC_PAIRS, ids=["CUDAIPCWrapper", "NVMLObserver"])
def test_td_exporter_file_is_identical(canonical: Path, derived: Path) -> None:
    """Verify each TD-exporter copy is byte-identical to its canonical source."""
    assert canonical.exists(), f"Canonical source not found: {canonical}"
    assert derived.exists(), f"Derived copy not found: {derived}\nRun: python scripts/sync_td_wrapper.py"

    canonical_content = canonical.read_text(encoding="utf-8")
    derived_content = derived.read_text(encoding="utf-8")

    assert canonical_content == derived_content, (
        f"{derived.name} is out of sync with {canonical.name}!\n"
        f"  canonical: {canonical} ({len(canonical_content)} chars)\n"
        f"  derived:   {derived} ({len(derived_content)} chars)\n"
        "\n"
        "Run: python scripts/sync_td_wrapper.py"
    )


def test_wrapper_contains_key_definitions() -> None:
    """Verify wrapper contains expected CUDA definitions."""
    project_root = Path(__file__).parent.parent
    pip_wrapper = project_root / "src" / "cuda_link" / "cuda_ipc_wrapper.py"

    content = pip_wrapper.read_text(encoding="utf-8")

    # Check for key structures and classes
    assert "class cudaIpcMemHandle_t(ctypes.Structure):" in content
    assert "class cudaIpcEventHandle_t(ctypes.Structure):" in content
    assert "class CUDARuntimeAPI:" in content
    assert "def get_cuda_runtime(" in content

    # Check for key methods
    assert "def malloc(" in content
    assert "def free(" in content
    assert "def malloc_host(" in content
    assert "def free_host(" in content
    assert "def memcpy_async(" in content
    assert "def ipc_get_mem_handle(" in content
    assert "def ipc_open_mem_handle(" in content
    assert "def ipc_close_mem_handle(" in content
    assert "def record_event(" in content
    assert "def stream_wait_event(" in content
