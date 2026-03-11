"""Test to verify that the CUDA wrapper is synchronized between TD and pip package.

The CUDA wrapper is intentionally duplicated:
- td_exporter/CUDAIPCWrapper.py (TD-side, loaded as Text DAT)
- src/cuda_link/cuda_ipc_wrapper.py (pip package, loaded as Python module)

This test ensures they remain byte-for-byte identical.
"""

from pathlib import Path


def test_wrapper_files_are_identical() -> None:
    """Verify TD and pip package wrappers are identical."""
    project_root = Path(__file__).parent.parent

    td_wrapper = project_root / "td_exporter" / "CUDAIPCWrapper.py"
    pip_wrapper = project_root / "src" / "cuda_link" / "cuda_ipc_wrapper.py"

    assert td_wrapper.exists(), f"TD wrapper not found: {td_wrapper}"
    assert pip_wrapper.exists(), f"Pip wrapper not found: {pip_wrapper}"

    td_content = td_wrapper.read_text(encoding="utf-8")
    pip_content = pip_wrapper.read_text(encoding="utf-8")

    assert td_content == pip_content, (
        "CUDA wrapper files are out of sync!\n"
        f"TD wrapper: {td_wrapper} ({len(td_content)} chars)\n"
        f"Pip wrapper: {pip_wrapper} ({len(pip_content)} chars)\n"
        "\n"
        "These files must be identical. If you modified one, copy it to the other:\n"
        f"  copy {td_wrapper} {pip_wrapper}\n"
        "or\n"
        f"  copy {pip_wrapper} {td_wrapper}\n"
    )


def test_wrapper_line_count() -> None:
    """Verify wrapper is the expected size (606 lines)."""
    project_root = Path(__file__).parent.parent
    pip_wrapper = project_root / "src" / "cuda_link" / "cuda_ipc_wrapper.py"

    content = pip_wrapper.read_text(encoding="utf-8")
    line_count = len(content.splitlines())

    # Allow some flexibility (690-730 lines) for minor changes
    assert 690 <= line_count <= 730, (
        f"Wrapper line count ({line_count}) is outside expected range (690-730). "
        "Has the wrapper changed significantly? Update this test if intentional."
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
    assert "def memcpy_async(" in content
    assert "def ipc_get_mem_handle(" in content
    assert "def ipc_open_mem_handle(" in content
    assert "def ipc_close_mem_handle(" in content
    assert "def record_event(" in content
    assert "def stream_wait_event(" in content
