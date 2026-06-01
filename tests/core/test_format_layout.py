"""Tests for Format.layout_differs_from().

Verifies that the method correctly detects layout changes while ignoring the
sentinel-zero kind/bits/flags fields set by Format.from_overrides().
"""

from cuda_link.importer import Format


def _shm_fmt(shape=(480, 640, 4), dtype_str="float32", kind=4, bits=32, flags=0) -> Format:
    """Helper: build a Format as if parsed from SHM (kind/bits/flags non-zero)."""
    h, w, c = shape
    return Format(
        width=w,
        height=h,
        num_comps=c,
        kind=kind,
        bits=bits,
        flags=flags,
        dtype_str=dtype_str,
        shape=shape,
        numpy_dtype=None,
        frame_nbytes=h * w * c * 4,
    )


def _override_fmt(shape=(480, 640, 4), dtype_str="float32") -> Format:
    """Helper: build a Format via from_overrides path (kind=bits=flags=0)."""
    return Format.from_overrides(shape, dtype_str)


class TestLayoutDiffersFrom:
    def test_identical_shmfmt_returns_false(self):
        a = _shm_fmt()
        b = _shm_fmt()
        assert not a.layout_differs_from(b)

    def test_identical_override_fmt_returns_false(self):
        a = _override_fmt()
        b = _override_fmt()
        assert not a.layout_differs_from(b)

    def test_shape_change_detected(self):
        a = _shm_fmt(shape=(480, 640, 4))
        b = _shm_fmt(shape=(1080, 1920, 4))
        assert a.layout_differs_from(b)
        assert b.layout_differs_from(a)

    def test_dtype_change_detected(self):
        a = _shm_fmt(dtype_str="float32")
        b = _shm_fmt(dtype_str="float16")
        assert a.layout_differs_from(b)

    def test_num_comps_change_detected(self):
        a = _shm_fmt(shape=(480, 640, 4))
        b = _shm_fmt(shape=(480, 640, 3))
        assert a.layout_differs_from(b)

    def test_sentinel_zeros_do_not_affect_result(self):
        """from_overrides sets kind=bits=flags=0; this must NOT false-positive against
        an SHM-derived Format that has real kind/bits/flags but the same shape+dtype."""
        override = _override_fmt(shape=(480, 640, 4), dtype_str="float32")
        shm = _shm_fmt(shape=(480, 640, 4), dtype_str="float32", kind=4, bits=32, flags=0)
        # Same shape + dtype → no layout change, even though kind/bits differ
        assert not override.layout_differs_from(shm)
        assert not shm.layout_differs_from(override)

    def test_format_eq_would_false_positive(self):
        """Document why layout_differs_from exists: __eq__ compares ALL fields.
        An override-derived Format (kind=bits=flags=0) != SHM-derived (real values),
        even when the layout is identical."""
        override = _override_fmt(shape=(480, 640, 4), dtype_str="float32")
        shm = _shm_fmt(shape=(480, 640, 4), dtype_str="float32", kind=4, bits=32, flags=0)
        # __eq__ false-positives on kind/bits mismatch
        assert override != shm
        # layout_differs_from correctly returns False (same layout)
        assert not override.layout_differs_from(shm)

    def test_reflexive(self):
        fmt = _shm_fmt()
        assert not fmt.layout_differs_from(fmt)

    def test_dtype_and_shape_both_change(self):
        a = _shm_fmt(shape=(480, 640, 4), dtype_str="float32")
        b = _shm_fmt(shape=(1080, 1920, 3), dtype_str="float16")
        assert a.layout_differs_from(b)
