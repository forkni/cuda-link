"""Unit tests for cuda_link._env helpers."""

from __future__ import annotations

import pytest

from cuda_link._env import env_bool, env_int, env_str


class TestEnvBool:
    def test_absent_returns_default_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDALINK_TEST_BOOL", raising=False)
        assert env_bool("CUDALINK_TEST_BOOL", default=True) is True

    def test_absent_returns_default_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDALINK_TEST_BOOL", raising=False)
        assert env_bool("CUDALINK_TEST_BOOL", default=False) is False

    def test_one_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_BOOL", "1")
        assert env_bool("CUDALINK_TEST_BOOL", default=False) is True

    def test_zero_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_BOOL", "0")
        assert env_bool("CUDALINK_TEST_BOOL", default=True) is False

    def test_empty_string_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_BOOL", "")
        assert env_bool("CUDALINK_TEST_BOOL", default=True) is False

    def test_arbitrary_string_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_BOOL", "yes")
        assert env_bool("CUDALINK_TEST_BOOL", default=False) is False


class TestEnvInt:
    def test_absent_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDALINK_TEST_INT", raising=False)
        assert env_int("CUDALINK_TEST_INT", default=42) == 42

    def test_valid_int(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_INT", "100")
        assert env_int("CUDALINK_TEST_INT", default=0) == 100

    def test_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_INT", "0")
        assert env_int("CUDALINK_TEST_INT", default=99) == 0

    def test_negative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_INT", "-5")
        assert env_int("CUDALINK_TEST_INT", default=0) == -5

    def test_non_numeric_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_INT", "abc")
        assert env_int("CUDALINK_TEST_INT", default=7) == 7

    def test_empty_string_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_INT", "")
        assert env_int("CUDALINK_TEST_INT", default=3) == 3


class TestEnvStr:
    def test_absent_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CUDALINK_TEST_STR", raising=False)
        assert env_str("CUDALINK_TEST_STR", default="fallback") == "fallback"

    def test_present_returns_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_STR", "hello")
        assert env_str("CUDALINK_TEST_STR", default="fallback") == "hello"

    def test_empty_string_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CUDALINK_TEST_STR", "")
        assert env_str("CUDALINK_TEST_STR", default="fallback") == ""
