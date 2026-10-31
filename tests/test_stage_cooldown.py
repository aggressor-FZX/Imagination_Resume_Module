"""Keep production memory protection while allowing explicitly faster tests."""
import pytest

from orchestrator import stage_cooldown_seconds


def test_production_default_is_unchanged(monkeypatch):
    monkeypatch.delenv("STAGE_COOLDOWN_SECONDS", raising=False)
    assert stage_cooldown_seconds() == 1.5


@pytest.mark.parametrize("setting,expected", [
    ("0", 0), ("0.25", 0.25), ("2", 2),
    ("invalid", 1.5), ("", 1.5), ("-1", 1.5),
    ("nan", 1.5), ("inf", 1.5), ("-inf", 1.5),
])
def test_configured_cooldown_is_finite_nonnegative(monkeypatch, setting, expected):
    monkeypatch.setenv("STAGE_COOLDOWN_SECONDS", setting)
    assert stage_cooldown_seconds() == expected