"""Offline regression coverage for the live-test consent/credential boundary."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import conftest as suite_config


@pytest.mark.parametrize("run_live", [False, True])
def test_collection_requires_explicit_opt_in(run_live):
    config = Mock()
    config.getoption.return_value = run_live
    live_item, offline_item = Mock(), Mock()
    live_item.get_closest_marker.return_value = pytest.mark.live.mark
    offline_item.get_closest_marker.return_value = None

    suite_config.pytest_collection_modifyitems(config, [live_item, offline_item])

    offline_item.add_marker.assert_not_called()
    if run_live:
        live_item.add_marker.assert_not_called()
    else:
        marker = live_item.add_marker.call_args.args[0]
        assert marker.name == "skip"
        assert "--run-live" in marker.kwargs["reason"]


@pytest.mark.parametrize("key", ["d", "test-dummy-key-1", "rctb-explicit-key"])
def test_credentials_snapshot_has_no_dummy_key_heuristics(monkeypatch, key):
    monkeypatch.setenv("OPENROUTER_API_KEY_1", key)
    monkeypatch.setenv("ONET_API_AUTH", "explicit-onet-auth")
    config = Mock()
    suite_config.pytest_configure(config)

    # Later dotenv/import mutations cannot qualify or replace explicit keys.
    monkeypatch.setenv("OPENROUTER_API_KEY_1", "loaded-later")
    assert config._live_credentials["OPENROUTER_API_KEY_1"] == key
    assert config._live_credentials["ONET_API_AUTH"] == "explicit-onet-auth"


@pytest.mark.parametrize("missing", ["OPENROUTER_API_KEY_1", "ONET_API_AUTH"])
def test_live_credentials_skip_explicitly_when_missing(monkeypatch, missing):
    credentials = {
        "OPENROUTER_API_KEY_1": "explicit-key",
        "OPENROUTER_API_KEY_2": "",
        "ONET_API_AUTH": "explicit-onet-auth",
    }
    credentials[missing] = ""
    config = Mock()
    config.getoption.return_value = True
    config._live_credentials = credentials
    # Even if an application import loaded the missing value, reject it.
    monkeypatch.setenv(missing, "implicitly-loaded-later")

    with pytest.raises(pytest.skip.Exception, match=missing):
        suite_config.live_credentials.__wrapped__(
            SimpleNamespace(config=config), monkeypatch
        )


def test_live_credentials_cannot_bypass_opt_in(monkeypatch):
    config = Mock()
    config.getoption.return_value = False
    with pytest.raises(pytest.skip.Exception, match="--run-live"):
        suite_config.live_credentials.__wrapped__(
            SimpleNamespace(config=config), monkeypatch
        )