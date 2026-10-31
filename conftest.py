"""Pytest configuration for the Imaginator test suite.

Phase 2 fix: the suite had no pytest config at all, so `pytest-asyncio` was
never registered even though it is installed. Consequences:

  * `@pytest.mark.asyncio` was an *unknown mark*, so async tests were collected
    as ordinary functions, returned an un-awaited coroutine, and failed with
    "async def functions are not natively supported".
  * That produced 3 permanent failures (2 in test_phase2_onet.py, 1 in
    test_logic_mocks.py) which were unrelated to any real defect but made the
    suite impossible to read as a pass/fail signal.

No credentials are required to collect or run the default offline tests.
Live tests require --run-live and credentials explicitly present at startup.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_addoption(parser):
    parser.addoption(
        "--run-live", action="store_true", default=False,
        help="Run live API tests (requires explicitly exported credentials).",
    )


def pytest_configure(config):
    # Snapshot before collection: application imports may load dotenv files.
    # Those implicit credentials must never qualify a test for a live run.
    config._live_credentials = {
        name: os.environ.get(name, "").strip()
        for name in ("OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "ONET_API_AUTH")
    }
    config.addinivalue_line(
        "markers", "asyncio: mark a test as an asyncio coroutine test"
    )
    config.addinivalue_line(
        "markers", "fast: quick unit test (declared by existing tests)"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-live"):
        skip_live = pytest.mark.skip(reason="live API test: opt in with --run-live")
        for item in items:
            if item.get_closest_marker("live") is not None:
                item.add_marker(skip_live)


@pytest.fixture
def live_credentials(request, monkeypatch):
    """Validate explicit credentials before importing/constructing real clients."""
    if not request.config.getoption("--run-live"):
        pytest.skip("live API test: opt in with --run-live")
    credentials = request.config._live_credentials
    missing = [
        name for name in ("OPENROUTER_API_KEY_1", "ONET_API_AUTH")
        if not credentials[name]
    ]
    if missing:
        pytest.skip("--run-live requires explicit environment credentials: " + ", ".join(missing))

    import imaginator_new_integration as integration

    # Override dotenv-derived keys and the application's built-in O*NET fallback.
    monkeypatch.setattr(integration, "OPENROUTER_API_KEYS", [
        credentials[name] for name in ("OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2")
        if credentials[name]
    ])
    monkeypatch.setattr(integration, "ONET_API_AUTH", credentials["ONET_API_AUTH"])
    monkeypatch.setitem(
        integration._ONET_HEADERS, "Authorization", "Basic " + credentials["ONET_API_AUTH"]
    )
    return credentials.copy()
