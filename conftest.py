"""Pytest configuration for the Imaginator test suite.

Phase 2 fix: the suite had no pytest config at all, so `pytest-asyncio` was
never registered even though it is installed. Consequences:

  * `@pytest.mark.asyncio` was an *unknown mark*, so async tests were collected
    as ordinary functions, returned an un-awaited coroutine, and failed with
    "async def functions are not natively supported".
  * That produced 3 permanent failures (2 in test_phase2_onet.py, 1 in
    test_logic_mocks.py) which were unrelated to any real defect but made the
    suite impossible to read as a pass/fail signal.

No credentials are required to collect or run these tests (Phase 1 item 1.7).
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark a test as an asyncio coroutine test"
    )
    config.addinivalue_line(
        "markers", "fast: quick unit test (declared by existing tests)"
    )
