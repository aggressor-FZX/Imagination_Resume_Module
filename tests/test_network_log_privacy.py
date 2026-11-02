"""Offline network logging checks; these do not establish a global privacy boundary."""
import asyncio
import logging
import socket

import httpx
import pytest

from logging_setup import configure_logging


SENTINEL = "SYNTHETIC_PRIVATE_QUERY_83A9"
NETWORK_LOGGERS = ("httpx", "httpcore", "urllib3", "aiohttp.client")


@pytest.fixture(autouse=True)
def isolated_logging_and_network(monkeypatch):
    root = logging.getLogger()
    root_handlers, root_filters, root_level = root.handlers[:], root.filters[:], root.level
    loggers = [logging.getLogger(name) for name in NETWORK_LOGGERS]
    levels = [logger.level for logger in loggers]

    def blocked(*args, **kwargs):
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    yield
    root.handlers[:] = root_handlers
    root.filters[:] = root_filters
    root.setLevel(root_level)
    for logger, level in zip(loggers, levels):
        logger.setLevel(level)


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
@pytest.mark.parametrize("log_format", ["text", "json"])
def test_httpx_mock_requests_do_not_emit_url_and_application_logs_remain(
    monkeypatch, capsys, level, log_format,
):
    monkeypatch.setenv("LOG_LEVEL", level)
    monkeypatch.setenv("LOG_FORMAT", log_format)
    configure_logging()
    sent = []

    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"ok": True})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        response = client.get("https://example.invalid/" + SENTINEL, params={"keyword": SENTINEL})
    assert response.status_code == 200
    assert len(sent) == 1 and sent[0].url.params["keyword"] == SENTINEL
    assert SENTINEL in sent[0].url.path

    for name in NETWORK_LOGGERS:
        logger = logging.getLogger(name)
        logger.info("URL https://example.invalid/?query=%s", SENTINEL)
        logger.debug("header %s", SENTINEL)
        logger.warning("network warning remains visible")
    logging.getLogger("httpcore.http11").debug("child header %s", SENTINEL)
    logging.getLogger("urllib3.connectionpool").info("child URL %s", SENTINEL)
    logging.getLogger("unrelated.application").info("application event remains visible")
    logging.getLogger("unrelated.application").info("application value %s", "unchanged-value")
    output = capsys.readouterr()
    combined = output.out + output.err
    assert SENTINEL not in combined
    assert "HTTP Request:" not in combined
    assert combined.count("network warning remains visible") == len(NETWORK_LOGGERS)
    assert "application event remains visible" in combined
    assert "application value unchanged-value" in combined
    assert logging.getLogger("unrelated.application").getEffectiveLevel() == getattr(logging, level)


def test_unprotected_httpx_info_reproduces_leak(monkeypatch, capsys):
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    monkeypatch.setenv("LOG_FORMAT", "text")
    configure_logging()
    # Demonstrate that the real HTTPX event would expose this value without the
    # network logger level, rather than merely asserting against a mock logger.
    logging.getLogger("httpx").setLevel(logging.INFO)
    sent = []

    def respond(request):
        sent.append(request)
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert client.get("https://example.invalid/search", params={"keyword": SENTINEL}).status_code == 200
    assert len(sent) == 1 and sent[0].url.params["keyword"] == SENTINEL
    assert SENTINEL in capsys.readouterr().out


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_real_onet_keyword_search_uses_mock_transport_without_logging_query(
    monkeypatch, capsys, level,
):
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    import imaginator_new_integration as integration

    monkeypatch.setenv("LOG_LEVEL", level)
    monkeypatch.setenv("LOG_FORMAT", "text")
    configure_logging()
    monkeypatch.setattr(integration, "_ONET_CACHE", {})
    sent = []
    async_client = httpx.AsyncClient

    def respond(request):
        sent.append(request)
        return httpx.Response(200, json={"occupation": [{"code": "15-1252.00", "title": "Engineer"}]})

    monkeypatch.setattr(integration.httpx, "AsyncClient", lambda **kwargs: async_client(
        transport=httpx.MockTransport(respond), **kwargs
    ))
    result = asyncio.run(integration._search_onet_keyword(SENTINEL))
    assert len(sent) == 1
    assert sent[0].method == "GET" and sent[0].url.path.endswith("/online/search")
    assert sent[0].url.params["keyword"] == SENTINEL
    assert result[0]["query"] == SENTINEL
    output = capsys.readouterr()
    assert SENTINEL not in output.out + output.err
    assert "HTTP Request:" not in output.out + output.err