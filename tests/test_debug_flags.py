"""
Tests for the /debug/flags endpoint returning current feature flag values
"""
from fastapi.testclient import TestClient
from app import app
from config import settings


client = TestClient(app, headers={"X-API-Key": settings.api_key})


def test_debug_flags_default():
    r = client.get("/debug/flags")
    assert r.status_code == 200
    data = r.json()
    assert set(["ENABLE_LOADER", "ENABLE_FASTSVM", "ENABLE_HERMES", "FINAL_WRITER_PROVIDER"]).issubset(set(data.keys()))
    assert isinstance(data["ENABLE_LOADER"], bool)
    assert isinstance(data["ENABLE_FASTSVM"], bool)
    assert isinstance(data["ENABLE_HERMES"], bool)


def test_debug_flags_toggles(monkeypatch):
    # Toggle flags to True and set base URLs
    monkeypatch.setattr(settings, "ENABLE_LOADER", True)
    monkeypatch.setattr(settings, "ENABLE_FASTSVM", True)
    monkeypatch.setattr(settings, "ENABLE_HERMES", True)
    monkeypatch.setattr(settings, "LOADER_BASE_URL", "https://loader.test")
    monkeypatch.setattr(settings, "FASTSVM_BASE_URL", "https://fastsvm.test")
    monkeypatch.setattr(settings, "HERMES_BASE_URL", "https://hermes.test")

    r = client.get("/debug/flags")
    assert r.status_code == 200
    data = r.json()
    assert data["ENABLE_LOADER"] is True
    assert data["ENABLE_FASTSVM"] is True
    assert data["ENABLE_HERMES"] is True
    assert data["LOADER_BASE_URL_SET"] is True
    assert data["FASTSVM_BASE_URL_SET"] is True
    assert data["HERMES_BASE_URL_SET"] is True
