"""Capture and offline replay of a real correction failure.

The correction adapter talks to a model whose answers cannot be reproduced, so a
production failure reported as ``source_completeness_uncertain`` was not
replayable: the evidence lived only in provider responses that were deliberately
never logged. This module makes one case replayable without weakening anything.

Design constraints it must satisfy:

* **Synthetic only.** A capture refuses to record unless every source is
  designated ``synthetic_only`` and carries the synthetic marker, so a real
  candidate document cannot be recorded by accident.
* **Access controlled.** Captures are written ``0600`` inside a directory that is
  ignored by git, because even synthetic captures contain a complete resume-shaped
  document and a model response.
* **No text in logs.** The recorder never prints or logs document or model text;
  only counts, hashes and fixed identifiers.
* **Exact replay.** Replay feeds the recorded response body back through the real
  ``CorrectionProvider`` and the real ``source_correction.correct`` validator, so
  the decision is recomputed rather than compared against a stored verdict.

A capture file is JSON with a schema version so a future wire change is detected
rather than silently replayed against the wrong contract.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SYNTHETIC_MARKER = "SYNTHETIC-ONLY"
DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parent / "captures"


class CaptureRefused(Exception):
    """The case is not eligible to be captured (non-synthetic or malformed)."""


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass
class Capture:
    """One synthetic correction case, recorded end to end."""

    case_id: str
    source_text: str
    draft_text: str
    calls: list[dict[str, Any]] = field(default_factory=list)
    results: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    schema: int = SCHEMA_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "case_id": self.case_id,
                "synthetic": SYNTHETIC_MARKER,
                "source_text": self.source_text, "draft_text": self.draft_text,
                "calls": self.calls, "results": self.results, "notes": self.notes,
                "source_sha256": _digest(self.source_text),
                "draft_sha256": _digest(self.draft_text)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Capture":
        if value.get("schema") != SCHEMA_VERSION:
            raise CaptureRefused("unsupported capture schema")
        if value.get("synthetic") != SYNTHETIC_MARKER:
            raise CaptureRefused("capture is not marked synthetic")
        source, draft = value.get("source_text"), value.get("draft_text")
        if not isinstance(source, str) or not isinstance(draft, str):
            raise CaptureRefused("capture text missing")
        if _digest(source) != value.get("source_sha256") or _digest(draft) != value.get("draft_sha256"):
            raise CaptureRefused("capture text does not match its digest")
        return cls(case_id=str(value.get("case_id")), source_text=source, draft_text=draft,
                   calls=list(value.get("calls") or []),
                   results=list(value.get("results") or []),
                   notes=str(value.get("notes") or ""))


def assert_synthetic(case_id: str, source_text: str, draft_text: str) -> None:
    """Refuse anything not explicitly marked as synthetic-only material."""
    if not case_id or not case_id.replace("-", "").replace("_", "").isalnum():
        raise CaptureRefused("case_id must be a simple identifier")
    for label, text in (("source", source_text), ("draft", draft_text)):
        if not isinstance(text, str) or not text.strip():
            raise CaptureRefused(f"{label} text missing")
        if SYNTHETIC_MARKER not in text:
            raise CaptureRefused(f"{label} is not marked {SYNTHETIC_MARKER}")


def save_capture(capture: Capture, directory: Path | None = None) -> Path:
    """Write a capture with owner-only permissions. Never logs its contents."""
    assert_synthetic(capture.case_id, capture.source_text, capture.draft_text)
    target = Path(directory or DEFAULT_CAPTURE_DIR)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{capture.case_id}.json"
    payload = json.dumps(capture.as_dict(), ensure_ascii=False, indent=1).encode("utf-8")
    # O_EXCL with 0600: the file is never world-readable, not even briefly.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    return path


def load_capture(case_id: str, directory: Path | None = None) -> Capture:
    path = Path(directory or DEFAULT_CAPTURE_DIR) / f"{case_id}.json"
    # Refuse group/world-readable captures: they hold a resume-shaped document.
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        raise CaptureRefused("capture permissions are not owner-only")
    return Capture.from_dict(json.loads(path.read_text("utf-8")))


class RecordingSession:
    """Inline aiohttp-shaped session that records each exchange verbatim.

    It wraps the real session: the request still goes to the provider, and the
    complete response body is stored so the decision can be recomputed later. The
    body is never placed in a log record.
    """

    def __init__(self, inner, capture: Capture, *, synthetic: bool = True):
        self.inner = inner
        self.capture = capture
        self.synthetic = synthetic
        self.closed = False

    async def close(self):
        return await self.inner.close()

    def post(self, url, **kwargs):
        request_text = kwargs["json"]["messages"][1]["content"]
        if self.synthetic and SYNTHETIC_MARKER not in json.dumps(kwargs["json"]):
            raise CaptureRefused("refusing to record a request with no synthetic marker")
        # aiohttp's post() returns a request context manager, not a response: it
        # must be entered first, so the wrapper enters it and records from the
        # live response object.
        inner = self.inner.post(url, **kwargs)
        capture, entry = self.capture, {"request": json.loads(request_text)}

        class Recorded:
            def __init__(self, response):
                self._response = response
                self.status = getattr(response, "status", 200)
                self._chunks: list[bytes] = []
                self.content = self

            async def iter_chunked(self, size):
                async for chunk in self._response.content.iter_chunked(size):
                    self._chunks.append(chunk)
                    yield chunk
                entry["response_body"] = b"".join(self._chunks).decode("utf-8", "replace")
                entry["status"] = self.status
                capture.calls.append(entry)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return await self._response.__aexit__(*args)

        class Pending:
            async def __aenter__(self):
                response = await inner.__aenter__()
                return Recorded(response)

            async def __aexit__(self, *args):
                return False

        return Pending()


class ReplaySession:
    """aiohttp-shaped session that answers from a capture, with no network.

    Responses are returned in recorded order. A request beyond the recording is an
    error rather than an empty success, so a replay cannot accidentally pass by
    returning fewer phases than production did.
    """

    def __init__(self, capture: Capture):
        self.capture = capture
        self.position = 0
        self.requests: list[dict[str, Any]] = []
        self.calls = 0
        self.closed = False

    async def close(self):
        pass

    def post(self, url, **kwargs):
        if self.position >= len(self.capture.calls):
            raise AssertionError("replay exhausted: production made fewer calls")
        recorded = self.capture.calls[self.position]
        self.position += 1
        self.calls += 1
        self.requests.append(kwargs)
        body = recorded["response_body"].encode("utf-8")

        class Replayed:
            status = recorded.get("status", 200)

            def __init__(self):
                self.content = self

            async def iter_chunked(self, size):
                for index in range(0, len(body), size):
                    yield body[index:index + size]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        return Replayed()


def replay(capture: Capture, *, disclosure_enabled: bool = True):
    """Replay a capture through the real adapter and validator.

    Returns ``(result, provider, session)`` where ``result`` is the real
    ``CorrectionResult``. No network call is made and the disclosure setting is
    forced on for the duration, matching the production configuration the capture
    was recorded under.
    """
    import correction_provider
    from correction_provider import CorrectionProvider
    from source_correction import SourceDocument, correct

    previous = os.environ.get(correction_provider.DISCLOSURE_SETTING)
    if disclosure_enabled:
        os.environ[correction_provider.DISCLOSURE_SETTING] = "1"
    try:
        session = ReplaySession(capture)
        provider = CorrectionProvider("replay-no-network", session)
        result = asyncio.run(correct(
            capture.draft_text,
            (SourceDocument("resume", capture.source_text),),
            provider))
        return result, provider, session
    finally:
        if previous is None:
            os.environ.pop(correction_provider.DISCLOSURE_SETTING, None)
        else:
            os.environ[correction_provider.DISCLOSURE_SETTING] = previous
