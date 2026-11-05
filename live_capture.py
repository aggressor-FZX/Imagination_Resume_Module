"""Access-controlled capture of one live website correction exchange.

The earlier probes recorded the correction engine in isolation, so they could not
answer "what did the website actually send and receive?". This module records the
**live path**: the same objects an `/analyze` request produces, for one synthetic
request, keyed by its correlation id.

What is recorded, in order:

* the extracted source documents after the disclosure redaction the request used;
* the writer-generated draft that entered correction (the draft is *not* constructed);
* every outgoing correction request, exactly as serialized;
* every provider response body, verbatim;
* the normalized review the engine derived, and its final gate decision;
* per-phase timings and the correlation id that joins the two services' logs.

Access control, all enforced in code rather than by convention:

* **opt-in only.** Nothing is recorded unless ``CORRECTION_LIVE_CAPTURE_DIR`` is set
  and ``CORRECTION_LIVE_CAPTURE`` is truthy. The default is off.
* **synthetic only.** Every source document must carry the synthetic marker, so a real
  candidate document cannot be recorded by accident.
* **never logged.** No text reaches a log record; only counts, digests and identifiers.
* **never in Git.** The directory must be outside the repository, or inside a
  git-ignored one; the writer refuses otherwise.
* **owner-only.** Files are written ``0600`` in a ``0700`` directory.
* **one request per file**, named by correlation id, so a capture cannot be appended to
  or confused with another run.

Recording failures never affect the request: a capture is diagnostics, and a bug in it
must not change what a user receives.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SYNTHETIC_MARKER = "SYNTHETIC-ONLY"
SETTING_ENABLED = "CORRECTION_LIVE_CAPTURE"
SETTING_DIR = "CORRECTION_LIVE_CAPTURE_DIR"
_REPO_ROOT = Path(__file__).resolve().parent
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
# Fields that are safe to keep verbatim in the summary: identifiers and numbers only.
_SAFE_SCALARS = (int, float, bool, type(None))
# A summary string must look like a machine identifier, not prose: lowercase, a small
# separator set, and no long runs of capitalised words. `unsupported_claims` and
# `provider_call_timeout_repair` pass; `PRIVATE_DRAFT_TEXT` and free text do not.
_IDENTIFIER_LIKE = re.compile(r"[a-z][a-z0-9]*(?:[._:-][a-z0-9]+){0,7}\Z")


class LiveCaptureRefused(Exception):
    """The capture cannot be written where or how it was asked to be."""


def enabled() -> bool:
    """True only when an operator explicitly turned capture on."""
    return (os.getenv(SETTING_ENABLED, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def capture_dir() -> Path | None:
    value = (os.getenv(SETTING_DIR, "") or "").strip()
    return Path(value) if value else None


def _is_inside_repo(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    return resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents


def _ignored_by_git(path: Path) -> bool:
    """True when git would ignore ``path``.

    ``git check-ignore`` answers for a *concrete path*, and returns 1 for a path
    that does not exist yet when the matching rule is directory-only: the pattern
    ``captures/`` does not match the string ``captures`` unless the directory is
    present. That made the guard refuse the **first** capture in a clean checkout
    even though the location is correctly ignored — the worst possible time for a
    false refusal.

    Fix: when the path does not exist, test the nearest existing ancestor AND the
    path's own name against the ignore rules directly. The authoritative question
    is "would a file written here be ignored", which is answered by matching the
    pattern, not by stat-ing a directory into existence.
    """
    probe = Path(path)
    ancestors = []
    for _ in range(64):
        ancestors.append(probe)
        if probe.exists():
            break
        parent = probe.parent
        if parent == probe:
            break
        probe = parent

    # 1. If the path (or an existing ancestor) is already ignored, done.
    for candidate in ancestors:
        try:
            result = subprocess.run(
                ["git", "check-ignore", "-q", str(candidate)],
                cwd=str(_REPO_ROOT), timeout=5, capture_output=True,
            )
        except Exception:
            return False
        if result.returncode == 0:
            return True

    # 2. The path does not exist yet and no ancestor is ignored. Ask git directly
    #    whether a file at this location would be ignored, using a dry-run add.
    #    That is exactly the question the guard exists to answer, and it works
    #    for directory-only patterns without creating anything.
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "--no-index", str(Path(path) / "_probe")],
            cwd=str(_REPO_ROOT), timeout=5, capture_output=True,
        )
    except Exception:
        return False
    return result.returncode == 0


def _device_of(path: Path) -> int | None:
    """Device id of ``path``, falling back to its nearest existing ancestor."""
    probe = Path(path)
    for _ in range(64):
        try:
            return probe.stat().st_dev
        except OSError:
            parent = probe.parent
            if parent == probe:
                return None
            probe = parent
    return None


def _separate_volume(path: Path, root: Path | None = None) -> bool:
    """True when ``path`` is on a different filesystem than the source tree.

    The deployed image mounts its persistent disk *inside* the application
    directory (``/app/data``), so every writable location there is technically
    "inside the repository". A separately mounted volume is not part of the deploy
    artifact: nothing populates it from the checkout and the container ships with no
    ``.git`` and no ``git`` binary at all, so no commit can reach a file written
    there. Comparing device ids keeps the guard meaningful in a container, where
    ``git check-ignore`` cannot answer.
    """
    path_device = _device_of(path)
    root_device = _device_of(Path(root or _REPO_ROOT))
    if path_device is None or root_device is None:
        return False
    return path_device != root_device


def assert_writable_directory(directory: Path) -> None:
    """Refuse a location that would put a resume-shaped document into Git."""
    if not _is_inside_repo(directory):
        return
    if _ignored_by_git(directory):
        return
    if _separate_volume(directory):
        return
    raise LiveCaptureRefused(
        "capture directory is inside the repository and not git-ignored")


def assert_synthetic(source_texts: list[str]) -> None:
    """Refuse anything not explicitly marked as synthetic-only material."""
    if not source_texts:
        raise LiveCaptureRefused("no source documents to record")
    for text in source_texts:
        if not isinstance(text, str) or SYNTHETIC_MARKER not in text:
            raise LiveCaptureRefused("a source document is not marked SYNTHETIC-ONLY")


def redact_for_capture(value: Any) -> Any:
    """Strip anything that is not a number or a fixed identifier.

    Used for the event/gate summaries. Document text must travel in the dedicated
    ``sources``/``draft``/``exchanges`` fields, never in a summary blob, so an
    accidentally logged summary cannot leak content.
    """
    if isinstance(value, _SAFE_SCALARS):
        return value
    if isinstance(value, str):
        # Keep only machine-identifier-shaped strings. Anything else is reported by
        # length so a summary can never carry document or model text.
        return value if _IDENTIFIER_LIKE.fullmatch(value) else f"<str:{len(value)}>"
    if isinstance(value, dict):
        return {str(key): redact_for_capture(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_for_capture(item) for item in value]
    return f"<{type(value).__name__}>"


class LiveCapture:
    """One live correction exchange. ``record`` methods never raise into the request."""

    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.data: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "synthetic": SYNTHETIC_MARKER,
            "correlation_id": correlation_id,
            "sources": [],
            "draft": None,
            "exchanges": [],
            "review": None,
            "gate": None,
            "timing": {},
        }

    # -- recorders ---------------------------------------------------------
    def record_sources(self, sources) -> None:
        self.data["sources"] = [{"source_id": sid, "text": text} for sid, text in sources]

    def record_draft(self, draft: str) -> None:
        self.data["draft"] = draft

    def record_request(self, phase: str, request: Any) -> int | None:
        """Record one outgoing request and return its index in ``exchanges``."""
        if not isinstance(request, dict):
            return None
        self.data["exchanges"].append({"phase": phase, "request": request,
                                       "response_body": None, "status": None})
        return len(self.data["exchanges"]) - 1

    def record_response(self, index: int | None, status: Any, body: str) -> None:
        if index is None or not 0 <= index < len(self.data["exchanges"]):
            return
        self.data["exchanges"][index]["status"] = status if isinstance(status, (int, str)) else None
        self.data["exchanges"][index]["response_body"] = body

    def record_review(self, review: Any) -> None:
        self.data["review"] = review

    def record_gate(self, decision: Any) -> None:
        self.data["gate"] = redact_for_capture(decision)

    def record_timing(self, **values: Any) -> None:
        self.data["timing"].update({"correlation_id": self.correlation_id,
                                    **redact_for_capture(values)})

    # -- persistence -------------------------------------------------------
    def payload(self) -> dict[str, Any]:
        return dict(self.data)

    def write(self, directory: Path | None = None) -> Path | None:
        """Persist the capture. Returns the path, or None when capture is off."""
        target = directory or capture_dir()
        if target is None or not enabled():
            return None
        if not _ID.fullmatch(self.correlation_id):
            raise LiveCaptureRefused("correlation id is not a safe filename")
        assert_synthetic([item["text"] for item in self.data["sources"]])
        assert_writable_directory(target)
        target = Path(target)
        target.mkdir(parents=True, exist_ok=True)
        os.chmod(target, stat.S_IRWXU)
        path = target / f"{self.correlation_id}.json"
        blob = json.dumps(self.payload(), ensure_ascii=False, indent=1).encode("utf-8")
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                            stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(blob)
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        return path


def start(correlation_id: str) -> LiveCapture | None:
    """Create a capture when enabled; otherwise return None so callers stay cheap."""
    if not enabled() or not correlation_id:
        return None
    if not _ID.fullmatch(correlation_id):
        return None
    return LiveCapture(correlation_id)


class CapturingSession:
    """aiohttp-shaped session wrapper that records each exchange.

    Wraps the real session so the request still reaches the provider; the complete
    response body is stored so the decision can be recomputed offline. Recording
    errors are swallowed: diagnostics must never break a user's request.
    """

    def __init__(self, inner, capture: LiveCapture):
        self.inner = inner
        self.capture = capture
        self.closed = getattr(inner, "closed", False)

    async def close(self):
        return await self.inner.close()

    @property
    def requests(self):
        return getattr(self.inner, "requests", [])

    def post(self, url, **kwargs):
        pending = self.inner.post(url, **kwargs)
        capture = self.capture
        try:
            phase = self._phase(kwargs)
            index = capture.record_request(phase, _strip_secrets(kwargs))
        except Exception:
            index = None

        class Recorded:
            def __init__(self, response):
                self._response = response
                self.status = getattr(response, "status", None)
                self._chunks: list[bytes] = []
                self.content = self

            async def iter_chunked(self, size):
                async for chunk in self._response.content.iter_chunked(size):
                    self._chunks.append(chunk)
                    yield chunk
                try:
                    capture.record_response(index, self.status,
                                            b"".join(self._chunks).decode("utf-8", "replace"))
                except Exception:
                    pass

            async def text(self, *args, **kwargs):
                # A provider that answers with an error status is read through
                # ``text()`` and never reaches ``iter_chunked``. Without this the
                # capture kept the request and dropped the answer, so a live 429
                # recorded as "no response at all" — the one shape that cannot be
                # diagnosed offline. Record it here as well.
                body = await self._response.text(*args, **kwargs)
                try:
                    capture.record_response(index, self.status, body)
                except Exception:
                    pass
                return body

            async def read(self):
                body = await self._response.read()
                try:
                    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
                    capture.record_response(index, self.status, text)
                except Exception:
                    pass
                return body

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return await self._response.__aexit__(*args)

        class Pending:
            async def __aenter__(self):
                return Recorded(await pending.__aenter__())

            async def __aexit__(self, *args):
                return False

        return Pending()

    @staticmethod
    def _phase(kwargs) -> str:
        """Name the phase from the request body without inspecting document text."""
        try:
            messages = kwargs["json"]["messages"]
            payload = json.loads(messages[1]["content"])
        except Exception:
            return "unknown"
        # Name the phase by the question being asked. Keying off "has lines" was
        # wrong for a restoration, which also carries lines: every restore was
        # recorded as a review, so a captured run could not be replayed through the
        # restoration path it actually took.
        contract = payload.get("contract")
        if type(contract) is str:
            if contract.startswith("Rewrite the named draft lines"):
                return "restore"
            if contract.startswith("Classify every numbered source segment"):
                return "source_inventory"
        if "flagged_claim_ids" in payload:
            return "repair"
        if "lines" in payload:
            return "review"
        return "source_inventory"


def _strip_secrets(kwargs) -> dict[str, Any]:
    """Keep the body, drop transport headers that carry credentials."""
    body = kwargs.get("json")
    return {"json": body} if isinstance(body, dict) else {"json": None}


def summarise(capture: LiveCapture | None) -> dict[str, Any]:
    """Log-safe description: counts and identifiers only, never text."""
    if capture is None:
        return {"live_capture": "off"}
    data = capture.data
    return {"live_capture": "on",
            "exchanges": len(data["exchanges"]),
            "source_documents": len(data["sources"]),
            "draft_chars": len(data["draft"] or ""),
            "responses_recorded": sum(1 for item in data["exchanges"]
                                      if item.get("response_body") is not None)}


# ---------------------------------------------------------------------------
# Offline replay of a live capture
# ---------------------------------------------------------------------------

class ReplayExhausted(AssertionError):
    """The replay needed a provider exchange the live run never performed."""


def load(path) -> dict:
    """Read a capture, refusing anything not owner-only or not synthetic."""
    path = Path(path)
    mode = path.stat().st_mode
    if mode & (stat.S_IRGRP | stat.S_IROTH | stat.S_IWGRP | stat.S_IWOTH):
        raise LiveCaptureRefused("capture permissions are not owner-only")
    data = json.loads(path.read_text("utf-8"))
    if data.get("schema") != SCHEMA_VERSION:
        raise LiveCaptureRefused("unsupported capture schema")
    if data.get("synthetic") != SYNTHETIC_MARKER:
        raise LiveCaptureRefused("capture is not marked synthetic")
    if not data.get("sources") or any(SYNTHETIC_MARKER not in item.get("text", "")
                                      for item in data["sources"]):
        raise LiveCaptureRefused("capture sources are not marked synthetic")
    return data


class ReplaySession:
    """aiohttp-shaped session answering from a live capture, with no network.

    Exchanges are returned in recorded order. A request past the end of the recording
    raises ``ReplayExhausted`` rather than returning an empty success, so a replay can
    never pass by quietly performing fewer phases than the live run did.
    """

    def __init__(self, recorded: dict):
        self.recorded = recorded
        self.position = 0
        self.requests: list[dict] = []
        self.calls = 0
        self.closed = False

    async def close(self):
        pass

    def post(self, url, **kwargs):
        exchanges = self.recorded.get("exchanges") or []
        if self.position >= len(exchanges):
            raise ReplayExhausted(
                f"replay exhausted after {self.position} of {len(exchanges)} exchanges")
        exchange = exchanges[self.position]
        self.position += 1
        self.calls += 1
        self.requests.append(kwargs)
        body = (exchange.get("response_body") or "").encode("utf-8")
        status = exchange.get("status") or 200

        class Replayed:
            def __init__(self):
                self.status = status
                self.content = self

            async def iter_chunked(self, size):
                for index in range(0, len(body), size):
                    yield body[index:index + size]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        return Replayed()


def replay(path, *, timeout_seconds: float = 120.0):
    """Replay a live capture through the real engine, adapter and validator.

    The draft is the writer's own output as recorded, not a reconstruction. The
    disclosure setting is forced on for the duration because that is the configuration
    the live run used. Returns ``(result, provider, session)``.
    """
    result, provider, session = replay_with_extensions(path, timeout_seconds=timeout_seconds)
    return result, provider, session


class ControlledExtension:
    """A separately labelled answer for a phase the recording never performed.

    A capture records what the provider was asked and what it answered. It cannot
    contain a phase the live run never reached — and delivery runs 4 through 7 all
    stopped at the same point, so none of them contains a repair, a re-review or a
    restore. Replaying new code against them therefore *ends* at the new behaviour
    rather than exercising it.

    This class supplies the missing answers, and is deliberately noisy about what it
    is: every exchange it provides is labelled ``controlled``, it can never overwrite
    a recorded exchange, and it refuses to answer a phase that the recording already
    covers. Anything proved with it is proved about the code, never about the live
    run — a distinction that matters because a controlled answer is written by the
    same person reading the same answer key, and so cannot independently corroborate
    the answer key.

    Ordering rule, and the reason this is not a plain queue: an extension answer is
    released for a phase only after every recorded exchange has been consumed. That
    keeps the recording authoritative for everything it actually contains, so the
    original rejection is reproduced by the original bytes before the extension is
    ever consulted.
    """

    CONTROLLED = "controlled"

    def __init__(self, answers: dict[str, str]):
        for phase, body in answers.items():
            if not isinstance(phase, str) or not isinstance(body, str) or not body:
                raise LiveCaptureRefused("extension answers must be non-empty strings")
        self.answers = dict(answers)
        self.served: list[str] = []

    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(self.answers)

    def prepare(self, recorded: dict) -> None:
        """Refuse an answer for a phase the recording already covers."""
        covered = {item.get("phase") for item in recorded.get("exchanges") or []}
        clash = sorted(covered.intersection(self.answers))
        if clash:
            raise LiveCaptureRefused(
                f"extension would overwrite recorded phase(s): {', '.join(clash)}")
        self.served.clear()


class ExtendedReplaySession(ReplaySession):
    """ReplaySession that serves labelled controlled answers once the recording ends.

    The parent's ``ReplayExhausted`` remains the behaviour when no controlled answer
    exists for the phase being asked, so an unplanned phase is still an error rather
    than an empty success.
    """

    def __init__(self, recorded: dict, extension: ControlledExtension):
        super().__init__(recorded)
        self.extension = extension
        self.controlled_requests: list[dict] = []
        # Phases the engine asked for that neither the recording nor the extension
        # could answer. Recorded explicitly because raising consumes nothing: without
        # this, "ran past the recording" and "never reached the end" look identical on
        # the session, and a test asserting the former would be asserting nothing.
        self.unanswered: list[str] = []

    # ``_phase`` lives on CapturingSession, the class that names a request for the
    # recorder. A replay must name phases exactly as the recorder did, so it borrows
    # that detector rather than keeping a second copy that could drift from it.
    _phase = staticmethod(CapturingSession._phase)

    def post(self, url, **kwargs):
        exchanges = self.recorded.get("exchanges") or []
        if self.position < len(exchanges):
            return super().post(url, **kwargs)
        phase = self._phase(kwargs)
        body = self.extension.answers.get(phase)
        if body is None:
            self.unanswered.append(phase)
            raise ReplayExhausted(
                f"replay exhausted after {self.position} of {len(exchanges)} exchanges"
                f" and no controlled answer for phase {phase!r}")
        self.calls += 1
        self.position += 1
        self.requests.append(kwargs)
        self.controlled_requests.append({"phase": phase, "origin": self.extension.CONTROLLED})
        self.extension.served.append(phase)
        encoded = body.encode("utf-8")
        status = 200

        class Controlled:
            def __init__(self):
                self.status = status
                self.content = self

            async def iter_chunked(self, size):
                for index in range(0, len(encoded), size):
                    yield encoded[index:index + size]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

        return Controlled()


def replay_with_extensions(path, *, answers: dict[str, str] | None = None,
                          timeout_seconds: float = 120.0):
    """Replay a capture, optionally answering phases it never performed.

    Returns ``(result, provider, session)``. Without ``answers`` this is exactly
    ``replay``: the recording decides every phase and running past its end raises
    ``ReplayExhausted``.
    """
    import correction_provider
    from correction_provider import CorrectionProvider
    from source_correction import SourceDocument, correct

    data = load(path)
    extension = ControlledExtension(answers or {})
    extension.prepare(data)
    previous = os.environ.get(correction_provider.DISCLOSURE_SETTING)
    os.environ[correction_provider.DISCLOSURE_SETTING] = "1"
    try:
        session = ExtendedReplaySession(data, extension)
        provider = CorrectionProvider("replay-no-network", session)
        sources = tuple(SourceDocument(item["source_id"], item["text"])
                        for item in data["sources"])
        result = asyncio.run(correct(data["draft"], sources, provider,
                                     timeout_seconds=timeout_seconds))
        return result, provider, session
    finally:
        if previous is None:
            os.environ.pop(correction_provider.DISCLOSURE_SETTING, None)
        else:
            os.environ[correction_provider.DISCLOSURE_SETTING] = previous


def recorded_stopping_point(path) -> dict[str, Any]:
    """The diagnosis the *original* code reached on a capture's own exchanges.

    Recomputing this with current code is wrong: the current code deliberately
    goes further (it repairs before judging missing facts), so it no longer stops
    where the live run stopped. The original rejection is evidence about a revision,
    so it is read from the capture's own ``timing.reason`` and exchange list rather
    than re-derived.
    """
    data = load(path)
    exchanges = data.get("exchanges") or []
    timing = data.get("timing") or {}
    review = data.get("review") or {}
    return {
        "phases": [item.get("phase") for item in exchanges],
        "phase_count": len(exchanges),
        "status": review.get("status"),
        "reason": review.get("reason") or timing.get("reason"),
        "review_calls": review.get("review_calls"),
        "repair_calls": review.get("repair_calls"),
        "restore_calls": review.get("restore_calls"),
        "correlation_id": data.get("correlation_id"),
        "revision": timing.get("revision") or (data.get("synthetic") and None),
    }

