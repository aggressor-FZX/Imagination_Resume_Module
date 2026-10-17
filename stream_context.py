"""Request-scoped token callback for stage streaming (draft + rewrite).

The stored callback is field-aware: ``cb(field, delta)`` where field is
"draft" (Drafter) or "rewrite" (StarEditor). Legacy arity-1 callbacks
``cb(delta)`` are wrapped and receive only "rewrite" deltas.
"""

from __future__ import annotations

import inspect
from contextvars import ContextVar, Token
from typing import Callable, Optional

FieldTokenCallback = Callable[[str, str], None]
TokenCallback = Callable[[str], None]  # legacy arity-1 (rewrite only)

_token_cb: ContextVar[Optional[FieldTokenCallback]] = ContextVar(
    "imaginator_token_cb", default=None
)


def _accepts_two_args(cb: Callable) -> bool:
    try:
        sig = inspect.signature(cb)
    except (TypeError, ValueError):
        return True
    params = list(sig.parameters.values())
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
        return True
    positional = [
        p
        for p in params
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _wrap_legacy(cb: TokenCallback) -> FieldTokenCallback:
    def _wrapped(field: str, delta: str) -> None:
        if field == "rewrite":
            cb(delta)

    return _wrapped


def set_token_callback(cb: Optional[Callable]) -> Token:
    """Install cb(field, delta); arity-1 callbacks are treated as rewrite-only."""
    if cb is not None and not _accepts_two_args(cb):
        cb = _wrap_legacy(cb)
    return _token_cb.set(cb)


def reset_token_callback(token: Token) -> None:
    _token_cb.reset(token)


def get_token_callback() -> Optional[FieldTokenCallback]:
    """Return the raw field-aware callback cb(field, delta), or None."""
    return _token_cb.get()


def get_field_callback(field: str) -> Optional[TokenCallback]:
    """Return an arity-1 callback bound to ``field``, or None if no stream active."""
    cb = _token_cb.get()
    if cb is None:
        return None

    def _bound(delta: str) -> None:
        cb(field, delta)

    return _bound
