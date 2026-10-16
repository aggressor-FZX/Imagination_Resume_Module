"""Request-scoped token callback for StarEditor streaming (Option C)."""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Callable, Optional

TokenCallback = Callable[[str], None]

_token_cb: ContextVar[Optional[TokenCallback]] = ContextVar(
    "imaginator_token_cb", default=None
)


def set_token_callback(cb: Optional[TokenCallback]) -> Token:
    return _token_cb.set(cb)


def reset_token_callback(token: Token) -> None:
    _token_cb.reset(token)


def get_token_callback() -> Optional[TokenCallback]:
    return _token_cb.get()
