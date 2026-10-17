from stream_context import (
    get_field_callback,
    get_token_callback,
    reset_token_callback,
    set_token_callback,
)


def test_field_aware_callback_receives_both_fields():
    seen = []
    tok = set_token_callback(lambda field, delta: seen.append((field, delta)))
    try:
        get_field_callback("draft")("a")
        get_field_callback("rewrite")("b")
    finally:
        reset_token_callback(tok)
    assert seen == [("draft", "a"), ("rewrite", "b")]


def test_legacy_arity1_callback_is_rewrite_only():
    seen = []
    tok = set_token_callback(seen.append)
    try:
        get_field_callback("draft")("a")
        get_field_callback("rewrite")("b")
    finally:
        reset_token_callback(tok)
    assert seen == ["b"]


def test_no_callback_returns_none():
    assert get_token_callback() is None
    assert get_field_callback("draft") is None


def test_sse_queue_emits_draft_then_rewrite():
    """Mirror app.py: field-aware on_token pushes (field, delta) in order."""
    import asyncio

    async def go():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def on_token(field, delta):
            loop.call_soon_threadsafe(queue.put_nowait, ("token", (field, delta)))

        tok = set_token_callback(on_token)
        try:
            get_field_callback("draft")("x")
            get_field_callback("rewrite")("y")
        finally:
            reset_token_callback(tok)
        return [await queue.get(), await queue.get()]

    items = asyncio.run(go())
    assert items == [("token", ("draft", "x")), ("token", ("rewrite", "y"))]
