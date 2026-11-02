#!/usr/bin/env python3
"""Re-send one captured request verbatim and report the provider's answer.

Diagnostic only. Prints the HTTP status, selected error fields and token counts.
It never prints document text, prompts or model content.

Usage (inside the container so the key never leaves it)::

    python tools/probe_captured_request.py /app/data/live-capture/<id>.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import aiohttp


async def main(path: str) -> int:
    with open(path, encoding="utf-8") as handle:
        capture = json.load(handle)
    exchanges = capture.get("exchanges") or []
    if not exchanges:
        print("no exchanges recorded")
        return 2
    request = exchanges[0].get("request", {}).get("json")
    if not isinstance(request, dict):
        print("captured exchange has no request body")
        return 2

    key = (os.getenv("OPENROUTER_API_KEY_1") or os.getenv("OPENROUTER_API_KEY_2") or "").strip()
    if not key:
        print("no provider key in the environment")
        return 2

    print(f"phase={exchanges[0].get('phase')} model={request.get('model')} "
          f"max_tokens={request.get('max_tokens')} "
          f"request_bytes={len(json.dumps(request, ensure_ascii=True).encode())}")

    timeout = aiohttp.ClientTimeout(total=90)
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=request, timeout=timeout, allow_redirects=False,
        ) as response:
            raw = await response.text()
            print(f"HTTP {response.status} bytes={len(raw)}")
            if response.status != 200:
                # Error envelopes carry no document text; show the diagnostic fields.
                try:
                    body = json.loads(raw)
                    error = body.get("error") if isinstance(body, dict) else None
                    if isinstance(error, dict):
                        for name in ("code", "message", "type", "param"):
                            if name in error:
                                print(f"  error.{name}={str(error[name])[:300]}")
                    else:
                        print(f"  body={raw[:400]}")
                except Exception:
                    print(f"  body={raw[:400]}")
                return 1
            body = json.loads(raw)
            usage = body.get("usage") or {}
            choices = body.get("choices") or []
            finish = choices[0].get("finish_reason") if choices else None
            content = (choices[0].get("message", {}) or {}).get("content") if choices else None
            print(f"finish_reason={finish} content_chars={len(content or '')} "
                  f"prompt_tokens={usage.get('prompt_tokens')} "
                  f"completion_tokens={usage.get('completion_tokens')}")
            try:
                parsed = json.loads(content or "{}")
                print(f"answer_keys={sorted(parsed)[:8]}")
            except Exception:
                print("answer_parsed=no")
            return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(main(sys.argv[1])))
