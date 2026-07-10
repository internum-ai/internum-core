import json
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from api.common.errors import UpstreamError

SseLineKind = Literal["data", "done", "comment", "blank"]

_DATA_PREFIX = "data:"
_DONE_MARKER = "[DONE]"


@dataclass
class StreamAssembly:
    content: str = ""
    model: str | None = None
    provider: str | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    choice_error: dict[str, Any] | None = None


def classify_sse_line(line: str) -> SseLineKind:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if stripped.startswith(":"):
        # SSE comment line, e.g. OpenRouter's ": OPENROUTER PROCESSING" keepalive.
        return "comment"
    if stripped.startswith(_DATA_PREFIX):
        payload = stripped[len(_DATA_PREFIX) :].strip()
        return "done" if payload == _DONE_MARKER else "data"
    # Any other SSE field (event:, id:, retry:) carries no completion delta.
    return "comment"


async def consume_completion_stream(response: httpx.Response) -> StreamAssembly:
    """Assemble a streamed OpenAI-compatible chat completion.

    Appends ``choices[0].delta.content`` across chunks, captures model/provider/usage/
    finish_reason from whichever chunk carries them, stops on ``[DONE]``, and raises
    ``UpstreamError`` if a chunk reports a provider-side choice error. Keepalive comment
    lines are ignored and never corrupt the assembled content. Deadlines/stall detection
    are enforced by the caller (``asyncio.wait_for`` + httpx read timeout)."""
    assembly = StreamAssembly()
    async for raw_line in response.aiter_lines():
        kind = classify_sse_line(raw_line)
        if kind in ("blank", "comment"):
            continue
        if kind == "done":
            break
        chunk = _parse_data_payload(raw_line)
        if chunk is None:
            continue
        _apply_chunk(assembly, chunk)
        if assembly.choice_error is not None:
            raise UpstreamError("OpenRouter provider returned an error")
    return assembly


def _parse_data_payload(line: str) -> dict[str, Any] | None:
    payload = line.strip()[len(_DATA_PREFIX) :].strip()
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def _apply_chunk(assembly: StreamAssembly, chunk: dict[str, Any]) -> None:
    if chunk.get("model"):
        assembly.model = str(chunk["model"])
    if chunk.get("provider") is not None:
        assembly.provider = str(chunk["provider"])
    usage = chunk.get("usage")
    if isinstance(usage, dict):
        assembly.usage = usage

    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice = choices[0]
    if not isinstance(choice, dict):
        return
    if choice.get("error"):
        assembly.choice_error = choice["error"]
        return
    delta = choice.get("delta")
    if isinstance(delta, dict):
        piece = delta.get("content")
        if isinstance(piece, str):
            assembly.content += piece
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None:
        assembly.finish_reason = finish_reason
