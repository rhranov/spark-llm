"""Pure logic shared with image_server.py, deliberately dependency-free (no
torch/diffusers/fastapi/pydantic) so it can be imported and unit-tested from
test_console_paths.py on a host that has none of those installed — they only
exist inside the spark-llm-diffusers container, never on the host running
spark_llm.py itself.

extract_prompt() takes anything with .role/.content attributes (a pydantic
ChatMessage in the real server, a plain types.SimpleNamespace in tests) —
never imports pydantic itself, so tests don't need it either.
"""


def parse_size(size: str, fallback: str) -> tuple[int, int]:
    for candidate in (size, fallback):
        if candidate and "x" in candidate:
            try:
                w, h = candidate.lower().split("x", 1)
                return int(w), int(h)
            except ValueError:
                continue
    return 1328, 1328


def extract_prompt(messages) -> str:
    """The chat-completions shim treats the latest user message as the image
    prompt — the standard convention other OpenAI-compatible image-model
    shims use to make a pure image generator "chattable" in a normal chat UI,
    since /v1/chat/completions has no notion of an image request otherwise."""
    for msg in reversed(messages):
        if msg.role != "user" or msg.content is None:
            continue
        if isinstance(msg.content, str):
            return msg.content
        parts = [p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text"]
        if parts:
            return " ".join(parts).strip()
    return ""


def is_auxiliary_task(prompt: str) -> bool:
    """MEASURED 2026-07-23: every one of Open WebUI's own background jobs
    (title generation, tag generation, follow-up-question suggestions) that
    get routed here when this model is the active chat model use a prompt
    starting with the literal string "### Task:" — confirmed against its
    title/tags/follow-up templates. A real user prompt is never going to
    start with that. Used to skip a ~5-minute generation these jobs can't
    even use the result of."""
    return prompt.strip().startswith("### Task:")


def chunk_text(content: str, chunk_size: int) -> list[str]:
    """Split content into chunk_size-or-smaller pieces, always yielding at
    least one piece (even "") so callers can always send an initial
    role-establishing delta. MEASURED 2026-07-23: a data-URI image is several
    MB of base64 with no newlines — sending it as one SSE line broke Open
    WebUI's client (a line-length cap common to SSE/streaming parsers), so
    chunk_size must stay safely under that limit (measured trigger: 131072
    bytes). A SEPARATE issue (also measured directly): too many small chunks
    makes a client that re-renders markdown on every incoming piece redraw a
    multi-MB string hundreds of times, which can be slow enough to show as a
    visible, if transient, wall of raw un-rendered text before the final
    chunk completes the markdown and it gets replaced by the real image —
    keep chunk_size as large as the line-length limit allows, not as small as
    correctness alone would require."""
    indices = list(range(0, len(content), chunk_size)) or [0]
    return [content[i:i + chunk_size] for i in indices]
