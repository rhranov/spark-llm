"""spark-llm diffusers engine — minimal OpenAI-images-API-compatible server.

Runs inside the spark-llm-diffusers container (see Dockerfile.diffusers), one
process per loaded declaration, same lifecycle as every other spark-llm slot
(started/stopped by vllm@<decl>.service via `spark-llm launch <decl>`).

Deliberately minimal (PROPOSAL.md §15 scope discipline: one operator, one GPU,
no queueing infrastructure): a single generation runs at a time behind one
lock; a second request waits rather than failing. No image-to-image editing,
no LoRA, no batching — those are separate, unbuilt features.

The pipeline is loaded to readiness BEFORE the HTTP listener starts, so any
successful connection to --port already implies the model is loaded — this is
what lets spark-llm's existing engine-agnostic readiness poll (GET /v1/models)
work unmodified for this engine.
"""
import argparse
import base64
import io
import json
import sys
import threading
import time
import uuid

import torch
import uvicorn
from diffusers import DiffusionPipeline
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from image_server_lib import parse_size, extract_prompt, is_auxiliary_task, chunk_text


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to the diffusers pipeline directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--served-name", required=True, help="Model id returned by /v1/models")
    p.add_argument("--default-size", default="1328x1328", help="WxH used when a request omits size")
    p.add_argument("--steps", type=int, default=50, help="num_inference_steps default")
    p.add_argument("--cfg-scale", type=float, default=4.0, help="true_cfg_scale default")
    return p.parse_args()


class ImageGenRequest(BaseModel):
    model: str | None = None
    prompt: str
    n: int = 1
    size: str | None = None
    response_format: str | None = None
    negative_prompt: str | None = None
    num_inference_steps: int | None = None
    true_cfg_scale: float | None = None
    seed: int | None = None


class ChatMessage(BaseModel):
    role: str
    content: str | list | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    stream: bool | None = False
    seed: int | None = None  # standard OpenAI chat-completions field, reused here for reproducible generation


def build_app(args) -> FastAPI:
    app = FastAPI()
    gen_lock = threading.Lock()
    state = {"pipe": None, "loaded_at": None}

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": args.served_name, "object": "model"}]}

    def _generate(prompt: str, negative_prompt: str | None, width: int, height: int,
                  steps: int, cfg_scale: float, seed: int | None):
        """Runs one generation under the shared lock. Returns (b64_png, error_message)."""
        pipe = state["pipe"]
        if pipe is None:
            return None, "model not loaded"
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(int(seed))
        acquired = gen_lock.acquire(timeout=600)
        if not acquired:
            return None, "generator busy — timed out waiting for the lock"
        try:
            t0 = time.time()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or " ",
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=cfg_scale,
                generator=generator,
            )
            image = result.images[0]
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            print(f"generated {width}x{height}, {steps} steps, {time.time()-t0:.1f}s", flush=True)
            return base64.b64encode(buf.getvalue()).decode("ascii"), None
        except Exception as e:
            print(f"generation failed: {e}", file=sys.stderr, flush=True)
            return None, str(e)
        finally:
            gen_lock.release()

    @app.post("/v1/images/generations")
    def generate(req: ImageGenRequest):
        width, height = parse_size(req.size or "", args.default_size)
        steps = req.num_inference_steps or args.steps
        cfg_scale = req.true_cfg_scale if req.true_cfg_scale is not None else args.cfg_scale
        images_out = []
        for _ in range(max(1, req.n)):
            b64, err = _generate(req.prompt, req.negative_prompt, width, height, steps, cfg_scale, req.seed)
            if err:
                status = 503 if "not loaded" in err or "busy" in err else 500
                return JSONResponse(status_code=status, content={"error": err})
            images_out.append({"b64_json": b64})
        return {"created": int(time.time()), "data": images_out}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        """Lets a normal chat UI 'talk to' this image model directly: the
        latest user message becomes the prompt, the reply is the generated
        image embedded as a markdown data URI — the standard shim shape for
        making an image-only backend chattable (no text understanding here,
        Qwen-Image's own text encoder is used only to condition the image).

        MEASURED 2026-07-23: when this model is the active chat model, Open
        WebUI also routes its own internal background jobs here (title
        generation, tag generation, follow-up-question suggestions) — each one
        otherwise burns a full ~5-minute image generation on text this model
        can't even use, roughly doubling perceived latency per real request.
        Every one of Open WebUI's own default templates for these starts with
        the literal string "### Task:" (confirmed against its title/tags/
        follow-up templates) — a real user prompt is never going to start with
        that. Detect it and return instantly with empty content instead of
        generating: this model has no text capability to genuinely serve
        these jobs anyway, so an honest empty answer is correct, not a
        workaround."""
        prompt = extract_prompt(req.messages)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if is_auxiliary_task(prompt):
            content = ""
        elif not prompt:
            content = "Image generation failed: no user message found to use as an image prompt"
        else:
            width, height = parse_size("", args.default_size)
            # req.seed: standard OpenAI chat-completions field — wired through
            # 2026-07-23 so a chat-based generation can be reproduced (was
            # hardcoded to None before, every chat generation was unseeded).
            b64, err = _generate(prompt, None, width, height, args.steps, args.cfg_scale, req.seed)
            content = f"![generated image](data:image/png;base64,{b64})" if b64 else f"Image generation failed: {err}"

        if req.stream:
            # MEASURED bug #1 (2026-07-23): a data-URI image is several MB of
            # base64 with no newlines — sending it as ONE SSE data line broke
            # Open WebUI's client ("Got more than 131072 bytes when reading"),
            # a line-length cap common to SSE/streaming parsers. Fix: chunk it
            # across multiple deltas so no single line approaches that limit.
            #
            # MEASURED bug #2 (2026-07-23, found by comparing the server
            # log side-by-side with the UI): the whole image is ALREADY fully
            # generated before the first chunk is even sent — there is no real
            # token-by-token streaming here, just delivery of a finished
            # result. The first fix chunked it into ~700 pieces of 4096 bytes,
            # and if the client re-renders the accumulating markdown text on
            # every incoming piece (typical "live typing" chat UX), that is
            # ~700 progressively-larger re-renders of a multi-MB string —
            # genuinely expensive to redraw. Observed symptom:
            # the raw, not-yet-valid markdown showed as literal text for a
            # visible period, then got replaced by the correctly rendered
            # image once the closing ")" chunk finally landed — not a
            # permanent failure, a slow transient one. Fix: use far fewer,
            # larger chunks (~100 KB instead of 4 KB) — still safely under the
            # 131072-byte line limit from bug #1, but roughly 25x fewer
            # re-render events for the client to fall behind on.
            CHUNK = 100_000

            def sse():
                first = True
                for piece in chunk_text(content, CHUNK):
                    delta = {"role": "assistant", "content": piece} if first else {"content": piece}
                    first = False
                    chunk = {
                        "id": completion_id, "object": "chat.completion.chunk", "created": created,
                        "model": args.served_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                done_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": args.served_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(done_chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        return {
            "id": completion_id, "object": "chat.completion", "created": created,
            "model": args.served_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def load_pipeline():
        print(f"loading pipeline from {args.weights} ...", flush=True)
        t0 = time.time()
        pipe = DiffusionPipeline.from_pretrained(args.weights, torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        state["pipe"] = pipe
        state["loaded_at"] = time.time()
        print(f"pipeline ready after {time.time()-t0:.1f}s", flush=True)

    app.state.load_pipeline = load_pipeline
    return app


def main():
    args = parse_args()
    app = build_app(args)
    # Load BEFORE the listener starts: any successful connection to --port
    # already implies the model is loaded, matching every other engine's
    # readiness semantics without needing a separate readiness code path.
    app.state.load_pipeline()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
