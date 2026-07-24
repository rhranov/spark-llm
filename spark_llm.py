#!/usr/bin/env python3
"""spark-llm — LLM model switcher and live console for the DGX Spark.

Three nouns, matching three distinct facts — nothing else is stored:
  - Declaration (models.d/*.toml)   what CAN run: weights, engine, flags
  - Consumer    (config.toml)        what NEEDS something: a stable alias at
                                      a fixed port, e.g. an agent @ 8000
  - Loaded      (loaded.toml)        what IS running: declaration -> port

"Which loaded model fulfills a consumer" is NEVER stored — it is derived by
asking which port each loaded model is on. Only one process can hold a given
port, so the fact always has exactly one true answer, live. (
rejected an earlier draft that stored a separate role/slot/assignment layer —
that was unnecessary abstraction. Assignment is a question you ask a port,
not a fact you write down.)

Config dir: $SPARK_LLM_DIR, default /etc/spark-llm. Layout:
  config.toml            operator settings + [[consumers]] (name/port/alias)
  models.d/<name>.toml    one declaration per model; filename is an opaque
                          label, chosen from the model's own metadata when it
                          carries one, else its folder name (standing rule)
  loaded.toml             declaration -> {port}; the only runtime state
  mode                    "test" or "live" (missing file = test)
  audit.log               every action, every composed command, timestamped
  logs/<decl>.log         server stdout/stderr for a loaded declaration

No model name, alias, or "role" literal appears anywhere in this file. (§5
goal 6 of the original design proposal — still holds.)
"""

import json
import os
import re
import shlex
import subprocess
import sys
import time
import tomllib
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------- paths / config

def config_dir() -> Path:
    return Path(os.environ.get("SPARK_LLM_DIR", "/etc/spark-llm"))


def read_mode() -> str:
    p = config_dir() / "mode"
    if not p.exists():
        return "test"  # safe default: never mutate unless explicitly switched to live
    v = p.read_text().strip().lower()
    return v if v in ("test", "live") else "test"


def load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config() -> dict:
    p = config_dir() / "config.toml"
    if not p.exists():
        die(f"missing {p} — the config dir is not initialised")
    return load_toml(p)


def load_declarations() -> dict:
    """name -> declaration dict. Filenames are opaque labels (naming rule)."""
    d = config_dir() / "models.d"
    decls = {}
    if d.is_dir():
        for f in sorted(d.glob("*.toml")):
            decls[f.stem] = load_toml(f)
    return decls


def toml_dump_flat_tables(d: dict) -> str:
    """Minimal TOML writer for {name: {k: v}} — avoids a dependency for the
    one shape this app ever writes back out. Table names are ALWAYS quoted
    (json.dumps gives a valid TOML basic string for the ASCII names this app
    uses): a bare, unquoted table header treats '.' as nesting syntax, which
    silently turns a declaration name like 'qwen3.6-35b-a3b-nvfp4' into a
    nested table on the next read — corrupting loaded.toml (2026-07-20)."""
    out = []
    for name, fields in sorted(d.items()):
        out.append(f"[{json.dumps(name)}]")
        for k, v in fields.items():
            out.append(f"{k} = {json.dumps(v)}")
        out.append("")
    return "\n".join(out)


def load_loaded() -> dict:
    """declaration_name -> {"port": int}. The only runtime state — what is
    actually meant to be running. Ground truth for HEALTH still comes from
    probing the port live, never from this file alone (§ authoritative-source)."""
    p = config_dir() / "loaded.toml"
    if not p.exists():
        return {}
    return load_toml(p)


def die(msg: str, code: int = 1):
    print(f"spark-llm: error: {msg}", file=sys.stderr)
    sys.exit(code)


# ---------------------------------------------------------------- audit + executor

def audit(tag: str, detail: str):
    """Append one line to the audit log. Always executes, in both modes."""
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} | {read_mode().upper()} | {tag} | {detail}\n"
    p = config_dir() / "audit.log"
    try:
        with open(p, "a") as f:
            f.write(line)
    except PermissionError:
        die(f"cannot write {p} — this account does not own the config dir. "
            f"Use the operator account, or re-run install.sh (ownership step).")


class Executor:
    """The single mutation path. TEST mode: log, don't execute.
    Read-only probes do NOT go through here — they are safe in both modes."""

    def __init__(self):
        self.mode = read_mode()

    @property
    def test(self) -> bool:
        return self.mode != "live"

    def run(self, argv: list[str], desc: str) -> bool:
        """Run a mutating command. Returns success. In TEST mode: audited, skipped.
        Every step announces itself before running and reports its own outcome."""
        cmd = shlex.join(argv)
        if self.test:
            audit("WOULD-EXEC", cmd)
            print(f"  [test] {desc} — would exec: {cmd}", flush=True)
            return True
        audit("EXEC", cmd)
        print(f"  → {desc} ...", flush=True)
        if argv[0] == "sudo" and argv[1] != "-n":
            # -n: fail fast if a password would be needed — a captive subprocess
            # can never answer a prompt, so hanging is the only alternative.
            argv = ["sudo", "-n", *argv[1:]]
        r = subprocess.run(argv, capture_output=True, text=True)
        if r.returncode != 0:
            audit("EXEC-FAIL", f"rc={r.returncode} :: {cmd} :: {r.stderr.strip()[:300]}")
            print(f"  ✗ {desc} FAILED (rc={r.returncode}): {r.stderr.strip()[:200]}",
                  file=sys.stderr, flush=True)
            return False
        print(f"  ✓ {desc}", flush=True)
        return True

    def write_loaded(self, loaded: dict, desc: str) -> bool:
        content = toml_dump_flat_tables(loaded)
        path = config_dir() / "loaded.toml"
        if self.test:
            audit("WOULD-WRITE", f"{path} <- {content!r} ({desc})")
            print(f"  [test] {desc} — would write {path}", flush=True)
            return True
        audit("WRITE", f"{path} <- {content!r} ({desc})")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content)
        tmp.rename(path)
        print(f"  ✓ {desc}", flush=True)
        return True

    def note(self, tag: str, detail: str):
        audit(tag, detail)


# ---------------------------------------------------------------- composition

KNOWN_ENGINES = {"vllm-docker", "llamacpp", "diffusers"}  # refuse anything not implemented


def decl_host_dir(decl: dict, cfg: dict) -> str:
    """Host models directory: per-declaration override, else the canonical
    models_dir from config.toml. RULE: all models live in one canonical
    folder on the Spark disk so the tool can detect them."""
    return decl.get("host_models_dir") or cfg["models_dir"]


def compose_serve_argv(decl: dict, port: int, alias: str | None) -> list[str]:
    """vllm serve arguments. `alias` (a consumer's stable name) is appended to
    served names only when this port is currently fulfilling that consumer —
    computed by the caller from ground truth, never stored on the declaration."""
    served = [decl["served_name"]] + ([alias] if alias else [])
    argv = [
        "vllm", "serve", decl["weights"],
        "--served-model-name", *served,
        "--host", "0.0.0.0", "--port", str(port),
        "--gpu-memory-utilization", str(decl["gpu_memory_utilization"]),
        "--max-model-len", str(decl["max_model_len"]),
    ]
    argv += [str(x) for x in decl.get("extra_flags", [])]
    return argv


def compose_docker_argv(decl_name: str, decl: dict, port: int, cfg: dict, alias: str | None, total_gb: float) -> list[str]:
    """docker run for the vllm-docker engine. --memory/--memory-swap give the
    container a hard cgroup ceiling — 2026-07-20 incident: with no limit set,
    two declarations' combined real usage exceeded the machine's physical
    memory and drove the WHOLE system into swap-thrashing unresponsiveness,
    rather than one container failing cleanly. Capped at the declaration's
    own estimated reserve x weights_overhead (the same fudge factor already
    used for the generic estimate elsewhere) — real usage exceeding a
    declaration's OWN stated budget should fail that one container, not
    degrade the whole machine. memory-swap == memory: zero swap headroom for
    this specific container, so an overrun is an immediate, clean, cgroup-
    level OOM kill instead of slow, system-wide thrashing. Passed in bytes,
    not a 'g' suffix — Docker's own g/m/k suffixes are binary (GiB-based)
    despite the decimal-sounding name, and this app's standing rule is
    decimal GB throughout; computing the byte count directly avoids that
    ambiguity entirely."""
    cap_bytes = int(reserve_estimate_gb(decl, total_gb, cfg)
                     * float(cfg.get("auto", {}).get("weights_overhead", 1.15)) * 1e9)
    return [
        "docker", "run", "--rm", "--name", f"vllm-{decl_name}",
        "--gpus", "all", "--ipc=host",
        "--memory", str(cap_bytes), "--memory-swap", str(cap_bytes),
        "-p", f"{cfg.get('listen_host', '127.0.0.1')}:{port}:{port}",
        "-v", f"{decl_host_dir(decl, cfg)}:/models:ro",
        decl["image"],
        *compose_serve_argv(decl, port, alias),
    ]


def compose_llamacpp_argv(decl: dict, port: int, cfg: dict, alias: str | None) -> list[str]:
    """llama-server runs as a host binary, no container, and accepts exactly
    ONE --alias. It serves the consumer's alias when fulfilling one, else its
    own served name — so it is always addressable, consumer or not."""
    argv = [
        cfg["engines"]["llamacpp_bin"],
        "-m", decl["weights"],
        "--host", str(cfg.get("listen_host", "127.0.0.1")), "--port", str(port),
        "--alias", alias or decl["served_name"],
        "-c", str(decl["max_model_len"]),
        "-ngl", str(decl.get("n_gpu_layers", 999)),
    ]
    argv += [str(x) for x in decl.get("extra_flags", [])]
    return argv


def compose_diffusers_argv(decl_name: str, decl: dict, port: int, cfg: dict, alias: str | None,
                            total_gb: float) -> list[str]:
    """docker run for the diffusers engine (image generation, not text). Reuses
    compose_docker_argv's proven mechanism line-for-line: same --memory/
    --memory-swap cgroup cap (item 38 — the one thing that actually would have
    prevented the 2026-07-20 freeze), same weights bind mount and /models/<folder>
    convention (so disk_size_gb/decl_folder_name/undeclared_on_disk/validate_load
    keep working unmodified — they only special-case llamacpp). Built FROM the
    already-proven vLLM base image (Dockerfile.diffusers) rather than a bare host
    process, to avoid gambling on a fresh torch install matching this machine's
    ARM64/CUDA13/compat-mode driver combination. image_server.py itself is
    bind-mounted in read-only and run as the entrypoint instead of `vllm serve`.

    MEASURED bug, fixed 2026-07-23: this cap used to ALSO multiply by
    weights_overhead on top of reserve_estimate_gb()'s own result — that is
    correct in compose_docker_argv (a vllm-docker declaration's estimate comes
    from gpu_memory_utilization x total, which reserve_estimate_gb does NOT
    itself apply overhead to), but a diffusers declaration has no
    gpu_memory_utilization key, so reserve_estimate_gb() already applies
    weights_overhead internally (the est_weights_gb branch: est x overhead +
    kv_margin) — multiplying again here inflated the real cap from ~74.7 GB to
    ~86 GB for no reason, and made it disagree with the pre-flight estimate
    cmd_load actually validated against. Fixed by not re-applying it here."""
    cap_bytes = int(reserve_estimate_gb(decl, total_gb, cfg) * 1e9)
    script_host_path = Path(__file__).resolve().with_name("image_server.py")
    lib_host_path = Path(__file__).resolve().with_name("image_server_lib.py")
    image = decl.get("image") or cfg.get("auto", {}).get("diffusers_image", "spark-llm-diffusers:latest")
    argv = [
        "docker", "run", "--rm", "--name", f"vllm-{decl_name}",
        "--gpus", "all", "--ipc=host",
        "--memory", str(cap_bytes), "--memory-swap", str(cap_bytes),
        "-p", f"{cfg.get('listen_host', '127.0.0.1')}:{port}:{port}",
        "-v", f"{decl_host_dir(decl, cfg)}:/models:ro",
        "-v", f"{script_host_path}:/opt/image_server.py:ro",
        # MEASURED bug (2026-07-23): image_server.py imports from
        # image_server_lib (extracted for testability) — only image_server.py
        # itself was bind-mounted, so the container crashed at startup with
        # ModuleNotFoundError the moment the lib module was split out. Both
        # files must be mounted into the same directory for the import to
        # resolve; caught by an actual failed load, not by inspection.
        "-v", f"{lib_host_path}:/opt/image_server_lib.py:ro",
        image,
        "python3", "/opt/image_server.py",
        "--weights", f"/models/{decl_folder_name(decl)}",
        "--host", "0.0.0.0", "--port", str(port),
        "--served-name", alias or decl["served_name"],
        "--default-size", str(decl.get("default_size", "1328x1328")),
        "--steps", str(decl.get("num_inference_steps", 50)),
        "--cfg-scale", str(decl.get("true_cfg_scale", 4.0)),
    ]
    # Consistency fix, 2026-07-23: compose_docker_argv and compose_llamacpp_argv
    # both let a declaration pass arbitrary extra_flags through; this engine
    # silently dropped the key. image_server.py's own argparse will reject any
    # flag it doesn't recognize (fail loud at container start), same as every
    # other engine's behavior for a bad extra_flags entry.
    argv += [str(x) for x in decl.get("extra_flags", [])]
    return argv


def compose_launch_argv(decl_name: str, decl: dict, port: int, cfg: dict, alias: str | None,
                         total_gb: float) -> list[str]:
    """Single entry point for launch composition — engine dispatch lives here
    and nowhere else."""
    eng = decl.get("engine")
    if eng == "vllm-docker":
        return compose_docker_argv(decl_name, decl, port, cfg, alias, total_gb)
    if eng == "llamacpp":
        return compose_llamacpp_argv(decl, port, cfg, alias)
    if eng == "diffusers":
        return compose_diffusers_argv(decl_name, decl, port, cfg, alias, total_gb)
    die(f"engine '{eng}' not implemented (known: {sorted(KNOWN_ENGINES)})")


# ---------------------------------------------------------------- consumers / ports

def consumer_for_port(cfg: dict, port: int) -> dict | None:
    return next((c for c in cfg.get("consumers", []) if int(c["port"]) == port), None)


def fulfiller_of(consumer: dict, loaded: dict) -> str | None:
    """Which loaded declaration currently occupies this consumer's port, or
    None. This IS the assignment — asked, never stored."""
    for dn, e in loaded.items():
        if int(e["port"]) == int(consumer["port"]):
            return dn
    return None


def consumer_compatible(decl: dict, consumer: dict) -> bool:
    """Whether `decl` could possibly satisfy `consumer`'s hard requirements.
    Shared by decide_port (so a fresh load is routed to a consumer it can
    actually serve, not just the first unfulfilled one in declaration order)
    and validate_load (so an incompatible pairing is refused, not silently
    assigned) — one place, so the two can never drift apart. Currently the
    supported hard requirements are engine type and tool-calling; add future
    ones here, never duplicate the condition at either call site."""
    required_engine = consumer.get("required_engine")
    if required_engine and decl.get("engine") != required_engine:
        return False
    required_engine = consumer.get("required_engine")
    if required_engine and decl.get("engine") != required_engine:
        return False
    if consumer.get("requires_tool_calling"):
        if decl.get("engine") != "vllm-docker":
            return False
        if "--enable-auto-tool-choice" not in decl.get("extra_flags", []):
            return False
    return True


def decide_port(decl_name: str, decl: dict, loaded: dict, cfg: dict) -> tuple[int, dict | None]:
    """Where should this declaration run, and is it fulfilling a consumer?

    Every call to this function comes from an explicit `load` request — there
    is no read-only caller — so "already loaded, asked for again" always means
    the operator wants to PROMOTE it, not merely re-confirm its current spot.

    Rule:
    - Already fulfilling some consumer at its current port -> stay there
      (idempotent: re-loading the current brain is a no-op).
    - Already loaded but free-floating, and asked for again -> PROMOTE: take
      over a consumer's port, displacing whoever currently holds it. Prefer
      an unfulfilled consumer it can actually satisfy if one exists, else the
      first declared one.
      ( this case previously fell through to "pick an unused
      pool port," which either did nothing or silently relocated the model to
      a random new port — the promote feature never actually worked when the
      consumer it should take over was healthy, which is the exact case it
      exists for.)
    - Never loaded before, and some consumer is unfulfilled AND this
      declaration could actually satisfy it -> take that port (this is also
      what a fresh load does when nothing is running yet). A consumer it
      structurally cannot serve (e.g. a tool-requiring text consumer offered
      an image-generation declaration) is skipped, not claimed — 2026-07-23:
      loading Qwen-Image while the primary consumer's slot happened to also be empty routed
      it onto the primary consumer's port and only failed by accident, at validate_load,
      instead of correctly falling through to a compatible consumer or a free
      pool port.
    - Never loaded before, and every consumer is already served (or none are
      compatible) -> next free pool port, as an independent candidate —
      loading something new must never silently evict a healthy brain; that
      is what promotion is for.
    """
    consumers = cfg.get("consumers", [])
    if decl_name in loaded:
        cur_port = int(loaded[decl_name]["port"])
        c = consumer_for_port(cfg, cur_port)
        if c is not None:
            return cur_port, c
        if consumers:
            unfulfilled = next((c for c in consumers
                                if fulfiller_of(c, loaded) is None and consumer_compatible(decl, c)), None)
            target = unfulfilled or consumers[0]
            return int(target["port"]), target
        return cur_port, None
    for c in consumers:
        if fulfiller_of(c, loaded) is None and consumer_compatible(decl, c):
            return int(c["port"]), c
    used = {int(e["port"]) for e in loaded.values()} | {int(c["port"]) for c in consumers}
    p = int(cfg.get("auto", {}).get("free_port_start", 8001))
    while p in used:
        p += 1
    return p, None


# ---------------------------------------------------------------- validation

def check_naming_rule(decls: dict, cfg: dict) -> list[str]:
    """STANDING RULE: a declaration's label is its own metadata name if it
    carries one, else its folder name — never an invented short name. Warns
    on hand-created drift; auto-declare always conforms by construction."""
    warnings = []
    for dname, d in decls.items():
        try:
            expected = derive_label(decl_folder_name(d), cfg)
            if dname != expected:
                warnings.append(f"declaration '{dname}' breaks the naming rule — the "
                                f"derived label is '{expected}'; rename the file in models.d/")
        except Exception:
            pass
    return warnings


def validate_load(decl_name: str, decl: dict, port: int, cfg: dict,
                   consumer: dict | None) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors block a load."""
    errors, warnings = [], []
    eng = decl.get("engine")
    if eng not in KNOWN_ENGINES:
        errors.append(f"declaration '{decl_name}': engine '{eng}' not implemented "
                      f"(known: {sorted(KNOWN_ENGINES)})")
    if consumer is not None:
        if decl.get("served_name") == consumer["alias"]:
            errors.append(f"declaration '{decl_name}': its own served name equals "
                          f"consumer alias '{consumer['alias']}' — must be distinct")
        # Headroom rule (incident B): only matters once this declaration is
        # actually facing a consumer — an idle candidate on a side port
        # doesn't need to satisfy it yet. Meaningless for the diffusers engine
        # (2026-07-23: image generation has no text-context concept at all —
        # every diffusers declaration has max_model_len 0, which isn't a real
        # deficiency, just a question that doesn't apply to it).
        if eng != "diffusers":
            min_ctx = int(cfg["consumer_context_length"]) * int(cfg.get("headroom_factor", 2))
            mml = int(decl.get("max_model_len", 0))
            if mml < min_ctx:
                errors.append(f"declaration '{decl_name}': max_model_len {mml} < required "
                              f"{min_ctx} (consumer_context_length x headroom_factor) — "
                              f"incident-B class")
        # Tool-calling rule (2026-07-20: an auto-declared model silently broke
        # the primary consumer — healthy, but incapable of the tool calls it needs on nearly
        # every message; vLLM only enables tool-calling with an explicit flag).
        # A consumer that requires it MUST be refused a declaration that
        # hasn't turned it on — caught here, before the load, not discovered
        # by the consumer failing after the fact. The exact --tool-call-parser
        # value for a new model family still needs one human check (its
        # chat_template.jinja against vLLM's ToolParserManager list) — that
        # part cannot be automated away; what this rule removes is the SILENT
        # failure, not the one-time research. Gate condition shared with
        # decide_port via consumer_compatible() so routing and refusal can
        # never drift apart (2026-07-23).
        if consumer.get("requires_tool_calling") and not consumer_compatible(decl, consumer):
            if eng == "vllm-docker":
                errors.append(
                        f"declaration '{decl_name}': consumer '{consumer['name']}' requires "
                        f"tool-calling, but this declaration does not enable it (no "
                        f"--enable-auto-tool-choice in extra_flags). Check the model's "
                        f"chat_template.jinja for its tool-call format, find the matching "
                        f"--tool-call-parser in vLLM's ToolParserManager list, and add both "
                        f"flags — or confirm this consumer genuinely does not need tools.")
            else:
                # Every other engine (llamacpp today; any future addition) is
                # refused outright for a tool-requiring consumer, not silently
                # let through. llamacpp's tool-calling mechanism has never been
                # verified end-to-end in this project — the vllm-docker branch
                # above exists only because that path WAS verified (2026-07-20
                # incident). An unverified engine facing a tool-requiring
                # consumer is the same silent-failure shape as that incident;
                # refusing it here is the fix, not a new rule.
                errors.append(
                    f"declaration '{decl_name}': consumer '{consumer['name']}' requires "
                    f"tool-calling, but engine '{eng}' has no verified tool-calling support "
                    f"in this project — refusing rather than risking a repeat of the "
                    f"2026-07-20 nemotron incident. Verify tool-calling end-to-end for this "
                    f"engine first, or serve this consumer from a vllm-docker declaration.")
    if eng == "llamacpp":
        binp = cfg.get("engines", {}).get("llamacpp_bin", "")
        if not (binp and Path(binp).exists()):
            errors.append(f"declaration '{decl_name}': llamacpp engine binary missing "
                          f"({binp or 'engines.llamacpp_bin unset'})")
        host_path = Path(decl["weights"])
    else:
        host_path = Path(decl_host_dir(decl, cfg)) / Path(decl["weights"]).name
    try:
        if not host_path.exists():
            errors.append(f"declaration '{decl_name}': weights not found at {host_path}")
    except PermissionError:
        warnings.append(f"declaration '{decl_name}': cannot read {host_path} from this "
                        f"account — weights existence UNVERIFIED (hard requirement live)")
    return errors, warnings


# ---------------------------------------------------------------- read-only probes

def probe_models(port: int, timeout: float = 3.0) -> dict | None:
    """GET /v1/models. Returns parsed JSON or None. Read-only, safe in all modes."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=timeout) as r:
            return json.load(r)
    except Exception:
        return None


def served_names(port: int) -> list[str]:
    """Model names from the live server, or []. (The 'created' field is NOT
    usable as uptime: MEASURED 2026-07-19, vLLM regenerates it per request.)"""
    data = probe_models(port)
    if not data:
        return []
    return [m["id"] for m in data.get("data", [])]


def unit_state(decl_name: str) -> str:
    """ActiveState of the declaration's unit: active/activating/failed/inactive.
    Used to abort a readiness wait early when the unit already died — without
    this, a model that crashes in 2 s wastes the full poll timeout."""
    try:
        r = subprocess.run(
            ["systemctl", "show", f"vllm@{decl_name}.service", "--property=ActiveState", "--value"],
            capture_output=True, text=True, timeout=3)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def unit_rss_gb(decl_name: str) -> float | None:
    """Resident memory of the declaration's main process, decimal GB, or None.
    Meaningful for host-binary engines (llamacpp); docker's MainPID is the
    client, so callers must not use this for docker engines."""
    try:
        r = subprocess.run(["systemctl", "show", f"vllm@{decl_name}.service",
                            "--property=MainPID", "--value"],
                           capture_output=True, text=True, timeout=3)
        pid = int(r.stdout.strip() or 0)
        if pid <= 0:
            return None
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * KB_TO_GB
    except Exception:
        return None
    return None


def unit_memory_gb(decl: dict) -> float | None:
    """Real, live memory usage of a loaded declaration's actual workload,
    decimal GB, or None. MEASURED 2026-07-20: neither systemd's own unit
    cgroup (17 MB — docker's container is NOT nested under it here) nor the
    vllm process's own RSS (~691 MB) remotely reflects its real ~111 GB
    unified-memory footprint; only nvidia-smi's per-process compute-app
    accounting reports it honestly. Finds the right process by matching the
    declaration's own weights path AND an engine-specific marker in its
    command line (excluding the docker CLI wrapper, whose own command line
    also embeds that same text as trailing args and would otherwise match
    first), then sums nvidia-smi's usage across that process and every
    descendant (vllm-docker forks worker/engine subprocesses that hold the
    actual GPU allocation; diffusers is a single process but the same walk is
    harmless for it). The marker is the only engine-aware piece of this
    function — kept here, in one place, rather than scattered (PROPOSAL.md
    §16.3: 'keep engine-specific knowledge in one place')."""
    weights = decl.get("weights", "")
    if not weights:
        return None
    marker = "image_server.py" if decl.get("engine") == "diffusers" else "vllm serve"
    try:
        ps_out = subprocess.run(["ps", "-e", "-o", "pid=,ppid=,args="],
                                 capture_output=True, text=True, timeout=5).stdout
        rows = []
        for line in ps_out.splitlines():
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
        target = next((pid for pid, _, args in rows
                       if weights in args and marker in args and not args.startswith("docker")),
                      None)
        if target is None:
            return None
        relevant = {target}
        changed = True
        while changed:
            changed = False
            for pid, ppid, _ in rows:
                if ppid in relevant and pid not in relevant:
                    relevant.add(pid)
                    changed = True
        smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        total_mib = sum(int(mem) for pid_s, mem in
                        (line.split(",") for line in smi.strip().splitlines())
                        if int(pid_s) in relevant)
        return total_mib * 1024 * 1024 / 1e9 if total_mib else None
    except Exception:
        return None


def unit_uptime_s(decl_name: str) -> float | None:
    """Uptime via systemd monotonic timestamp, or None if the unit doesn't exist."""
    try:
        r = subprocess.run(
            ["systemctl", "show", f"vllm@{decl_name}.service",
             "--property=ActiveEnterTimestampMonotonic", "--value"],
            capture_output=True, text=True, timeout=3)
        usec = int(r.stdout.strip() or 0)
        if usec <= 0:
            return None
        now_uptime = float(Path("/proc/uptime").read_text().split()[0])
        return now_uptime - usec / 1e6
    except Exception:
        return None


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        out[k] = int(v.strip().split()[0])  # kB
    return out


# UNITS: the whole app displays DECIMAL GB (x10^9), matching the NVIDIA Sync
# dashboard, which used as the reference. 108.9 GiB == 116.9 GB — same
# measurement (MemTotal - MemAvailable), different unit; not a discrepancy.
KB_TO_GB = 1024 / 1e9


def mem_gb() -> tuple[float, float, float]:
    """(total, available, swap_used) in decimal GB. `total` is capped at
    config.toml's max_memory_gb (the configured safety policy, 2026-07-20) — never the
    raw OS-reported total. The raw total (130.6 GB on this machine) proved
    optimistic enough, combined with an unbounded docker container, to freeze
    the whole machine that night; every downstream estimate (pre-flight,
    the docker --memory cap, the status display) must work from one
    operator-set ceiling, all consistently, never hardware's own report.
    `used` is always the real, measured figure — only the ceiling used to
    derive 'available' is capped. Missing config key -> no cap (the raw
    total), so an un-migrated config.toml behaves exactly as before."""
    m = meminfo()
    real_total = m["MemTotal"] * KB_TO_GB
    real_avail = m["MemAvailable"] * KB_TO_GB
    cap = load_config().get("max_memory_gb")
    total = min(real_total, float(cap)) if cap is not None else real_total
    avail = max(0.0, total - (real_total - real_avail))
    return (total, avail, (m.get("SwapTotal", 0) - m.get("SwapFree", 0)) * KB_TO_GB)


def fmt_gb(value_gb: float) -> str:
    """Format a quantity already expressed in decimal GB, 2 decimal places,
    auto-scaling to TB/PB as it grows. This app should keep reading correctly
    on far larger memory/disk than today's without a display rewrite."""
    if value_gb >= 1e6:
        return f"{value_gb / 1e6:.2f} PB"
    if value_gb >= 1e3:
        return f"{value_gb / 1e3:.2f} TB"
    return f"{value_gb:.2f} GB"


_host_ip_cache = [None]


def host_ip() -> str:
    """This machine's LAN IP (cached per process) — for displaying served URLs."""
    if _host_ip_cache[0] is None:
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            _host_ip_cache[0] = s.getsockname()[0]
            s.close()
        except Exception:
            _host_ip_cache[0] = "localhost"
    return _host_ip_cache[0]


def endpoint_host(cfg: dict) -> str:
    """Host shown to clients. Loopback is the safe default; wildcard listeners
    are displayed using the machine's LAN address."""
    configured = str(cfg.get("listen_host", "127.0.0.1"))
    if configured in ("127.0.0.1", "::1", "localhost"):
        return "localhost"
    if configured in ("0.0.0.0", "::"):
        return host_ip()
    return configured


def gpu_util_pct() -> int | None:
    """GPU utilisation percent via nvidia-smi, or None when unavailable.
    (memory.used is N/A on GB10; utilization.gpu is MEASURED working.)"""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=3)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


_models_size_cache = {"t": 0.0, "v": None}


def disk_stats_gb(cfg: dict) -> tuple[float, float, float | None]:
    """(disk_total, disk_used, models_folder_size) in decimal GB. Folder size
    is cached 120 s — it walks the whole tree; None when unreadable."""
    import shutil as _shutil
    d = _shutil.disk_usage(cfg["models_dir"])
    if time.time() - _models_size_cache["t"] > 120:
        try:
            total = sum(f.stat().st_size for f in Path(cfg["models_dir"]).rglob("*")
                        if f.is_file())
            _models_size_cache["v"] = total / 1e9
        except (PermissionError, OSError):
            _models_size_cache["v"] = None
        _models_size_cache["t"] = time.time()
    return d.total / 1e9, (d.total - d.free) / 1e9, _models_size_cache["v"]


def disk_size_gb(decl: dict, cfg: dict) -> float | None:
    """Weights size on disk in decimal GB, or None if unreadable/absent —
    never a silent 0 for a path that does not exist."""
    if decl.get("engine") == "llamacpp":
        host_path = Path(decl["weights"]).parent
    else:
        host_path = Path(decl_host_dir(decl, cfg)) / Path(decl["weights"]).name
    try:
        if not host_path.exists():
            return None
        total = sum(f.stat().st_size for f in host_path.rglob("*") if f.is_file())
        return total / 1e9
    except (PermissionError, OSError):
        return None


def decl_folder_name(decl: dict) -> str:
    """The models_dir folder a declaration covers. vllm-docker weights name the
    model DIRECTORY; llamacpp weights name a FILE inside it — the folder is
    its parent."""
    w = Path(decl["weights"])
    return w.parent.name if decl.get("engine") == "llamacpp" else w.name


def undeclared_on_disk(cfg: dict, decls: dict) -> list[str] | None:
    """RULE: every directory in the canonical models_dir is a model. Returns
    those with no declaration yet, or None when the folder is unreadable from
    this account — reported honestly, never silently empty."""
    declared = {decl_folder_name(d) for d in decls.values()}
    try:
        return sorted(p.name for p in Path(cfg["models_dir"]).iterdir()
                      if p.is_dir() and p.name not in declared)
    except (PermissionError, FileNotFoundError, OSError):
        return None


def reserve_estimate_gb(decl: dict, total_gb: float, cfg: dict) -> float:
    """Estimated memory reservation in decimal GB, engine-aware: vLLM
    pre-reserves utilisation x total; llama.cpp roughly weights + margin."""
    if "gpu_memory_utilization" in decl:
        return float(decl["gpu_memory_utilization"]) * total_gb
    a = cfg.get("auto", {})
    est = (float(decl["est_weights_gb"]) if "est_weights_gb" in decl
           else float(decl.get("est_weights_gib", 0)) * 1.073741824)
    return est * float(a.get("weights_overhead", 1.15)) + float(a.get("kv_margin_gib", 8))


# ---------------------------------------------------------------- live state

def consumer_states(cfg: dict, loaded: dict | None = None) -> list[dict]:
    """Reality check for every declared consumer endpoint — probed live,
    independent of anything selected. Rendered ALWAYS, above everything else:
    'one true source' must mean true about the machine (the operator, 2026-07-20,
    after the primary consumer was down with nothing on screen saying so)."""
    loaded = loaded if loaded is not None else load_loaded()
    out = []
    for c in cfg.get("consumers", []):
        names = served_names(int(c["port"]))
        state = "down" if not names else ("ok" if c["alias"] in names else "alias-missing")
        out.append({**c, "state": state, "served": names,
                    "decl_name": fulfiller_of(c, loaded)})
    return out


def loaded_states(decls: dict, cfg: dict) -> list[dict]:
    """What is actually running, for every entry in loaded.toml, matched
    against live reality — never trusted blindly (§ authoritative-source)."""
    loaded = load_loaded()
    result = []
    for decl_name, entry in loaded.items():
        port = int(entry["port"])
        names = served_names(port)
        consumer = consumer_for_port(cfg, port)
        decl = decls.get(decl_name)
        if decl is None:
            memory_gb = None
        elif decl.get("engine") == "llamacpp":
            memory_gb = unit_rss_gb(decl_name)  # host binary: MainPID's own RSS is meaningful
        else:
            memory_gb = unit_memory_gb(decl)  # docker: MainPID is just the CLI client, not the workload
        result.append({
            "decl_name": decl_name, "port": port, "served": names,
            "uptime_s": unit_uptime_s(decl_name),
            "up": bool(names),
            "unit_state": unit_state(decl_name),
            "memory_gb": memory_gb,
            "consumer": consumer,
            "alias_ok": (consumer is None) or (consumer["alias"] in names),
        })
    return result


SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def loaded_status_text(s: dict) -> tuple[str, str]:
    """The ONE display status for a loaded-model entry — text + style — used
    by every table that shows one, so they can never disagree again. (the operator,
    2026-07-20: the Models table showed 'LOADED' in green for a model that
    had been mid-load for 130s and was correctly shown DOWN elsewhere.)"""
    if s["up"]:
        if s["consumer"] and not s["alias_ok"]:
            return "✗ alias NOT served", "bold red"
        u = s["uptime_s"]
        up = f", up {u/3600:.1f}h" if u and u > 3600 else (f", up {u/60:.0f}m" if u else "")
        return f"✓ healthy{up}", "green"
    if s["unit_state"] in ("failed", "inactive"):
        return "✗ DOWN", "bold red"
    spin = SPIN[int(time.time() * 8) % len(SPIN)]
    return f"{spin} LOADING", "yellow"


# ---------------------------------------------------------------- load / unload

def cmd_load(args):
    if not args:
        die("usage: spark-llm load <declaration>")
    decl_name = args[0]
    cfg = load_config()
    decls = load_declarations()
    if decl_name not in decls:
        die(f"declaration '{decl_name}' not found in models.d/")
    decl = decls[decl_name]
    loaded = load_loaded()

    port, consumer = decide_port(decl_name, decl, loaded, cfg)

    if decl_name in loaded and int(loaded[decl_name]["port"]) == port:
        msg = f"'{decl_name}' already loaded on port {port}"
        msg += f" — fulfilling {consumer['name']}" if consumer else " (not fulfilling any consumer)"
        print(msg)
        return

    ex = Executor()
    ex.note("LOAD-BEGIN", f"target '{decl_name}' -> port {port}"
                          f"{' fulfilling ' + consumer['name'] if consumer else ''} (mode={ex.mode})")

    for w in check_naming_rule(decls, cfg):
        print(f"  warning: {w}")
        ex.note("VALIDATE-WARN", w)
    errors, warnings = validate_load(decl_name, decl, port, cfg, consumer)
    for w in warnings:
        print(f"  warning: {w}")
        ex.note("VALIDATE-WARN", w)
    if errors:
        for e in errors:
            print(f"  error: {e}", file=sys.stderr)
            ex.note("VALIDATE-FAIL", e)
        die("validation failed — nothing was changed")
    if not ex.test and any("UNVERIFIED" in w for w in warnings):
        die("weights unverifiable in live mode — refusing to proceed")

    total, avail, _ = mem_gb()
    displaced = next((dn for dn, e in loaded.items()
                      if int(e["port"]) == port and dn != decl_name), None)
    was_loaded_before = decl_name in loaded
    need = reserve_estimate_gb(decl, total, cfg)
    freed = reserve_estimate_gb(decls[displaced], total, cfg) if displaced in decls else 0.0
    # Promoting an already-loaded declaration stops its OWN current instance
    # before rebinding to the new port (below) — that frees its own reserve
    # first, so it must count toward what's available, not just what's needed.
    if was_loaded_before:
        freed += need
    # Cumulative check (2026-07-20 incident): the check above only ever
    # compared `need` against the ONE declaration being displaced or this
    # declaration's own freed reservation — a second, DIFFERENT declaration
    # already loaded elsewhere was never counted at all. That gap let a
    # 45.71 GB load proceed alongside an already-loaded 52.24 GB one with no
    # check whatsoever that they'd fit together, and the combination froze
    # the whole machine. `still_loaded` is every declaration that keeps
    # running through this operation (everyone except the one being loaded
    # and whichever it displaces); `committed` is their summed estimate.
    still_loaded = {dn: e for dn, e in loaded.items() if dn != decl_name and dn != displaced}
    committed = sum(reserve_estimate_gb(decls[dn], total, cfg) for dn in still_loaded if dn in decls)
    safety_margin = float(cfg.get("min_free_gib_after_load", 4))
    msg = f"est. reserve {fmt_gb(need)}; available now {fmt_gb(avail)}" + (f" + est. freed {fmt_gb(freed)} (unloading {displaced})" if displaced else "") \
        + (" + own reservation (rebinding, not a second instance)" if was_loaded_before else "") \
        + (f"; {len(still_loaded)} other declaration(s) staying loaded commit {fmt_gb(committed)}" if still_loaded else "")
    print(f"  pre-flight: {msg}")
    ex.note("PREFLIGHT", msg)
    if need > avail + freed:
        ex.note("PREFLIGHT-FAIL", msg)
        die("pre-flight estimate says this cannot fit — refusing")
    if need + committed + safety_margin > total:
        ex.note("PREFLIGHT-FAIL", msg)
        die(f"pre-flight estimate says this cannot fit alongside {len(still_loaded)} other "
            f"loaded declaration(s) ({', '.join(sorted(still_loaded))}) without leaving at "
            f"least {fmt_gb(safety_margin)} free — refusing")

    # Rollback scope: exactly the two declarations this operation touches.
    prev_loaded_snapshot = dict(loaded)

    if displaced:
        ex.run(["sudo", "systemctl", "stop", f"vllm@{displaced}.service"],
               f"Stopping {displaced} (freeing port {port})")
    if was_loaded_before:
        ex.run(["sudo", "systemctl", "stop", f"vllm@{decl_name}.service"],
               f"Stopping {decl_name} (rebinding to port {port})")

    new_loaded = dict(loaded)
    new_loaded.pop(displaced, None)
    new_loaded[decl_name] = {"port": port}
    ex.write_loaded(new_loaded, f"Recording '{decl_name}' on port {port}")
    ex.run(["sudo", "systemctl", "start", f"vllm@{decl_name}.service"],
           f"Starting {decl_name}" + (f" as '{consumer['alias']}'" if consumer else ""))

    wait_for = consumer["alias"] if consumer else decl["served_name"]
    timeout = int(cfg.get("readiness_timeout_s", 900))

    def do_rollback():
        ex.note("ROLLBACK-BEGIN", f"restoring pre-load state for '{decl_name}'")
        ex.run(["sudo", "systemctl", "stop", f"vllm@{decl_name}.service"], "Stopping the failed model")
        ex.write_loaded(prev_loaded_snapshot, "Restoring previous loaded-models state")
        if displaced:
            ex.run(["sudo", "systemctl", "start", f"vllm@{displaced}.service"], f"Restarting {displaced}")
        if was_loaded_before:
            ex.run(["sudo", "systemctl", "start", f"vllm@{decl_name}.service"], f"Restarting {decl_name} at its previous port")
        ex.note("ROLLBACK-END", "restored — verify readiness manually if this was live")

    if ex.test:
        ex.note("WOULD-POLL", f"/v1/models on :{port} until '{wait_for}' appears, "
                              f"timeout {timeout}s; on timeout: rollback")
        print(f"  [test] would poll :{port} for '{wait_for}' (timeout {timeout}s), roll back on failure")
        ex.note("LOAD-END", f"'{decl_name}' simulated on port {port}")
        print(f"[test] simulated load of '{decl_name}' on port {port} — nothing was changed")
        return

    print(f"  waiting for '{wait_for}' on :{port} ...", flush=True)
    t0 = time.time()
    ok = died = False
    last_tick = 0.0
    while time.time() - t0 < timeout:
        if wait_for in served_names(port):
            ok = True
            break
        state = unit_state(decl_name)
        if state in ("failed", "inactive"):
            died = True
            print(f"  ✗ unit vllm@{decl_name} is '{state}' — crashed or could not "
                  f"start; aborting the wait", flush=True)
            ex.note("READY-UNIT-DEAD", f"'{decl_name}' unit state '{state}' after {time.time()-t0:.0f}s")
            break
        elapsed = time.time() - t0
        if elapsed - last_tick >= 15:
            print(f"  ... still loading, {elapsed:.0f}s elapsed (timeout {timeout}s)", flush=True)
            last_tick = elapsed
        time.sleep(5)

    if not ok:
        why = "the model process died during startup" if died else f"'{wait_for}' absent after {timeout}s"
        ex.note("READY-TIMEOUT", f"'{decl_name}': {why}")
        print(f"  ✗ READINESS FAILED ({why}) — rolling back", flush=True)
        do_rollback()
        die("load failed and was rolled back")

    elapsed = int(time.time() - t0)
    ex.note("READY", f"'{decl_name}' serving '{wait_for}' after {elapsed}s")
    print(f"  ready after {elapsed}s")
    _, avail_after, _ = mem_gb()
    floor = float(cfg.get("min_free_gib_after_load", 4))
    ex.note("HEADROOM", f"available after load: {fmt_gb(avail_after)} (floor {fmt_gb(floor)})")
    if avail_after < floor:
        print(f"  headroom violated ({fmt_gb(avail_after)} < {fmt_gb(floor)}) — rolling back")
        do_rollback()
        die("headroom violated after load — rolled back")

    ex.note("LOAD-END", f"'{decl_name}' active on port {port}")
    print(f"DONE: '{decl_name}' serving on port {port}"
          + (f" as '{consumer['alias']}' for {consumer['name']}" if consumer else "")
          + f" — http://{endpoint_host(cfg)}:{port}/v1")


def cmd_unload(args):
    if not args:
        die("usage: spark-llm unload <declaration>")
    decl_name = args[0]
    loaded = load_loaded()
    if decl_name not in loaded:
        print(f"'{decl_name}' is not loaded — nothing to do")
        return
    ex = Executor()
    ex.note("UNLOAD-BEGIN", f"'{decl_name}'")
    ex.run(["sudo", "systemctl", "stop", f"vllm@{decl_name}.service"], f"Stopping {decl_name}")
    new_loaded = dict(loaded)
    del new_loaded[decl_name]
    ex.write_loaded(new_loaded, f"Removing '{decl_name}' from loaded models")
    ex.note("UNLOAD-END", f"'{decl_name}' {'(simulated)' if ex.test else 'unloaded'}")


def cmd_launch(args):
    """Internal: exec the composed launch command for a declaration. Used by
    vllm@.service (%i = declaration name). Test mode: print + audit instead."""
    if not args:
        die("usage: spark-llm launch <declaration>")
    decl_name = args[0]
    loaded = load_loaded()
    if decl_name not in loaded:
        die(f"'{decl_name}' is not in loaded.toml — nothing to launch")
    port = int(loaded[decl_name]["port"])
    decls = load_declarations()
    if decl_name not in decls:
        die(f"declaration '{decl_name}' not found in models.d/")
    decl = decls[decl_name]
    cfg = load_config()
    consumer = consumer_for_port(cfg, port)
    errors, _ = validate_load(decl_name, decl, port, cfg, consumer)
    if errors:
        for e in errors:
            print(f"  error: {e}", file=sys.stderr)
        audit("LAUNCH-BLOCKED", f"'{decl_name}': {len(errors)} validation errors")
        sys.exit(1)
    alias = consumer["alias"] if consumer else None
    total, _, _ = mem_gb()
    argv = compose_launch_argv(decl_name, decl, port, cfg, alias, total)
    if read_mode() != "live":
        audit("WOULD-LAUNCH", shlex.join(argv))
        print(f"[test] would launch:\n  {shlex.join(argv)}")
        return
    audit("LAUNCH", shlex.join(argv))
    os.execvp(argv[0], argv)


# ---------------------------------------------------------------- info commands

def cmd_status(args):
    cfg = load_config()
    decls = load_declarations()
    mode = read_mode()
    total, avail, swap_used = mem_gb()
    loaded = load_loaded()

    for c in consumer_states(cfg, loaded):
        tag = {"down": "✗ DOWN — NOTHING SERVING", "alias-missing": "⚠ serving, but alias NOT served",
               "ok": "✓ ok"}[c["state"]]
        print(f"Consumer: {c['name']} — '{c['alias']}' @ :{c['port']} — {tag}")
    print(f"Mode: {mode.upper()}" + ("  (commands are logged, not executed)" if mode == "test" else ""))

    states = loaded_states(decls, cfg)
    if not states:
        print("Loaded models: none")
    else:
        print("Loaded models:")
        for st in states:
            if not st["up"]:
                text, _style = loaded_status_text(st)
                print(f"  {text} {st['decl_name']} (port :{st['port']})")
                continue
            up = f", uptime {st['uptime_s']/60:.0f}m" if st["uptime_s"] else ""
            who = f" — fulfilling {st['consumer']['name']} as '{st['consumer']['alias']}'" if st["consumer"] else " — not fulfilling any consumer"
            print(f"  ✓ {st['decl_name']} @ http://{endpoint_host(cfg)}:{st['port']}/v1{who}{up}")
            if st["consumer"] and not st["alias_ok"]:
                print(f"    !! consumer alias '{st['consumer']['alias']}' is NOT served — incident-D condition")

    print(f"Memory: {fmt_gb(total-avail)}/{fmt_gb(total)} used, {fmt_gb(avail)} available, "
          f"swap {fmt_gb(swap_used)}")
    gpu = gpu_util_pct()
    print(f"GPU: {gpu} %" if gpu is not None else "GPU: n/a (nvidia-smi unavailable)")
    dtotal, dused, msize = disk_stats_gb(cfg)
    print(f"Disk: {fmt_gb(dused)}/{fmt_gb(dtotal)} used, "
          + (f"Models folder {fmt_gb(msize)}" if msize is not None else "Models folder unreadable"))


def cmd_list(args):
    cfg = load_config()
    decls = load_declarations()
    loaded = load_loaded()
    loaded_by_name = {s["decl_name"]: s for s in loaded_states(decls, cfg)}
    total, avail, _ = mem_gb()
    freed_total = sum(reserve_estimate_gb(decls[dn], total, cfg) for dn in loaded if dn in decls)
    print(f"{'':2}{'declaration':<22}{'reserve(est)':>14}{'disk':>14}   status")
    for name, d in decls.items():
        res = reserve_estimate_gb(d, total, cfg)
        size = disk_size_gb(d, cfg)
        size_s = fmt_gb(size) if size is not None else "unreadable"
        if name in loaded:
            s = loaded_by_name[name]
            text, _style = loaded_status_text(s)
            status = f"{text} :{loaded[name]['port']}" + (f" ({s['consumer']['name']})" if s["consumer"] else "")
        elif res <= avail:
            status = "fits now (est)"
        elif res <= avail + freed_total:
            status = "requires freeing memory (est)"
        else:
            status = "does not fit (est)"
        mark = "●" if name in loaded else " "
        print(f"{mark:2}{name:<22}{fmt_gb(res):>12}  {size_s:>12}   {status}")
    undecl = undeclared_on_disk(cfg, decls)
    if undecl is None:
        print(f"\n  (models_dir {cfg['models_dir']} unreadable from this account — "
              f"on-disk detection unavailable)")
    elif undecl:
        print(f"\n  on disk, no declaration yet:")
        for name in undecl:
            status, reason = probe_folder(name, cfg)
            print(f"    {name}  — {'loadable' if status == 'ok' else 'BLOCKED: ' + reason}")


def cmd_mode(args):
    if not args:
        print(read_mode())
        return
    target = args[0].lower()
    if target not in ("test", "live"):
        die("usage: spark-llm mode [test|live]")
    if target == "live":
        print("Switching to LIVE means commands will actually stop and start models.")
        confirm = input("Type LIVE to confirm: ").strip()
        if confirm != "LIVE":
            die("not confirmed — staying in test mode")
    (config_dir() / "mode").write_text(target + "\n")
    audit("MODE-SET", f"mode set to {target}")
    print(f"mode: {target.upper()}")


def cmd_audit(args):
    n = int(args[0]) if args else 30
    p = config_dir() / "audit.log"
    if not p.exists():
        print("(no audit entries)")
        return
    for line in p.read_text().splitlines()[-n:]:
        print(line)


def gguf_kv(path: Path, exact: set = frozenset(), suffix: str | None = None) -> dict:
    """Read selected keys from a GGUF header (v2/v3). Returns {key: value};
    empty dict on any parse trouble — callers treat unknown honestly."""
    import struct
    SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    out = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            struct.unpack("<I", f.read(4))
            struct.unpack("<Q", f.read(8))
            n_kv = struct.unpack("<Q", f.read(8))[0]

            def rstr():
                n = struct.unpack("<Q", f.read(8))[0]
                return f.read(n).decode("utf-8", "replace")

            def rval(t):
                if t == 8:
                    return rstr()
                if t == 9:
                    et = struct.unpack("<I", f.read(4))[0]
                    n = struct.unpack("<Q", f.read(8))[0]
                    if et == 8:
                        for _ in range(n):
                            rstr()
                    elif et == 9:
                        for _ in range(n):
                            rval(9)
                    else:
                        f.seek(SIZES[et] * n, 1)
                    return None
                raw = f.read(SIZES[t])
                if t == 6:
                    return struct.unpack("<f", raw)[0]
                if t == 12:
                    return struct.unpack("<d", raw)[0]
                return int.from_bytes(raw, "little", signed=t in (1, 3, 5, 11))

            wanted = len(exact) + (1 if suffix else 0)
            for _ in range(n_kv):
                key = rstr()
                t = struct.unpack("<I", f.read(4))[0]
                v = rval(t)
                if v is None:
                    continue
                if key in exact or (suffix and key.endswith(suffix)):
                    out[key if key in exact else suffix] = v
                    if len(out) >= wanted:
                        break
    except Exception:
        return out
    return out


def gguf_context_length(path: Path) -> int | None:
    v = gguf_kv(path, suffix=".context_length").get(".context_length")
    return int(v) if v is not None else None


def model_identity(root: Path) -> str | None:
    """The model's OWN name from its data sheet, when it carries one.
    Priority: GGUF general.name, then HF config.json _name_or_path."""
    try:
        files = {p.name for p in root.iterdir() if p.is_file()}
    except OSError:
        return None
    gguf = main_gguf(files)
    if gguf:
        name = gguf_kv(root / gguf, exact={"general.name"}).get("general.name")
        return str(name).strip() or None if name else None
    try:
        with open(root / "config.json") as f:
            nop = json.load(f).get("_name_or_path")
        if nop and str(nop).strip():
            return str(nop).rstrip("/").split("/")[-1]
    except Exception:
        pass
    return None


def derive_label(folder: str, cfg: dict) -> str:
    """NAMING RULE (standing): a model's label comes from its own metadata
    when present, else its folder name — always sanitized."""
    identity = model_identity(Path(cfg["models_dir"]) / folder)
    return sanitize_label(identity or folder)


def main_gguf(files: set[str]) -> str | None:
    """The file to hand llama-server: first shard of a split, else the single gguf."""
    ggufs = sorted(f for f in files if f.endswith(".gguf"))
    if not ggufs:
        return None
    first_shards = [f for f in ggufs if "00001-of-" in f]
    return first_shards[0] if first_shards else ggufs[0]


def probe_folder(folder: str, cfg: dict) -> tuple[str, str]:
    """Pre-flight status for an undeclared folder, from disk facts alone.
    Returns ('ok'|'blocked', reason). Anything inspection can PROVE will fail
    must be marked blocked at display time — never shown as plainly loadable."""
    root = Path(cfg["models_dir"]) / folder
    try:
        files = {p.name for p in root.iterdir() if p.is_file()}
    except OSError as e:
        return "blocked", f"unreadable ({e.__class__.__name__})"
    need = int(cfg["consumer_context_length"]) * int(cfg.get("headroom_factor", 2))
    gguf = main_gguf(files)
    if gguf:
        bin_ = cfg.get("engines", {}).get("llamacpp_bin", "")
        if not (bin_ and Path(bin_).exists()):
            return "blocked", "GGUF format — llamacpp engine not installed (engines.llamacpp_bin)"
        try:
            with open(root / gguf, "rb") as f:
                if f.read(4) != b"GGUF":
                    return "blocked", "corrupt GGUF — bad magic bytes"
        except OSError as e:
            return "blocked", f"GGUF unreadable ({e.__class__.__name__})"
        native = gguf_context_length(root / gguf)
        if native and native < need:
            return "blocked", f"native context {native} < required {need} (headroom rule, incident B)"
        return "ok", ""
    if "model_index.json" in files:
        # diffusers pipeline (image generation, e.g. Qwen-Image): weights live
        # in nested component subfolders, not flat at root, so the HF-checkpoint
        # branch below does not apply. Validate the manifest actually parses
        # and every component it names is really on disk — never pass an
        # interrupted/incomplete download as loadable.
        try:
            with open(root / "model_index.json") as f:
                mi = json.load(f)
        except Exception as e:
            return "blocked", f"unreadable model_index.json ({e.__class__.__name__})"
        missing = [k for k, v in mi.items()
                   if isinstance(v, list) and len(v) == 2 and v[0] and not (root / k).is_dir()]
        if missing:
            return "blocked", f"incomplete diffusers pipeline — missing component folder(s): {', '.join(missing)}"
        return "ok", ""
    if "config.json" not in files:
        return "blocked", "no config.json — not a servable checkpoint layout"
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        try:
            with open(idx) as f:
                wanted = set(json.load(f)["weight_map"].values())
            missing = wanted - files
            if missing:
                return "blocked", f"incomplete — {len(missing)} weight shard(s) missing"
        except Exception:
            return "blocked", "unreadable safetensors index"
    elif not any(f.endswith(".safetensors") for f in files):
        return "blocked", "no safetensors weights found"
    try:
        with open(root / "config.json") as f:
            mc = json.load(f)
        native = (mc.get("max_position_embeddings")
                  or mc.get("text_config", {}).get("max_position_embeddings"))
        if native and int(native) < need:
            return "blocked", f"native context {native} < required {need} (headroom rule, incident B)"
    except Exception:
        pass
    return "ok", ""


def sanitize_label(folder: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", folder.lower()).strip("-")[:48]


def synth_declaration(folder: str, cfg: dict) -> tuple[str, str]:
    """Generate a declaration for a model folder from disk facts + [auto]
    config defaults. Returns (label, toml_text). Raises on unreadable folder."""
    a = cfg["auto"]
    root = Path(cfg["models_dir"]) / folder
    size_gb = sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 1e9
    files = {p.name for p in root.iterdir() if p.is_file()}
    total, _, _ = mem_gb()
    gguf = main_gguf(files)
    if gguf:
        native = gguf_context_length(root / gguf)
        target_ctx = int(cfg["consumer_context_length"]) * int(cfg.get("headroom_factor", 2))
        mml = min(int(native), target_ctx) if native else target_ctx
        label = derive_label(folder, cfg)
        flags = "".join(f'  "{f}",\n' for f in a.get("llamacpp_extra_flags", []))
        text = (
            f"# AUTO-GENERATED {datetime.now().date()} — GGUF via llamacpp engine.\n"
            f"# Generic, NOT tuned. Edit freely; this file is now the owner.\n"
            f"# gguf native context: {native}; disk: {fmt_gb(size_gb)}\n"
            f"# TOOL-CALLING: unverified. This engine's tool-calling mechanism is not\n"
            f"# checked by validate_load (§ tool-calling rule) — if this model will serve\n"
            f"# a consumer that needs tools, confirm it works before relying on it.\n"
            f'engine = "llamacpp"\n'
            f'weights = "{root / gguf}"\n'
            f'served_name = "{label}"\n'
            f"max_model_len = {mml}\n"
            f"n_gpu_layers = 999\n"
            f"est_weights_gb = {size_gb:.0f}\n"
            f"extra_flags = [\n{flags}]\n"
        )
        return label, text
    if (root / "model_index.json").exists():
        # diffusers pipeline (probe_folder already validated it's complete).
        # No native-context/architecture concept applies here — this is image
        # generation, not text — so none of the vLLM-specific fields below fit;
        # a distinct declaration shape, same AUTO-GENERATED/editable contract.
        label = derive_label(folder, cfg)
        text = (
            f"# AUTO-GENERATED {datetime.now().date()} — diffusers image-gen pipeline.\n"
            f"# Generic defaults (1328x1328, 50 steps, cfg 4.0) — edit freely; this\n"
            f"# file is now the owner. disk: {fmt_gb(size_gb)}\n"
            f'engine = "diffusers"\n'
            f'weights = "/models/{folder}"\n'
            f'served_name = "{label}"\n'
            f"est_weights_gb = {size_gb:.0f}\n"
            f'default_size = "1328x1328"\n'
            f"num_inference_steps = 50\n"
            f"true_cfg_scale = 4.0\n"
        )
        return label, text
    native = None
    arch = "unknown"
    try:
        with open(root / "config.json") as f:
            mc = json.load(f)
        native = (mc.get("max_position_embeddings")
                  or mc.get("text_config", {}).get("max_position_embeddings"))
        arch = ", ".join(mc.get("architectures", []) or ["unknown"])
    except Exception:
        pass
    util = max(0.2, min(0.85, round((size_gb * float(a["weights_overhead"])
                                     + float(a["kv_margin_gib"])) / total, 2)))
    target_ctx = int(cfg["consumer_context_length"]) * int(cfg.get("headroom_factor", 2))
    mml = min(int(native), target_ctx) if native else target_ctx
    label = derive_label(folder, cfg)
    flags = "".join(f'  "{f}",\n' for f in a.get("extra_flags", []))
    text = (
        f"# AUTO-GENERATED {datetime.now().date()} from on-disk facts + [auto] config\n"
        f"# defaults — generic, NOT tuned. Edit freely; this file is now the owner.\n"
        f"# architecture: {arch}; native context: {native}; disk: {fmt_gb(size_gb)}\n"
        f"# TOOL-CALLING NOT CONFIGURED (2026-07-20: this silently broke the primary consumer once).\n"
        f"# If this model will serve a consumer with requires_tool_calling=true, add\n"
        f"# --tool-call-parser <name> --enable-auto-tool-choice to extra_flags below.\n"
        f"# Find <name> by checking this model's chat_template.jinja against vLLM's\n"
        f"# ToolParserManager list — do not guess it. Until then, validate_load will\n"
        f"# correctly REFUSE to load this as that consumer's brain.\n"
        f'engine = "vllm-docker"\n'
        f'image = "{a["image"]}"\n'
        f'weights = "/models/{folder}"\n'
        f'served_name = "{label}"\n'
        f"gpu_memory_utilization = {util}\n"
        f"max_model_len = {mml}\n"
        f"est_weights_gb = {size_gb:.0f}\n"
        f"extra_flags = [\n{flags}]\n"
    )
    return label, text


def auto_declare(folder: str, cfg: dict) -> str:
    """Write a declaration for an undeclared folder. Config write (audited),
    not a machine mutation — allowed in test mode; only load/unload touch the
    machine."""
    status, reason = probe_folder(folder, cfg)
    if status != "ok":
        audit("DECLARE-BLOCKED", f"'{folder}': {reason}")
        raise RuntimeError(f"'{folder}' cannot be served: {reason}")
    label, text = synth_declaration(folder, cfg)
    dpath = config_dir() / "models.d" / f"{label}.toml"
    if dpath.exists():
        raise RuntimeError(f"declaration {dpath} already exists but does not cover "
                           f"'{folder}' — resolve by hand")
    dpath.write_text(text)
    audit("CONFIG-WRITE", f"auto-generated declaration {dpath} for folder '{folder}'")
    return label



def cmd_selftest(args):
    """Run the regression suite that ships with the app. Use after ANY change —
    to the app, the declarations, or the machine — to confirm nothing broke."""
    t = Path(__file__).resolve().with_name("test_console_paths.py")
    if not t.exists():
        die(f"selftest suite not found at {t} — deploy it alongside the app")
    sys.exit(subprocess.run([sys.executable, str(t)]).returncode)


def cmd_declare(args):
    """Make an on-disk model loadable: generate its declaration."""
    if not args:
        die("usage: spark-llm declare <folder-name-in-models_dir>")
    cfg = load_config()
    folder = args[0]
    undecl = undeclared_on_disk(cfg, load_declarations())
    if undecl is None:
        die(f"models_dir {cfg['models_dir']} is unreadable from this account")
    if folder not in undecl:
        die(f"'{folder}' is not an undeclared folder in {cfg['models_dir']} "
            f"(undeclared: {', '.join(undecl) or 'none'})")
    try:
        label = auto_declare(folder, cfg)
    except RuntimeError as e:
        die(str(e))
    print(f"declared '{folder}' as '{label}' — review models.d/{label}.toml, "
          f"then `spark-llm load {label}`")


def cmd_logs(args):
    """Server logs for a loaded declaration. Read-only. Tries the app-owned
    log file, then docker/journal; reports honestly when nothing is readable."""
    if not args:
        loaded = load_loaded()
        if not loaded:
            die("nothing is loaded — usage: spark-llm logs <declaration>")
        args = list(loaded)
    n = "30"
    for decl_name in args:
        slot_log = config_dir() / "logs" / f"{decl_name}.log"
        if slot_log.exists() and slot_log.stat().st_size > 0:
            print(f"--- {decl_name} ({slot_log}) ---")
            print("\n".join(slot_log.read_text(errors="replace").splitlines()[-int(n):]))
            continue
        printed = False
        for argv in (["docker", "logs", "--tail", n, f"vllm-{decl_name}"],
                     ["journalctl", "-u", f"vllm@{decl_name}.service", "-n", n, "--no-pager"]):
            try:
                r = subprocess.run(argv, capture_output=True, text=True, timeout=10)
                out = (r.stdout + r.stderr).strip()
                if "No entries" in out:
                    continue
                if r.returncode == 0 and out:
                    print(f"--- {decl_name} ({argv[0]}) ---")
                    print(out)
                    printed = True
                    break
            except Exception:
                continue
        if not printed:
            print(f"--- {decl_name}: logs unreadable from this account "
                  f"(docker and journal both refused; needs group membership or root) ---")


# ---------------------------------------------------------------- console

def cmd_console(args):
    frames = None
    if args and args[0] == "--frames":
        frames = int(args[1])
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table

    console = Console()
    selection = [0]
    message = [""]
    action_out = [None]
    pending = [None]
    LOGS_TITLE = "Server logs"
    AUDIT_TITLE = "Audit log"
    running = [None]
    state_cache = {"t": 0.0, "loaded": [], "cons": []}
    last_probe = [time.time()]

    def start_subcommand(title: str, argv: list[str]):
        audit("CONSOLE-INVOKE", shlex.join(argv))
        p = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), *argv],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        os.set_blocking(p.stdout.fileno(), False)
        running[0] = {"title": title, "proc": p, "lines": [], "buf": "", "t0": time.time()}
        action_out[0] = None

    def pump_running():
        r = running[0]
        if not r:
            return
        try:
            chunk = r["proc"].stdout.read()
            if chunk:
                r["buf"] += chunk
                *complete, r["buf"] = r["buf"].split("\n")
                r["lines"].extend(complete)
        except (BlockingIOError, TypeError, ValueError):
            pass
        if r["proc"].poll() is not None:
            if r["buf"].strip():
                r["lines"].append(r["buf"])
            action_out[0] = (r["title"], r["lines"][-14:], r["proc"].returncode == 0)
            running[0] = None

    def cached(decls, cfg):
        if time.time() - state_cache["t"] > 2.0:
            state_cache["loaded"] = loaded_states(decls, cfg)
            state_cache["cons"] = consumer_states(cfg)
            state_cache["t"] = time.time()
            last_probe[0] = time.time()
        return state_cache["loaded"], state_cache["cons"]

    def entries_list(cfg, decls):
        """Rows, in order: loaded declarations, then not-yet-loaded
        declarations, then undeclared on-disk folders. One flat cursor list —
        Enter's meaning is contextual per kind."""
        loaded, _ = cached(decls, cfg)
        loaded_names = {s["decl_name"] for s in loaded}
        out = [(s["decl_name"], "loaded", s) for s in loaded]
        out += [(n, "decl", None) for n in decls if n not in loaded_names]
        for f in (undeclared_on_disk(cfg, decls) or []):
            status, reason = probe_folder(f, cfg)
            out.append((f, "new" if status == "ok" else "blocked", reason))
        return out

    def request_new(folder: str):
        try:
            cfg = load_config()
            label = auto_declare(folder, cfg)
            message[0] = f"Declared '{folder}' as '{label}' — loading it now"
            request_load(label)
        except RuntimeError as e:
            message[0] = str(e)

    def request_load(decl_name: str):
        cfg = load_config()
        decls = load_declarations()
        loaded = load_loaded()
        port, consumer = decide_port(decl_name, decls[decl_name], loaded, cfg)
        if decl_name in loaded and int(loaded[decl_name]["port"]) == port:
            message[0] = f"'{decl_name}' already loaded" + (f" — fulfilling {consumer['name']}" if consumer else "")
            return
        displaced = next((dn for dn, e in loaded.items() if int(e["port"]) == port and dn != decl_name), None)
        promoting = decl_name in loaded  # already running elsewhere: this reassigns its port, not a fresh start
        verb = ("Simulate promote" if promoting else "Simulate load") if read_mode() != "live" \
            else ("Promote" if promoting else "Load")
        title = f"{verb}: {decl_name}" + (f" (as {consumer['name']})" if consumer else f" (port {port})")
        warning = (f"This stops {displaced} first — it will be unavailable in the meantime" if displaced
                   else "This can take several minutes and reserves GPU memory for the duration")
        pending[0] = (title, ["load", decl_name], warning)

    def request_unload(decl_name: str):
        mode_word = "Simulate unload" if read_mode() != "live" else "Unload"
        pending[0] = (f"{mode_word}: {decl_name}", ["unload", decl_name],
                      f"This stops {decl_name} — anything it was fulfilling goes down")

    def build():
        cfg = load_config()
        decls = load_declarations()
        mode = read_mode()
        total, avail, swap_used = mem_gb()
        loaded_st, cons_st = cached(decls, cfg)
        loaded = load_loaded()

        parts = []
        if mode == "test":
            parts.append(Panel(Text("TEST MODE — commands are logged to audit.log, not executed",
                                    style="bold black on yellow", justify="center"), style="yellow"))
        if str(config_dir()) != "/etc/spark-llm":
            parts.append(Text(f"  ⚠ DEV CONFIG: {config_dir()} — this is NOT the "
                              f"production console", style="bold yellow"))

        # Consumers: machine reality, shown ALWAYS — before anything else.
        for c in cons_st:
            if c["state"] == "down":
                parts.append(Panel(Text(
                    f"✗ {c['name']} endpoint :{c['port']} — NOTHING IS SERVING. "
                    f"This consumer is DOWN until a model serving '{c['alias']}' "
                    f"is loaded or promoted onto that port.",
                    style="bold white on red", justify="center"), style="red"))
            elif c["state"] == "alias-missing":
                parts.append(Text(
                    f"  ⚠ {c['name']} :{c['port']} is serving {c['served']} but NOT "
                    f"'{c['alias']}' — this consumer breaks if it calls that name",
                    style="bold yellow"))
            else:
                parts.append(Text(f"  ✓ {c['name']} · '{c['alias']}' @ :{c['port']}",
                                  style="green dim"))

        # Status panel — ONE bordered box: app title, hostname, usage gauges.
        # ( restore the border removed in the multi-slot
        # rewrite; fold Memory/GPU/Disk into it instead of a floating strip.)
        def gauge(label, frac, value_text, color, extra="", warn=None, spacer=True):
            width = 40
            filled = max(0, min(width, int(width * frac)))
            g = Text(f"  {label:<8}")
            g.append("▐" + "█" * filled + "░" * (width - filled) + "▌",
                     style="bold red" if (warn is not None and frac > warn) else color)
            g.append(f"  {value_text}", style="bold")
            if extra:
                g.append(f"    {extra}", style="dim")
            g.append("\n\n" if spacer else "\n")
            return g

        used = total - avail
        gauges = Text()
        gauges.append_text(gauge("Memory", used / total, f"{fmt_gb(used)} / {fmt_gb(total)}",
                                 "cyan", f"Swap {fmt_gb(swap_used)}", warn=0.92))
        gpu = gpu_util_pct()
        if gpu is not None:
            gauges.append_text(gauge("GPU", gpu / 100, f"{gpu} %", "green"))
        else:
            gauges.append(Text("  GPU      n/a (nvidia-smi unavailable)\n\n", style="dim"))
        dtotal, dused, msize = disk_stats_gb(cfg)
        gauges.append_text(gauge("Disk", dused / dtotal, f"{fmt_gb(dused)} / {fmt_gb(dtotal)}",
                                 "magenta",
                                 (f"Models folder {fmt_gb(msize)} ({cfg['models_dir']})"
                                  if msize is not None
                                  else f"Models folder unreadable ({cfg['models_dir']})"),
                                 warn=0.9, spacer=False))
        parts.append(Panel(gauges, border_style="dim", title=" spark-llm ", title_align="left",
                           subtitle=f" {os.uname().nodename} ", subtitle_align="right"))

        parts.append(Text(""))
        # Loaded models — one row each, however many there are.
        if loaded_st:
            lt = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
            lt.add_column("  Loaded")
            lt.add_column("Endpoint")
            lt.add_column("Fulfilling")
            lt.add_column("Status")
            for s in loaded_st:
                url = f"http://{endpoint_host(cfg)}:{s['port']}/v1"
                fulfilling = (f"'{s['consumer']['alias']}' ({s['consumer']['name']})"
                             if s["consumer"] else Text("— (free)", style="dim"))
                text, style = loaded_status_text(s)
                lt.add_row(f"  ● {s['decl_name']}", url,
                          fulfilling if s["up"] else "—", Text(text, style=style))
            parts.append(lt)
        else:
            parts.append(Text("  Loaded: nothing — pick a model below\n", style="dim"))

        # Selectable list: loaded models, then not-yet-loaded declarations,
        # then undeclared folders.
        ACCENT = "bold rgb(215,119,87)"

        def col_cell(txt, selected: bool, semantic: str | None = None):
            # Prefix width must be CONSTANT regardless of selection, or the
            # column auto-sizes differently depending on which row currently
            # holds the "❯ " marker — visibly shifting Memory/Status left or
            # right as selection moves . Matches how
            # name_cell already pads unselected rows with "  " to match.
            out = Text()
            out.append("❯ " if selected else "  ", style=ACCENT if selected else "")
            style = semantic if semantic else (ACCENT if selected else "")
            out.append(txt, style=(style + " bold").strip() if selected else style)
            return out

        parts.append(Text(""))
        tbl = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
        tbl.add_column("  Models")
        tbl.add_column("Memory", justify="right")
        tbl.add_column("Status")
        entries = entries_list(cfg, decls)
        sel = selection[0] % max(len(entries), 1)

        for i, (name, kind, extra) in enumerate(entries):
            selected = i == sel
            name_cell = col_cell(name, selected) if selected else Text("  " + name)
            if kind == "blocked":
                tbl.add_row(name_cell, col_cell("✗", selected, semantic="red"),
                            col_cell(f"Cannot load — {extra}", selected, semantic="red"))
            elif kind == "new":
                tbl.add_row(name_cell, col_cell("—", selected),
                            col_cell("New — press ⏎ to set it up and load it", selected))
            elif kind == "loaded":
                s = extra
                text, sem = loaded_status_text(s)
                fits = f"{text} :{s['port']}" + (f" ({s['consumer']['name']})" if s["consumer"] else "")
                mem_cell = fmt_gb(s["memory_gb"]) if s["memory_gb"] is not None else "—"
                tbl.add_row(name_cell, col_cell(mem_cell, selected),
                            col_cell(fits, selected, semantic=sem))
            else:  # "decl": not currently loaded
                d = decls[name]
                res = reserve_estimate_gb(d, total, cfg)
                freed = sum(reserve_estimate_gb(decls[s["decl_name"]], total, cfg)
                           for s in loaded_st if s["decl_name"] in decls)
                if res <= avail:
                    fits, sem = "Fits now", None
                elif res <= avail + freed:
                    fits, sem = "Requires freeing memory", None
                else:
                    fits, sem = "Does not fit", "red"
                tbl.add_row(name_cell, col_cell(fmt_gb(res), selected),
                            col_cell(fits, selected, semantic=sem))
        parts.append(tbl)
        if undeclared_on_disk(cfg, decls) is None:
            parts.append(Text(f"  Models folder {cfg['models_dir']} is unreadable from this "
                              f"account — on-disk detection unavailable", style="dim yellow"))

        if running[0]:
            r = running[0]
            spin = SPIN[int(time.time() * 8) % len(SPIN)]
            body = Text("\n".join(r["lines"][-12:]) or "Starting ...")
            parts.append(Panel(body, title=f"{spin} {r['title']} · {time.time()-r['t0']:.0f} s",
                               border_style="cyan"))
        elif pending[0]:
            title, _argv, warning = pending[0]
            t = Text()
            t.append(f"  {title}\n", style="bold")
            t.append(f"  ⚠ {warning}\n\n", style="yellow")
            t.append("  Press ", style="dim")
            t.append("⏎ or Y", style="bold green")
            t.append(" to proceed · ", style="dim")
            t.append("N or Esc", style="bold red")
            t.append(" to cancel", style="dim")
            parts.append(Panel(t, border_style="yellow", title="Confirm"))
        elif action_out[0]:
            title, lines, ok = action_out[0]
            body = Text("\n".join(lines) or "(no output)")
            if mode == "test" and ok:
                body.append("\n\nTEST MODE: nothing was changed. This is the exact "
                            "sequence LIVE mode will execute.", style="bold yellow")
            parts.append(Panel(body, title=("✓ " if ok else "✗ ") + title,
                               border_style="green" if ok else "red"))
        if message[0]:
            parts.append(Text(f"\n  {message[0]}", style="yellow"))

        logs_open = bool(action_out[0]) and action_out[0][0] == LOGS_TITLE
        audit_open = bool(action_out[0]) and action_out[0][0] == AUDIT_TITLE
        panel_open = bool(action_out[0])
        foot = Text(f"\n  ↑↓ Select · ⏎ Load/Promote · U Unload"
                    f" · L {'Close server logs' if logs_open else 'Server logs'}"
                    f" · A {'Close audit' if audit_open else 'Audit'}"
                    + (" · X Close panel" if panel_open and not (logs_open or audit_open) else "")
                    + " · Q Quit", style="dim")
        age = time.time() - last_probe[0]
        foot.append(f"{'':>12}● Live · {age:.0f} s ago", style="green dim")
        parts.append(foot)
        return Group(*parts)

    if frames is not None:
        for _ in range(frames):
            console.print(build())
        return

    import termios, tty, select
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        with Live(build(), console=console, refresh_per_second=8, screen=True) as live:
            while True:
                try:
                    ready = select.select([sys.stdin], [], [], 0.25)[0]
                except InterruptedError:
                    ready = []
                if ready:
                    ch = sys.stdin.read(1)
                    if pending[0]:
                        title, argv, _w = pending[0]
                        if ch in ("y", "Y", "\n", "\r"):
                            pending[0] = None
                            message[0] = ""
                            start_subcommand(title, argv)
                        elif ch in ("n", "N", "q", "\x1b"):
                            pending[0] = None
                            message[0] = "Cancelled — nothing done"
                            audit("CONSOLE-CANCEL", title)
                            if ch == "\x1b":
                                select.select([sys.stdin], [], [], 0.05)[0] and sys.stdin.read(2)
                    elif ch in ("q", "Q"):
                        if running[0]:
                            message[0] = "An action is still running — wait for it to finish"
                        else:
                            break
                    elif ch == "\x1b":
                        seq = sys.stdin.read(2)
                        if seq == "[A":
                            selection[0] -= 1
                        elif seq == "[B":
                            selection[0] += 1
                    elif running[0]:
                        message[0] = "An action is running — keys are ignored until it finishes"
                    elif ch in ("\n", "\r"):
                        message[0] = ""
                        decls = load_declarations()
                        entries = entries_list(load_config(), decls)
                        if entries:
                            name, kind, extra = entries[selection[0] % len(entries)]
                            if kind == "blocked":
                                message[0] = f"'{name}' cannot be loaded: {extra}"
                            elif kind == "new":
                                request_new(name)
                            elif kind == "loaded":
                                if not extra["up"]:
                                    message[0] = f"'{name}' is still starting up — wait for it to finish"
                                elif extra["consumer"]:
                                    message[0] = f"'{name}' is already fulfilling {extra['consumer']['name']} — U to unload"
                                else:
                                    request_load(name)  # loaded, healthy, free-floating: promote to a consumer
                            else:
                                request_load(name)
                    elif ch in ("x", "X"):
                        if action_out[0]:
                            action_out[0] = None
                    elif ch in ("u", "U"):
                        message[0] = ""
                        decls = load_declarations()
                        entries = entries_list(load_config(), decls)
                        if entries:
                            name, kind, _extra = entries[selection[0] % len(entries)]
                            if kind == "loaded":
                                request_unload(name)
                            else:
                                message[0] = "Select a loaded model to unload it"
                    elif ch in ("l", "L"):
                        message[0] = ""
                        if action_out[0] and action_out[0][0] == LOGS_TITLE:
                            action_out[0] = None
                        else:
                            start_subcommand(LOGS_TITLE, ["logs"])
                    elif ch in ("a", "A"):
                        message[0] = ""
                        if action_out[0] and action_out[0][0] == AUDIT_TITLE:
                            action_out[0] = None
                        else:
                            p = config_dir() / "audit.log"
                            lines = p.read_text().splitlines()[-10:] if p.exists() else ["(no audit entries)"]
                            action_out[0] = (AUDIT_TITLE, lines, True)
                pump_running()
                live.update(build())
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------- main

COMMANDS = {
    "status": cmd_status,
    "list": cmd_list,
    "load": cmd_load,
    "unload": cmd_unload,
    "launch": cmd_launch,
    "mode": cmd_mode,
    "audit": cmd_audit,
    "logs": cmd_logs,
    "declare": cmd_declare,
    "selftest": cmd_selftest,
    "console": cmd_console,
}


def main():
    args = sys.argv[1:]
    if not args:
        cmd_console([])
        return
    cmd = args[0]
    if cmd not in COMMANDS:
        die(f"unknown command '{cmd}' (known: {', '.join(COMMANDS)})")
    COMMANDS[cmd](args[1:])


if __name__ == "__main__":
    main()
