#!/usr/bin/env python3
"""Regression suite for spark-llm. Run after ANY change — to the app, the
declarations, or the machine — via `spark-llm selftest`.

Exercises the real declaration/consumer/loaded-model state on the live
machine (read-only checks) plus the load/unload logic in isolation (using
temp dirs — never mutates the real config)."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spark_llm as S

failures = []

# HARD SAFETY GATE: this suite calls real subcommands including `load`/`unload`
# against a TEMP config — but also reads live state from the real config for
# some checks. In LIVE mode against the REAL config, never let anything here
# risk a real mutation.
if S.read_mode() == "live":
    print("REFUSING to run: this config dir is in LIVE mode — refusing as a "
          "precaution. Point SPARK_LLM_DIR at a test-mode config, or "
          "`spark-llm mode test` first.")
    sys.exit(2)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    if not cond:
        failures.append(name)


cfg = S.load_config()
decls = S.load_declarations()

# 1. Declarations exist and are readable
check("declarations load without error", isinstance(decls, dict) and len(decls) > 0,
      f"{len(decls)} declarations")

# 2. Consumers: probed live, independent of anything selected
cons = S.consumer_states(cfg)
check("consumer_states returns the declared consumers", len(cons) == len(cfg.get("consumers", [])))
for c in cons:
    check(f"consumer '{c['name']}' has a valid state", c["state"] in ("ok", "down", "alias-missing"),
          c["state"])

# 3. loaded_states reflects live reality, not a stale file
loaded = S.load_loaded()
lstates = S.loaded_states(decls, cfg)
check("loaded_states returns one row per loaded.toml entry", len(lstates) == len(loaded))
for s in lstates:
    check(f"loaded '{s['decl_name']}': up is consistent with served names",
          s["up"] == bool(s["served"]))

# 4. Naming rule: every declaration's label matches its derived name
for name, d in decls.items():
    expected = S.derive_label(S.decl_folder_name(d), cfg)
    check(f"naming rule: '{name}'", name == expected,
          name if name == expected else f"should be '{expected}'")

# 5. Units and gauges: decimal GB (NVIDIA Sync dashboard = ground truth)
total_gb, avail_gb, _sw = S.mem_gb()
check("units: total is decimal GB, capped at config's max_memory_gb, not GiB (~122) "
      "and not the raw uncapped OS total (~131)", 122 < total_gb <= 128, f"{total_gb:.1f}")
gpu = S.gpu_util_pct()
check("gpu util: int 0-100 or honest None", gpu is None or (isinstance(gpu, int) and 0 <= gpu <= 100), str(gpu))
dt, du_, ms = S.disk_stats_gb(cfg)
check("disk stats: used < total, both sane", 0 < du_ < dt < 2000, f"{du_:.0f}/{dt:.0f} GB")
check("disk stats: models folder measured", ms is not None and ms > 100, f"{ms:.0f} GB" if ms else "None")

# 6. On-disk detection: every real folder gets a clear verdict
undecl = S.undeclared_on_disk(cfg, decls)
check("detection: models_dir readable", undecl is not None)
if undecl:
    for name in undecl:
        status, reason = S.probe_folder(name, cfg)
        check(f"probe verdict on real model {name[:40]}", status in ("ok", "blocked"),
              f"{status}: {reason}" if status == "blocked" else "ok: servable")

# 7. GGUF engine + header parsing
gguf_available = bool(cfg.get("engines", {}).get("llamacpp_bin")) and \
    os.path.exists(cfg["engines"]["llamacpp_bin"])
check("llamacpp engine binary present", gguf_available, cfg.get("engines", {}).get("llamacpp_bin", "unset"))

staged = os.path.expanduser("~/staging/Gemma4-Uncensored/supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf")
if os.path.exists(staged):
    ctx = S.gguf_context_length(S.Path(staged))
    check("real GGUF header parses", ctx is not None and ctx > 0, f"context_length={ctx}")


# ============================================================ isolated sandbox
# Everything below uses a TEMP config dir — never touches the real one.

tmp = tempfile.mkdtemp(prefix="spark-llm-test-")
os.makedirs(os.path.join(tmp, "models.d"))
os.makedirs(os.path.join(tmp, "logs"))


def write(path, content):
    with open(os.path.join(tmp, path), "w") as f:
        f.write(content)


# Hand-write real TOML (avoid a tomli_w dependency for the test harness)
sandbox_config_text = f'''
models_dir = "{tmp}"
consumer_context_length = 1000
headroom_factor = 2
readiness_timeout_s = 5
min_free_gib_after_load = 0

[[consumers]]
name = "TestConsumer"
port = 9000
alias = "test-brain"

[auto]
free_port_start = 9001
weights_overhead = 1.15
kv_margin_gib = 1
image = "test-image"
extra_flags = []
llamacpp_extra_flags = []

[engines]
llamacpp_bin = "/bin/true"
'''
with open(os.path.join(tmp, "config.toml"), "w") as f:
    f.write(sandbox_config_text)

# Two fake declarations: one "small" (fits, satisfies headroom), one "huge" (never fits)
os.makedirs(os.path.join(tmp, "model-small"))
with open(os.path.join(tmp, "model-small", "config.json"), "w") as f:
    json.dump({"max_position_embeddings": 4096}, f)
with open(os.path.join(tmp, "model-small", "model.safetensors"), "w") as f:
    f.write("x")

os.makedirs(os.path.join(tmp, "model-huge"))
with open(os.path.join(tmp, "model-huge", "config.json"), "w") as f:
    json.dump({"max_position_embeddings": 4096}, f)
with open(os.path.join(tmp, "model-huge", "model.safetensors"), "w") as f:
    f.write("x")

os.makedirs(os.path.join(tmp, "model-large"))
with open(os.path.join(tmp, "model-large", "config.json"), "w") as f:
    json.dump({"max_position_embeddings": 4096}, f)
with open(os.path.join(tmp, "model-large", "model.safetensors"), "w") as f:
    f.write("x")

write("models.d/model-small.toml", '''
engine = "vllm-docker"
image = "test-image"
weights = "/models/model-small"
served_name = "model-small"
gpu_memory_utilization = 0.01
max_model_len = 4096
est_weights_gb = 1
extra_flags = []
''')
write("models.d/model-huge.toml", '''
engine = "vllm-docker"
image = "test-image"
weights = "/models/model-huge"
served_name = "model-huge"
gpu_memory_utilization = 5.0
max_model_len = 4096
est_weights_gb = 999999
extra_flags = []
''')
write("models.d/model-large.toml", '''
engine = "vllm-docker"
image = "test-image"
weights = "/models/model-large"
served_name = "model-large"
gpu_memory_utilization = 0.85
max_model_len = 4096
est_weights_gb = 1
extra_flags = []
''')

os.environ["SPARK_LLM_DIR"] = tmp
tcfg = S.load_config()
tdecls = S.load_declarations()

check("sandbox: three declarations loaded", len(tdecls) == 3, str(sorted(tdecls)))
check("sandbox: naming rule holds for hand-written declarations",
      all(n == S.derive_label(S.decl_folder_name(d), tcfg) for n, d in tdecls.items()))

# Memory ceiling (2026-07-20, the operator's instruction): every calculation must be
# capped at config.toml's max_memory_gb, never the raw OS-reported total.
# Proven with an absurdly small cap (1 GB) that no real test machine could
# possibly have naturally — passing only makes sense if the cap is actually
# being enforced, regardless of the host's real memory.
with open(os.path.join(tmp, "config.toml"), "w") as f:
    f.write("max_memory_gb = 1\n" + sandbox_config_text)
capped_total, capped_avail, _ = S.mem_gb()
check("mem_gb: total is capped at config's max_memory_gb, never the raw OS total",
      capped_total <= 1.0001, f"total={capped_total}")
check("mem_gb: available never exceeds the capped total",
      capped_avail <= capped_total + 0.0001, f"avail={capped_avail}, total={capped_total}")
# restore: no cap for the rest of the suite (keeps every other test's
# arithmetic exactly as already verified against the real host total)
with open(os.path.join(tmp, "config.toml"), "w") as f:
    f.write(sandbox_config_text)

# 8. decide_port: the core of "which one is local-brain" without a stored mapping
empty_loaded = {}
port, consumer = S.decide_port("model-small", tdecls["model-small"], empty_loaded, tcfg)
check("decide_port: fresh load with unfulfilled consumer takes the consumer's port",
      port == 9000 and consumer is not None and consumer["name"] == "TestConsumer")

one_loaded = {"model-small": {"port": 9000}}
port2, consumer2 = S.decide_port("model-huge", tdecls["model-huge"], one_loaded, tcfg)
check("decide_port: second declaration goes to the free pool, not the taken consumer port",
      port2 == 9001 and consumer2 is None)

port3, consumer3 = S.decide_port("model-small", tdecls["model-small"], one_loaded, tcfg)
check("decide_port: a declaration already fulfilling stays put",
      port3 == 9000 and consumer3 is not None)

# PROMOTE, the exact bug the operator hit (2026-07-20): a second model already loaded
# free-floating (fits alongside the first), re-requested via `load` again,
# must take over the consumer's port and displace the current fulfiller —
# not silently stay put, and not get relocated to some other random pool port.
two_loaded = {"model-small": {"port": 9000}, "model-huge": {"port": 9001}}
port4, consumer4 = S.decide_port("model-huge", tdecls["model-huge"], two_loaded, tcfg)
check("decide_port: re-requesting an already-loaded free-floating model PROMOTES it "
      "(takes over the consumer's port from the current fulfiller)",
      port4 == 9000 and consumer4 is not None and consumer4["name"] == "TestConsumer",
      f"got port={port4}")

# Compatibility-aware routing (2026-07-23, the Qwen-Image incident): a fresh
# declaration must not be routed to a consumer it structurally cannot serve.
# Discovered live: loading a diffusers (image-gen) declaration onto the primary consumer's
# empty, tool-requiring slot got refused only by accident, at validate_load —
# decide_port itself had no idea the pairing was hopeless and offered the slot
# anyway. It must skip past an incompatible consumer to a compatible one, or
# to a free pool port if none exists.
tool_only_cfg = {**tcfg, "consumers": [{**tcfg["consumers"][0], "requires_tool_calling": True}]}
incompatible_decl = {**tdecls["model-small"], "engine": "diffusers"}
port5, consumer5 = S.decide_port("incompatible-model", incompatible_decl, {}, tool_only_cfg)
check("decide_port: a declaration that cannot satisfy the only consumer's tool-calling "
      "requirement is not routed there — falls through to a free pool port",
      consumer5 is None and port5 == int(tool_only_cfg["auto"]["free_port_start"]),
      f"got port={port5}, consumer={consumer5}")

mixed_cfg = {**tcfg, "consumers": [{**tcfg["consumers"][0], "requires_tool_calling": True},
                                    {"name": "Image Generation", "port": 9002, "alias": "local-image"}]}
port6, consumer6 = S.decide_port("incompatible-model", incompatible_decl, {}, mixed_cfg)
check("decide_port: skips an incompatible unfulfilled consumer and routes to the next "
      "one this declaration CAN satisfy",
      consumer6 is not None and consumer6["name"] == "Image Generation" and port6 == 9002,
      f"got port={port6}, consumer={consumer6}")

# fulfiller_of / consumer_for_port: the "assignment" that is never stored
c = tcfg["consumers"][0]
check("fulfiller_of: derives correctly from port match, no stored mapping needed",
      S.fulfiller_of(c, one_loaded) == "model-small")
check("consumer_for_port: finds the consumer by port", S.consumer_for_port(tcfg, 9000)["name"] == "TestConsumer")
check("consumer_for_port: None for an unrelated port", S.consumer_for_port(tcfg, 12345) is None)

# 9. validate_load: headroom rule applies only when facing a consumer
errs, warns = S.validate_load("model-small", tdecls["model-small"], 9000, tcfg, c)
check("validate_load: small model facing the consumer passes", errs == [], "; ".join(errs))
errs2, _ = S.validate_load("model-small", tdecls["model-small"], 9001, tcfg, None)
check("validate_load: same model as a free-floating candidate also passes (no headroom check)",
      errs2 == [], "; ".join(errs2))

# Tool-calling rule (2026-07-20): the exact failure class that silently broke
# the primary consumer — an auto-declared model, healthy, but never given --enable-auto-tool-
# choice, facing a consumer that needs it. Must be REFUSED, not discovered later.
tool_consumer = {**c, "requires_tool_calling": True}
errs_tc1, _ = S.validate_load("model-small", tdecls["model-small"], 9000, tcfg, tool_consumer)
check("validate_load: refuses a tool-requiring consumer when the declaration "
      "never enabled tool-calling", any("tool-calling" in e for e in errs_tc1), "; ".join(errs_tc1))

tool_ok_decl = {**tdecls["model-small"], "extra_flags": ["--enable-auto-tool-choice"]}
errs_tc2, _ = S.validate_load("model-small", tool_ok_decl, 9000, tcfg, tool_consumer)
check("validate_load: passes once --enable-auto-tool-choice is present",
      not any("tool-calling" in e for e in errs_tc2), "; ".join(errs_tc2))

# a consumer that does NOT require tools must be unaffected either way
errs_tc3, _ = S.validate_load("model-small", tdecls["model-small"], 9000, tcfg, c)
check("validate_load: a consumer with no tool-calling requirement is unaffected",
      not any("tool-calling" in e for e in errs_tc3), "; ".join(errs_tc3))

# The llamacpp gap this incident class left open: the vllm-docker check above
# only fires for eng == "vllm-docker" — a llamacpp declaration facing a
# tool-requiring consumer must be refused too (unverified tool-calling for
# that engine), not silently let through the one path the earlier fix missed.
llamacpp_decl = {**tdecls["model-small"], "engine": "llamacpp"}
errs_tc4, _ = S.validate_load("model-small", llamacpp_decl, 9000, tcfg, tool_consumer)
check("validate_load: refuses a llamacpp declaration facing a tool-requiring consumer "
      "(unverified engine, not silently permitted)",
      any("tool-calling" in e for e in errs_tc4), "; ".join(errs_tc4))

# unknown engine / missing weights
write("models.d/bad-engine.toml", '''
engine = "not-a-real-engine"
weights = "/models/model-small"
served_name = "bad"
max_model_len = 4096
''')
bad_decls = S.load_declarations()
errs3, _ = S.validate_load("bad-engine", bad_decls["bad-engine"], 9001, tcfg, None)
check("validate_load: unknown engine is rejected", any("not implemented" in e for e in errs3))
os.remove(os.path.join(tmp, "models.d", "bad-engine.toml"))

# loaded.toml round-trip: a declaration name containing a '.' (a real name on
# this machine — 'qwen3.6-35b-a3b-nvfp4') broke live 2026-07-20. An unquoted
# TOML table header treats '.' as nesting syntax, so writing it and reading
# it back silently turned one flat entry into a nested table, and every
# reader that does entry["port"] on the top-level rows then KeyErrors.
dotted_loaded = {"qwen3.6-35b-a3b-nvfp4": {"port": 8000}, "plain-name": {"port": 8001}}
dotted_tmp = os.path.join(tmp, "loaded-dotted.toml")
with open(dotted_tmp, "w") as f:
    f.write(S.toml_dump_flat_tables(dotted_loaded))
dotted_parsed = S.load_toml(S.Path(dotted_tmp))
check("toml_dump_flat_tables: a dotted declaration name round-trips as one flat "
      "entry, not a nested table", dotted_parsed == dotted_loaded, str(dotted_parsed))
os.remove(dotted_tmp)

# 10. compose: alias only appears when facing a consumer; never a stored slot concept
TOTAL_GB_FOR_TESTS = 130.6
argv_facing = S.compose_launch_argv("model-small", tdecls["model-small"], 9000, tcfg, "test-brain", TOTAL_GB_FOR_TESTS)
check("compose: serves both its own name and the consumer alias when facing it",
      "test-brain" in argv_facing and "model-small" in argv_facing)
argv_free = S.compose_launch_argv("model-small", tdecls["model-small"], 9001, tcfg, None, TOTAL_GB_FOR_TESTS)
check("compose: serves only its own name when free-floating (no consumer alias)",
      "test-brain" not in argv_free and "model-small" in argv_free)
check("compose: docker engine composes a docker run", argv_facing[0] == "docker")

# Hard cgroup ceiling (2026-07-20 incident): every vllm-docker container must
# carry --memory/--memory-swap, sized from the declaration's own estimate —
# with none set, one container's overrun degraded the whole machine instead
# of failing cleanly by itself.
expected_cap = int(S.reserve_estimate_gb(tdecls["model-small"], TOTAL_GB_FOR_TESTS, tcfg)
                   * float(tcfg["auto"]["weights_overhead"]) * 1e9)
check("compose: docker run carries a --memory ceiling sized from the declaration's own estimate",
      "--memory" in argv_facing and argv_facing[argv_facing.index("--memory") + 1] == str(expected_cap),
      f"expected {expected_cap}")
check("compose: --memory-swap equals --memory (zero swap headroom for this container)",
      "--memory-swap" in argv_facing
      and argv_facing[argv_facing.index("--memory-swap") + 1] == argv_facing[argv_facing.index("--memory") + 1])

# 11. cmd_load / cmd_unload end-to-end (TEST mode: simulated, but the full
# decision+validation+audit path runs for real)
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "spark_llm.py"), "load", "model-small"],
                   capture_output=True, text=True, env={**os.environ})
check("load (fresh, facing consumer): exits 0", r.returncode == 0, r.stdout[-200:])
check("load: simulates, does not silently claim success without the word 'simulated'",
      "simulated" in r.stdout)

loaded_after = S.load_loaded()
check("load in TEST mode does not actually write loaded.toml (nothing executes)",
      loaded_after == {})  # TEST mode: write_loaded is also simulated — confirms Executor gating

# Force a real loaded.toml write for the next set of checks (bypassing the
# Executor's test-mode gate, since these checks are about STATE LOGIC, not
# about whether execution happened — that's covered above).
with open(os.path.join(tmp, "loaded.toml"), "w") as f:
    f.write('[model-small]\nport = 9000\n')
loaded_manual = S.load_loaded()
check("loaded.toml round-trips", loaded_manual == {"model-small": {"port": 9000}})

r2 = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "spark_llm.py"), "load", "model-small"],
                    capture_output=True, text=True, env={**os.environ})
check("load: idempotent when already fulfilling — no-op message, not a reload",
      "already loaded" in r2.stdout)

r3 = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "spark_llm.py"), "load", "model-huge"],
                    capture_output=True, text=True, env={**os.environ})
check("load: a model that cannot fit is refused, not silently attempted",
      r3.returncode != 0 and "cannot fit" in (r3.stdout + r3.stderr))

# Promoting an already-loaded declaration must not be refused just because its
# OWN reservation, taken alone against current availability, looks like it
# doesn't fit — it frees that exact reservation by stopping itself first.
# (2026-07-20: this refused a real 111 GB promote with only ~13 GB "available",
# because the check didn't know the declaration being promoted would free its
# own memory before needing it again.) model-large (0.85 utilization, the
# same figure as the real qwen3-coder declaration this incident happened
# with) is large enough to fail this check on any real machine WITHOUT the
# self-credit, and passes with it — regardless of the host's actual
# available memory. (model-huge is deliberately impossible even alone —
# 500% of total — so it is the wrong fixture for this specific check: the
# cumulative check added below correctly refuses it regardless of promotion,
# which is separately covered.)
with open(os.path.join(tmp, "loaded.toml"), "w") as f:
    f.write('[model-small]\nport = 9000\n\n[model-large]\nport = 9001\n')
r3b = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "spark_llm.py"), "load", "model-large"],
                     capture_output=True, text=True, env={**os.environ})
check("load: promoting an already-loaded declaration credits its own freed "
      "reservation, not refused as if it needed the memory twice",
      r3b.returncode == 0 and "cannot fit" not in (r3b.stdout + r3b.stderr),
      (r3b.stdout + r3b.stderr)[-200:])
with open(os.path.join(tmp, "loaded.toml"), "w") as f:
    f.write('[model-small]\nport = 9000\n')

# Cumulative check (2026-07-20 incident): a SECOND, DIFFERENT declaration
# already loaded elsewhere was never counted at all before this fix — only
# the one thing being displaced (or a declaration's own prior instance, just
# above) ever factored into the pre-flight math. With model-huge already
# loaded free-floating, loading model-small — trivially small on its own —
# must now be refused, because nothing before this fix ever added
# model-huge's committed reservation into the equation at all. (Honest
# caveat: this proves the cumulative check closes a real gap — it does NOT
# by itself reproduce the exact 2026-07-20 freeze, since that pair's naive
# reservations summed to well under the total; the docker --memory ceiling
# tested above is the fix for that specific case.)
with open(os.path.join(tmp, "loaded.toml"), "w") as f:
    f.write('[model-huge]\nport = 9001\n')
r3c = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                   "spark_llm.py"), "load", "model-small"],
                     capture_output=True, text=True, env={**os.environ})
check("load: refuses a fresh load that would not fit alongside an already-loaded "
      "DIFFERENT declaration, not just the one being displaced",
      r3c.returncode != 0 and "cannot fit alongside" in (r3c.stdout + r3c.stderr),
      (r3c.stdout + r3c.stderr)[-300:])
with open(os.path.join(tmp, "loaded.toml"), "w") as f:
    f.write('[model-small]\nport = 9000\n')

r4 = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "spark_llm.py"), "unload", "model-small"],
                    capture_output=True, text=True, env={**os.environ})
check("unload: exits 0", r4.returncode == 0, r4.stdout[-200:])

r5 = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "spark_llm.py"), "unload", "not-loaded-anything"],
                    capture_output=True, text=True, env={**os.environ})
check("unload: unloading something not loaded is a clean no-op, not an error",
      r5.returncode == 0 and "nothing to do" in r5.stdout)

# 12. status/list run clean against the sandbox
r6 = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "spark_llm.py"), "status"],
                    capture_output=True, text=True, env={**os.environ})
check("status: exits 0 and reports the consumer", r6.returncode == 0 and "TestConsumer" in r6.stdout)

# 13. loaded_status_text: the ONE status shown by every table for a loaded
# entry. Must never claim LOADED/healthy while a model is still starting —
# this is the exact bug the operator caught (2026-07-20): the Models table showed
# "LOADED" in green for a model 130s into loading, while the Loaded-models
# table correctly showed it DOWN.
base = {"decl_name": "x", "consumer": None, "alias_ok": True, "uptime_s": None}
mid_load = {**base, "up": False, "unit_state": "activating"}
text, style = S.loaded_status_text(mid_load)
check("loaded_status_text: mid-load shows LOADING, not LOADED/healthy",
      "LOADING" in text and "LOADED" not in text and "healthy" not in text, text)
check("loaded_status_text: mid-load is NOT styled green (which reads as success)",
      "green" not in style, style)

crashed = {**base, "up": False, "unit_state": "failed"}
text2, style2 = S.loaded_status_text(crashed)
check("loaded_status_text: a failed unit shows DOWN, not LOADING", "DOWN" in text2, text2)

healthy = {**base, "up": True, "unit_state": "active"}
text3, style3 = S.loaded_status_text(healthy)
check("loaded_status_text: genuinely up shows healthy", "healthy" in text3, text3)

# 14. diffusers engine (image generation, e.g. Qwen-Image) — added alongside
# vllm-docker and llamacpp. No real GPU/model needed: these checks exercise
# detection, declaration synthesis, and command composition, not generation.
check("KNOWN_ENGINES includes diffusers", "diffusers" in S.KNOWN_ENGINES)

os.makedirs(os.path.join(tmp, "model-diffusion", "transformer"))
os.makedirs(os.path.join(tmp, "model-diffusion", "vae"))
with open(os.path.join(tmp, "model-diffusion", "model_index.json"), "w") as f:
    json.dump({
        "_class_name": "FakePipeline",
        "transformer": ["diffusers", "FakeTransformer"],
        "vae": ["diffusers", "FakeVAE"],
    }, f)
with open(os.path.join(tmp, "model-diffusion", "transformer", "dummy.safetensors"), "w") as f:
    f.write("x")

status, reason = S.probe_folder("model-diffusion", tcfg)
check("probe_folder: recognizes a complete diffusers pipeline as ok", status == "ok", reason)

label, decl_text = S.synth_declaration("model-diffusion", tcfg)
check('synth_declaration: writes engine = "diffusers" for a diffusers pipeline',
      'engine = "diffusers"' in decl_text, decl_text[:120])

# Incomplete pipeline (a component folder the manifest names is missing) must
# be blocked, not silently accepted — same discipline as the safetensors-index
# completeness check already applied to vllm-docker checkpoints.
os.makedirs(os.path.join(tmp, "model-diffusion-incomplete"))
with open(os.path.join(tmp, "model-diffusion-incomplete", "model_index.json"), "w") as f:
    json.dump({"_class_name": "FakePipeline", "transformer": ["diffusers", "FakeTransformer"]}, f)
status2, reason2 = S.probe_folder("model-diffusion-incomplete", tcfg)
check("probe_folder: blocks a diffusers pipeline missing a manifest-referenced component folder",
      status2 == "blocked" and "missing component" in reason2, reason2)

write("models.d/model-diffusion.toml", '''
engine = "diffusers"
weights = "/models/model-diffusion"
served_name = "model-diffusion"
est_weights_gb = 1
default_size = "512x512"
num_inference_steps = 10
true_cfg_scale = 2.0
''')
tdecls2 = S.load_declarations()
argv_diff = S.compose_launch_argv("model-diffusion", tdecls2["model-diffusion"], 9002, tcfg, None, TOTAL_GB_FOR_TESTS)
check("compose: diffusers engine composes a docker run", argv_diff[0] == "docker")
check("compose: diffusers container bind-mounts image_server.py read-only",
      any(a.endswith(":/opt/image_server.py:ro") for a in argv_diff))
check("compose: diffusers container ALSO bind-mounts image_server_lib.py (2026-07-23 "
      "regression: the container crashed with ModuleNotFoundError when this was missing)",
      any(a.endswith(":/opt/image_server_lib.py:ro") for a in argv_diff))
check("compose: diffusers passes --weights via the /models/<folder> bind-mount convention",
      "/models/model-diffusion" in argv_diff)
check("compose: diffusers carries a --memory ceiling too (same mechanism as vllm-docker, item 38)",
      "--memory" in argv_diff)
check("compose: diffusers uses the configured diffusers_image (falls back to the hardcoded default "
      "since the sandbox config doesn't set one)",
      "spark-llm-diffusers:latest" in argv_diff)

# Memory-cap double-count bug (2026-07-23): compose_diffusers_argv used to
# ALSO multiply by weights_overhead on top of reserve_estimate_gb()'s own
# result, which already applies it internally for a declaration with no
# gpu_memory_utilization key (every diffusers declaration) — inflating the
# real cgroup cap ~15% above the pre-flight estimate cmd_load actually
# validated against. Fixed: the cap must equal reserve_estimate_gb(...)
# exactly, in bytes, no second multiplication.
expected_diff_cap = int(S.reserve_estimate_gb(tdecls2["model-diffusion"], TOTAL_GB_FOR_TESTS, tcfg) * 1e9)
check("compose: diffusers --memory cap is NOT double-counted against reserve_estimate_gb",
      "--memory" in argv_diff and argv_diff[argv_diff.index("--memory") + 1] == str(expected_diff_cap),
      f"expected {expected_diff_cap}, argv has {argv_diff[argv_diff.index('--memory') + 1] if '--memory' in argv_diff else 'MISSING'}")

# extra_flags consistency fix (2026-07-23): compose_docker_argv and
# compose_llamacpp_argv both pass a declaration's extra_flags through;
# compose_diffusers_argv silently dropped the key. Same declaration, same
# check pattern as the other two engines already use elsewhere in this file.
write("models.d/model-diffusion-flags.toml", '''
engine = "diffusers"
weights = "/models/model-diffusion"
served_name = "model-diffusion-flags"
est_weights_gb = 1
extra_flags = ["--some-future-flag", "value"]
''')
tdecls_flags = S.load_declarations()
argv_diff_flags = S.compose_launch_argv("model-diffusion-flags", tdecls_flags["model-diffusion-flags"],
                                         9003, tcfg, None, TOTAL_GB_FOR_TESTS)
check("compose: diffusers engine now passes extra_flags through (was silently dropped)",
      "--some-future-flag" in argv_diff_flags and "value" in argv_diff_flags)
os.remove(os.path.join(tmp, "models.d", "model-diffusion-flags.toml"))

# 15. image_server_lib — pure logic shared with image_server.py, deliberately
# dependency-free (no torch/diffusers/fastapi/pydantic) so it's importable and
# testable here even though image_server.py itself only runs inside the
# diffusers container. extract_prompt() takes anything with .role/.content
# attributes — a plain SimpleNamespace stands in for a pydantic ChatMessage.
import image_server_lib as ISL
import types as _types

check("image_server_lib.parse_size: parses a real WxH string",
      ISL.parse_size("1024x1024", "1328x1328") == (1024, 1024))
check("image_server_lib.parse_size: falls back to the declaration default when empty",
      ISL.parse_size("", "1328x1328") == (1328, 1328))
check("image_server_lib.parse_size: falls back on garbage input, never raises",
      ISL.parse_size("garbage", "1328x1328") == (1328, 1328))
check("image_server_lib.parse_size: hardcoded final fallback when both are empty",
      ISL.parse_size("", "") == (1328, 1328))

_ns = _types.SimpleNamespace
check("image_server_lib.extract_prompt: plain string content",
      ISL.extract_prompt([_ns(role="user", content="hello")]) == "hello")
check("image_server_lib.extract_prompt: takes the LATEST user message, not the first",
      ISL.extract_prompt([_ns(role="user", content="first"),
                          _ns(role="assistant", content="reply"),
                          _ns(role="user", content="second")]) == "second")
check("image_server_lib.extract_prompt: no user message at all returns empty, not a crash",
      ISL.extract_prompt([_ns(role="assistant", content="only assistant")]) == "")
check("image_server_lib.extract_prompt: multi-part content (vision-style messages) joins text parts",
      ISL.extract_prompt([_ns(role="user", content=[{"type": "text", "text": "part1"},
                                                     {"type": "text", "text": "part2"}])]) == "part1 part2")
check("image_server_lib.extract_prompt: empty message list returns empty, not a crash",
      ISL.extract_prompt([]) == "")

check("image_server_lib.is_auxiliary_task: detects Open WebUI's internal job prefix",
      ISL.is_auxiliary_task("### Task:\nSuggest 3-5 follow-ups") is True)
check("image_server_lib.is_auxiliary_task: detects it past leading whitespace",
      ISL.is_auxiliary_task("   ### Task:\nwith leading space") is True)
check("image_server_lib.is_auxiliary_task: a real user prompt is never flagged",
      ISL.is_auxiliary_task("a fat alien cat astronaut floating in space") is False)

check("image_server_lib.chunk_text: empty content still yields one (empty) piece — "
      "needed so the SSE path can always send an initial role-establishing delta",
      ISL.chunk_text("", 4) == [""])
check("image_server_lib.chunk_text: content shorter than chunk_size is a single piece",
      ISL.chunk_text("hello", 100) == ["hello"])
check("image_server_lib.chunk_text: splits correctly on a non-even boundary",
      ISL.chunk_text("abcdefgh", 3) == ["abc", "def", "gh"])
check("image_server_lib.chunk_text: reassembling every piece reproduces the original content exactly",
      "".join(ISL.chunk_text("x" * 250_000, 100_000)) == "x" * 250_000)
check("image_server_lib.chunk_text: every piece stays at or under chunk_size — the actual "
      "line-length-limit safety property this function exists for",
      all(len(p) <= 100_000 for p in ISL.chunk_text("y" * 250_000, 100_000)))

# unit_memory_gb: engine-aware marker selection must not crash for either
# engine, even when — as in this sandbox — no real process matches.
mem_vllm = S.unit_memory_gb(tdecls["model-small"])
mem_diff = S.unit_memory_gb(tdecls2["model-diffusion"])
check("unit_memory_gb: returns None (not a crash) for a vllm-docker decl with no running process",
      mem_vllm is None)
check("unit_memory_gb: returns None (not a crash) for a diffusers decl with no running process",
      mem_diff is None)

shutil.rmtree(tmp)

print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    sys.exit(1)
print("all checks passed")
