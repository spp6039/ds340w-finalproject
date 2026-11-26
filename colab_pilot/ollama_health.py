"""Ollama server-health checks: readiness, model availability, and a smoke test.

The first pilot failed with 36x "Connection refused" because requests went out
before the server was listening. These helpers replace a blind sleep with a real
poll + one trivial request. Network calls accept an injectable opener/transport so
the tests run without a live server. Nothing here runs at import time.
"""
import json
import time
import urllib.request
import urllib.error
from dataclasses import replace

from config import DEFAULT_CONFIG
import agent_a  # reuse the agents' transport for the smoke test

EXPECTED_MODEL = DEFAULT_CONFIG.model  # "qwen3:4b", exact tag -- never substituted

# Start enforcing the fail-fast error rate only after this many requests, so one
# early hiccup can't kill a healthy run.
MIN_SAMPLES_BEFORE_ABORT = 3


def should_abort(n_done, n_err, threshold=0.20, min_samples=MIN_SAMPLES_BEFORE_ABORT):
    """True once enough requests are in and the request_error rate exceeds threshold."""
    if n_done < min_samples:
        return False
    return (n_err / n_done) > threshold


def _default_get(url, timeout):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def tags_url(config=DEFAULT_CONFIG):
    return config.host.rstrip("/") + "/api/tags"


def diagnose(config=DEFAULT_CONFIG, exc=None):
    """A message naming the endpoints and what to check when the server is down."""
    lines = [
        "Ollama server health check FAILED.",
        f"  generate endpoint : {config.url}",
        f"  tags endpoint     : {tags_url(config)}",
        f"  expected model    : {EXPECTED_MODEL}",
    ]
    if exc is not None:
        lines.append(f"  error             : {type(exc).__name__}: {exc}")
    lines += [
        "  Likely cause: the Ollama server is not running or not yet listening.",
        "  Check, in the Colab runtime:",
        "    * is the process alive?        ->  !pgrep -a ollama   (or: !ollama ps)",
        "    * did `ollama serve` start?    ->  re-run the start cell, then wait for readiness",
        "    * can you reach the port?      ->  !curl -s http://localhost:11434/api/tags",
        "  Do NOT start the pilot until the readiness check and smoke test both PASS.",
    ]
    return "\n".join(lines)


def check_ready(config=DEFAULT_CONFIG, timeout=180.0, interval=2.0,
                http_get=_default_get, sleep=time.sleep, now=time.monotonic):
    """Poll /api/tags until it returns 200 or we time out.

    Returns (ready, info). info has 'elapsed', 'attempts', and either 'models' or
    'last_error'. A connection error just means not-ready-yet; we never raise.
    """
    start = now()
    attempts = 0
    last_err = None
    while True:
        attempts += 1
        try:
            status, body = http_get(tags_url(config), timeout=min(10.0, timeout))
            if status == 200:
                try:
                    models = [m.get("name") for m in json.loads(body).get("models", [])]
                except Exception:
                    models = []
                return True, {"elapsed": now() - start, "attempts": attempts, "models": models}
            last_err = f"HTTP {status}"
        except (urllib.error.URLError, ConnectionError, OSError) as e:
            last_err = f"{type(e).__name__}: {e}"
        if now() - start >= timeout:
            return False, {"elapsed": now() - start, "attempts": attempts, "last_error": last_err}
        sleep(interval)


def list_models(config=DEFAULT_CONFIG, timeout=10.0, http_get=_default_get):
    status, body = http_get(tags_url(config), timeout=timeout)
    if status != 200:
        raise RuntimeError(f"/api/tags returned HTTP {status}")
    return [m.get("name") for m in json.loads(body).get("models", [])]


def model_available(config=DEFAULT_CONFIG, model=None, http_get=_default_get, timeout=10.0):
    """(present, installed_tags). Exact-tag match only; never pulls or substitutes."""
    want = model or EXPECTED_MODEL
    try:
        installed = list_models(config, timeout=timeout, http_get=http_get)
    except Exception:
        return False, []
    return (want in installed), installed


# A short-timeout, small-budget copy of the shared config so the probe returns
# fast. num_predict=512 (not 16): with think=True Qwen3 reasons first, and a tiny
# budget gets spent on reasoning before any answer (a 128-token probe came back
# empty with done_reason=length). This is a new object; DEFAULT_CONFIG is untouched
# and this variant is never recorded in results.
SMOKE_PROMPT = "Reply with the single word: ready"


def smoke_config(config=DEFAULT_CONFIG, timeout=60.0, num_predict=512):
    return replace(config, request_timeout=timeout, num_predict=num_predict)


def smoke_test(config=DEFAULT_CONFIG, transport=None, timeout=60.0):
    """Send one trivial-prompt request through the agents' transport and validate it.

    Passes only if: the request returned a dict, it was not truncated
    (done_reason != "length"), `response` has non-empty text, and the served model
    is the expected tag. Writes nothing; never counts as a record.
    """
    cfg = smoke_config(config, timeout=timeout)
    tx = transport or agent_a._ollama_transport
    out = {"ok": False, "model": None, "text": None, "done_reason": None, "error": None}
    try:
        resp = tx(SMOKE_PROMPT, cfg)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    if not isinstance(resp, dict):
        out["error"] = f"transport returned {type(resp).__name__}, expected dict"
        return out
    text = (resp.get("response") or "").strip()
    model = resp.get("model")
    done_reason = resp.get("done_reason")
    out.update(model=model, text=text, done_reason=done_reason)
    if done_reason == "length":
        out["error"] = (f"truncated (done_reason='length'); num_predict="
                        f"{cfg.num_predict} too small to finish thinking + answer")
        return out
    if not text:
        out["error"] = "empty generated text (no `response`)"
        return out
    if model is not None and model != EXPECTED_MODEL:
        out["error"] = f"served model {model!r} != expected {EXPECTED_MODEL!r}"
        return out
    out["ok"] = True
    return out


def preflight(config=DEFAULT_CONFIG, ready_timeout=180.0, smoke_timeout=60.0, verbose=True):
    """readiness -> model availability -> smoke, in order. Returns (ok, detail)."""
    ready, info = check_ready(config, timeout=ready_timeout)
    if not ready:
        if verbose:
            print(diagnose(config, exc=info.get("last_error")))
        return False, {"stage": "readiness", **info}
    if verbose:
        print(f"[readiness] OK in {info['elapsed']:.1f}s ({info['attempts']} attempt(s)); "
              f"models={info.get('models')}")

    present, installed = model_available(config)
    if not present:
        msg = (f"Expected model {EXPECTED_MODEL!r} is NOT installed. Installed tags: "
               f"{installed}. Pull it with `ollama pull {EXPECTED_MODEL}` -- do not "
               f"substitute a different model or tag.")
        if verbose:
            print(msg)
        return False, {"stage": "model", "installed": installed}
    if verbose:
        print(f"[model] OK: {EXPECTED_MODEL} is installed")

    s = smoke_test(config, timeout=smoke_timeout)
    if not s["ok"]:
        if verbose:
            print(f"[smoke] FAILED: {s['error']}")
            print(diagnose(config, exc=s["error"]))
        return False, {"stage": "smoke", **s}
    if verbose:
        print(f"[smoke] OK: model={s['model']} text={s['text']!r}")
    return True, {"stage": "ok", "smoke": s}
