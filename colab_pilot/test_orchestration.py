"""Tests for the orchestration helpers (readiness, model check, smoke, fail-fast).
No live Ollama: every network path uses an injected opener/transport."""
import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT,
          os.path.join(ROOT, "option_c", "agents"),
          os.path.join(ROOT, "option_c", "eval"),
          os.path.join(ROOT, "option_c")):
    if p not in sys.path:
        sys.path.insert(0, p)

import urllib.error  # noqa: E402
import ollama_health as H  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402

FAILS = []
def check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond:
        FAILS.append(name)


def make_get(sequence):
    """http_get that returns each item of `sequence` per call; an Exception item is raised."""
    calls = {"i": 0}

    def _get(url, timeout):
        i = calls["i"]
        item = sequence[min(i, len(sequence) - 1)]
        calls["i"] = i + 1
        if isinstance(item, Exception):
            raise item
        return item
    _get.calls = calls
    return _get


TAGS_OK = (200, json.dumps({"models": [{"name": "qwen3:4b"}, {"name": "llama3:8b"}]}))
TAGS_NO_MODEL = (200, json.dumps({"models": [{"name": "llama3:8b"}]}))
REFUSED = urllib.error.URLError(OSError(111, "Connection refused"))


print("[1] readiness poll")
TICK = {"t": 0.0}
def fake_now(): return TICK["t"]
def fake_sleep(s): TICK["t"] += s

ready, info = H.check_ready(timeout=10, interval=1, http_get=make_get([TAGS_OK]),
                            sleep=fake_sleep, now=fake_now)
check("ready immediately when /api/tags returns 200", ready and "qwen3:4b" in info["models"])

TICK["t"] = 0.0
g = make_get([REFUSED, REFUSED, TAGS_OK])
ready, info = H.check_ready(timeout=10, interval=1, http_get=g, sleep=fake_sleep, now=fake_now)
check("becomes ready after transient connection-refused", ready and info["attempts"] == 3)

TICK["t"] = 0.0
g = make_get([REFUSED])
ready, info = H.check_ready(timeout=5, interval=1, http_get=g, sleep=fake_sleep, now=fake_now)
check("not-ready on persistent connection-refused (no raise)",
      (not ready) and "Connection refused" in info["last_error"])


print("[2] model availability")
present, installed = H.model_available(http_get=make_get([TAGS_OK]))
check("exact tag qwen3:4b present -> True", present and "qwen3:4b" in installed)

present, installed = H.model_available(http_get=make_get([TAGS_NO_MODEL]))
check("qwen3:4b absent -> False (does not substitute)", (not present) and "qwen3:4b" not in installed)

present, installed = H.model_available(http_get=make_get([REFUSED]))
check("unreachable tags endpoint -> False, no raise", present is False)


print("[3] one-request smoke test")
scfg = H.smoke_config()
check("smoke_config num_predict == 512", scfg.num_predict == 512)
check("smoke_config preserves think=True", scfg.think is DEFAULT_CONFIG.think is True)
check("smoke_config preserves model/temp/top_p/endpoint",
      scfg.model == DEFAULT_CONFIG.model and scfg.temperature == DEFAULT_CONFIG.temperature
      and scfg.top_p == DEFAULT_CONFIG.top_p and scfg.url == DEFAULT_CONFIG.url)
check("DEFAULT_CONFIG untouched (num_predict still 2048)", DEFAULT_CONFIG.num_predict == 2048)

def tx_ok(prompt, cfg):
    assert prompt == H.SMOKE_PROMPT
    assert cfg.url.endswith("/api/generate")
    assert cfg.num_predict == 512
    assert cfg.think is True
    return {"response": "ready", "model": "qwen3:4b", "done_reason": "stop"}
s = H.smoke_test(transport=tx_ok)
check("smoke SUCCESS: nonempty text + correct model + done_reason=stop",
      s["ok"] and s["text"] == "ready" and s["model"] == "qwen3:4b" and s["done_reason"] == "stop")

s = H.smoke_test(transport=lambda p, c: {"response": "rea", "model": "qwen3:4b", "done_reason": "length"})
check("smoke FAIL on truncation (done_reason=length) despite partial text",
      (not s["ok"]) and "truncated" in s["error"] and s["done_reason"] == "length")

s = H.smoke_test(transport=lambda p, c: {"response": "", "model": "qwen3:4b", "done_reason": "length"})
check("smoke FAIL on empty+length (the live num_predict-too-small case)",
      (not s["ok"]) and "truncated" in s["error"])

s = H.smoke_test(transport=lambda p, c: (_ for _ in ()).throw(RuntimeError("boom")))
check("smoke FAIL on transport exception", (not s["ok"]) and "boom" in s["error"])

s = H.smoke_test(transport=lambda p, c: {"response": "   ", "model": "qwen3:4b", "done_reason": "stop"})
check("smoke FAIL on empty generated text (not truncation)",
      (not s["ok"]) and "empty" in s["error"])

s = H.smoke_test(transport=lambda p, c: {"response": "hi", "model": "qwen3:14b", "done_reason": "stop"})
check("smoke FAIL on wrong served model", (not s["ok"]) and "!=" in s["error"])

s = H.smoke_test(transport=lambda p, c: {"response": "hi"})
check("smoke OK when model/done_reason absent but text present", s["ok"])


print("[4] fail-fast predicate should_abort")
check("no abort before min samples (2/2 fail)", H.should_abort(2, 2) is False)
check("abort when all fail (3/3)", H.should_abort(3, 3) is True)
check("abort when >20% (e.g. 5/10)", H.should_abort(10, 5) is True)
check("no abort at exactly 20% (2/10)", H.should_abort(10, 2) is False)
check("no abort for healthy run (1/36)", H.should_abort(36, 1) is False)
check("abort honors custom threshold (3/10 > 0.25)", H.should_abort(10, 3, threshold=0.25) is True)


print("[5] diagnostic message")
d = H.diagnose()
check("diagnostic names generate endpoint", "/api/generate" in d)
check("diagnostic names tags endpoint", "/api/tags" in d)
check("diagnostic suggests checking the ollama process", "ollama" in d.lower() and "process" in d.lower())


print("\nRESULT:", "ALL ORCHESTRATION TESTS PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(0 if not FAILS else 1)
