"""Live server gate: wait for Ollama, confirm qwen3:4b, send one trivial request.

Run this after starting the server and before run_pilot.py. Exits 0 only if all
three checks pass; otherwise non-zero with a diagnostic naming the endpoint.

    python smoke_test.py [--ready-timeout 180] [--smoke-timeout 60]
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT,
          os.path.join(ROOT, "option_c", "agents"),
          os.path.join(ROOT, "option_c", "eval"),
          os.path.join(ROOT, "option_c")):
    if p not in sys.path:
        sys.path.insert(0, p)

import ollama_health as H  # noqa: E402
from config import DEFAULT_CONFIG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ready-timeout", type=float, default=180.0)
    ap.add_argument("--smoke-timeout", type=float, default=60.0)
    a = ap.parse_args()

    print(f"Endpoint: {DEFAULT_CONFIG.url}  | expected model: {H.EXPECTED_MODEL}")
    ok, detail = H.preflight(DEFAULT_CONFIG, ready_timeout=a.ready_timeout,
                             smoke_timeout=a.smoke_timeout, verbose=True)
    print("\nRESULT:", "SMOKE TEST PASS" if ok else f"SMOKE TEST FAIL (stage={detail.get('stage')})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
