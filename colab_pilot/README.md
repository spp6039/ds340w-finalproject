# AmbiQT Three-Agent Colab Pilot

> **For the final demo / reproduction, follow the top-level [`README.md`](../README.md) and `DS340W_Demo.ipynb`.** This file documents the bundle internals. The upload-the-ZIP instructions below are from development and are no longer needed, because the notebook clones this repo directly. The paper uses `--num-predict 8192`.

Runs **Agent A**, **Agent A2**, and **Agent B** on the **same 12 eligible AmbiQT
validation records** using **Qwen3:4B** via local Ollama, and scores every
candidate with the real Stage 2 pipeline. Self-contained: no network access to
the original repo is needed.

> **Orchestration hardened (v2).** The first pilot produced 36×
> `URLError: <urlopen error [Errno 111] Connection refused>` because the runner
> fired requests before the Ollama server was listening (a fixed `sleep` is not a
> readiness guarantee). v2 adds a **readiness poll**, an **exact-tag model check**,
> a **one-request smoke test** over the agents' real transport, and a **fail-fast**
> guard in the runner. **Agent prompts, the shared generation config, scoring,
> golds, benchmark data, and the 12-record selection are unchanged.**

## What's in here

```
colab_pilot/
├── README.md                     # this file
├── requirements.txt              # sqlglot==30.31.0 (rest is Python stdlib)
├── pilot_records.json            # the 12 selected record IDs + selection method
├── ollama_health.py              # NEW: readiness poll, model check, smoke test, fail-fast, diagnostics
├── smoke_test.py                 # NEW: standalone live server gate (readiness→model→1 request)
├── test_orchestration.py         # NEW: unit tests for the above (NO live Ollama)
├── prepare_and_verify.py         # dry check: imports, DB paths, gold prep, isolation (NO model)
├── run_pilot.py                  # the live pilot runner (num_predict=4096 default, --num-predict; preflight + fail-fast)
├── test_run_pilot_config.py      # unit tests for the configurable budget (NO live Ollama)
├── run_budget_diagnostic.py      # OPTIONAL: num_predict=4096 completion diagnostic (separate outputs)
├── test_budget_diagnostic.py     # unit tests for the diagnostic (NO live Ollama)
├── run_reasoning_probe.py        # OPTIONAL: 1-record (#328) probe capturing the `thinking` field at num_predict=8192
├── test_reasoning_probe.py       # mock tests for the probe (NO live Ollama)
├── option_c/
│   ├── manifest_unified.json     # MINIMAL manifest: the 12 selected records (all fields)
│   ├── compare.py  execute.py    # result comparison + base DB connection
│   ├── eval/   golds.py  score.py  interpret.py  safe_execute.py
│   ├── agents/ config.py  agent_a.py  agent_a2.py  agent_b.py  adapter.py  __init__.py
│   └── perturbed_dbs/<category>/<db_id>__<key>.sqlite   # the 12 reconstructed DBs (~1 MB total)
└── AmbiQT-master/AmbiQT-master/benchmark/<category>/validation.json
                                  # MINIMAL, index-preserving: selected records sit at their
                                  # ORIGINAL indices; all other slots are null (unused)
```

### How paths resolve (do not rename folders)
The bundle root is treated as the project root. `option_c/eval/golds.py` computes
`PROJECT` from its own location, then loads:
- manifest  → `<root>/option_c/manifest_unified.json`
- validation → `<root>/AmbiQT-master/AmbiQT-master/benchmark/<cat>/validation.json`
- each DB    → `<root>/<perturbed_db_path>` (as stored in the manifest)

`compare.py` and `execute.py` live in `option_c/` (not `eval/`) because the
modules add that directory to `sys.path`. `ollama_health.py` and `smoke_test.py`
live at the bundle root. Keep the tree exactly as shipped.

### Agent isolation (why this is a fair test)
`run_pilot.py` calls `agent.generate(question, schema_without_content)` only. The
agent never receives gold queries, gold results, interpretation metadata, the
perturbed-DB contents, or Stage 2 labels. Gold data is used **only** by the
scorer, after generation. The smoke test uses a **trivial prompt**, writes nothing
to results, and is never counted as an experimental record.

---

## Colab setup (copy-paste cells, in this order)

**Recommended:** `Runtime ▸ Change runtime type ▸ T4 GPU`. Ollama auto-detects the
GPU, which makes Qwen3:4B dramatically faster than CPU.

> **Order matters.** Start the server, then **clear the smoke test (Cell 5)**
> before running the pilot (Cell 7). Do not run the pilot on a red smoke test.

### Cell 1 — upload & unzip the bundle
```python
from google.colab import files
up = files.upload()                       # choose ambiqt_colab_pilot.zip
!unzip -q ambiqt_colab_pilot.zip -d /content
%cd /content/colab_pilot
!ls
```

### Cell 2 — Python requirements
```python
!pip -q install -r requirements.txt       # sqlglot==30.31.0
```

### Cell 3 — verify the bundle (NO model inference)
```python
!python prepare_and_verify.py
```
Expect `RESULT: ALL CHECKS PASS` (imports, 12 DB paths, 12 golds prepared,
validation mapping, agent isolation, scorer access). This checks *data/code*
health and does not need Ollama.

### Cell 4 — install Ollama and start the server, then pull Qwen3:4B
```python
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess
subprocess.Popen(["ollama", "serve"])     # background server (do NOT rely on a fixed sleep)
!ollama pull qwen3:4b                      # pull blocks until the model is present
```
(No `time.sleep` gate here on purpose — the next cell *polls* for readiness.)

### Cell 5 — SERVER GATE: readiness + model check + one-request smoke test
```python
!python smoke_test.py
```
This **polls** `http://localhost:11434/api/tags` until the server answers, confirms
the **exact tag `qwen3:4b`** is installed (it will not substitute another model),
and fires **one trivial-prompt request** through the agents' real transport.
Expect `RESULT: SMOKE TEST PASS`. **If it fails, do not run the pilot** — the
output names the endpoint and tells you what to check (is `ollama serve` alive? is
the port reachable?). Re-run Cell 4, wait, and re-run Cell 5.

### Cell 6 — mount Google Drive for persistent results
```python
from google.colab import drive
drive.mount('/content/drive')
import os
OUT = '/content/drive/MyDrive/ambiqt_pilot'
os.makedirs(OUT, exist_ok=True)
```

### Cell 7 — run the live pilot (12 records × 3 agents = 36 generations)
```python
!python run_pilot.py --out "/content/drive/MyDrive/ambiqt_pilot"
```
The pilot runs at a **generation budget of `num_predict=4096`** by default (one
shared `dataclasses.replace(DEFAULT_CONFIG, num_predict=4096)` config passed to all
three agents — `DEFAULT_CONFIG` stays 2048 and is not modified; the budget is
recorded in `pilot_meta_*.json` as `num_predict`). Override with
`--num-predict N` if needed (e.g. `--num-predict 2048` to reproduce the first run).

`run_pilot.py` runs its **own preflight** (readiness + model + smoke) first and
**aborts with exit code 2** if the server is unreachable, before touching any
record. During the run it **fails fast**: if more than 20% of requests return
`request_error` (all-fail being the limiting case), it stops with a prominent
diagnostic and **exit code 3**, preserving whatever JSONL rows were already
written. A clean run ends with `RUN OK: 36/36 requests written, 0 request_error.`

Outputs land in your Drive:
- `pilot_generations_<run_id>.jsonl` — one row per (record, agent): candidates,
  raw response, status, `done_reason`, prompt/total-generated tokens (total
  **includes** Qwen3 thinking tokens), latency, candidate counts, and the Stage 2
  score (`record_score` + per-candidate `candidate_scores`).
- `pilot_meta_<run_id>.json` — run metadata (model, generation settings, host,
  `error_threshold`).

### Cell 8 — quick look at budget adequacy
```python
import json, collections
run = sorted([f for f in os.listdir(OUT) if f.startswith("pilot_generations_")])[-1]
rows = [json.loads(l) for l in open(os.path.join(OUT, run))]
by = collections.defaultdict(list)
for r in rows: by[r["agent"]].append(r)
for a, rs in by.items():
    tok = [r["total_generated_tokens"] or 0 for r in rs]
    trunc = sum(1 for r in rs if r["done_reason"] == "length")
    perr  = sum(1 for r in rs if r["generation_status"] == "parse_error")
    rerr  = sum(1 for r in rs if r["generation_status"] == "request_error")
    print(f"{a}: req_err={rerr}/{len(rs)} trunc(len)={trunc}/{len(rs)} parse_err={perr} "
          f"max_tok={max(tok)} mean_lat={sum(r['latency'] for r in rs)/len(rs):.1f}s")
```
`request_error > 0` means the server dropped mid-run (investigate before trusting
results). `done_reason == "length"` or `max_tok` near **2048** = the shared
`num_predict` budget is too small (watch Agent B first).

---

## Optional: generation-budget diagnostic (num_predict 2048 → 4096)

A **separate, isolated** diagnostic to test whether a larger token budget improves
generation *completion* (fewer `done_reason="length"` truncations). It reuses the
real agents/transport/scoring and the same 12 records, but runs them under a
throwaway `dataclasses.replace(DEFAULT_CONFIG, num_predict=4096, think=True)` copy.
**It does not change the official experiment** — `DEFAULT_CONFIG` stays at 2048,
and results go to a separate directory with a `budget_diag_*` filename prefix so
the `pilot_*` outputs are never touched.

Run it *after* the server gate (Cell 5) passes, into its **own** Drive folder:
```python
!python run_budget_diagnostic.py --out "/content/drive/MyDrive/ambiqt_budget_diag"
```
It runs its own preflight (exit 2 if the server is down), attempts all 36
generations, and prints a per-agent **ok / truncated / parse_err / req_err** table.
Exit `0` = all 36 generated with the server available (truncations are a valid
signal, not a failure); exit `3` = fewer than 36 rows or any `request_error`.
Outputs: `budget_diag_generations_<run_id>.jsonl`, `budget_diag_meta_<run_id>.json`.
Compare its truncation counts against the 2048 pilot before deciding anything —
and do **not** change the official config based on it without a separate decision.

## Local / CI checks (no live Ollama)
```python
!python test_orchestration.py       # readiness, model check, smoke, fail-fast — mocked
!python test_budget_diagnostic.py   # 4096 config copy, immutability, truncation — mocked
```
Expect `ALL ORCHESTRATION TESTS PASS` and `ALL BUDGET-DIAGNOSTIC TESTS PASS`.

## Notes
- The shared model/decoding config (`qwen3:4b`, temperature 0.0, top_p 1.0,
  `num_predict=2048`, `think=True`, 600 s request timeout) is identical for all
  three agents — do not change it; the comparison depends on it. The smoke test
  uses a short-timeout, small-`num_predict` **copy** (`dataclasses.replace`) and
  never mutates the shared config.
- This is a **pilot** (12 records). It surfaces formatting/execution/truncation/
  diversity behavior; it is not a benchmark measurement.
- Re-running appends a new `run_id`; nothing is overwritten.
- Exit codes from `run_pilot.py`: `0` clean · `2` preflight failed (no records
  written) · `3` fail-fast abort / unhealthy run (partial results preserved).
