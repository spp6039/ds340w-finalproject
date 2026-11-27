# Evaluating Answer Sensitivity to Multiple Interpretations in Text-to-SQL Agents

**DS 340W, Group 28, Penn State, Fall 2026**
Sanjana Menon (ssm5808@psu.edu) · Sharvari Purighalla (spp6039@psu.edu)

We test whether **interpretation-first prompting** helps a Text-to-SQL agent recover more of the
valid readings of an ambiguous question than **plain multi-candidate generation**, under the same
model, candidate count, and token budget. We use 12 questions from the
[AmbiQT](https://github.com/testzer0/AmbiQT) benchmark (3 from each ambiguity category).

| Agent | What it does | Max SQL candidates |
|---|---|---|
| **A**: single baseline | Generates one SQL query | 1 |
| **A2**: multi baseline | Generates up to K SQL queries directly | K = 2 |
| **B**: interpretation-first | Detects ambiguity, lists natural-language readings, writes SQL for each (JSON output) | K = 2 |

All three agents share the same settings: **Qwen3:4B** via Ollama, temperature 0.0, top_p 1.0,
`num_predict = 8192`, `think = True`, 600 s timeout, and one LLM call per question.

---

## How to reproduce our results (≈ 45 minutes)

Everything runs in **Google Colab** on a free **T4 GPU**. Nothing needs to be installed on the
local computer, and no files need to be uploaded.

1. Open **<https://colab.research.google.com>** and sign in with any Google account.
2. **File → Open notebook → GitHub tab**, paste `spp6039/ds340w-finalproject`, and open
   **`DS340W_Demo.ipynb`**.
   *(Or download this repo as a ZIP from GitHub and use **File → Upload notebook**.)*
3. **Runtime → Change runtime type → T4 GPU → Save.**
4. Run the cells **one at a time, top to bottom**. Don't use *Run all*, so each output can be checked as it appears.

| Notebook step | What you should see |
|---|---|
| 2. Clone repo | `git clone` succeeds and lists `run_pilot.py` etc. |
| 3. Offline checks | `RESULT: ALL CHECKS PASS` plus two test suites that `PASS` |
| 4. Ollama + model | `ollama list` shows `qwen3:4b  359d7dd4bcda` |
| 5. Smoke test | `RESULT: SMOKE TEST PASS` |
| 6. Full run (~40 min) | 36 progress lines, then `RUN OK: 36/36 requests written, 0 request_error (num_predict=8192)` |
| 7. Compare to paper | The results table, plus a per-question comparison with the committed paper run |
| 8. Paper tables | The 4,096- and 8,192-token tables used in the paper |

### Results reported in the paper (8,192-token run `20261124T194002Z`)

| Agent | Completed | Any-alt coverage | Both-alt coverage | Alt-interp recall | Generated tokens |
|---|---|---|---|---|---|
| A | 12/12 | 8/12 | 0/12 | 8/24 (33%) | 28,425 |
| A2 | 12/12 | 8/12 | 3/12 | 11/24 (46%) | 39,188 |
| B | 11/12 (1 parse error) | 7/12 | 4/12 | 11/24 (46%) | 47,070 |

To print these tables without a GPU or model (any machine with Python 3.10+):

```bash
git clone https://github.com/spp6039/ds340w-finalproject.git
cd ds340w-finalproject/colab_pilot
pip install -r requirements.txt
python summarize_results.py ../results/pilot_generations_20261119T191703Z.jsonl \
                            ../results/pilot_generations_20261124T194002Z.jsonl
```

### A note on exact reproducibility
Decoding is greedy (temperature 0), and the prompts, data, and scoring are fixed. In our own runs, two
separate Colab sessions (Nov 19 and Nov 24) produced **identical SQL for every question that was not
cut off by the token limit**. LLM inference on GPUs is still not guaranteed to be bit-for-bit
deterministic across hardware or Ollama versions. For that reason we pin Ollama `v0.40.18`, print the model
digest, commit our raw outputs in `results/`, and Step 7 of the notebook reports any (agent, question)
pair whose outcome differs from the paper run.

---

## Repository layout

```
DS340W_Demo.ipynb            ← START HERE: the end-to-end Colab notebook
results/                     ← raw outputs of the two runs used in the paper
  pilot_generations_20261124T194002Z.jsonl   8,192-token final run (paper results)
  pilot_generations_20261119T191703Z.jsonl   4,096-token preliminary budget run
  pilot_meta_*.json                          exact settings for each run
colab_pilot/                 ← all experiment code
  run_pilot.py               runs A, A2, B on the 12 questions and scores every candidate
  summarize_results.py       turns a run file into the paper's metrics table
  prepare_and_verify.py      offline data/integrity checks (no model)
  smoke_test.py, ollama_health.py   Ollama readiness / model / smoke checks
  test_*.py                  unit tests (mocked model, no GPU)
  pilot_records.json         the 12 selected AmbiQT records and how they were chosen
  option_c/agents/           config.py (shared settings) · agent_a.py · agent_a2.py · agent_b.py · adapter.py
  option_c/eval/             golds.py (gold prep) · safe_execute.py · interpret.py (structural labels) · score.py
  option_c/compare.py        exact result comparison policy
  option_c/perturbed_dbs/    the 12 SQLite databases
  AmbiQT-master/.../validation.json   AmbiQT benchmark records (only the 12 used are filled in)
archive/                     ← earlier course milestones, NOT part of the final experiment
```

## How scoring works (`option_c/eval/score.py`)
A generated SQL candidate gets credit for gold interpretation A or B only if **all** of these hold:
1. it executes without error or timeout (10 s, read-only) on the question's SQLite database,
2. its result rows **exactly** equal the gold query's rows (row order checked only if the gold query has `ORDER BY`), and
3. its **structural label** (`interpret.py`, via `sqlglot`) is exactly that interpretation.

A matching result alone is not enough. *Alternative-interpretation recall* = gold alternatives
recovered ÷ (2 × 12), with each alternative counted once. *Both-alt coverage* = questions where two distinct
candidates recover A and B. Agents receive only the question and schema. Gold data is used only by
the scorer after generation (this is checked in `prepare_and_verify.py`, step 5).

## Limitations
This is a 12-question pilot drawn from AmbiQT's `validation.json` files, not a held-out benchmark
evaluation. Results on 12 questions are directional only.
