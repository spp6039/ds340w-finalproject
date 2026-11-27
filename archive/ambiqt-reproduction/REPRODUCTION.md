# AmbiQT Baseline Reproduction

## DS 340W Final Project

This folder contains our reproduction of the AmbiQT
column-synonyms benchmark evaluation.

Original repository:
https://github.com/tomartushar/AmbiQT

## Objective

We reproduced the evaluation pipeline using the authors'
saved LogicalBeam predictions to establish a baseline
for our ambiguity-aware text-to-SQL research.

## Dataset

- Benchmark: AmbiQT column-synonyms
- Split: Validation
- Number of examples: 1,240
- Predictions: Authors' saved LogicalBeam outputs
- Evaluation: Top-5

## Running the Evaluation

Set up the Python dependencies and obtain the original
AmbiQT SQLite database files.

The extracted databases must be available under:
db-content/database/

From the root of this reproduction folder, run:

    ./scripts/run_col_synonyms.sh

## Reproduced Results

Direct Execution Match (non-test-suite):

| Metric | Accuracy |
|---|---:|
| Top-1 Original Query | 8.31% |
| Top-5 Original Query | 20.48% |
| Top-5 Alternative 1 | 53.55% |
| Top-5 Alternative 2 | 39.44% |
| Top-5 Either Alternative | 65.56% |
| Top-5 Both Alternatives | 27.42% |

The evaluation completed successfully on all 1,240 examples.

The original evaluation script also reported 100%
for exact-set and execution-match metrics.
These values require further verification.

## Reproduction Scope

We reproduced the evaluation of the authors' saved
predictions. We did not independently regenerate
predictions using the original model.

## Planned Extension

Our research will compare two approaches:

Agent A: Direct SQL generation.

Agent B: Explicit interpretation identification
before SQL generation.

We plan to evaluate whether identifying alternative
interpretations improves the reliability of text-to-SQL
agents.
