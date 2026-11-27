import json
from pathlib import Path

from evaluation.edit_f1 import get_edit_f1

VERBALIZERS = {
    "entailment": "Then the claim is true.",
    "neutral": "Then the claim is inconclusive.",
    "contradiction": "Then the claim is false.",
}

def parse_response(text):
    """Parse the numbered lines expected by the parent paper."""
    lines = text.strip().splitlines()
    predictions = {}

    for line in lines:
        line = line.strip()
        if not line.startswith(("1. ", "2. ", "3. ")):
            continue

        content = line[3:].strip()
        for label, ending in VERBALIZERS.items():
            if content.endswith(ending):
                predictions[label] = content[:-len(ending)].strip()
                break

    return predictions

def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f]

prompts = read_jsonl("ios_results/pilot_prompts.jsonl")
response_path = Path("ios_results/pilot_responses.jsonl")

if not response_path.exists():
    raise SystemExit(
        "No responses yet. Expected ios_results/pilot_responses.jsonl; "
        "we'll create it when we connect a model."
    )

responses = {str(r["id"]): r for r in read_jsonl(response_path)}
results = []

for item in prompts:
    response = responses.get(str(item["id"]))
    if response is None:
        raise SystemExit(f"Missing response for example {item['id']}")

    key = item["ambiguous_sentence_key"]
    original = next(
        # The ambiguous original appears in the final prompt's Context or Claim.
        line.split(": ", 1)[1]
        for line in item["prompt"].splitlines()[-3:]
        if line.startswith("Context: " if key == "premise" else "Claim: ")
    )
    gold = {
        d["label"]: d[key]
        for d in item["gold_disambiguations"]
    }
    predicted = parse_response(response["generation"])

    scores = [
        get_edit_f1(original, rewrite, predicted[label])
        if label in predicted else 0.0
        for label, rewrite in gold.items()
    ]
    results.append({
        "id": item["id"],
        "edit_f1": sum(scores) / len(scores),
        "predicted_rewrites": predicted,
    })

output = Path("ios_results/pilot_scores.jsonl")
with output.open("w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")

print(f"Scored {len(results)} responses")
print(f"Mean Edit F1: {sum(r['edit_f1'] for r in results) / len(results):.3f}")
print(f"Saved results to {output}")