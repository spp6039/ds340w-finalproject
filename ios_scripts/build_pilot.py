import json
import random
from pathlib import Path

SEED = 340
PILOT_SIZE = 10
NUM_EXAMPLES = 4

with Path("ios_data/train.jsonl").open() as f:
    train = [json.loads(line) for line in f]

template = {
    "premise": {
        "instruction": (
            "In each example, you will be given some context and a claim, where the correctness"
            " of the claim is affected by some ambiguity in the context. Enumerate two or three"
            " interpretations of the context that lead to different judgments about the claim."
        ),
        "bridge": "We don't know, because the context can be interpreted in many different ways:\n",
    },
    "hypothesis": {
        "instruction": (
            "In each example, you will be given some context and a claim. Unfortunately, the"
            " claim has some ambiguity that affects whether it is correct. Enumerate two or"
            " three interpretations of the claim that lead to different judgments about its correctness."
        ),
        "bridge": "We don't know, because the claim can be interpreted in many different ways:\n",
    },
}

verbalizer = {
    "entailment": "Then the claim is true.",
    "neutral": "Then the claim is inconclusive.",
    "contradiction": "Then the claim is false.",
}
question = "Given the context alone, is this claim true, false, or inconclusive?\n"

pilot = []
for target in train[:PILOT_SIZE]:
    key = "premise" if target["premise_ambiguous"] else "hypothesis"

    # Demonstrations come from train and exclude the target itself.
    pool = [
        row for row in train
        if row["id"] != target["id"]
        and row["premise_ambiguous"] == target["premise_ambiguous"]
    ]
    examples = random.Random(f"{SEED}:{target['id']}").sample(pool, NUM_EXAMPLES)

    prompt = template[key]["instruction"] + "\n\n"
    for row in examples:
        prompt += (
            f'Context: {row["premise"]}\n'
            f'Claim: {row["hypothesis"]} {question}{template[key]["bridge"]}'
        )
        for number, d in enumerate(row["disambiguations"], 1):
            prompt += f'{number}. {d[key]} {verbalizer[d["label"]]}\n'
        prompt += "\n"

    prompt += (
        f'Context: {target["premise"]}\n'
        f'Claim: {target["hypothesis"]} {question}{template[key]["bridge"]}'
    )

    pilot.append({
        "id": target["id"],
        "ambiguous_sentence_key": key,
        "demonstration_ids": [row["id"] for row in examples],
        "prompt": prompt,
        "gold_disambiguations": target["disambiguations"],
    })

output = Path("ios_results/pilot_prompts.jsonl")
with output.open("w") as f:
    for item in pilot:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(pilot)} prompts to {output}")
print("Target IDs:", [item["id"] for item in pilot])
