from transformers import AutoTokenizer
from itertools import islice

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3-1.7B"
)

dataset = load_hotpotqa("distractor")

lengths = []

for sample in islice(dataset["train"], 1000):

    text = build_input(sample)

    n_tokens = len(
        tokenizer(
            text,
            add_special_tokens=False
        )["input_ids"]
    )

    lengths.append(n_tokens)

print(f"avg   : {sum(lengths)/len(lengths):.1f}")
print(f"max   : {max(lengths)}")
print(f"min   : {min(lengths)}")

lengths = sorted(lengths)

print(f"p90   : {lengths[int(0.9*len(lengths))]}")
print(f"p95   : {lengths[int(0.95*len(lengths))]}")
print(f"p99   : {lengths[int(0.99*len(lengths))]}")