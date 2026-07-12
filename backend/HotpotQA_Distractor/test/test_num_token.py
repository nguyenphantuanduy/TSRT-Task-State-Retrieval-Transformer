from itertools import islice
from collections import defaultdict

from transformers import AutoTokenizer

from HotpotQA_Distractor.data.load_data import load_hotpotqa


MODEL_NAME = "Qwen/Qwen3-1.7B"
NUM_SAMPLES = 1000


def build_input(sample):

    docs = []

    for title, sentences in zip(
        sample["context"]["title"],
        sample["context"]["sentences"]
    ):
        docs.append(
            f"{title}\n{' '.join(sentences)}"
        )

    context = "\n\n".join(docs)

    text = (
        f"Question: {sample['question']}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer: {sample['answer']}"
    )

    return text


def percentile(sorted_values, p):

    if len(sorted_values) == 0:
        return 0

    idx = int(p * len(sorted_values))
    idx = min(idx, len(sorted_values) - 1)

    return sorted_values[idx]


def print_stats(name, lengths):

    if len(lengths) == 0:
        print(f"\n===== {name} =====")
        print("No samples.")
        return

    lengths = sorted(lengths)

    avg_len = sum(lengths) / len(lengths)

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")

    print(f"Samples : {len(lengths)}")
    print(f"Avg     : {avg_len:.1f}")
    print(f"Min     : {min(lengths)}")
    print(f"Max     : {max(lengths)}")
    print(f"P50     : {percentile(lengths, 0.50)}")
    print(f"P90     : {percentile(lengths, 0.90)}")
    print(f"P95     : {percentile(lengths, 0.95)}")
    print(f"P99     : {percentile(lengths, 0.99)}")

    print("\nTruncation estimate:")

    for max_length in [1024, 2048, 4096, 8192]:

        kept = sum(
            1 for x in lengths
            if x <= max_length
        )

        ratio = kept / len(lengths) * 100

        print(
            f"  MAX_LENGTH={max_length:<5}"
            f" -> {ratio:.2f}% kept"
        )


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    dataset = load_hotpotqa("distractor")

    stats = defaultdict(list)

    print(f"Analyzing first {NUM_SAMPLES} samples...")

    for sample in islice(dataset["train"], NUM_SAMPLES):

        text = build_input(sample)

        n_tokens = len(
            tokenizer(
                text,
                add_special_tokens=False
            )["input_ids"]
        )

        level = sample["level"].lower()

        stats["all"].append(n_tokens)
        stats[level].append(n_tokens)

    print_stats("ALL", stats["all"])
    print_stats("EASY", stats["easy"])
    print_stats("MEDIUM", stats["medium"])
    print_stats("HARD", stats["hard"])


if __name__ == "__main__":
    main()