from datasets import load_dataset


def load_hotpotqa(config="distractor"):
    """
    Load HotpotQA dataset using streaming mode.

    Args:
        config (str): "distractor" or "fullwiki"

    Returns:
        dict[str, IterableDataset]
    """

    dataset = {}

    # ===== TRAIN =====
    dataset["train"] = load_dataset(
        "hotpotqa/hotpot_qa",
        config,
        split="train",
        streaming=True
    )

    # ===== VALIDATION =====
    dataset["validation"] = load_dataset(
        "hotpotqa/hotpot_qa",
        config,
        split="validation",
        streaming=True
    )

    # ===== TEST =====
    if config == "fullwiki":
        dataset["test"] = load_dataset(
            "hotpotqa/hotpot_qa",
            config,
            split="test",
            streaming=True
        )

    return dataset


def print_sample(sample):
    print("\n" + "=" * 80)

    print("\nQuestion:")
    print(sample["question"])

    print("\nAnswer:")
    print(sample["answer"])

    print("\nType:", sample["type"])
    print("Level:", sample["level"])

    support_titles = set(sample["supporting_facts"]["title"])

    print("\nSupporting Facts:")
    for title, sent_id in zip(
        sample["supporting_facts"]["title"],
        sample["supporting_facts"]["sent_id"]
    ):
        print(f"  - {title} | sentence {sent_id}")

    print("\nDocuments:")
    print("-" * 80)

    for idx, (title, sentences) in enumerate(
        zip(
            sample["context"]["title"],
            sample["context"]["sentences"]
        )
    ):
        label = "POS" if title in support_titles else "NEG"

        print(f"\n[{idx}] [{label}] {title}")
        print("-" * 40)

        doc_text = " ".join(sentences)
        print(doc_text)

    print("\n" + "=" * 80)


def main():
    dataset = load_hotpotqa(config="distractor")

    for split_name in dataset.keys():
        print(f"\n{'=' * 20}")
        print(f"{split_name.upper()} SAMPLE")
        print(f"{'=' * 20}")

        sample = next(iter(dataset[split_name]))

        print("\nKeys:")
        print(sample.keys())

        print_sample(sample)


if __name__ == "__main__":
    main()