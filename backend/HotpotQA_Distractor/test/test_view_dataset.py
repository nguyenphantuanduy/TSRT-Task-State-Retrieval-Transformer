from datasets import load_dataset
from pprint import pprint


def inspect_tsrt_hotpotqa_teacher(
    split: str = "train",
    num_samples: int = 3,
):
    dataset = load_dataset(
        "nguyenphantuanduy/TSRT-HotpotQA-Teacher",
        split=split,
        streaming=True,
    )

    print(f"Dataset: {dataset}\n")

    for i, sample in enumerate(dataset):
        print("=" * 80)
        print(f"Sample {i}")

        for key, value in sample.items():
            print(f"\n[{key}]")
            pprint(value)

        print()

        if i + 1 >= num_samples:
            break

inspect_tsrt_hotpotqa_teacher()