from __future__ import annotations

import torch
from transformers import AutoTokenizer

from ..retriever_collator import TSRTRetrieverCollator
from ..data.load_data import load_tsrt_hotpotqa_teacher


MODEL_NAME = "tsrt-lab/TSRT-Qwen3-1.7B"


def main():

    # =====================================================
    # Dataset
    # =====================================================

    print("Loading dataset...")

    dataset = load_tsrt_hotpotqa_teacher()

    train_dataset = dataset["train"].shuffle(seed=42)

    # Sample thứ 6 = index 5
    sample = train_dataset[185]

    print("\n" + "=" * 80)
    print("RAW SAMPLE #6")
    print("=" * 80)

    for key, value in sample.items():
        print(f"\n[{key}]")
        print(value)

    # =====================================================
    # Tokenizer
    # =====================================================

    print("\n" + "=" * 80)
    print("Loading tokenizer...")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================
    # Collator
    # =====================================================

    collator = TSRTRetrieverCollator(
        tokenizer=tokenizer,
        document_max_length=512,
    )

    # =====================================================
    # Run sample through collator
    # =====================================================

    print("\n" + "=" * 80)
    print("RUNNING SAMPLE #6 THROUGH COLLATOR")
    print("=" * 80)

    batch = collator([sample])

    # =====================================================
    # Print outputs
    # =====================================================

    print("\n" + "=" * 80)
    print("COLLATOR OUTPUT")
    print("=" * 80)

    for key, value in batch.items():

        print(f"\n{'-' * 80}")
        print(f"KEY: {key}")
        print(f"TYPE: {type(value)}")

        if isinstance(value, torch.Tensor):

            print(f"SHAPE: {tuple(value.shape)}")
            print(f"DTYPE: {value.dtype}")

            print("VALUE:")

            if value.numel() < 200:
                print(value)
            else:
                print(value)

        else:
            print("VALUE:")
            print(value)

    # =====================================================
    # Decode tokenized inputs
    # =====================================================

    print("\n" + "=" * 80)
    print("DECODED INPUTS")
    print("=" * 80)

    for key, value in batch.items():

        if not isinstance(value, torch.Tensor):
            continue

        if value.dtype not in (
            torch.int64,
            torch.int32,
            torch.long,
        ):
            continue

        if "input_ids" not in key:
            continue

        print("\n" + "-" * 80)
        print(f"{key}")
        print("-" * 80)

        ids = value[0]

        print(tokenizer.decode(
            ids,
            skip_special_tokens=False,
        ))

    # =====================================================
    # Basic tensor diagnostics
    # =====================================================

    print("\n" + "=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)

    for key, value in batch.items():

        if isinstance(value, torch.Tensor):

            print(
                f"{key:35s} "
                f"shape={str(tuple(value.shape)):20s} "
                f"dtype={str(value.dtype):15s}"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()