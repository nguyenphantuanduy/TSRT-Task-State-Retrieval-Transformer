from __future__ import annotations

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

from HotpotQA_Distractor.data.load_data import load_tsrt_hotpotqa_teacher
from HotpotQA_Distractor.collator import TSRTDataCollator

MODEL_NAME = "nguyenphantuanduy/TSRT-Qwen3-1.7B"


def main():

    # =====================================================
    # DATASET
    # =====================================================

    print("Loading dataset...")

    dataset = load_tsrt_hotpotqa_teacher()

    train_dataset = dataset["train"]

    # =====================================================
    # TOKENIZER
    # =====================================================

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================
    # COLLATOR
    # =====================================================

    collator = TSRTDataCollator(
        tokenizer=tokenizer,
        document_max_length=512,
    )

    # =====================================================
    # STATISTICS
    # =====================================================

    doc_lengths = []

    for sample in tqdm(train_dataset):

        batch = collator([sample])

        # (1, D, L)
        document_ids = batch["document_ids"][0]

        # (1, D, L)
        document_attention_mask = batch["document_attention_mask"][0]

        num_docs = document_ids.size(0)

        for doc_idx in range(num_docs):

            length = (
                document_attention_mask[doc_idx]
                .sum()
                .item()
            )

            # bỏ document padding
            if length > 0:
                doc_lengths.append(length)

    doc_lengths = np.array(doc_lengths)

    print("=" * 60)
    print(f"Total documents : {len(doc_lengths)}")
    print(f"Mean            : {doc_lengths.mean():.2f}")
    print(f"Std             : {doc_lengths.std():.2f}")
    print(f"Min             : {doc_lengths.min()}")
    print(f"Max             : {doc_lengths.max()}")

    for p in [50, 75, 90, 95, 99]:
        print(f"P{p:<2}             : {np.percentile(doc_lengths, p):.0f}")


if __name__ == "__main__":
    main()