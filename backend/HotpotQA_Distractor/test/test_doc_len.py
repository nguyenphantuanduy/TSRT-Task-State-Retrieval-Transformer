from __future__ import annotations

import numpy as np
from tqdm import tqdm

from transformers import AutoTokenizer

from HotpotQA_Distractor.data.load_data import (
    load_tsrt_hotpotqa_teacher,
)

MODEL_NAME = "nguyenphantuanduy/TSRT-Qwen3-1.7B"


def main():

    print("Loading dataset...")
    dataset = load_tsrt_hotpotqa_teacher()
    train_dataset = dataset["train"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    document_lengths = []
    documents_per_sample = []

    for sample in tqdm(train_dataset):

        titles = sample["context"]["title"]
        paragraphs = sample["context"]["sentences"]

        documents_per_sample.append(len(paragraphs))

        for title, sentences in zip(titles, paragraphs):

            # giống cách build document của HotpotQA
            document = title + "\n" + "\n".join(sentences)

            token_ids = tokenizer(
                document,
                add_special_tokens=False,
            )["input_ids"]

            document_lengths.append(len(token_ids))

    document_lengths = np.asarray(document_lengths)

    print("=" * 60)
    print(f"Total samples         : {len(train_dataset)}")
    print(f"Total documents       : {len(document_lengths)}")
    print(f"Docs / sample (mean)  : {np.mean(documents_per_sample):.2f}")

    print()

    print(f"Mean length           : {document_lengths.mean():.2f}")
    print(f"Std                   : {document_lengths.std():.2f}")
    print(f"Min                   : {document_lengths.min()}")
    print(f"Max                   : {document_lengths.max()}")

    for p in [50, 75, 90, 95, 99]:
        print(
            f"P{p:<2}                   : "
            f"{np.percentile(document_lengths, p):.0f}"
        )


if __name__ == "__main__":
    main()