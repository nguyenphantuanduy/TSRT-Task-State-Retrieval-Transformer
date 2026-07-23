import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from utils.utils import (
    build_tsrt_question_answer_batch,
    build_tsrt_document_batch,
)


MODEL_NAME = "Qwen/Qwen3-1.7B"

BATCH_SIZE = 4


def main():

    # =====================================================
    # TOKENIZER
    # =====================================================

    print("=" * 80)
    print("LOAD TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # =====================================================
    # LOAD DATA
    # =====================================================

    print("=" * 80)
    print("LOAD DATASET")


    dataset = load_dataset(
        "nguyenphantuanduy/TSRT-HotpotQA-Teacher",
        split="train",
        streaming=True,
    )


    samples = []

    for sample in dataset:

        samples.append(sample)

        if len(samples) == BATCH_SIZE:
            break


    print(
        "Loaded samples:",
        len(samples)
    )


    # =====================================================
    # BUILD QA
    # =====================================================

    batch = {}


    batch = build_tsrt_question_answer_batch(
        samples=samples,
        tokenizer=tokenizer,
        batch=batch,
    )


    print("=" * 80)
    print("QA SHAPES")


    for k,v in batch.items():

        if isinstance(v, torch.Tensor):

            print(
                f"{k:<35}",
                tuple(v.shape)
            )


    # =====================================================
    # BUILD DOCUMENT
    # =====================================================

    print("=" * 80)
    print("BUILD DOCUMENT")


    batch = build_tsrt_document_batch(
        samples=samples,
        tokenizer=tokenizer,
        batch=batch,
        max_length=512,
    )


    print("\nDOCUMENT SHAPES")


    for k,v in batch.items():

        if isinstance(v, torch.Tensor):

            print(
                f"{k:<35}",
                tuple(v.shape)
            )


    # =====================================================
    # DOCUMENT COUNT INSPECTION
    # =====================================================

    print("=" * 80)
    print("DOCUMENT COUNT")


    document_ids = batch[
        "document_ids"
    ]

    document_attention_mask = batch[
        "document_padding_mask"
    ]


    B,D,L_doc = document_ids.shape


    print(
        "B:",
        B
    )

    print(
        "D(max docs):",
        D
    )

    print(
        "L_doc:",
        L_doc
    )


    # =====================================================
    # COUNT REAL DOCUMENTS
    # =====================================================

    print("\nReal document count per sample")


    for i in range(B):

        # document empty nếu attention toàn 0

        doc_mask = (
            document_attention_mask[i]
            .sum(dim=-1)
            > 0
        )


        print(
            f"sample {i}:",
            int(doc_mask.sum()),
            "/",
            D,
            "documents"
        )


    # =====================================================
    # USEFULNESS MATRIX
    # =====================================================

    print("=" * 80)
    print("USEFULNESS")


    usefulness = batch[
        "usefulness_score_matrix"
    ]


    print(
        "shape:",
        tuple(usefulness.shape)
    )


    for i in range(B):

        print(
            f"\nSample {i}"
        )

        print(
            usefulness[i]
            .unique(
                return_counts=True
            )
        )


    # =====================================================
    # CHECK DOCUMENT PADDING
    # =====================================================

    print("=" * 80)
    print("DOCUMENT PADDING CHECK")


    padding_docs = (
        document_attention_mask
        .sum(dim=-1)
        ==
        0
    )


    print(
        "Number of padded docs:",
        padding_docs.sum().item()
    )


    if padding_docs.sum() > 0:

        print(
            "Padding doc positions:"
        )

        print(
            torch.where(
                padding_docs
            )
        )

    else:

        print(
            "No document padding in this batch"
        )


    # =====================================================
    # DONE
    # =====================================================

    print("=" * 80)
    print("DONE")



if __name__ == "__main__":
    main()