import torch

from datasets import load_dataset
from transformers import AutoTokenizer

from utils.utils import (
    build_tsrt_question_answer_batch,
)


MODEL_NAME = "Qwen/Qwen3-1.7B"

BATCH_SIZE = 4


def test_build_tsrt_question_answer_batch():

    # =====================================================
    # TOKENIZER
    # =====================================================

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # =====================================================
    # LOAD DATASET STREAMING
    # =====================================================

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


    # =====================================================
    # BUILD EMPTY BATCH
    # =====================================================

    batch = {}


    batch = build_tsrt_question_answer_batch(
        samples=samples,
        tokenizer=tokenizer,
        batch=batch,
    )


    # =====================================================
    # KEYS
    # =====================================================

    print("=" * 80)

    print("Batch keys")

    print(batch.keys())


    # =====================================================
    # SHAPES
    # =====================================================

    print("\nShapes")

    for k, v in batch.items():

        if isinstance(v, torch.Tensor):

            print(
                f"{k:<30}",
                tuple(v.shape)
            )


    # =====================================================
    # SAMPLE 0 DECODE
    # =====================================================

    idx = 0

    print("\n" + "=" * 80)

    print("Decoded sample 0")

    text = tokenizer.decode(
        batch["input_ids"][idx],
        skip_special_tokens=False,
    )

    print(text)


    # =====================================================
    # QUESTION MASK
    # =====================================================

    print("\nQuestion mask")

    print(
        batch["question_mask"][idx]
    )


    # =====================================================
    # LABEL CHECK
    # =====================================================

    print("\nLabels")

    labels = batch["labels"][idx]

    print(
        "Ignore tokens:",
        (labels == -100).sum().item()
    )

    print(
        "Train tokens:",
        (labels != -100).sum().item()
    )


    # =====================================================
    # RETRIEVAL LABEL CHECK
    # =====================================================

    print("\nRetrieval decision labels")

    retrieval_labels = (
        batch["retrieval_decision_labels"][idx]
    )

    print(retrieval_labels)


    print("\nRetrieval positions")

    positions = torch.where(
        retrieval_labels == 1
    )[0]


    print(positions)


    # =====================================================
    # TOKEN LEVEL INSPECTION
    # =====================================================

    print("\nToken level")

    input_ids = batch["input_ids"][idx]


    for i, token_id in enumerate(input_ids):

        if batch["attention_mask"][idx][i] == 0:
            break


        token = tokenizer.decode(
            [token_id]
        )


        q = batch["question_mask"][idx][i].item()

        r = retrieval_labels[i].item()

        print(
            f"{i:4d}",
            f"Q={q}",
            f"R={r}",
            repr(token)
        )


    # =====================================================
    # ASSERTIONS
    # =====================================================

    print("\nAssertions")


    assert (
        batch["input_ids"].shape
        ==
        batch["attention_mask"].shape
    )


    assert (
        batch["input_ids"].shape
        ==
        batch["labels"].shape
    )


    assert (
        batch["input_ids"].shape
        ==
        batch["retrieval_decision_labels"].shape
    )


    # padding retrieval = -1

    padding_positions = (
        batch["attention_mask"] == 0
    )


    assert torch.all(
        batch["retrieval_decision_labels"]
        [padding_positions]
        ==
        -1
    )


    # có ít nhất 1 retrieval point

    assert torch.all(
        (
            batch["retrieval_decision_labels"]
            ==
            1
        ).sum(dim=1)
        >
        0
    )


    print("\nALL TEST PASSED")


if __name__ == "__main__":

    test_build_tsrt_question_answer_batch()