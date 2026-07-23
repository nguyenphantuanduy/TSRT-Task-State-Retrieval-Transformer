from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from utils.utils import (
    build_tsrt_document_batch,
)


MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_LENGTH = 256
BATCH_SIZE = 4


def test_build_tsrt_document_batch():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    questions = [
        sample["question"]
        for sample in samples
    ]

    encoded = tokenizer(
        questions,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    batch = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

    batch = build_tsrt_document_batch(
        samples=samples,
        tokenizer=tokenizer,
        batch=batch,
        max_length=MAX_LENGTH,
    )

    print("=" * 80)
    print("Batch Keys")
    print(batch.keys())

    print("\nTensor Shapes")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:<28} {tuple(v.shape)}")

    print("\nDocument Padding Mask")
    print(batch["document_padding_mask"].int())

    print("\nUsefulness Score Matrix Shape")
    print(batch["usefulness_score_matrix"].shape)

    print("\nSample 0 usefulness labels")
    print(batch["usefulness_score_matrix"][0, 0])

    print("\n" + "=" * 80)
    print("Sample 0 Documents")

    doc_input_ids = batch["document_input_ids"][0]
    doc_attention_mask = batch["document_attention_mask"][0]
    labels = batch["usefulness_score_matrix"][0, 0]

    for i in range(doc_input_ids.shape[0]):

        if not batch["document_padding_mask"][0, i]:
            continue

        length = doc_attention_mask[i].sum().item()

        text = tokenizer.decode(
            doc_input_ids[i][:length],
            skip_special_tokens=True,
        )

    print(f"\nDocument {i}")
    print(f"Label: {labels[i].item()}")

    print("-" * 80)
    print(text[:800])      # chỉ in 800 ký tự đầu cho đỡ dài

    if len(text) > 800:
        print("...")

    print("\nPositive docs per sample")
    positive = (batch["usefulness_score_matrix"][:, 0] == 1).sum(dim=1)
    print(positive)

    print("\nNegative docs per sample")
    negative = (batch["usefulness_score_matrix"][:, 0] == 0).sum(dim=1)
    print(negative)

    print("\nPadding docs per sample")
    padding = (batch["usefulness_score_matrix"][:, 0] == -1).sum(dim=1)
    print(padding)


if __name__ == "__main__":
    test_build_tsrt_document_batch()