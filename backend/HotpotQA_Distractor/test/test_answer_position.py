from datasets import load_dataset
from transformers import AutoTokenizer

from HotpotQA_Distractor.collator import (
    build_tsrt_question_answer_batch,
)


def main():

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-1.7B"
    )

    dataset = load_dataset(
        "nguyenphantuanduy/TSRT-HotpotQA-Teacher",
        split="train",
        streaming=True,
    )

    sample = next(iter(dataset))

    batch = {}

    batch = build_tsrt_question_answer_batch(
        samples=[sample],
        tokenizer=tokenizer,
        batch=batch,
    )

    input_ids = batch["input_ids"][0]
    answer_position = batch["answer_position"][0].item()

    print("=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(sample["question"])

    print()

    print("=" * 80)
    print("TEACHER ANSWER")
    print("=" * 80)
    print(sample["teacher_answer"])

    print()

    print("=" * 80)
    print("ANSWER POSITION")
    print("=" * 80)
    print(answer_position)

    print()

    before = tokenizer.decode(
        input_ids[
            max(0, answer_position - 20):answer_position
        ],
        skip_special_tokens=False,
    )

    after = tokenizer.decode(
        input_ids[
            answer_position:answer_position + 30
        ],
        skip_special_tokens=False,
    )

    print("=" * 80)
    print("20 TOKENS BEFORE")
    print("=" * 80)
    print(before)

    print()

    print("=" * 80)
    print("FROM ANSWER POSITION")
    print("=" * 80)
    print(after)

    print()

    print("=" * 80)
    print("FULL CONTEXT")
    print("=" * 80)

    tokens = [
        tokenizer.decode([token])
        for token in input_ids.tolist()
    ]

    tokens.insert(
        answer_position,
        "\n<<<< ANSWER_POSITION >>>>\n",
    )

    print("".join(tokens))


if __name__ == "__main__":
    main()