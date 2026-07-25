# test.py

from datasets import load_dataset


def load_tsrt_hotpotqa_teacher(
    split="train",
):
    dataset = load_dataset(
        "nguyenphantuanduy/TSRT-HotpotQA-Teacher",
        split=split,
        streaming=True,
    )

    return dataset


def main():

    dataset = load_tsrt_hotpotqa_teacher()

    # lấy sample đầu tiên
    sample = next(iter(dataset))

    print("=" * 80)
    print("KEYS")
    print("=" * 80)

    print(sample.keys())

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)

    print(sample["question"])

    print("\n" + "=" * 80)
    print("TEACHER ANSWER")
    print("=" * 80)

    print(sample["teacher_answer"])

    print("\n" + "=" * 80)
    print("CHECK Answer:")
    print("=" * 80)

    answer_text = sample["teacher_answer"]

    pos = answer_text.find("Answer:")

    print("Answer char position:", pos)

    if pos != -1:
        print("Before Answer:")
        print(answer_text[:pos])

        print("\nAfter Answer:")
        print(answer_text[pos:])


if __name__ == "__main__":
    main()