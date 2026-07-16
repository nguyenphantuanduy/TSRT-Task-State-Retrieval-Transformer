from datasets import load_dataset
import re

HF_DATASET_REPO = (
    "nguyenphantuanduy/TSRT-HotpotQA-Teacher"
)

MAX_EXAMPLES = 5


def is_answer_only(text):
    """
    Chỉ chứa:

    Answer: xxx

    hoặc

    Answer:
    xxx

    và không có reasoning phía trước.
    """

    if text is None:
        return False

    text = text.strip()

    return re.fullmatch(
        r"\s*Answer:\s*.*",
        text,
        flags=re.DOTALL,
    ) is not None


def analyze_split(dataset, split_name):

    total = len(dataset)

    empty_count = 0
    answer_only_count = 0
    user_answer_says_count = 0

    answer_only_examples = []
    user_answer_says_examples = []

    for idx, sample in enumerate(dataset):

        teacher = sample["teacher_answer"]

        if teacher is None:
            teacher = ""

        teacher = teacher.strip()

        # empty
        if len(teacher) == 0:
            empty_count += 1

        # Answer: ...
        if is_answer_only(teacher):

            answer_only_count += 1

            if len(answer_only_examples) < MAX_EXAMPLES:

                answer_only_examples.append(
                    (
                        idx,
                        sample["question"],
                        teacher,
                    )
                )

        # But the user's answer says ...
        if (
            "but the user's answer says"
            in teacher.lower()
        ):

            user_answer_says_count += 1

            if len(user_answer_says_examples) < MAX_EXAMPLES:

                user_answer_says_examples.append(
                    (
                        idx,
                        sample["question"],
                        teacher[:1500],
                    )
                )

    print()
    print("=" * 100)
    print(split_name.upper())
    print("=" * 100)

    print(f"Total samples: {total}")

    print()

    print(
        f"Empty teacher_answer: "
        f"{empty_count} "
        f"({100 * empty_count / total:.4f}%)"
    )

    print(
        f"Answer-only: "
        f"{answer_only_count} "
        f"({100 * answer_only_count / total:.4f}%)"
    )

    print(
        f"Contains 'But the user's answer says': "
        f"{user_answer_says_count} "
        f"({100 * user_answer_says_count / total:.4f}%)"
    )

    print()
    print("-" * 100)
    print("Answer-only examples")
    print("-" * 100)

    if len(answer_only_examples) == 0:
        print("None")

    for idx, question, teacher in answer_only_examples:

        print()
        print(f"[{idx}]")
        print("Question:")
        print(question)

        print("\nTeacher:")
        print(teacher)

    print()
    print("-" * 100)
    print("User-answer-says examples")
    print("-" * 100)

    if len(user_answer_says_examples) == 0:
        print("None")

    for idx, question, teacher in user_answer_says_examples:

        print()
        print(f"[{idx}]")
        print("Question:")
        print(question)

        print("\nTeacher:")
        print(teacher)

    print()
    print("=" * 100)
    print(
        f"{split_name.upper()} SUMMARY"
    )
    print("=" * 100)

    print(
        f"Empty ratio      : "
        f"{100 * empty_count / total:.4f}%"
    )

    print(
        f"Answer-only ratio: "
        f"{100 * answer_only_count / total:.4f}%"
    )

    print(
        f"User-answer ratio: "
        f"{100 * user_answer_says_count / total:.4f}%"
    )


def main():

    print("Loading dataset...")

    dataset = load_dataset(
        HF_DATASET_REPO
    )

    analyze_split(
        dataset["train"],
        "train",
    )

    analyze_split(
        dataset["validation"],
        "validation",
    )


if __name__ == "__main__":
    main()