from datasets import load_dataset

HF_DATASET_REPO = (
    "nguyenphantuanduy/TSRT-HotpotQA-Teacher"
)


def print_sample(sample, idx):
    print("\n" + "=" * 100)
    print(f"Sample {idx}")
    print("=" * 100)

    print("Question:")
    print(sample["question"])

    print("\nAnswer:")
    print(sample["answer"])

    print("\nTeacher Answer:")
    print(sample["teacher_answer"])

    print("\nKeys:")
    print(list(sample.keys()))


def main():

    print("Downloading dataset...")

    dataset = load_dataset(
        HF_DATASET_REPO
    )

    print("\nDataset:")
    print(dataset)

    print("\nTrain size:", len(dataset["train"]))
    print("Validation size:", len(dataset["validation"]))

    print("\n\nTRAIN SAMPLES")

    for i in range(min(3, len(dataset["train"]))):
        print_sample(
            dataset["train"][i],
            i,
        )

    print("\n\nVALIDATION SAMPLES")

    for i in range(min(3, len(dataset["validation"]))):
        print_sample(
            dataset["validation"][i],
            i,
        )


if __name__ == "__main__":
    main()