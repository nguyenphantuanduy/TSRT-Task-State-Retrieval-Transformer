from datasets import Dataset, DatasetDict

from HotpotQA_Distractor.data.load_data import (
    load_hotpotqa,
)

from HotpotQA_Distractor.data.teacher_forcing import (
    teacher_labeling,
)


HF_DATASET_REPO = (
    "nguyenphantuanduy/TSRT-HotpotQA-Teacher"
)


def build_teacher_dataset():

    print(
        "Loading original HotpotQA..."
    )

    original_dataset = load_hotpotqa(
        "distractor"
    )


    generated_splits = {}


    for mode in [
        "train",
        "validation",
    ]:

        print()
        print("=" * 100)
        print(
            f"Generating teacher labels for {mode}"
        )
        print("=" * 100)


        teacher_records = []

        original_keys = set(
            original_dataset[mode].features.keys()
        )


        for idx, item in enumerate(
            teacher_labeling(
                mode=mode
            )
        ):

            sample = dict(
                item["raw_sample"]
            )

            # check không mất field gốc
            assert set(sample.keys()) == original_keys, (
                f"Schema changed before adding teacher_answer\n"
                f"Expected: {original_keys}\n"
                f"Got: {sample.keys()}"
            )


            sample["teacher_answer"] = (
                item["teacher_text"]
            )


            assert set(sample.keys()) == (
                original_keys | {"teacher_answer"}
            )


            teacher_records.append(
                sample
            )


            if (idx + 1) % 100 == 0:
                print(
                    f"{mode}: Generated {idx+1}"
                )

        assert len(teacher_records) == len(original_dataset[mode]), (
            f"{mode} size mismatch: "
            f"{len(teacher_records)} vs "
            f"{len(original_dataset[mode])}"
        )

        generated_splits[mode] = Dataset.from_list(
            teacher_records
        )


    return DatasetDict(
        generated_splits
    )



def upload_dataset(dataset):

    print(
        f"Uploading to {HF_DATASET_REPO}"
    )


    dataset.push_to_hub(
        HF_DATASET_REPO,
        private=False,
    )


    print(
        "Upload completed."
    )



def main():

    dataset = build_teacher_dataset()


    print()
    print(dataset)


    print()
    print(
        dataset["train"][0].keys()
    )


    upload_dataset(
        dataset
    )



if __name__ == "__main__":

    main()