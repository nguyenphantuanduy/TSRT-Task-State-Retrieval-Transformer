from datasets import Dataset, DatasetDict
import os
import json

from HotpotQA_Distractor.data.load_data import (
    load_hotpotqa,
)

from HotpotQA_Distractor.data.teacher_forcing import (
    teacher_labeling,
)


HF_DATASET_REPO = (
    "nguyenphantuanduy/TSRT-HotpotQA-Teacher"
)

CHUNK_SIZE = 1000

SAVE_DIR = "teacher_chunks"


def build_split(mode):

    print()
    print("=" * 100)
    print(
        f"Generating teacher labels for {mode}"
    )
    print("=" * 100)


    os.makedirs(
        SAVE_DIR,
        exist_ok=True
    )


    original_dataset = load_hotpotqa(
        "distractor"
    )

    original_keys = set(
        original_dataset[mode].features.keys()
    )


    buffer = []

    chunk_idx = 0

    chunk_files = []


    for idx, item in enumerate(
        teacher_labeling(
            mode=mode
        )
    ):

        sample = dict(
            item["raw_sample"]
        )


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


        buffer.append(
            sample
        )


        if (idx + 1) % 100 == 0:
            print(
                f"{mode}: Generated {idx+1}"
            )


        if len(buffer) >= CHUNK_SIZE:

            file_path = os.path.join(
                SAVE_DIR,
                f"{mode}_{chunk_idx:05d}.jsonl"
            )


            with open(
                file_path,
                "w",
                encoding="utf-8",
            ) as f:

                for row in buffer:
                    f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )


            chunk_files.append(
                file_path
            )


            print(
                f"Saved {file_path}"
            )


            buffer.clear()

            chunk_idx += 1



    # save phần còn lại
    if len(buffer) > 0:

        file_path = os.path.join(
            SAVE_DIR,
            f"{mode}_{chunk_idx:05d}.jsonl"
        )


        with open(
            file_path,
            "w",
            encoding="utf-8",
        ) as f:

            for row in buffer:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )


        chunk_files.append(
            file_path
        )


    print(
        f"{mode}: total chunks = {len(chunk_files)}"
    )


    return chunk_files



def load_chunks(files):

    records = []

    for file in files:

        print(
            f"Loading {file}"
        )

        with open(
            file,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:
                records.append(
                    json.loads(line)
                )


    return Dataset.from_list(
        records
    )



def build_teacher_dataset():

    train_files = build_split(
        "train"
    )

    val_files = build_split(
        "validation"
    )


    print(
        "Loading all chunks..."
    )


    dataset = DatasetDict(
        {
            "train": load_chunks(
                train_files
            ),

            "validation": load_chunks(
                val_files
            ),
        }
    )


    return dataset



def main():

    dataset = build_teacher_dataset()


    print(dataset)

    print(
        dataset["train"][0].keys()
    )


    dataset.push_to_hub(
        HF_DATASET_REPO,
        private=False,
    )


    print(
        "Upload completed."
    )


if __name__ == "__main__":
    main()