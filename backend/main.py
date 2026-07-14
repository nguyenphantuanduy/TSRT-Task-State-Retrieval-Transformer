# from transformers import AutoModelForCausalLM, AutoTokenizer

# MODEL_NAME = "Qwen/Qwen3-1.7B"
# CACHE_DIR = "weights/Qwen3-1.7B"

# tokenizer = AutoTokenizer.from_pretrained(
#     MODEL_NAME,
#     cache_dir=CACHE_DIR
# )

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     cache_dir=CACHE_DIR
# )

# import inspect
# from transformers import GenerationMixin

# print(
#     "stop_strings" in inspect.signature(
#         GenerationMixin.generate
#     ).parameters
# )

# import time
# from datasets import load_dataset

# DATASET_REPO = "nguyenphantuanduy/temp-dataset"

# # ==========================================================
# # DOWNLOAD
# # ==========================================================

# start = time.time()

# dataset = load_dataset(
#     "hotpotqa/hotpot_qa",
#     "distractor",
# )

# download_time = time.time() - start

# print(
#     f"Download + load: "
#     f"{download_time:.2f}s"
# )

# print(dataset)

# # ==========================================================
# # UPLOAD
# # ==========================================================

# start = time.time()

# dataset.push_to_hub(
#     DATASET_REPO,
#     private=False,
# )

# upload_time = time.time() - start

# print(
#     f"Upload time: "
#     f"{upload_time:.2f}s"
# )

# print(
#     f"Total time: "
#     f"{download_time + upload_time:.2f}s"
# )

import json

FILE_PATH = "train_00017.jsonl"


def load_jsonl(path):
    samples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    return samples


def main():
    data = load_jsonl(FILE_PATH)

    print("=" * 100)
    print(f"Total samples: {len(data)}")
    print("=" * 100)

    for i, sample in enumerate(data[:5]):
        print()
        print("#" * 100)
        print(f"SAMPLE {i}")
        print("#" * 100)

        print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()